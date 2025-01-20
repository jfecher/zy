//! parser/mod.rs - This file defines parsing, the second phase of the compiler.
//! The goal of parsing is to take the `Vec<Token>` output from the lexing phase
//! and validate the grammar/syntax of the program. If the syntax is invalid,
//! a parse error is printed out. Otherwise, the resulting Ast is returned and
//! the compiler moves onto the name resolution pass.
//!
//! This parser itself is built up from parser combinators. The basic combinators
//! (as well as the parser! macro) are defined in the parser/combinators.rs module.
//! These combinators backtrack by default though !<- can be used to prevent backtracking
//! to speed up parsing.
//!
//! This file makes heavy use of the parser! macro which combines parsers in a
//! sequence, threading the `input` parameter between each step, returning early if
//! there was an error, and handles getting the starting and end Locations for the
//! current parse rule, and union-ing them. This resulting Location for the whole
//! rule is accessible via the location/loc parameter.
#[macro_use]
mod combinators;
pub mod error;

#[macro_use]
pub mod ast;
mod desugar;
pub mod pretty_printer;

use std::{collections::HashSet, iter::FromIterator};

use crate::lexer::token::Token;
use crate::{error::location::Location, parser::ast::Mutability};
use ast::{Ast, Trait, Type, TypeDefinitionBody};
use combinators::*;
use error::{ParseError, ParseResult};

use self::ast::Sharedness;

type AstResult<'a, 'b> = ParseResult<'a, 'b, Ast<'b>>;

/// The entry point to parsing. Parses an entire file, printing any
/// error found, or returns the Ast if there was no error.
pub fn parse<'b>(input: Input<'_, 'b>) -> Result<Ast<'b>, ParseError<'b>> {
    parse_file(input)
}

/// A file is a sequence of statements, separated by newlines.
pub fn parse_file<'b>(input: Input<'_, 'b>) -> Result<Ast<'b>, ParseError<'b>> {
    let (input, _, _) = maybe_newline(input)?;
    let (input, ast, _) = statement_list(input)?;
    let (input, _, _) = maybe_newline(input)?;
    let _ = expect(Token::EndOfInput)(input)?;
    Ok(ast)
}

fn maybe_newline<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Option<Token>> {
    maybe(expect(Token::Newline))(input)
}

parser!(statement_list loc =
    first <- statement;
    rest <- many0(pair( expect(Token::Newline), statement ));
    if rest.is_empty() {
        first
    } else {
        let mut statements = Vec::with_capacity(rest.len() + 1);
        statements.push(first);
        for (_, b) in rest.into_iter() {
            statements.push(b);
        }
        Ast::sequence(statements, loc)
    }
);

fn statement<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::ParenthesisLeft | Token::Identifier(_) => or(&[definition, assignment, expression], "statement")(input),
        Token::Boxed => type_definition(input),
        Token::Type => or(&[type_definition, type_alias], "statement")(input),
        Token::Import => import(input),
        Token::Trait => trait_definition(input),
        Token::Effect => effect_definition(input),
        Token::Impl => trait_impl(input),
        Token::Return => return_expr(input),
        Token::Extern => parse_extern(input),
        _ => expression(input),
    }
}

fn definition<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    raw_definition(input).map(|(input, definition, location)| (input, Ast::Definition(definition), location))
}

fn raw_definition<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, ast::Definition<'b>> {
    or(&[function_definition, variable_definition], "definition")(input)
}

parser!(function_definition location -> 'b ast::Definition<'b> =
    name <- pattern_argument;
    args <- many1(pattern_argument);
    return_type <- maybe(function_return_type);
    effects <- maybe(effect_clause);
    _ <- expect(Token::Equal);
    body !<- block_or_statement;
    ast::Definition {
        pattern: Box::new(name),
        expr: Box::new(Ast::lambda(args, return_type, effects, body, location)),
        mutable: false,
        location,
        level: None,
        info: None,
        typ: None,
    }
);

parser!(effect_clause location -> 'b Vec<(String, Vec<Type<'b>>)> =
    _ <- expect(Token::Can);
    effects <- many1(effect);
    effects
);

parser!(effect location -> 'b (String, Vec<Type<'b>>) =
    name <- typename;
    args <- many0(basic_type);
    (name, args)
);

parser!(varargs location -> 'b () =
    _ <- expect(Token::Range);
    _ <- expect(Token::MemberAccess);
    ()
);

parser!(function_return_type location -> 'b ast::Type<'b> =
    _ <- expect(Token::Colon);
    typ <- parse_type;
    typ
);

parser!(variable_definition location -> 'b ast::Definition<'b> =
    name <- pattern;
    _ <- expect(Token::Equal);
    mutable <- maybe(expect(Token::Mut));
    expr !<- block_or_statement;
    ast::Definition {
        pattern: Box::new(name),
        expr: Box::new(expr),
        mutable: mutable.is_some(),
        location,
        level: None,
        info: None,
        typ: None,
    }
);

parser!(assignment location =
    lhs <- expression;
    _ <- expect(Token::Assignment);
    rhs !<- expression;
    Ast::assignment(lhs, rhs, location)
);

fn pattern<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    or(&[pattern_pair, type_annotation_pattern, pattern_function_call, pattern_argument], "pattern")(input)
}

// TODO: There's a lot of repeated parsing done in patterns due to or combinators
// being used to express the pair -> type annotation -> call -> argument  lattice.
parser!(pattern_pair loc =
    first <- or(&[type_annotation_pattern, pattern_function_call, pattern_argument], "pattern");
    _ <- expect(Token::Comma);
    rest !<- pattern;
    Ast::function_call(Ast::operator(Token::Comma, loc), vec![first, rest], loc)
);

parser!(type_annotation_pattern loc =
    lhs <- or(&[pattern_function_call, pattern_argument], "pattern");
    _ <- expect(Token::Colon);
    rhs !<- parse_type;
    Ast::type_annotation(lhs, rhs, loc)
);

fn parenthesized_irrefutable_pattern<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    parenthesized(or(&[operator, pattern], "pattern"))(input)
}

parser!(type_definition loc =
    boxed <- maybe(expect(Token::Boxed));
    _ <- expect(Token::Type);
    name <- typename;
    args <- many0(identifier);
    _ <- expect(Token::Equal);
    body <- type_definition_body;
    Ast::type_definition(boxed.is_some(), name, args, body, loc)
);

parser!(type_alias loc =
    _ <- expect(Token::Type);
    name <- typename;
    args <- many0(identifier);
    _ <- expect(Token::Equal);
    body <- parse_type;
    Ast::type_definition(false, name, args, TypeDefinitionBody::Alias(body), loc)
);

fn type_definition_body<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, ast::TypeDefinitionBody<'b>> {
    match input[0].0 {
        Token::Indent => or(&[union_block_body, struct_block_body], "type_definition_body")(input),
        Token::Pipe => union_inline_body(input),
        _ => struct_inline_body(input),
    }
}

parser!(union_variant loc -> 'b (String, Vec<Type<'b>>, Location<'b>) =
    _ <- expect(Token::Pipe);
    variant !<- typename;
    args !<- many0(basic_type);
    (variant, args, loc)
);

parser!(union_block_body _loc -> 'b ast::TypeDefinitionBody<'b> =
    _ <- expect(Token::Indent);
    variants <- delimited_trailing(union_variant, expect(Token::Newline), false);
    _ !<- expect(Token::Unindent);
    TypeDefinitionBody::Union(variants)
);

parser!(union_inline_body _loc -> 'b ast::TypeDefinitionBody<'b> =
    variants <- many1(union_variant);
    TypeDefinitionBody::Union(variants)
);

parser!(struct_field loc -> 'b (String, Type<'b>, Location<'b>) =
    field_name <- identifier;
    _ !<- expect(Token::Colon);
    field_type !<- parse_type_no_pair;
    (field_name, field_type, loc)
);

parser!(struct_block_body _loc -> 'b ast::TypeDefinitionBody<'b> =
    _ <- expect(Token::Indent);
    fields <- delimited_trailing(struct_field, expect(Token::Newline), false);
    _ !<- expect(Token::Unindent);
    TypeDefinitionBody::Struct(fields)
);

parser!(struct_inline_body _loc -> 'b ast::TypeDefinitionBody<'b> =
    fields <- delimited(struct_field, expect(Token::Comma));
    TypeDefinitionBody::Struct(fields)
);

parser!(import loc =
    _ <- expect(Token::Import);
    path !<- delimited_trailing(typename, expect(Token::MemberAccess), false);
    symbols !<- many0(imported_item);
    Ast::import(path, loc, HashSet::from_iter(symbols))
);

parser!(trait_definition loc =
    _ <- expect(Token::Trait);
    name !<- typename;
    args !<- many1(identifier);
    _ !<- maybe(expect(Token::RightArrow));
    fundeps !<- many0(identifier);
    body <- maybe(trait_body);
    Ast::trait_definition(name, args, fundeps, body.unwrap_or_default(), loc)
);

parser!(trait_body loc -> 'b Vec<ast::TypeAnnotation<'b>> =
    _ <- expect(Token::With);
    body <- or(&[trait_body_block, trait_body_single], "trait body");
    body
);

parser!(effect_definition loc =
    _ <- expect(Token::Effect);
    name !<- typename;
    args !<- many0(identifier);
    body <- trait_body;
    Ast::effect_definition(name, args, body, loc)
);

parser!(trait_body_single loc -> 'b Vec<ast::TypeAnnotation<'b>> =
    body <- declaration;
    vec![body]
);

parser!(trait_body_block loc -> 'b Vec<ast::TypeAnnotation<'b>> =
    _ <- expect(Token::Indent);
    body !<- delimited_trailing(declaration, expect(Token::Newline), false);
    _ !<- expect(Token::Unindent);
    body
);

parser!(declaration loc -> 'b ast::TypeAnnotation<'b> =
    lhs <- pattern_argument;
    _ <- expect(Token::Colon);
    rhs !<- parse_type;
    ast::TypeAnnotation { lhs: Box::new(lhs), rhs, location: loc, typ: None }
);

parser!(trait_impl loc =
    _ <- expect(Token::Impl);
    name !<- typename;
    args !<- many1(basic_type);
    given !<- maybe(given);
    definitions !<- maybe(impl_body);
    Ast::trait_impl(name, args, given.unwrap_or_default(), definitions.unwrap_or_default(), loc)
);

parser!(impl_body loc -> 'b Vec<ast::Definition<'b>> =
    _ <- maybe(expect(Token::Newline));
    _ <- expect(Token::With);
    definitions <- or(&[impl_body_block, impl_body_single], "impl body");
    definitions
);

parser!(impl_body_single loc -> 'b Vec<ast::Definition<'b>> =
    definition <- raw_definition;
    vec![definition]
);

parser!(impl_body_block loc -> 'b Vec<ast::Definition<'b>> =
    _ <- expect(Token::Indent);
    definitions !<- delimited_trailing(raw_definition, expect(Token::Newline), false);
    _ !<- expect(Token::Unindent);
    definitions
);

parser!(given loc -> 'b Vec<Trait<'b>> =
    _ <- expect(Token::Given);
    traits <- delimited(required_trait, expect(Token::Comma));
    traits
);

parser!(required_trait location -> 'b Trait<'b> =
    name <- typename;
    args <- many1(basic_type);
    Trait { name, args, location }
);

parser!(return_expr loc =
    _ <- expect(Token::Return);
    expr !<- expression;
    Ast::return_expr(expr, loc)
);

parser!(parse_extern loc =
    _ <- expect(Token::Extern);
    declarations <- or(&[extern_block, extern_single], "extern");
    Ast::extern_expr(declarations, loc)
);

parser!(extern_block _loc -> 'b Vec<ast::TypeAnnotation<'b>> =
    _ <- expect(Token::Indent);
    declarations !<- delimited_trailing(declaration, expect(Token::Newline), false);
    _ !<- expect(Token::Unindent);
    declarations
);

parser!(extern_single _loc -> 'b Vec<ast::TypeAnnotation<'b>> =
    declaration <- declaration;
    vec![declaration]
);

fn block_or_statement<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::Indent => block(input),
        _ => statement(input),
    }
}

parser!(block _loc =
    _ <- expect(Token::Indent);
    expr !<- statement_list;
    _ !<- maybe_newline;
    _ !<- expect(Token::Unindent);
    expr
);

/// Returns the precedence of an operator along with
/// whether or not it is right-associative.
/// Returns None if the given Token is not an operator
fn precedence(token: &Token) -> Option<(i8, bool)> {
    match token {
        Token::Semicolon => Some((0, false)),
        Token::ApplyRight => Some((1, false)),
        Token::ApplyLeft => Some((2, true)),
        Token::Comma => Some((3, true)),
        Token::Or => Some((4, false)),
        Token::And => Some((5, false)),
        Token::EqualEqual
        | Token::NotEqual
        | Token::GreaterThan
        | Token::LessThan
        | Token::GreaterThanOrEqual
        | Token::LessThanOrEqual => Some((7, false)),
        Token::In => Some((8, false)),
        Token::Append => Some((9, false)),
        Token::Range => Some((10, false)),
        Token::Else => Some((11, false)),
        Token::Add | Token::Subtract => Some((12, false)),
        Token::Multiply | Token::Divide | Token::Modulus => Some((13, false)),
        Token::Index => Some((14, false)),
        Token::As => Some((15, false)),
        _ => None,
    }
}

/// Should we push this operator onto our operator stack and keep parsing our expression?
/// This handles the operator precedence and associativity parts of the shunting-yard algorithm.
fn should_continue(operator_on_stack: &Token, r_prec: i8, r_is_right_assoc: bool) -> bool {
    let (l_prec, _) = precedence(operator_on_stack).unwrap();

    l_prec > r_prec || (l_prec == r_prec && !r_is_right_assoc)
}

fn pop_operator<'c>(operator_stack: &mut Vec<&Token>, results: &mut Vec<(Ast<'c>, Location<'c>)>) {
    let (rhs, rhs_location) = results.pop().unwrap();
    let (lhs, lhs_location) = results.pop().unwrap();
    let location = lhs_location.union(rhs_location);
    let operator = operator_stack.pop().unwrap().clone();
    let call = desugar::desugar_operators(operator, lhs, rhs, location);
    results.push((call, location));
}

/// Parse an arbitrary expression using the shunting-yard algorithm
fn expression<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    let (mut input, value, location) = term(input)?;

    let mut operator_stack = vec![];
    let mut results = vec![(value, location)];

    // loop while the next token is an operator
    while let Some((prec, right_associative)) = precedence(&input[0].0) {
        while !operator_stack.is_empty()
            && should_continue(operator_stack[operator_stack.len() - 1], prec, right_associative)
        {
            pop_operator(&mut operator_stack, &mut results);
        }

        operator_stack.push(&input[0].0);
        input = &input[1..];

        let (new_input, value, location) = no_backtracking(term)(input)?;
        results.push((value, location));
        input = new_input;
    }

    while !operator_stack.is_empty() {
        assert!(results.len() >= 2);
        pop_operator(&mut operator_stack, &mut results);
    }

    assert!(operator_stack.is_empty());
    assert!(results.len() == 1);
    let (value, location) = results.pop().unwrap();
    Ok((input, value, location))
}

fn term<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::If => if_expr(input),
        Token::Loop => loop_expr(input),
        Token::Match => match_expr(input),
        Token::Handle => handle_expr(input),
        _ => or(&[type_annotation, named_constructor_expr, function_call, function_argument], "term")(input),
    }
}

parser!(function_call loc =
    function <- member_access;
    args <- many1(function_argument);
    desugar::desugar_explicit_currying(function, args, Ast::function_call, loc)
);

parser!(named_constructor_expr loc =
    constructor <- variant;
    _ <- expect(Token::With);
    sequence !<- named_constructor_args;
    Ast::named_constructor(constructor, sequence, loc)
);

fn named_constructor_args<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Ast<'b>> {
    if let Token::Indent = input[0].0 {
        named_constructor_block_args(input)
    } else {
        named_constructor_inline_args(input)
    }
}

parser!(named_constructor_block_args loc -> 'b Ast<'b> =
    _ <- expect(Token::Indent);
    statements <- delimited_trailing(named_constructor_arg, expect(Token::Newline), false);
    _ !<- expect(Token::Unindent);
    Ast::sequence(statements, loc)
);

parser!(named_constructor_inline_args loc -> 'b Ast<'b> =
    statements <- delimited(named_constructor_arg, expect(Token::Comma));
    Ast::sequence(statements, loc)
);

fn named_constructor_arg<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Ast<'b>> {
    let (input, ident, start) = identifier(input)?;
    let field_name = Ast::variable(vec![], ident, start);
    let (input, maybe_expr, end) = maybe(pair(expect(Token::Equal), function_argument))(input)?;
    let expr = match maybe_expr {
        Some((_, expr)) => Ast::definition(field_name, expr, start.union(end)),
        None => field_name,
    };
    Ok((input, expr, start.union(end)))
}

parser!(pattern_function_call loc =
    function <- pattern_function_argument;
    args <- many1(pattern_function_argument);
    Ast::function_call(function, args, loc)
);

parser!(if_expr loc =
    _ <- expect(Token::If);
    condition !<- block_or_statement;
    _ !<- maybe_newline;
    _ !<- expect(Token::Then);
    then !<- block_or_statement;
    otherwise !<- maybe(else_expr);
    Ast::if_expr(condition, then, otherwise, loc)
);

parser!(match_expr loc =
    _ <- expect(Token::Match);
    expression !<- block_or_statement;
    branches !<- many0(match_branch);
    Ast::match_expr(expression, branches, loc)
);

parser!(loop_expr loc =
    _ <- expect(Token::Loop);
    args !<- many1(loop_param);
    _ !<- expect(Token::RightArrow);
    body !<- block_or_statement;
    desugar::desugar_loop(args, body, loc)
);

fn loop_param<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, (Ast<'b>, Ast<'b>)> {
    or(&[loop_param_shorthand, loop_param_longform], "loop parameter")(input)
}

parser!(loop_param_shorthand loc -> 'b (Ast<'b>, Ast<'b>) =
    arg <- pattern_argument;
    (arg.clone(), arg)
);

parser!(loop_param_longform loc -> 'b (Ast<'b>, Ast<'b>) =
    _ <- expect(Token::ParenthesisLeft);
    parameter !<- pattern;
    _ !<- expect(Token::Equal);
    argument !<- expression;
    _ !<- expect(Token::ParenthesisRight);
    (parameter, argument)
);

parser!(handle_expr loc =
    _ <- expect(Token::Handle);
    expression !<- block_or_statement;
    branches !<- many1(handle_branch);
    Ast::handle(expression, branches, loc)
);

parser!(handle_branch _loc -> 'b (Ast<'b>, Ast<'b>) =
    _ <- maybe_newline;
    _ <- expect(Token::Pipe);
    pattern !<- handle_pattern;
    _ !<- expect(Token::RightArrow);
    branch !<- block_or_statement;
    (pattern, branch)
);

// This gives a type error when inlined into handle_branch for some reason
fn handle_pattern<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    or(&[pattern, return_expr], "pattern")(input)
}

parser!(not_expr loc =
    not <- expect(Token::Not);
    expr !<- term;
    Ast::function_call(Ast::operator(not, loc), vec![expr], loc)
);

parser!(ref_expr loc =
    token <- or(&[expect(Token::Ampersand), expect(Token::ExclamationMark)], "expression");
    expr !<- term;
    Ast::function_call(Ast::operator(token, loc), vec![expr], loc)
);

parser!(at_expr loc =
    token <- expect(Token::At);
    expr !<- term;
    Ast::function_call(Ast::operator(token, loc), vec![expr], loc)
);

parser!(type_annotation loc =
    lhs <- or(&[function_call, function_argument], "term");
    _ <- expect(Token::Colon);
    rhs <- parse_type;
    Ast::type_annotation(lhs, rhs, loc)
);

fn parse_type<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Type<'b>> {
    or(&[function_type, pair_type, reference_type, type_application, basic_type], "type")(input)
}

fn function_arg_type<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Type<'b>> {
    or(&[type_application, pair_type, basic_type], "type")(input)
}

fn parse_type_no_pair<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Type<'b>> {
    or(&[function_type, reference_type, type_application, basic_type], "type")(input)
}

fn basic_type<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Type<'b>> {
    match input[0].0 {
        Token::IntegerType(_) => int_type(input),
        Token::FloatType(_) => float_type(input),
        Token::PolymorphicIntType => polymorphic_int_type(input),
        Token::PolymorphicFloatType => polymorphic_float_type(input),
        Token::CharType => char_type(input),
        Token::StringType => string_type(input),
        Token::PointerType => pointer_type(input),
        Token::BooleanType => boolean_type(input),
        Token::UnitType => unit_type(input),
        Token::Ampersand => basic_reference_type(input),
        Token::Identifier(_) => type_variable(input),
        Token::TypeName(_) => user_defined_type(input),
        Token::ParenthesisLeft => parenthesized_type(input),
        _ => Err(ParseError::InRule("type", input[0].1)),
    }
}

fn parenthesized_type<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Type<'b>> {
    parenthesized(parse_type)(input)
}

parser!(match_branch _loc -> 'b (Ast<'b>, Ast<'b>) =
    _ <- maybe_newline;
    _ <- expect(Token::Pipe);
    pattern !<- pattern;
    _ !<- expect(Token::RightArrow);
    branch !<- block_or_statement;
    (pattern, branch)
);

parser!(else_expr _loc =
    _ <- maybe_newline;
    _ <- expect(Token::Else);
    otherwise !<- block_or_statement;
    otherwise
);

/// A function_argument is a unary expr or a member_access of
/// 1-n arguments.
fn function_argument<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::Not => not_expr(input),
        Token::Ampersand => ref_expr(input),
        Token::ExclamationMark => ref_expr(input),
        Token::At => at_expr(input),
        _ => member_access(input),
    }
}

fn pattern_function_argument<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::Ampersand => ref_expr(input),
        Token::ExclamationMark => ref_expr(input),
        Token::At => at_expr(input),
        _ => pattern_argument(input),
    }
}

/// member_access = argument ('.' identifier)*
fn member_access<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    let (mut input, mut arg, mut location) = argument(input)?;

    while input[0].0 == Token::MemberAccess || input[0].0 == Token::MemberRef || input[0].0 == Token::MemberMutRef {
        let is_reference = match input[0].0 {
            Token::MemberMutRef => Some(Mutability::Mutable),
            Token::MemberRef => Some(Mutability::Immutable),
            _ => None,
        };
        input = &input[1..];

        let (new_input, field, field_location) = no_backtracking(identifier)(input)?;
        input = new_input;
        location = location.union(field_location);
        arg = Ast::member_access(arg, field, is_reference, location);
    }

    Ok((input, arg, location))
}

fn argument<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::StringType => variable(input),
        Token::Identifier(_) => variable(input),
        Token::TypeName(_) => or(&[variable, variant], "argument")(input),
        Token::StringLiteral(_) => string(input),
        Token::IntegerLiteral(..) => integer(input),
        Token::FloatLiteral(..) => float(input),
        Token::CharLiteral(_) => parse_char(input),
        Token::BooleanLiteral(_) => parse_bool(input),
        Token::UnitLiteral => unit(input),
        Token::Fn => lambda(input),
        Token::ParenthesisLeft => parenthesized_expression(input),
        _ => Err(ParseError::InRule("argument", input[0].1)),
    }
}

fn pattern_argument<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    match input[0].0 {
        Token::Identifier(_) => variable(input),
        Token::StringLiteral(_) => string(input),
        Token::IntegerLiteral(..) => integer(input),
        Token::FloatLiteral(..) => float(input),
        Token::CharLiteral(_) => parse_char(input),
        Token::BooleanLiteral(_) => parse_bool(input),
        Token::UnitLiteral => unit(input),
        Token::ParenthesisLeft => parenthesized_irrefutable_pattern(input),
        Token::TypeName(_) => variant(input),
        _ => Err(ParseError::InRule("pattern argument", input[0].1)),
    }
}

parser!(lambda loc =
    _ <- expect(Token::Fn);
    args !<- many1(pattern_argument);
    return_type <- maybe(function_return_type);
    effects <- maybe(effect_clause);
    _ !<- expect(Token::RightArrow);
    body !<- block_or_statement;
    Ast::lambda(args, return_type, effects, body, loc)
);

parser!(operator loc =
    op <- expect_if("operator", |op| op.is_overloadable_operator());
    Ast::operator(op, loc)
);

fn parenthesized_expression<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    parenthesized(or(&[expression, operator], "operator or expression"))(input)
}

parser!(variant loc =
    mut module_prefix <- delimited(typename, expect(Token::MemberAccess));
    {
        let name = module_prefix.pop().unwrap();
        Ast::type_constructor(module_prefix, name, loc)
    }
);

parser!(variable loc =
    module_prefix <- maybe(delimited_trailing(typename, expect(Token::MemberAccess), true));
    name <- identifier;
    Ast::variable(module_prefix.unwrap_or_default(), name, loc)
);

parser!(string_literal loc =
    contents <- string_literal_token;
    Ast::string(contents, loc)
);

fn interpolated_expression<'a, 'b>(input: Input<'a, 'b>) -> AstResult<'a, 'b> {
    bounded(Token::InterpolateLeft, expression, Token::InterpolateRight)(input)
}

parser!(interpolation loc =
    lhs <- interpolated_expression;
    rhs <- string_literal;
    desugar::interpolate(lhs, rhs, loc)
);

parser!(string loc =
    head <- string_literal;
    tail <- many0(interpolation);
    desugar::concatenate_strings(head, tail, loc)
);

parser!(integer loc =
    (value, kind) <- integer_literal_token;
    Ast::integer(value, kind, loc)
);

parser!(float loc =
    (value, kind) <- float_literal_token;
    Ast::float(value, kind, loc)
);

parser!(parse_char loc =
    contents <- char_literal_token;
    Ast::char_literal(contents, loc)
);

parser!(parse_bool loc =
    value <- bool_literal_token;
    Ast::bool_literal(value, loc)
);

parser!(unit loc =
    _ <- expect(Token::UnitLiteral);
    Ast::unit_literal(loc)
);

parser!(function_type loc -> 'b Type<'b> =
    args <- delimited_trailing(function_arg_type, expect(Token::Subtract), false);
    varargs <- maybe(varargs);
    is_closure <- function_arrow;
    return_type <- parse_type;
    Type::Function(args, Box::new(return_type), varargs.is_some(), is_closure, loc)
);

// Returns true if this function is a closure
fn function_arrow<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, bool> {
    match input[0].0 {
        Token::RightArrow => Ok((&input[1..], false, input[0].1)),
        Token::FatArrow => Ok((&input[1..], true, input[0].1)),
        _ => Err(ParseError::InRule("function type", input[0].1)),
    }
}

parser!(type_application loc -> 'b Type<'b> =
    type_constructor <- basic_type;
    args <- many1(basic_type);
    Type::TypeApplication(Box::new(type_constructor), args, loc)
);

parser!(pair_type loc -> 'b Type<'b> =
    first <- or(&[type_application, basic_type], "type");
    _ <- expect(Token::Comma);
    rest !<- parse_type;
    Type::Pair(Box::new(first), Box::new(rest), loc)
);

parser!(int_type loc -> 'b Type<'b> =
    kind <- int_type_token;
    Type::Integer(Some(kind), loc)
);

parser!(float_type loc -> 'b Type<'b> =
    kind <- float_type_token;
    Type::Float(Some(kind), loc)
);

parser!(polymorphic_int_type loc -> 'b Type<'b> =
    _ <- expect(Token::PolymorphicIntType);
    Type::Integer(None, loc)
);

parser!(polymorphic_float_type loc -> 'b Type<'b> =
    _ <- expect(Token::PolymorphicFloatType);
    Type::Float(None, loc)
);

parser!(char_type loc -> 'b Type<'b> =
    _ <- expect(Token::CharType);
    Type::Char(loc)
);

parser!(string_type loc -> 'b Type<'b> =
    _ <- expect(Token::StringType);
    Type::String(loc)
);

parser!(pointer_type loc -> 'b Type<'b> =
    _ <- expect(Token::PointerType);
    Type::Pointer(loc)
);

parser!(boolean_type loc -> 'b Type<'b> =
    _ <- expect(Token::BooleanType);
    Type::Boolean(loc)
);

parser!(unit_type loc -> 'b Type<'b> =
    _ <- expect(Token::UnitType);
    Type::Unit(loc)
);

parser!(reference_type loc -> 'b Type<'b> =
    mutability <- reference_operator;
    sharedness <- sharedness;
    element <- maybe(reference_element_type);
    make_reference_type(Type::Reference(sharedness, mutability, loc), element, loc)
);

parser!(reference_operator loc -> 'b Mutability =
    token <- or(&[expect(Token::Ampersand), expect(Token::ExclamationMark)], "type");
    match token {
        Token::Ampersand => Mutability::Immutable,
        Token::ExclamationMark => Mutability::Mutable,
        Token::QuestionMark => Mutability::Polymorphic,
        _ => unreachable!(),
    }
);

// The basic reference type `&t` can be used without parenthesis in a type application
parser!(basic_reference_type loc -> 'b Type<'b> =
    mutability <- reference_operator;
    element <- maybe(basic_type);
    make_reference_type(Type::Reference(Sharedness::Polymorphic, mutability, loc), element, loc)
);

parser!(reference_element_type loc -> 'b Type<'b> =
    typ <- or(&[type_application, basic_type], "type");
    typ
);

fn make_reference_type<'b>(reference: Type<'b>, element: Option<Type<'b>>, loc: Location<'b>) -> Type<'b> {
    match element {
        Some(element) => Type::TypeApplication(Box::new(reference), vec![element], loc),
        None => reference,
    }
}

// Parses 'owned' or 'shared' on a reference type
fn sharedness<'a, 'b>(input: Input<'a, 'b>) -> ParseResult<'a, 'b, Sharedness> {
    match input[0].0 {
        Token::Shared => Ok((&input[1..], Sharedness::Shared, input[0].1)),
        Token::Owned => Ok((&input[1..], Sharedness::Owned, input[0].1)),
        _ => Ok((input, Sharedness::Polymorphic, input[0].1)),
    }
}

parser!(type_variable loc -> 'b Type<'b> =
    name <- identifier;
    Type::TypeVariable(name, loc)
);

parser!(user_defined_type loc -> 'b Type<'b> =
    name <- typename;
    Type::UserDefined(name, loc)
);
