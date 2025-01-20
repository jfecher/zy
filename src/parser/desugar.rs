use std::collections::BTreeMap;

use crate::error::location::Location;
use crate::parser::ast::Ast;
use crate::{error::location::Locatable, lexer::token::Token, parser::ast, util::fmap};

/// Turns `(foo _  _ 2)` into `(fn $1 $2 -> (foo $1 $2 2))`
pub fn desugar_explicit_currying<'a, F>(
    function: Ast<'a>, args: Vec<Ast<'a>>, make_function_call: F, loc: Location<'a>,
) -> Ast<'a>
where
    F: FnOnce(Ast<'a>, Vec<Ast<'a>>, Location<'a>) -> Ast<'a>,
{
    if args.iter().any(matches_underscore) {
        curried_function_call(function, args, make_function_call, loc)
    } else {
        make_function_call(function, args, loc)
    }
}

fn curried_function_call<'a, F>(function: Ast<'a>, args: Vec<Ast<'a>>, call_function: F, loc: Location<'a>) -> Ast<'a>
where
    F: FnOnce(Ast<'a>, Vec<Ast<'a>>, Location<'a>) -> Ast<'a>,
{
    let mut curried_args = vec![];
    let mut curried_arg_count = 0;
    let args: Vec<Ast<'a>> = fmap(args, |arg| {
        if matches_underscore(&arg) {
            curried_arg_count += 1;
            let curried_arg = format!("${}", curried_arg_count);
            curried_args.push(Ast::variable(vec![], curried_arg.clone(), arg.locate()));
            Ast::variable(vec![], curried_arg, arg.locate()) // TODO: add correct module prefix
        } else {
            arg
        }
    });

    let function_call = call_function(function, args, loc);
    Ast::lambda(curried_args, None, None, function_call, loc)
}

fn matches_underscore(arg: &Ast) -> bool {
    matches!(arg, Ast::Variable(ast::Variable{ kind: ast::VariableKind::Identifier(x), ..}) if x == "_")
}

/// Turns:
/// - `bar |> foo` into `foo bar` (applies to <| as well)
/// - `a and b` into `if a then b else false`
/// - `a or b` into `if a then true else b`
///
/// Also handles explicitly curried operators. E.g. `_ or false` will
/// be translated as `fn $1 -> if $1 then true else false`
pub fn desugar_operators<'a>(operator: Token, lhs: Ast<'a>, rhs: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    let call_operator_function = |function: Ast<'a>, mut arguments: Vec<Ast<'a>>, location| {
        let rhs = arguments.pop().unwrap();
        let lhs = arguments.pop().unwrap();

        match function.get_operator() {
            Some(Token::ApplyLeft) => prepend_argument_to_function(lhs, rhs, location),
            Some(Token::ApplyRight) => prepend_argument_to_function(rhs, lhs, location),
            Some(Token::And) => Ast::if_expr(lhs, rhs, Some(Ast::bool_literal(false, location)), location),
            Some(Token::Or) => Ast::if_expr(lhs, Ast::bool_literal(true, location), Some(rhs), location),
            Some(Token::Else) => create_else_match(lhs, rhs, location),
            Some(operator_token) => {
                let operator = Ast::operator(operator_token, location);
                Ast::function_call(operator, vec![lhs, rhs], location)
            },
            None => unreachable!(),
        }
    };

    let operator_symbol = Ast::operator(operator, location);
    desugar_explicit_currying(operator_symbol, vec![lhs, rhs], call_operator_function, location)
}

fn create_else_match<'a>(lhs: Ast<'a>, rhs: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    let x = Ast::variable(vec![], "x".to_owned(), location);
    let some = Ast::type_constructor(vec!["prelude".to_owned()], "Some".to_owned(), location);
    let some_x = Ast::function_call(some, vec![x.clone()], location);
    let none = Ast::type_constructor(vec!["prelude".to_owned()], "None".to_owned(), location);
    Ast::match_expr(lhs, vec![(some_x, x), (none, rhs)], location)
}

fn prepend_argument_to_function<'a>(f: Ast<'a>, arg: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    match f {
        Ast::FunctionCall(mut call) => {
            call.args.insert(0, arg);
            Ast::FunctionCall(call)
        },
        _ => Ast::function_call(f, vec![arg], location),
    }
}

pub fn desugar_loop<'a>(params_defaults: Vec<(Ast<'a>, Ast<'a>)>, body: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    let (params, args) = params_defaults.into_iter().unzip();
    let recur_name = || Ast::variable(vec![], "recur".to_owned(), location);
    let recur_def = Ast::definition(recur_name(), Ast::lambda(params, None, None, body, location), location);
    let recur_call = Ast::function_call(recur_name(), args, location);
    Ast::new_scope(Ast::sequence(vec![recur_def, recur_call], location), location)
}

/// Desugar:
///
/// handle foo + bar
/// | set 0 a -> resume ()
/// | get () -> foo resume 1 // test 'resume' is parsed as a normal identifier
/// | set _ b -> resume ()
///
/// To:
/// handle foo + bar
/// | set _$1 _$2 ->
///     match (_$1, _$2)
///     | (0, a) -> resume ()
///     | (_, b) -> resume ()
/// | get () -> resume ()
///
/// So that we do not need to duplicate pattern matching logic inside Ast::Handle
pub fn desugar_handle_branches_into_matches<'a>(branches: Vec<(Ast<'a>, Ast<'a>)>) -> Vec<(Ast<'a>, Ast<'a>)> {
    // BTreeMap is used here for a deterministic ordering for tests
    let mut cases = BTreeMap::new();

    for (pattern, branch) in branches {
        let (name, match_pattern, args_len, location) = match pattern {
            Ast::FunctionCall(call) => match call.function.as_ref() {
                Ast::Variable(name) => {
                    let arg_len = call.args.len();
                    let args = tuplify(call.args, call.location);
                    (name.to_string(), args, arg_len, call.location)
                },
                _ => unreachable!("Invalid syntax in pattern of 'handle' expression"),
            },
            Ast::Return(return_) => ("return".into(), *return_.expression, 1, return_.location),
            _ => unreachable!("Invalid syntax in pattern of 'handle' expression"),
        };

        cases.entry((name, args_len)).or_insert((vec![], location)).0.push((match_pattern, branch))
    }

    fmap(cases, |((name, args_len), (branches, location))| {
        // _$0, _$1, ...
        let new_args1 = fmap(0..args_len, |i| Ast::variable(vec![], format!("_${}", i), location));
        // Ast doesn't impl Clone currently
        let new_args2 = fmap(0..args_len, |i| Ast::variable(vec![], format!("_${}", i), location));

        let expr = tuplify(new_args1, location);
        let match_expr = Ast::match_expr(expr, branches, location);

        // TODO: Do we need to forward the module prefix here?
        let handle_effect = Ast::variable(vec![], name, location);
        let handle_pattern = Ast::function_call(handle_effect, new_args2, location);
        (handle_pattern, match_expr)
    })
}

fn empty_string_literal(ast: &Ast) -> bool {
    match ast {
        Ast::Literal(super::ast::Literal { kind: ast::LiteralKind::String(s), .. }) => s.is_empty(),
        _ => false,
    }
}

fn append<'a>(lhs: Ast<'a>, rhs: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    let append = Ast::operator(Token::Append, location);
    Ast::function_call(append, vec![lhs, rhs], location)
}

pub fn interpolate<'a>(lhs: Ast<'a>, rhs: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    let cast = Ast::variable(vec![], String::from("cast"), location);
    let lhs = Ast::function_call(cast, vec![lhs], location);
    match rhs {
        // Don't append empty strings
        ast if empty_string_literal(&ast) => lhs,
        _ => append(lhs, rhs, location),
    }
}

pub fn concatenate_strings<'a>(head: Ast<'a>, tail: Vec<Ast<'a>>, location: Location<'a>) -> Ast<'a> {
    let mut tail = tail.into_iter();
    match tail.next() {
        Some(expr) => {
            // Use first value from the iterator in case the head is an empty string.
            // Explicitly using the iterator rather than recursion,
            // since this optimisation can only apply to the head
            let mut ret = match head {
                ast if empty_string_literal(&ast) => Ast::type_annotation(expr, ast::Type::String(location), location),
                _ => append(head, expr, location),
            };
            for expr in tail {
                ret = append(ret, expr, location);
            }
            ret
        },
        None => head,
    }
}

/// Wrap all arguments in a tuple of nested pairs.
/// This could be more efficient, using e.g. a VecDeque
fn tuplify<'a>(mut args: Vec<Ast<'a>>, location: Location<'a>) -> Ast<'a> {
    assert!(!args.is_empty());

    if args.len() == 1 {
        args.remove(0)
    } else {
        let first = args.remove(0);
        let rest = tuplify(args, location);
        let function = Ast::operator(Token::Comma, location);
        Ast::function_call(function, vec![first, rest], location)
    }
}

pub fn desugar_if_with_no_else<'a>(condition: Ast<'a>, then: Ast<'a>, location: Location<'a>) -> Ast<'a> {
    let then = Box::new(Ast::sequence(vec![then, Ast::unit_literal(location)], location));
    let otherwise = Box::new(Ast::unit_literal(location));

    Ast::If(ast::If { condition: Box::new(condition), then, otherwise, location, typ: None })
}
