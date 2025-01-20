//! nameresolution/builtin.rs - Helpers for importing the prelude
//! and defining some builtin symbols such as the `string` type (which
//! is builtin because otherwise string literals may be used before the
//! actual string type is defined) and the `builtin` function which is
//! used by codegen to stand in place of primitive operations like adding
//! integers together.
use crate::cache::{DefinitionInfoId, DefinitionKind, ModuleCache};
use crate::error::location::Location;
use crate::lexer::token::{IntegerKind, Token};
use crate::nameresolution::{declare_module, define_module, NameResolver};
use crate::types::{
    Field, FunctionType, GeneralizedType, LetBindingLevel, PrimitiveType, Type, TypeInfoBody, PAIR_TYPE, STRING_TYPE,
};

use std::collections::HashSet;
use std::path::PathBuf;

/// The DefinitionInfoId of the `builtin` symbol is defined to be
/// the first id. This invariant needs to be maintained by the
/// `define_builtins` function here being called before any other
/// symbol is defined. This is asserted to be the case within that function.
pub const BUILTIN_ID: DefinitionInfoId = DefinitionInfoId(0);

/// DefinitionInfoId for the string constructor to construct a string from its raw parts
pub const STRING_ID: DefinitionInfoId = DefinitionInfoId(1);

/// DefinitionInfoId for the pair constructor `,` to construct values like (1, 2)
pub const PAIR_ID: DefinitionInfoId = DefinitionInfoId(2);

/// Defines the builtin symbols:
/// - `type string = c_string: ptr char, len: usz`
/// - `builtin : string -> a` used by the codegen pass to implement
///   codegen of builtin operations such as adding integers.
///
/// This function needs to be called before any other DefinitionInfoId is
/// created, otherwise the `builtin` symbol will have the wrong id. If this
/// happens, this function will assert at runtime.
pub fn define_builtins(cache: &mut ModuleCache) {
    // Define builtin : forall a. string -> a imported only into the prelude to define
    // builtin operations by name. The specific string arguments are matched on in src/llvm/builtin.rs
    let id = cache.push_definition("builtin", Location::builtin());
    assert_eq!(id, BUILTIN_ID);

    let string_type = define_string(cache);
    define_pair(cache);

    let a = cache.next_type_variable_id(LetBindingLevel(1));
    let e = cache.next_type_variable_id(LetBindingLevel(1));

    let builtin_fn_type = Type::Function(FunctionType {
        parameters: vec![string_type],
        return_type: Box::new(Type::TypeVariable(a)),
        environment: Box::new(Type::UNIT),
        effects: Box::new(Type::TypeVariable(e)),
        is_varargs: true,
    });

    let builtin_type = GeneralizedType::PolyType(vec![a, e], builtin_fn_type);
    cache.definition_infos[id.0].typ = Some(builtin_type);
}

/// The prelude is currently stored (along with the rest of the stdlib) in the
/// user's config directory since it is a cross-platform concept that doesn't
/// require administrator priviledges.
pub fn prelude_path() -> PathBuf {
    crate::util::stdlib_dir().join("prelude.an")
}

pub fn import_prelude(resolver: &mut NameResolver, cache: &mut ModuleCache<'_>) {
    if resolver.filepath == prelude_path() {
        // If we're in the prelude include the built-in symbol "builtin" to define primitives
        resolver.current_scope().definitions.insert("builtin".into(), BUILTIN_ID);
    } else {
        // Otherwise, import the prelude itself
        let prelude_dir = prelude_path();
        if let Some(id) = declare_module(&prelude_dir, cache, Location::builtin()) {
            let exports = define_module(id, cache, Location::builtin()).unwrap();
            resolver.current_scope().import(exports, cache, Location::builtin(), &HashSet::new());
            let maybe_id = resolver.current_scope().types.get("Maybe").unwrap().clone();
            cache.maybe_type = Some(maybe_id);
        }
    }

    // Manually insert some builtins as if they were defined in the prelude
    resolver.current_scope().types.insert(Token::Comma.to_string(), PAIR_TYPE);
    resolver.current_scope().definitions.insert(Token::Comma.to_string(), PAIR_ID);
    resolver.current_scope().definitions.insert("String".into(), STRING_ID);
}

/// Defining the 'string' type is a bit different than most other builtins. Since 'string' has
/// its own dedicated keyword it need not be imported into scope like each impl of + or - does.
///
/// The builtin string type is defined here as:
///
/// type string = c_string: Ptr char, length: usz
fn define_string(cache: &mut ModuleCache) -> Type {
    let location = Location::builtin();

    let ptr_type = Type::Primitive(PrimitiveType::Ptr);
    let char_type = Type::Primitive(PrimitiveType::CharType);
    let c_string_type = Type::TypeApplication(Box::new(ptr_type), vec![char_type]);

    let length_type = Type::int(IntegerKind::Usz);

    let name = "String".to_string();
    let string_id = cache.push_type_info(name.clone(), vec![], location);
    assert_eq!(string_id, STRING_TYPE);
    let string = Type::UserDefined(STRING_TYPE);

    let fields = TypeInfoBody::Struct(vec![
        Field { name: "c_string".into(), field_type: c_string_type.clone(), location },
        Field { name: "length".into(), field_type: length_type.clone(), location },
    ]);

    let constructor = cache.push_definition(&name, location);
    assert_eq!(constructor, STRING_ID);

    let effects = cache.next_type_variable_id(LetBindingLevel(1));

    let constructor_type = Type::Function(FunctionType {
        parameters: vec![c_string_type, length_type],
        return_type: Box::new(string.clone()),
        environment: Box::new(Type::UNIT),
        effects: Box::new(Type::TypeVariable(effects)),
        is_varargs: false,
    });

    let polytype = GeneralizedType::PolyType(vec![effects], constructor_type);

    cache.definition_infos[constructor.0].typ = Some(polytype);
    cache.definition_infos[constructor.0].definition = Some(DefinitionKind::TypeConstructor { name, tag: None });

    cache.type_infos[string_id.0].body = fields;
    string
}

/// The builtin pair type is defined here as:
///
/// type (,) a b = first: a, second: b
fn define_pair(cache: &mut ModuleCache) {
    let location = Location::builtin();

    let level = LetBindingLevel(0);
    let a = cache.next_type_variable_id(level);
    let b = cache.next_type_variable_id(level);

    let name = Token::Comma.to_string();
    let pair = cache.push_type_info(name.clone(), vec![a, b], location);
    assert_eq!(pair, PAIR_TYPE);

    cache.type_infos[pair.0].body = TypeInfoBody::Struct(vec![
        Field { name: "first".into(), field_type: Type::TypeVariable(a), location },
        Field { name: "second".into(), field_type: Type::TypeVariable(b), location },
    ]);

    // The type is defined, now we define the constructor
    let parameters = vec![Type::TypeVariable(a), Type::TypeVariable(b)];
    let pair = Box::new(Type::UserDefined(pair));
    let pair_a_b = Box::new(Type::TypeApplication(pair, parameters.clone()));

    let e = cache.next_type_variable_id(level);

    let constructor_type = Type::Function(FunctionType {
        parameters,
        return_type: pair_a_b,
        environment: Box::new(Type::UNIT),
        effects: Box::new(Type::TypeVariable(e)),
        is_varargs: false,
    });

    let constructor_type = GeneralizedType::PolyType(vec![a, b, e], constructor_type);

    // and now register a new type constructor in the cache with the given type
    let id = cache.push_definition(&name, location);
    assert_eq!(id, PAIR_ID);
    let constructor = DefinitionKind::TypeConstructor { name, tag: None };

    cache.definition_infos[id.0].typ = Some(constructor_type);
    cache.definition_infos[id.0].definition = Some(constructor);
}
