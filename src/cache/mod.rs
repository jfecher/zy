//! cache/mod.rs - Provides the ModuleCache struct which the ante compiler
//! pervasively uses to cache and store results from various compiler phases.
//! The most important things the cache stores are additional information about
//! the parse tree. For example, for each variable definition, there is a corresponding
//! `DefinitionInfo` struct and a `DefinitionInfoId` key that can be used on the
//! cache to access this struct. The `DefinitionInfo` struct stores additional information
//! about a definition like the `ast::Definition` node it was defined in, its name,
//! and how many times it is referenced in the program.
//! This XXXInfo and XXXInfoId pattern is also used for TraitDefinitions, TraitImpls, and Types.
//! See the corresponding structs further down in this file for more information.
//!
//! The ModuleCache itself is kept the entirely of compilation - its contents may be
//! used in any phase and thus nothing is freed until the program is fully linked.
//! Any pass-specific information that isn't needed for later phases shouldn't be
//! kept in the ModuleCache and instead should be in a special data structure for
//! the relevant phase. An example is the `llvm::Generator` in the llvm codegen phase.
use crate::cache::unsafecache::UnsafeCache;
use crate::error::location::{Locatable, Location};
use crate::error::{Diagnostic, DiagnosticKind, ErrorType};
use crate::nameresolution::NameResolver;
use crate::parser::ast::{Ast, Definition, EffectDefinition, Extern, TraitDefinition, TraitImpl};
use crate::types::traits::{ConstraintSignature, RequiredImpl, RequiredTrait, TraitConstraintId};
use crate::types::{GeneralizedType, Kind, LetBindingLevel, TypeBinding};
use crate::types::{Type, TypeInfo, TypeInfoBody, TypeInfoId, TypeVariableId};
use crate::util::stdlib_dir;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use self::dependency_graph::DependencyGraph;

mod counter;
mod dependency_graph;
mod unsafecache;

/// The ModuleCache is for information needed until compilation is completely finished
/// (ie. not just for one phase). Accessing each `Vec` inside the `ModuleCache` is done
/// only through the XXXInfoId keys returned as a result of each of the `push_xxx` methods
/// on the `ModuleCache`. These keys are also often stored in the AST itself as a result
/// of various passes. For example, name resolution will fill in the `definition: Option<DefinitionInfoId>`
/// field of each `ast::Variable` node, which has the effect of pointing each variable to
/// where it was defined.
#[derive(Debug)]
pub struct ModuleCache<'a> {
    /// All the 'root' directories for imports. In practice this will contain
    /// the directory of the driver module as well as all directories containing
    /// any libraries used by the program, including the standard library.
    pub relative_roots: Vec<PathBuf>,

    /// Maps ModuleId -> Ast
    /// Contains all the parse trees parsed by the program.
    pub parse_trees: UnsafeCache<'a, Ast<'a>>,

    /// Used to map paths to parse trees or name resolvers
    pub modules: HashMap<PathBuf, ModuleId>,

    /// Maps ModuleId -> NameResolver
    pub name_resolvers: UnsafeCache<'a, NameResolver>,

    /// Holds all the previously seen filenames referenced by Locations
    /// Used to lengthen the lifetime of Locations and the parse tree past
    /// the lifetime of the file that was read from.
    pub filepaths: Vec<PathBuf>,

    /// Maps DefinitionInfoId -> DefinitionInfo
    /// Filled out during name resolution.
    pub definition_infos: Vec<DefinitionInfo<'a>>,

    /// Maps VariableInfoId -> VariableInfo
    /// Each ast::Variable node stores the required impls for use while
    /// codegening the variable's definition. These impls are filled out
    /// during type inference (see typechecker::find_impl). Unlike
    /// DefinitionInfos, VariableInfos are per usage of the variable.
    pub variable_infos: Vec<VariableInfo<'a>>,

    /// Maps TypeVariableId -> Type
    /// Unique TypeVariableIds are generated during name
    /// resolution and are unified during type inference
    pub type_bindings: Vec<TypeBinding>,

    /// Maps TypeInfoId -> TypeInfo
    /// Filled out during name resolution
    pub type_infos: Vec<TypeInfo<'a>>,

    /// Maps TraitInfoId -> TraitInfo
    /// Filled out during name resolution
    pub trait_infos: Vec<TraitInfo<'a>>,

    /// Maps ImplInfoId -> ImplInfo
    /// Filled out during name resolution, though
    /// definitions within impls aren't publically exposed.
    pub impl_infos: Vec<ImplInfo<'a>>,

    /// Maps EffectInfoId -> EffectInfo
    /// Filled out during name resolution
    pub effect_infos: Vec<EffectInfo<'a>>,

    /// Maps ModuleId -> Vec<ImplInfo>
    /// Name resolution needs to store the impls visible to
    /// each variable so when impls are resolved
    /// during type inference the inferencer can quickly get the
    /// impls that should be in scope and select an instance.
    pub impl_scopes: Vec<Vec<ImplInfoId>>,

    /// A monotonically-increasing counter to uniquely identify trait constraints.
    pub current_trait_constraint_id: counter::TraitConstraintCounter,

    /// Call stack of functions traversed during type inference. Used to find
    /// mutually recursive functions and delay generalization of them until after
    /// all the functions in the mutually recursive set are finished.
    pub call_stack: Vec<DefinitionInfoId>,

    /// Map from the first function reachable in a mutually recursive set of functions
    /// to all functions in that set. Once the key'd function finishes compiling we can
    /// generalize all the functions in the set and add trait constraints at once.
    pub mutual_recursion_sets: Vec<MutualRecursionSet>,

    /// Dependency graph for all global definitions.
    ///
    /// This is generally rather lax, the only variant that is enforced is that
    /// there is no dependency cycle between two non-function globals.
    pub global_dependency_graph: DependencyGraph,

    /// Any diagnostics (errors, warnings, or notes) emitted by the program
    pub diagnostics: Vec<Diagnostic<'a>>,

    /// The number of errors emitted by the program
    pub error_count: usize,

    pub file_cache: FileCache,

    pub maybe_type: Option<TypeInfoId>,
}

pub type FileCache = HashMap<PathBuf, String>;

#[derive(Debug)]
pub struct MutualRecursionSet {
    pub root_definition: DefinitionInfoId,
    pub definitions: HashSet<DefinitionInfoId>,

    /// Index in ModuleCache.call_stack of the root_definition.
    /// While we are collecting mutually recursive definitions,
    /// the definition with the lowest place in the callstack is
    /// chosen as the root. This can change if more recursion is
    /// found later on in a function after creating the initial
    /// MutualRecursionSet.
    call_stack_index_of_root: usize,
}

/// TODO: Remove. This is used for experimenting with ante-lsp
unsafe impl<'c> Send for ModuleCache<'c> {}

/// The key for accessing parse trees or `NameResolver`s
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ModuleId(pub usize);

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct DefinitionInfoId(pub usize);

impl std::fmt::Debug for DefinitionInfoId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "${}", self.0)
    }
}

#[derive(Debug)]
pub enum DefinitionKind<'a> {
    /// A variable/function definition in the form `a = b`
    Definition(&'a mut Definition<'a>),

    /// A trait definition in the form `trait A a with ...`
    TraitDefinition(&'a mut TraitDefinition<'a>),

    /// An effect definition in the form `effect E with ...`
    EffectDefinition(&'a mut EffectDefinition<'a>),

    /// An extern FFI definition with no body
    Extern(&'a mut Extern<'a>),

    /// A TypeConstructor function to construct a type.
    /// If the constructed type is a tagged union, tag will
    /// be Some, otherwise if it is a struct, tag is None.
    TypeConstructor {
        name: String,
        tag: Option<u8>,
    },

    Parameter,

    /// Any variable declared in a match pattern. E.g. 'a' in
    /// match None with
    /// | a -> ()
    MatchPattern,
}

/// Carries additional information about a variable's definition.
/// Note that two variables defined in the same pattern, e.g: `(a, b) = c`
/// will have their own unique `DefinitionInfo`s, but each DefinitionInfo
/// will refer to the same `ast::Definition` in its definition field.
///
/// The corresponding DefinitionInfoId is attatched to each
/// `ast::Variable` during name resolution.
#[derive(Debug)]
pub struct DefinitionInfo<'a> {
    pub name: String,
    pub location: Location<'a>,

    /// Where this name was defined. It is expected that type checking
    /// this Definition kind should result in self.typ being filled out.
    pub definition: Option<DefinitionKind<'a>>,

    /// Some((trait_id, trait_args)) if this is a definition from a trait.
    /// Note that this is still None for definitions from trait impls.
    pub trait_info: Option<(TraitInfoId, Vec<Type>)>,

    /// For a given definition like:
    /// foo (a: a) -> a
    ///   given Add a, Print a = ...
    /// required_traits is the "given ..." part of the signature
    pub required_traits: Vec<RequiredTrait>,

    /// The trait impl, if any, that this definition belongs to.
    pub trait_impl: Option<ImplInfoId>,

    /// If this definition is in a mutually recursive set of functions,
    /// this will be set to Some(key) where key is the first reachable
    /// of functions (from main) in this set and the key in the
    /// mutually_recursive_definitions HashMap in the cache.
    pub mutually_recursive_set: Option<MutualRecursionId>,

    /// Remember which variable links lead to mutually recursive calls.
    /// Since traits need to be solved all at once for mutually recursive
    /// definitions, we need to remember where to link them to later.
    pub mutually_recursive_variables: Vec<VariableId>,

    /// Flag for whether we're currently inferring the type of this definition.
    /// Used to find mutual recursion sets. Technically unneeded since we can also
    /// check the call_graph, but this is faster and more readable.
    pub undergoing_type_inference: bool,

    /// If true, avoid issuing a 'variable x is unused' warning
    /// False by default.
    pub ignore_unused_warning: bool,

    /// The type of this definition. Filled out during type inference,
    /// and is guarenteed to be Some afterward.
    pub typ: Option<GeneralizedType>,

    /// A count of how many times was this variable referenced in the program.
    /// Used primarily for issuing unused warnings.
    pub uses: u32,
}

impl<'a> Locatable<'a> for DefinitionInfo<'a> {
    fn locate(&self) -> Location<'a> {
        self.location
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct MutualRecursionId(pub usize);

/// Each `ast::Variable` node corresponds to a VariableId that identifies it,
/// filled out during name resolution. These are currently used to identify the
/// origin/callsites of traits for trait dispatch.
/// `TraitConstraints` are passed around during type inference carrying these so
/// that once they're finally resolved, the correct variable can be linked to the
/// correct impl definition.
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct VariableId(pub usize);

/// Contains extra information for each variable node.
/// Used to map specific variable instances to trait impls.
#[derive(Debug)]
pub struct VariableInfo<'a> {
    pub required_impls: Vec<RequiredImpl>,
    pub name: String,
    pub location: Location<'a>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct TraitInfoId(pub usize);

/// Additional information on the definition of a trait.
/// The corresponding TraitInfoId is attatched to each
/// `ast::TraitDefinition` during name resolution.
///
/// Note that the builtin member access family of traits have their own TraitInfo.
#[derive(Debug)]
pub struct TraitInfo<'a> {
    pub name: String,

    /// The type arguments of this trait. These are the
    /// `a b c` in `trait Foo a b c -> d e f with ...`
    /// Note that all traits must have at least 1 type
    /// argument, otherwise there is no type to implement
    /// the trait for.
    pub typeargs: Vec<TypeVariableId>,

    /// The possibly-empty functional dependencies of this trait.
    /// These are the `d e f` in `trait Foo a b c -> d e f with ...`
    pub fundeps: Vec<TypeVariableId>,

    pub location: Location<'a>,

    /// The definitions included in this trait defintion.
    /// The term `defintion` is used somewhat loosely here
    /// since none of these functions/variables have bodies.
    /// They're merely declarations that impl definitions will
    /// later have to conform to.
    pub definitions: Vec<DefinitionInfoId>,

    /// The Ast node that defines this trait.
    /// A value of None means this trait was builtin to the compiler
    pub trait_node: Option<&'a mut TraitDefinition<'a>>,

    pub uses: u32,
}

impl<'a> Locatable<'a> for TraitInfo<'a> {
    fn locate(&self) -> Location<'a> {
        self.location
    }
}

/// An ImplScopeId is attached to an ast::Variable to remember
/// the impls that were in scope when it was used since scopes are
/// thrown away after name resolution but the impls in scope are still
/// needed during type inference.
/// TODO: The concept of an ImplScope is somewhat of a wart in the trait inference
/// algorithm. Getting rid of them would likely make it both cleaner and faster.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ImplScopeId(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ImplInfoId(pub usize);

/// Corresponds to a `ast::TraitImpl` node, carrying extra information
/// on it. These are filled out during name resolution.
#[derive(Debug)]
pub struct ImplInfo<'a> {
    pub trait_id: TraitInfoId,
    pub typeargs: Vec<Type>,
    pub location: Location<'a>,
    pub definitions: Vec<DefinitionInfoId>,

    /// These constraints are from the 'given' clause of a trait impl.
    /// They contain a unique TraitConstraintId that is used to map the
    /// constraints inside the impl's definitions.
    pub given: Vec<ConstraintSignature>,
    pub trait_impl: &'a mut TraitImpl<'a>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EffectInfoId(pub usize);

/// Corresponds to a `ast::EffectDefinition` node, carrying extra information
/// on it. These are filled out during name resolution.
#[derive(Debug)]
pub struct EffectInfo<'a> {
    pub name: String,

    pub effect_node: &'a mut EffectDefinition<'a>,

    /// Type variables on the effect declaration itself.
    /// Unlike traits, this may be empty.
    pub typeargs: Vec<TypeVariableId>,

    pub location: Location<'a>,

    pub declarations: Vec<DefinitionInfoId>,
}

impl<'a> Locatable<'a> for EffectInfo<'a> {
    fn locate(&self) -> Location<'a> {
        self.location
    }
}

pub fn cached_read<'a>(file_cache: &'a FileCache, path: &Path) -> Option<Cow<'a, str>> {
    match file_cache.get(path) {
        Some(contents) => Some(Cow::Borrowed(contents)),
        None => {
            let file = File::open(path).ok()?;
            let mut reader = BufReader::new(file);
            let mut contents = String::new();
            reader.read_to_string(&mut contents).ok()?;
            Some(Cow::Owned(contents))
        },
    }
}

impl<'a> ModuleCache<'a> {
    /// For consistency, all paths should be absolute and canonical.
    /// They can be converted to relative paths for displaying errors later.
    pub fn new(project_directory: &Path, file_cache: FileCache) -> ModuleCache<'a> {
        ModuleCache {
            relative_roots: vec![project_directory.to_owned(), stdlib_dir()],
            // Really wish you could do ..Default::default() for the remaining fields
            modules: HashMap::default(),
            parse_trees: UnsafeCache::default(),
            name_resolvers: UnsafeCache::default(),
            filepaths: Vec::new(),
            definition_infos: Vec::new(),
            variable_infos: Vec::new(),
            type_bindings: Vec::new(),
            type_infos: Vec::new(),
            trait_infos: Vec::new(),
            impl_infos: Vec::new(),
            impl_scopes: Vec::new(),
            current_trait_constraint_id: Default::default(),
            call_stack: Vec::new(),
            mutual_recursion_sets: Vec::new(),
            effect_infos: Vec::new(),
            global_dependency_graph: DependencyGraph::default(),
            diagnostics: Vec::new(),
            error_count: 0,
            maybe_type: None,
            file_cache,
        }
    }

    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Push a diagnostic and increment the error count if it was an error.
    /// This does not display the diagnostic.
    pub fn push_diagnostic(&mut self, location: Location<'a>, msg: DiagnosticKind) {
        self.push_full_diagnostic(Diagnostic::new(location, msg));
    }

    pub fn push_full_diagnostic(&mut self, diagnostic: Diagnostic<'a>) {
        if diagnostic.error_type() == ErrorType::Error {
            self.error_count += 1;
        }
        self.diagnostics.push(diagnostic);
    }

    pub fn display_diagnostics(&self) {
        for diagnostic in &self.diagnostics {
            let diagnostic = diagnostic.display(self);
            eprintln!("{}", diagnostic);
        }
    }

    pub fn get_contents<'local>(&'local mut self, path: &'a Path) -> Option<&'local str> {
        let contains_path = self.file_cache.contains_key(path);
        let contents = cached_read(&self.file_cache, path)?;

        if !contains_path {
            let contents = contents.into_owned();
            self.file_cache.insert(path.to_path_buf(), contents);
        }

        self.file_cache.get(path).map(|s| s.as_str())
    }

    pub fn strip_root<'b>(&self, path: &'b Path) -> Option<&'b Path> {
        self.relative_roots.iter().find_map(move |root| path.strip_prefix(root).ok())
    }

    #[allow(unused)]
    pub fn get_diagnostics(&self) -> &[Diagnostic<'a>] {
        &self.diagnostics
    }

    pub fn push_filepath(&mut self, path: PathBuf) -> &'a Path {
        let index = self.filepaths.len();
        self.filepaths.push(path);
        let path: &Path = &self.filepaths[index];
        unsafe { std::mem::transmute(path) }
    }

    pub fn push_definition(&mut self, name: &str, location: Location<'a>) -> DefinitionInfoId {
        let id = self.definition_infos.len();
        self.definition_infos.push(DefinitionInfo {
            name: name.to_string(),
            definition: None,
            trait_info: None,
            required_traits: vec![],
            location,
            typ: None,
            uses: 0,
            trait_impl: None,
            mutually_recursive_set: None,
            undergoing_type_inference: false,
            mutually_recursive_variables: vec![],
            ignore_unused_warning: name.starts_with('_'),
        });
        DefinitionInfoId(id)
    }

    pub fn push_ast(&mut self, ast: Ast<'a>) -> ModuleId {
        ModuleId(self.parse_trees.push(ast))
    }

    pub fn push_type_info(&mut self, name: String, args: Vec<TypeVariableId>, location: Location<'a>) -> TypeInfoId {
        let id = self.type_infos.len();
        let type_info = TypeInfo { name, args, location, uses: 0, body: TypeInfoBody::Unknown };
        self.type_infos.push(type_info);
        TypeInfoId(id)
    }

    pub fn get_name_resolver_by_path(&self, path: &Path) -> Option<&mut NameResolver> {
        let id = self.modules.get(path)?;
        self.name_resolvers.get_mut(id.0)
    }

    pub fn next_type_variable_id(&mut self, level: LetBindingLevel) -> TypeVariableId {
        let id = self.type_bindings.len();
        self.type_bindings.push(TypeBinding::Unbound(level, Kind::Normal(0)));
        TypeVariableId(id)
    }

    pub fn next_type_variable(&mut self, level: LetBindingLevel) -> Type {
        let id = self.next_type_variable_id(level);
        Type::TypeVariable(id)
    }

    pub fn push_trait_definition(
        &mut self, name: String, typeargs: Vec<TypeVariableId>, fundeps: Vec<TypeVariableId>,
        trait_node: Option<&'a mut TraitDefinition<'a>>, location: Location<'a>,
    ) -> TraitInfoId {
        let id = self.trait_infos.len();
        self.trait_infos.push(TraitInfo {
            name,
            typeargs,
            fundeps,
            definitions: vec![],
            trait_node,
            location,
            uses: 0,
        });
        TraitInfoId(id)
    }

    pub fn push_trait_impl(
        &mut self, trait_id: TraitInfoId, typeargs: Vec<Type>, definitions: Vec<DefinitionInfoId>,
        trait_impl: &'a mut TraitImpl<'a>, given: Vec<ConstraintSignature>, location: Location<'a>,
    ) -> ImplInfoId {
        let id = self.impl_infos.len();

        // Mark each definition as part of this impl.
        // This is used during type inference to retrieve the `given` constraints from this
        // impl for any definition within it.
        for definition in &definitions {
            self[*definition].trait_impl = Some(ImplInfoId(id));
        }

        self.impl_infos.push(ImplInfo { trait_id, typeargs, definitions, location, given, trait_impl });
        ImplInfoId(id)
    }

    pub fn push_effect_definition(
        &mut self, name: String, typeargs: Vec<TypeVariableId>, effect_node: &'a mut EffectDefinition<'a>,
        location: Location<'a>,
    ) -> EffectInfoId {
        let id = self.effect_infos.len();
        self.effect_infos.push(EffectInfo { name, typeargs, effect_node, declarations: vec![], location });
        EffectInfoId(id)
    }

    pub fn push_impl_scope(&mut self) -> ImplScopeId {
        let id = self.impl_scopes.len();
        self.impl_scopes.push(vec![]);
        ImplScopeId(id)
    }

    pub fn push_variable(&mut self, name: String, location: Location<'a>) -> VariableId {
        let id = self.variable_infos.len();
        self.variable_infos.push(VariableInfo { required_impls: vec![], name, location });
        VariableId(id)
    }

    pub fn next_trait_constraint_id(&mut self) -> TraitConstraintId {
        self.current_trait_constraint_id.next()
    }

    pub fn find_method_in_impl(&self, callsite: VariableId, binding: ImplInfoId) -> DefinitionInfoId {
        let name = &self[callsite].name;

        for definition in &self[binding].definitions {
            if self[*definition].name == *name {
                return *definition;
            }
        }

        unreachable!("No definition for '{}' found in trait impl {}", name, binding.0)
    }

    fn new_recursion_set(
        &mut self, root_definition: DefinitionInfoId, call_stack_index_of_root: usize,
    ) -> MutualRecursionId {
        let id = self.mutual_recursion_sets.len();
        let set = MutualRecursionSet { root_definition, definitions: HashSet::new(), call_stack_index_of_root };
        self.mutual_recursion_sets.push(set);
        let id = MutualRecursionId(id);
        self[root_definition].mutually_recursive_set = Some(id);
        id
    }

    fn find_mutual_recursion_root(&mut self, definition_id: DefinitionInfoId) -> Option<MutualRecursionId> {
        let root = self[definition_id].mutually_recursive_set;
        if self.call_stack.last() == Some(&definition_id) {
            return root;
        }

        let call_stack_index = self.call_stack.iter().position(|id| *id == definition_id);

        let mut root = root.unwrap_or_else(|| self.new_recursion_set(definition_id, call_stack_index.unwrap()));

        let count = call_stack_index.map_or(0, |n| self.call_stack.len() - n);
        for id in self.call_stack.iter().rev().copied().take(count) {
            if id == definition_id {
                break;
            }

            // Mark all the definitions in the set as recursive
            let info = &mut self.definition_infos[id.0];
            match info.mutually_recursive_set {
                Some(set) if set != root => {
                    let (all_sets, all_infos) = (&mut self.mutual_recursion_sets, &mut self.definition_infos);
                    root = Self::merge_recursion_sets(root, set, all_sets, all_infos);
                },
                Some(_root) => (),
                None => {
                    info.mutually_recursive_set = Some(root);
                    self.mutual_recursion_sets[root.0].definitions.insert(id);
                },
            }
        }

        Some(root)
    }

    /// Merge two recursion sets together, returning the dominant one that subsumes both
    fn merge_recursion_sets(
        id1: MutualRecursionId, id2: MutualRecursionId, mutual_recursion_sets: &mut [MutualRecursionSet],
        definition_infos: &mut [DefinitionInfo<'a>],
    ) -> MutualRecursionId {
        let set1 = &mutual_recursion_sets[id1.0];
        let set2 = &mutual_recursion_sets[id2.0];

        assert_ne!(set1.call_stack_index_of_root, set2.call_stack_index_of_root);

        let (dominant_id, other_set) =
            if set1.call_stack_index_of_root < set2.call_stack_index_of_root { (id1, set2) } else { (id2, set1) };

        // Merge all definitions from the other set into the set with the root that is lower on the call stack
        for definition_id in other_set.definitions.iter() {
            definition_infos[definition_id.0].mutually_recursive_set = Some(dominant_id);
        }

        let other_definitions = other_set.definitions.clone();
        let old_root = other_set.root_definition;

        let dominant_set = &mut mutual_recursion_sets[dominant_id.0];
        let new_root = dominant_set.root_definition;
        let definitions = &mut dominant_set.definitions;

        definitions.extend(other_definitions);
        definitions.insert(old_root);
        definitions.remove(&new_root);

        dominant_id
    }

    pub fn update_mutual_recursion_sets(&mut self, definition_id: DefinitionInfoId, variable_id: VariableId) {
        // Type inference checks in depth-first search order, so if a given id is currently
        // undergoing type inference then we can assume it is below us on the callstack somewhere.
        if self[definition_id].undergoing_type_inference {
            self.find_mutual_recursion_root(definition_id);

            // Remember recursive variables regardless of whether this is a mutually recursive set or not.
            // These may be needed if it becomes a mutually recursive set later.
            match self.call_stack.last().copied() {
                Some(top) => {
                    self[top].mutually_recursive_variables.push(variable_id);
                },
                None => {
                    eprintln!("WARNING: {} is used recursively, but callstack is empty", self[variable_id].name);
                },
            }
        }
    }

    pub fn bind(&mut self, id: TypeVariableId, binding: Type) {
        self.type_bindings[id.0] = TypeBinding::Bound(binding);
    }

    pub fn follow_typebindings_shallow<'b>(&'b self, typ: &'b Type) -> &'b Type {
        match typ {
            Type::TypeVariable(id) => match &self.type_bindings[id.0] {
                TypeBinding::Bound(typ) => self.follow_typebindings_shallow(typ),
                TypeBinding::Unbound(_, _) => typ,
            },
            other => other,
        }
    }
}

macro_rules! impl_index_for {
    ( $index_type:ty, $elem_type:tt, $field_name:tt ) => {
        impl<'c> std::ops::Index<$index_type> for ModuleCache<'c> {
            type Output = $elem_type<'c>;

            fn index(&self, index: $index_type) -> &Self::Output {
                &self.$field_name[index.0]
            }
        }

        impl<'c> std::ops::IndexMut<$index_type> for ModuleCache<'c> {
            fn index_mut(&mut self, index: $index_type) -> &mut Self::Output {
                &mut self.$field_name[index.0]
            }
        }
    };
}

impl_index_for!(DefinitionInfoId, DefinitionInfo, definition_infos);
impl_index_for!(TypeInfoId, TypeInfo, type_infos);
impl_index_for!(TraitInfoId, TraitInfo, trait_infos);
impl_index_for!(EffectInfoId, EffectInfo, effect_infos);
impl_index_for!(ImplInfoId, ImplInfo, impl_infos);
impl_index_for!(VariableId, VariableInfo, variable_infos);

impl<'c> std::ops::Index<ImplScopeId> for ModuleCache<'c> {
    type Output = Vec<ImplInfoId>;

    fn index(&self, index: ImplScopeId) -> &Self::Output {
        &self.impl_scopes[index.0]
    }
}

impl<'c> std::ops::IndexMut<ImplScopeId> for ModuleCache<'c> {
    fn index_mut(&mut self, index: ImplScopeId) -> &mut Self::Output {
        &mut self.impl_scopes[index.0]
    }
}
