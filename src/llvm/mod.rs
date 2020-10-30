//! Llvm backend for ante.
//! At the time of writing this is the only backend though in the future there is a cranelift
//! backend planned for faster debug build times and faster build times for the compiler itself
//! so that new users won't have to subject themselves to building llvm.

use crate::cache::{ ModuleCache, DefinitionInfoId, DefinitionKind, VariableId };
use crate::parser::{ ast, ast::Ast };
use crate::nameresolution::builtin::BUILTIN_ID;
use crate::types::{ self, typechecker, TypeVariableId, TypeBinding, TypeInfoId };
use crate::types::traits::RequiredImpl;
use crate::types::typed::Typed;
use crate::util::{ fmap, trustme, reinterpret_from_bits };

use inkwell::module::{ Module, Linkage };
use inkwell::builder::Builder;
use inkwell::basic_block::BasicBlock;
use inkwell::context::Context;
use inkwell::values::{ AggregateValue, BasicValueEnum, BasicValue, FunctionValue, InstructionOpcode };
use inkwell::types::{ BasicTypeEnum, BasicType };
use inkwell::AddressSpace;
use inkwell::targets::{ RelocMode, CodeModel, FileType, TargetTriple };
use inkwell::OptimizationLevel;
use inkwell::passes::{PassManager, PassManagerBuilder};
use inkwell::targets::{InitializationConfig, Target, TargetMachine };

use std::collections::{ HashMap, HashSet };
use std::path::{ Path, PathBuf };
use std::process::Command;

mod builtin;
mod decisiontree;

#[derive(Debug)]
pub struct Generator<'context> {
    context: &'context Context,
    module: Module<'context>,
    builder: Builder<'context>,

    /// Cache of already compiled, monomorphised definitions
    definitions: HashMap<(DefinitionInfoId, types::Type), BasicValueEnum<'context>>,

    /// Cache of mappings from types::Type to LLVM types
    types: HashMap<(types::TypeInfoId, Vec<types::Type>), BasicTypeEnum<'context>>,

    /// Compile-time mapping of variable -> definition for impls that were resolved
    /// after type inference. This is needed for definitions that are polymorphic in
    /// the impls they may use within.
    impl_mappings: HashMap<VariableId, DefinitionInfoId>,

    /// A stack of the current typevar bindings during monomorphisation. Unlike normal bindings,
    /// these are meant to be easily undone. Since ante doesn't support polymorphic recursion,
    /// we also don't have to worry about encountering the same typevar with a different
    /// monomorphisation binding.
    monomorphisation_bindings: Vec<typechecker::TypeBindings>,

    /// Contains all the definition ids that should be automatically dereferenced because they're
    /// either stored locally in an alloca or in a global.
    auto_derefs: HashSet<DefinitionInfoId>,

    current_function_info: Option<DefinitionInfoId>,
}

pub fn run<'c>(path: &Path, ast: &Ast<'c>, cache: &mut ModuleCache<'c>, show_ir: bool,
    run_program: bool, delete_binary: bool, optimization_level: &str)
{
    let context = Context::create();
    let module_name = path_to_module_name(path);
    let module = context.create_module(&module_name);

    let target_triple = TargetMachine::get_default_triple();
    module.set_triple(&target_triple);
    let mut codegen = Generator {
        context: &context,
        module,
        builder: context.create_builder(),
        definitions: HashMap::new(),
        types: HashMap::new(),
        impl_mappings: HashMap::new(),
        monomorphisation_bindings: vec![],
        auto_derefs: HashSet::new(),
        current_function_info: None,
    };

    codegen.codegen_main(ast, cache);

    codegen.module.verify().map_err(|error| {
        codegen.module.print_to_stderr();
        println!("{}", error);
    }).unwrap();

    codegen.optimize(optimization_level);

    // --show-llvm-ir: Dump the LLVM-IR of the generated module to stderr.
    // Useful to debug codegen
    if show_ir {
        codegen.module.print_to_stderr();
    }

    let binary_name = module_name_to_program_name(&module_name);
    codegen.output(module_name, &binary_name, &target_triple, &codegen.module);

    // --run: compile and run the program
    if run_program {
        let program_command = PathBuf::from("./".to_string() + &binary_name);
        Command::new(&program_command).spawn().unwrap().wait().unwrap();
    }

    // --delete-binary: remove the binary after running the program to
    // avoid littering a testing directory with temporary binaries
    if delete_binary {
        std::fs::remove_file(binary_name).unwrap();
    }
}

fn path_to_module_name(path: &Path) -> String {
    path.with_extension("").to_string_lossy().into()
}

fn module_name_to_program_name(module: &str) -> String {
    if cfg!(target_os = "windows") {
        PathBuf::from(module).with_extension("exe").to_string_lossy().into()
    } else {
        PathBuf::from(module).with_extension("").to_string_lossy().into()
    }
}

fn remove_forall(typ: &types::Type) -> &types::Type {
    match typ {
        types::Type::ForAll(_, t) => t,
        _ => typ,
    }
}

// TODO: remove
const UNBOUND_TYPE: types::Type = types::Type::Primitive(types::PrimitiveType::UnitType);

fn to_optimization_level(optimization_argument: &str) -> OptimizationLevel {
    match optimization_argument {
        "1" => OptimizationLevel::Less,
        "2" => OptimizationLevel::Default,
        "3" => OptimizationLevel::Aggressive,
        _ => OptimizationLevel::None,
    }
}

fn to_size_level(optimization_argument: &str) -> u32 {
    match optimization_argument {
        "s" => 1,
        "z" => 2,
        _ => 0,
    }
}

impl<'g> Generator<'g> {
    fn codegen_main<'c>(&mut self, ast: &Ast<'c>, cache: &mut ModuleCache<'c>) {
        let i32_type = self.context.i32_type();
        let main_type = i32_type.fn_type(&[], false);
        let function = self.module.add_function("main", main_type, Some(Linkage::External));
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        ast.codegen(self, cache);

        let success = i32_type.const_int(0, true);
        self.build_return(success.into());
    }

    fn optimize(&self, optimization_argument: &str) {
        let config = InitializationConfig::default();
        Target::initialize_native(&config).unwrap();
        let pass_manager_builder = PassManagerBuilder::create();

        let optimization_level = to_optimization_level(optimization_argument);
        let size_level = to_size_level(optimization_argument);
        pass_manager_builder.set_optimization_level(optimization_level);
        pass_manager_builder.set_size_level(size_level);

        let pass_manager = PassManager::create(());
        pass_manager_builder.populate_module_pass_manager(&pass_manager);
        pass_manager.add_tail_call_elimination_pass();
        pass_manager.run_on(&self.module);

        // Do LTO optimizations afterward mosty for function inlining
        let link_time_optimizations = PassManager::create(());
        pass_manager_builder.populate_lto_pass_manager(&link_time_optimizations, false, true);
        link_time_optimizations.run_on(&self.module);
    }

    fn output(&self, module_name: String, binary_name: &str, target_triple: &TargetTriple, module: &Module) {
        // generate the bitcode to a .bc file
        let path = Path::new(&module_name).with_extension("o");
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target.create_target_machine(&target_triple, "x86-64", "+avx2",
                OptimizationLevel::None, RelocMode::PIC, CodeModel::Default).unwrap();

        target_machine.write_to_file(&module, FileType::Object, &path).unwrap();

        // call gcc to compile the bitcode to a binary
        let output = "-o".to_string() + binary_name;
        let mut child = Command::new("gcc")
            .arg(path.to_string_lossy().as_ref())
            .arg("-Wno-everything")
            .arg("-O0")
            .arg("-lm")
            .arg(output)
            .spawn().unwrap();

        // remove the temporary bitcode file
        child.wait().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    fn lookup<'c>(&mut self, id: DefinitionInfoId, typ: &types::Type, cache: &mut ModuleCache<'c>) -> Option<BasicValueEnum<'g>> {
        let typ = self.follow_bindings(typ, cache);
        self.definitions.get(&(id, typ)).map(|value| *value)
    }

    /// Return the inkwell function we're currently inserting into
    fn current_function(&self) -> FunctionValue<'g> {
        self.current_block().get_parent().unwrap()
    }

    /// Return the llvm block we're currently inserting into
    fn current_block(&self) -> BasicBlock<'g> {
        self.builder.get_insert_block().unwrap()
    }

    /// Append a new BasicBlock into the current function and set it
    /// as the current insert point.
    fn insert_into_new_block(&self, block_name: &str) -> BasicBlock<'g> {
        let current_function = self.current_function();
        let block = self.context.append_basic_block(current_function, block_name);
        self.builder.position_at_end(block);
        block
    }

    /// Create a new function with the given name and type and set
    /// its entry block as the current insert point. Returns the
    /// function value as a pointer.
    fn function<'c>(&mut self, name: &str, typ: &types::Type, cache: &ModuleCache<'c>) -> (FunctionValue<'g>, BasicValueEnum<'g>) {
        let llvm_type = self.convert_type(&typ, cache).into_pointer_type().get_element_type();

        let function = self.module.add_function(name, llvm_type.into_function_type(), Some(Linkage::Internal));
        let function_pointer = function.as_global_value().as_pointer_value().into();

        if let Some(id) = self.current_function_info {
            let typ = self.follow_bindings(typ, cache);
            self.definitions.insert((id, typ), function_pointer);
            self.current_function_info = None;
        }

        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);
        (function, function_pointer)
    }

    fn add_required_impls<'c>(&mut self, required_impls: &[RequiredImpl]) {
        for required_impl in required_impls {
            assert!(!self.impl_mappings.contains_key(&required_impl.origin));
            self.impl_mappings.insert(required_impl.origin, required_impl.binding);
        }
    }

    fn remove_required_impls<'c>(&mut self, required_impls: &[RequiredImpl]) {
        for required_impl in required_impls {
            self.impl_mappings.remove(&required_impl.origin);
        }
    }

    /// Codegen a given definition unless it has been already.
    /// If it has been already codegen'd, return the cached value instead.
    fn codegen_definition<'c>(&mut self, id: DefinitionInfoId, typ: &types::Type, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        match self.lookup(id, typ, cache) {
            Some(value) => value,
            None => self.monomorphise(id, typ, cache).unwrap()
        }
    }

    /// Get the DefinitionInfoId this variable should point to. This is usually
    /// given by variable.definition but in the case of static trait dispatch,
    /// self.impl_mappings may be set to bind a given variable id to another
    /// definition. This is currently only done for trait functions/values to
    /// point them to impls that actually have definitions.
    fn get_definition_id<'c>(&self, variable: &ast::Variable<'c>) -> DefinitionInfoId {
        self.impl_mappings.get(&variable.id.unwrap())
            .copied().unwrap_or(variable.definition.unwrap())
    }

    fn monomorphise<'c>(&mut self, id: DefinitionInfoId, typ: &types::Type, cache: &mut ModuleCache<'c>) -> Option<BasicValueEnum<'g>> {
        let definition = &mut cache.definition_infos[id.0];
        let definition = trustme::extend_lifetime(definition);
        let definition_type = remove_forall(definition.typ.as_ref().unwrap());

        let mut bindings = HashMap::new();
        typechecker::try_unify(typ, definition_type, &mut bindings, definition.location, cache)
            .map_err(|error| println!("{}", error))
            .expect("Unification error during monomorphisation");

        self.monomorphisation_bindings.push(bindings);

        // Compile the definition with the bindings in scope. Each definition is expected to
        // add itself to Generator.definitions
        let value = match &definition.definition {
            Some(DefinitionKind::Definition(definition)) => {
                Some(self.codegen_monomorphise(*definition, cache))
            }
            Some(DefinitionKind::Extern(_)) => {
                Some(self.codegen_extern(id, typ, cache))
            }
            Some(DefinitionKind::TypeConstructor { name, tag }) => {
                Some(self.codegen_type_constructor(name, tag, typ, cache))
            },
            Some(DefinitionKind::TraitDefinition(_)) => {
                unreachable!("There is no code in a trait definition that can be codegen'd.\n\
                             No cached impl for {}: {}", definition.name, typ.display(cache))
            },
            Some(DefinitionKind::Parameter) => {
                unreachable!("There is no code to (lazily) codegen for parameters.\n\
                             Encountered while compiling {}: {}", definition.name, typ.display(cache))
            },
            Some(DefinitionKind::MatchPattern) => {
                unreachable!("There is no code to (lazily) codegen for match patterns.\n
                             Encountered while compiling {}: {}", definition.name, typ.display(cache))
            },
            None => unreachable!("No definition for {}", definition.name),
        };

        self.monomorphisation_bindings.pop();
        value
    }

    fn find_binding<'c, 'b>(&'b self, id: TypeVariableId, cache: &'b ModuleCache<'c>) -> &types::Type {
        use types::TypeBinding::*;
        use types::Type::TypeVariable;

        match &cache.type_bindings[id.0] {
            Bound(TypeVariable(id)) => self.find_binding(*id, cache),
            Bound(binding) => binding,
            Unbound(..) => {
                for bindings in self.monomorphisation_bindings.iter().rev() {
                    if let Some(binding) = bindings.get(&id) {
                        return binding;
                    }
                }
                // println!("Unbound type variable found during code generation");
                &UNBOUND_TYPE
            },
        }
    }

    fn size_of_type<'c>(&self, typ: &types::Type, cache: &ModuleCache<'c>) -> usize {
        use types::Type::*;
        use types::PrimitiveType::*;
        match typ {
            Primitive(IntegerType) => 4,
            Primitive(FloatType) => 8,
            Primitive(CharType) => 1,
            Primitive(BooleanType) => 1,
            Primitive(UnitType) => 1,
            Primitive(ReferenceType) => 8,

            Function(..) => 8,

            TypeVariable(id) => {
                let binding = self.find_binding(*id, cache);
                self.size_of_type(binding, cache)
            },

            UserDefinedType(id) => {
                let _info = &cache.type_infos[id.0];
                unimplemented!("size_of_type(UserDefinedType) is unimplemented");
            },

            TypeApplication(_typ, _args) => {
                unimplemented!("size_of_type(TypeApplication) is unimplemented");
            },

            Tuple(elements) => {
                elements.iter().map(|element| self.size_of_type(element, cache)).sum()
            }

            ForAll(_, typ) => self.size_of_type(typ, cache),
        }
    }

    fn convert_primitive_type(&self, typ: &types::PrimitiveType) -> BasicTypeEnum<'g> {
        use types::PrimitiveType::*;
        match typ {
            IntegerType => self.context.i32_type().into(),
            FloatType => self.context.f64_type().into(),
            CharType => self.context.i8_type().into(),
            BooleanType => self.context.bool_type().into(),
            UnitType => self.context.bool_type().into(),
            ReferenceType => unreachable!("Kind error during code generation"),
        }
    }

    fn convert_struct_type<'c>(&mut self, id: TypeInfoId, info: &types::TypeInfo, fields: &[types::Field<'c>],
        args: Vec<types::Type>, cache: &ModuleCache<'c>) -> BasicTypeEnum<'g>
    {
        let bindings = info.args.iter().copied().zip(args.iter().cloned()).collect();

        let typ = self.context.opaque_struct_type(&info.name);
        self.types.insert((id, args), typ.into());

        let fields = fmap(&fields, |field| {
            let field_type = typechecker::bind_typevars(&field.field_type, &bindings, cache);
            self.convert_type(&field_type, cache)
        });

        typ.set_body(&fields, false);
        typ.into()
    }

    fn convert_union_type<'c>(&mut self, id: TypeInfoId, info: &types::TypeInfo, variants: &[types::TypeConstructor<'c>],
        args: Vec<types::Type>, cache: &ModuleCache<'c>) -> BasicTypeEnum<'g>
    {
        let bindings = info.args.iter().copied().zip(args.iter().cloned()).collect();

        let typ = self.context.opaque_struct_type(&info.name);
        self.types.insert((id, args), typ.into());

        let variants: Vec<Vec<types::Type>> = fmap(&variants, |variant| {
            fmap(&variant.args, |arg| typechecker::bind_typevars(arg, &bindings, cache))
        });

        let mut max_size = 0;
        let mut largest_variant = None;
        for variant in variants.into_iter() {
            let size: usize = variant.iter().map(|arg| self.size_of_type(arg, cache)).sum();
            if size >= max_size {
                largest_variant = Some(variant);
                max_size = size;
            }
        }

        if let Some(variant) = largest_variant {
            let mut fields = vec![self.tag_type()];
            for typ in variant {
                fields.push(self.convert_type(&typ, cache));
            }
            typ.set_body(&fields, false);
        }

        typ.into()
    }

    fn convert_user_defined_type<'c>(&mut self, id: TypeInfoId, args: Vec<types::Type>, cache: &ModuleCache<'c>) -> BasicTypeEnum<'g> {
        let info = &cache.type_infos[id.0];
        assert!(info.args.len() == args.len(), "Kind error during llvm code generation");

        if let Some(typ) = self.types.get(&(id, args.clone())) {
            return *typ;
        }

        use types::TypeInfoBody::*;
        let typ = match &info.body {
            Union(variants) => self.convert_union_type(id, info, variants, args, cache),
            Struct(fields) => self.convert_struct_type(id, info, fields, args, cache),

            // TODO: handle aliases with type arguments
            Alias(typ) => {
                let converted = self.convert_type(typ, cache);
                self.types.insert((id, args), converted);
                converted
            },
            Unknown => unreachable!(),
        };

        typ
    }

    fn convert_type<'c>(&mut self, typ: &types::Type, cache: &ModuleCache<'c>) -> BasicTypeEnum<'g> {
        use types::Type::*;
        use types::PrimitiveType::ReferenceType;
        match typ {
            Primitive(primitive) => self.convert_primitive_type(primitive),

            Function(arg_types, return_type) => {
                let args = fmap(arg_types, |typ| self.convert_type(typ, cache));
                let return_type = self.convert_type(return_type, cache);
                return_type.fn_type(&args, false).ptr_type(AddressSpace::Global).into()
            },

            TypeVariable(id) => self.convert_type(&self.find_binding(*id, cache).clone(), cache),

            UserDefinedType(id) => self.convert_user_defined_type(*id, vec![], cache),

            Tuple(elements) => {
                let element_types = fmap(elements, |element| self.convert_type(element, cache));
                self.context.struct_type(&element_types, false).as_basic_type_enum()
            },

            TypeApplication(typ, args) => {
                let args = fmap(args, |arg| self.follow_bindings(arg, cache));
                let typ = self.follow_bindings(typ, cache);

                match &typ {
                    Primitive(ReferenceType) => {
                        assert!(args.len() == 1);
                        self.convert_type(&args[0], cache).ptr_type(AddressSpace::Global).into()
                    },
                    UserDefinedType(id) => self.convert_user_defined_type(*id, args, cache),
                    _ => {
                        unreachable!("Type {} requires 0 type args but was applied to {:?}", typ.display(cache), args);
                    }
                }
            },

            ForAll(_, typ) => self.convert_type(typ, cache),
        }
    }

    fn unit_value(&self) -> BasicValueEnum<'g> {
        // TODO: compile () to void, mainly higher-order functions and struct/tuple
        // indexing need to be addressed for this.
        let i1 = self.context.bool_type();
        i1.const_int(0, false).into()
    }

    fn integer_value(&self, value: u64) -> BasicValueEnum<'g> {
        self.context.i32_type().const_int(value, true).as_basic_value_enum()
    }

    fn char_value(&self, value: u64) -> BasicValueEnum<'g> {
        self.context.i8_type().const_int(value, false).into()
    }

    fn bool_value(&self, value: bool) -> BasicValueEnum<'g> {
        self.context.bool_type().const_int(value as u64, false).into()
    }

    fn float_value(&self, value: f64) -> BasicValueEnum<'g> {
        self.context.f64_type().const_float(value).into()
    }

    fn string_value<'c>(&mut self, contents: &str, cache: &ModuleCache<'c>) -> BasicValueEnum<'g> {
        let literal = self.context.const_string(contents.as_bytes(), true);
        let global = self.module.add_global(literal.get_type(), None, "string_literal");
        global.set_initializer(&literal);
        let value = global.as_pointer_value();
        let cstring_type = self.context.i8_type().ptr_type(AddressSpace::Global);
        let cast = self.builder.build_pointer_cast(value, cstring_type, "string_cast");

        let string_type = types::Type::UserDefinedType(types::STRING_TYPE);
        let string_type = self.convert_type(&string_type, cache).into_struct_type();
        let length = self.context.i32_type().const_int(contents.len() as u64, false);

        string_type.const_named_struct(&[cast.into(), length.into()]).into()
    }

    fn follow_bindings<'c>(&self, typ: &types::Type, cache: &ModuleCache<'c>) -> types::Type {
        use types::Type::*;
        match typ {
            Primitive(primitive) => Primitive(*primitive),

            Function(arg_types, return_type) => {
                let args = fmap(arg_types, |typ| self.follow_bindings(typ, cache));
                let return_type = self.follow_bindings(return_type, cache);
                Function(args, Box::new(return_type))
            },

            TypeVariable(id) => self.follow_bindings(self.find_binding(*id, cache), cache),

            UserDefinedType(id) => UserDefinedType(*id),

            TypeApplication(typ, args) => {
                let typ = self.follow_bindings(typ, cache);
                let args = fmap(args, |arg| self.follow_bindings(arg, cache));
                TypeApplication(Box::new(typ), args)
            },

            Tuple(elements) => {
                Tuple(fmap(elements, |element| self.follow_bindings(element, cache)))
            },

            // unwrap foralls
            ForAll(_, typ) => self.follow_bindings(typ, cache),
        }
    }

    fn bind_irrefutable_pattern<'c>(&mut self, ast: &Ast<'c>, mut value: BasicValueEnum<'g>, cache: &mut ModuleCache<'c>) {
        use { ast::LiteralKind, Ast::* };
        match ast {
            Literal(literal) => {
                assert!(literal.kind == LiteralKind::Unit)
                // pass, we don't need to actually do any assignment when ignoring unit values
            },
            Variable(variable) => {
                let id = variable.definition.unwrap();
                let typ = self.follow_bindings(variable.typ.as_ref().unwrap(), cache);

                let definition = &cache.definition_infos[id.0];
                if definition.mutable {
                    let alloca = self.builder.build_alloca(value.get_type(), &definition.name);
                    self.builder.build_store(alloca, value);
                    self.auto_derefs.insert(id);
                    value = alloca.as_basic_value_enum();
                } else {
                    // This line isn't currently needed but will be if ante is ever
                    // generic over mutability
                    self.auto_derefs.remove(&id);
                }

                self.definitions.insert((id, typ), value);
            },
            TypeAnnotation(annotation) => {
                self.bind_irrefutable_pattern(annotation.lhs.as_ref(), value, cache);
            },
            Tuple(tuple) => {
                for (i, element) in tuple.elements.iter().enumerate() {
                    let element_value = self.builder.build_extract_value(value.into_struct_value(), i as u32, "extract").unwrap();
                    self.bind_irrefutable_pattern(element, element_value, cache);
                }
            },
            _ => {
                unreachable!();
            }
        }
    }

    // codegen a Definition that should be monomorphised.
    // Really all definitions should be monomorphised, this is just used as a wrapper so
    // we only compilie function definitions when they're used at their call sites so that
    // we have all the monomorphisation bindings in scope.
    fn codegen_monomorphise<'c>(&mut self, definition: &ast::Definition<'c>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        // If we're defining a lambda, give the lambda info on DefinitionInfoId so that it knows
        // what to name itself in the IR and so recursive functions can properly codegen without
        // attempting to re-compile themselves over and over.
        match (definition.pattern.as_ref(), definition.expr.as_ref()) {
            (Ast::Variable(variable), Ast::Lambda(_)) => {
                self.current_function_info = Some(variable.definition.unwrap());
            }
            _ => (),
        }

        let value = definition.expr.codegen(self, cache);
        self.bind_irrefutable_pattern(definition.pattern.as_ref(), value, cache);
        value
    }

    // Is this a (possibly generalized) function type?
    // Used when to differentiate extern C functions/values when compiling Extern declarations.
    fn is_function_type<'c>(&self, typ: &types::Type, cache: &ModuleCache<'c>) -> bool {
        use types::Type::*;
        let typ = self.follow_bindings(typ, cache);
        match typ {
            Function(..) => true,
            ForAll(_, typ) => self.is_function_type(typ.as_ref(), cache),
            _ => false,
        }
    }

    fn codegen_extern<'c>(&mut self, id: DefinitionInfoId, typ: &types::Type, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        // extern definitions should only be declared once - never duplicated & monomorphised.
        // For this reason their value is always stored with the Unit type in the definitions map.
        if let Some(value) = self.lookup(id, &UNBOUND_TYPE, cache) {
            self.definitions.insert((id, typ.clone()), value);
            return value;
        }

        let llvm_type = self.convert_type(typ, cache);
        let name = &cache.definition_infos[id.0].name;

        let global = if self.is_function_type(typ, cache) {
            let function_type = llvm_type.into_pointer_type().get_element_type().into_function_type();
            self.module.add_function(name, function_type, Some(Linkage::External)).as_global_value().as_basic_value_enum()
        } else {
            self.auto_derefs.insert(id);
            self.module.add_global(llvm_type, None, name).as_basic_value_enum()
        };

        // Insert the global for both the current type and the unit type
        self.definitions.insert((id, typ.clone()), global);
        self.definitions.insert((id, UNBOUND_TYPE.clone()), global);
        global
    }

    fn codegen_type_constructor<'c>(&mut self, name: &str, tag: &Option<u8>, typ: &types::Type, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        use types::Type::*;
        let typ = self.follow_bindings(typ, cache);
        match &typ {
            Function(_, return_type) => {
                let caller_block = self.current_block();
                let (function, function_pointer) = self.function(name, &typ, cache);

                let mut elements = vec![];
                let mut element_types = vec![];

                if let Some(tag) = tag {
                    let tag_value = self.tag_value(*tag);
                    elements.push(tag_value);
                    element_types.push(tag_value.get_type());
                }

                for parameter in function.get_param_iter() {
                    elements.push(parameter);
                    element_types.push(parameter.get_type());
                }

                let tuple = self.tuple(elements, element_types);
                let value = self.reinterpret_cast(tuple, &return_type, cache);

                self.build_return(value);
                self.builder.position_at_end(caller_block);

                function_pointer
            },
            // Since this is not a function type, we know it has no bundled data and we can
            // thus ignore the additional type arguments, extract the tag value, and
            // reinterpret_cast to the appropriate type.
            UserDefinedType(_) | TypeApplication(_, _) => {
                let value = tag.map_or(self.unit_value(), |tag| self.tag_value(tag));
                self.reinterpret_cast(value, &typ, cache)
            },
            ForAll(_, typ) => {
                self.codegen_type_constructor(name, tag, &typ, cache)
            },
            _ => unreachable!("Type constructor's type is neither a Function or a  UserDefinedType, {}: {}", name, typ.display(cache)),
        }
    }

    /// Does the given llvm instruction terminate its BasicBlock?
    /// This currently only checks for cases that can actually occur
    /// while codegening an arbitrary Ast node.
    fn current_instruction_is_block_terminator(&self) -> bool {
        let instruction = self.current_block().get_last_instruction();
        match instruction.map(|instruction| instruction.get_opcode()) {
            Some(InstructionOpcode::Return) => true,
            Some(InstructionOpcode::Unreachable) => true,
            _ => false,
        }
    }

    fn build_return(&mut self, return_value: BasicValueEnum<'g>) {
        if !self.current_instruction_is_block_terminator() {
            self.builder.build_return(Some(&return_value));
        }
    }

    /// It is an error in llvm to insert a block terminator (like a br) after
    /// the block has already ended from another block terminator (like a return).
    ///
    /// Since returns can happen within a branch, this function should be used to
    /// check that the branch hasn't yet terminated before inserting a br after
    /// a then/else branch, pattern match, or looping construct.
    fn codegen_branch<'c>(&mut self, branch: &ast::Ast<'c>, end_block: BasicBlock<'g>,
        cache: &mut ModuleCache<'c>) -> Option<(BasicBlock<'g>, BasicValueEnum<'g>)>
    {
        let branch_value = branch.codegen(self, cache);
        let branch_block = self.current_block();

        if self.current_instruction_is_block_terminator() {
            None
        } else {
            self.builder.build_unconditional_branch(end_block);
            Some((branch_block, branch_value))
        }
    }

    /// Returns the type of a tag in an unoptimized tagged union
    fn tag_type(&self) -> BasicTypeEnum<'g> {
        self.context.i8_type().as_basic_type_enum()
    }

    /// Returns the value of a tag for a given variant of a tagged union
    fn tag_value(&self, tag: u8) -> BasicValueEnum<'g> {
        self.context.i8_type().const_int(tag as u64, false).as_basic_value_enum()
    }

    fn reinterpret_cast_llvm_type<'c>(&mut self, value: BasicValueEnum<'g>, target_type: BasicTypeEnum<'g>) -> BasicValueEnum<'g> {
        let source_type = value.get_type();
        let alloca = self.builder.build_alloca(source_type, "alloca");
        self.builder.build_store(alloca, value);

        let target_type = target_type.ptr_type(AddressSpace::Global);
        let cast = self.builder.build_pointer_cast(alloca, target_type, "cast");
        self.builder.build_load(cast, "union_cast")
    }

    fn reinterpret_cast<'c>(&mut self, value: BasicValueEnum<'g>, target_type: &types::Type, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let target_type = self.convert_type(target_type, cache);
        self.reinterpret_cast_llvm_type(value, target_type)
    }

    fn tuple<'c>(&mut self, elements: Vec<BasicValueEnum<'g>>, element_types: Vec<BasicTypeEnum<'g>>) -> BasicValueEnum<'g> {
        let tuple_type = self.context.struct_type(&element_types, false);

        // LLVM wants the const elements to be included in the struct literal itself.
        // Attempting to do build_insert_value would a const value will return the struct as-is
        // without mutating the existing struct.
        let const_elements = fmap(&elements, |element| {
            if Self::is_const(*element) {
                *element
            } else {
                Self::undef_value(element.get_type())
            }
        });

        let mut tuple = tuple_type.const_named_struct(&const_elements).as_aggregate_value_enum();

        // Now insert all the non-const values
        for (i, element) in elements.into_iter().enumerate() {
            if !Self::is_const(element) {
                tuple = self.builder.build_insert_value(tuple, element, i as u32, "insert").unwrap();
            }
        }

        tuple.as_basic_value_enum()
    }

    fn is_const(value: BasicValueEnum<'g>) -> bool {
        match value {
            BasicValueEnum::ArrayValue(array) => array.is_const(),
            BasicValueEnum::FloatValue(float) => float.is_const(),
            BasicValueEnum::IntValue(int) => int.is_const(),
            BasicValueEnum::PointerValue(pointer) => pointer.is_const(),
            BasicValueEnum::StructValue(_) => false,
            BasicValueEnum::VectorValue(vector) => vector.is_const(),
        }
    }

    fn undef_value(typ: BasicTypeEnum<'g>) -> BasicValueEnum<'g> {
        match typ {
            BasicTypeEnum::ArrayType(array) => array.get_undef().into(),
            BasicTypeEnum::FloatType(float) => float.get_undef().into(),
            BasicTypeEnum::IntType(int) => int.get_undef().into(),
            BasicTypeEnum::PointerType(pointer) => pointer.get_undef().into(),
            BasicTypeEnum::StructType(tuple) => tuple.get_undef().into(),
            BasicTypeEnum::VectorType(vector) => vector.get_undef().into(),
        }
    }

    fn get_field_index<'c>(&self, field_name: &str, typ: &types::Type, cache: &ModuleCache<'c>) -> u32 {
        use types::Type::*;
        match self.follow_bindings(typ, cache) {
            UserDefinedType(id) => {
                cache.type_infos[id.0].find_field(field_name).map(|(i, _)| i).unwrap()
            },
            TypeVariable(id) => {
                match &cache.type_bindings[id.0] {
                    TypeBinding::Bound(_) => unreachable!("Type variable {} is bound but its binding wasn't found by follow_bindings", id.0),
                    TypeBinding::Unbound(..) => unreachable!("Type variable {} is unbound", id.0),
                }
            },
            _ => {
                unreachable!("get_field_index called with a type that clearly doesn't have a {} field: {}", field_name, typ.display(cache));
            }
        }
    }
}

trait CodeGen<'g, 'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g>;
}

impl<'g, 'c> CodeGen<'g, 'c> for Ast<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        dispatch_on_expr!(self, CodeGen::codegen, generator, cache)
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Literal<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        self.kind.codegen(generator, cache)
    }
}

impl <'g, 'c> CodeGen<'g, 'c> for ast::LiteralKind {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        match self {
            ast::LiteralKind::Char(c) => generator.char_value(*c as u64),
            ast::LiteralKind::Bool(b) => generator.bool_value(*b),
            ast::LiteralKind::Float(f) => generator.float_value(reinterpret_from_bits(*f)),
            ast::LiteralKind::Integer(i) => generator.integer_value(*i),
            ast::LiteralKind::String(s) => generator.string_value(s, cache),
            ast::LiteralKind::Unit => generator.unit_value(),
        }
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Variable<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let required_impls = &cache.trait_bindings[self.trait_binding.unwrap().0].required_impls.clone();
        generator.add_required_impls(&required_impls);

        // The definition to compile is either the corresponding impl definition if this
        // variable refers to a trait function, or otherwise it is the regular definition of this variable.
        let id = generator.get_definition_id(self);
        let mut value = generator.codegen_definition(id, self.typ.as_ref().unwrap(), cache);

        generator.remove_required_impls(&required_impls);

        if generator.auto_derefs.contains(&id) {
            value = generator.builder.build_load(value.into_pointer_value(), &self.to_string());
        }

        value
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Lambda<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let function_name = match &generator.current_function_info {
            Some(id) => &cache.definition_infos[id.0].name,
            None => "lambda",
        };

        let caller_block = generator.current_block();
        let function_type = self.typ.as_ref().unwrap();
        let (function, function_pointer) = generator.function(&function_name, function_type, cache);

        // Bind each parameter node to the nth parameter of `function`
        for (i, parameter) in self.args.iter().enumerate() {
            let value = function.get_nth_param(i as u32).unwrap();
            generator.bind_irrefutable_pattern(parameter, value, cache);
        }

        let return_value = self.body.codegen(generator, cache);

        generator.build_return(return_value);
        generator.builder.position_at_end(caller_block);

        function_pointer
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::FunctionCall<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        match self.function.as_ref() {
            Ast::Variable(variable) if variable.definition == Some(BUILTIN_ID) => {
                // TODO: improve this control flow so that the fast path of normal function calls
                // doesn't have to check the rare case of a builtin function call.
                builtin::call_builtin(&self.args, generator)
            },
            _ => {
                let function = self.function.codegen(generator, cache);
                let args = fmap(&self.args, |arg| arg.codegen(generator, cache));
                generator.builder.build_call(function.into_pointer_value(), &args, "")
                    .try_as_basic_value().left().unwrap()
            },
        }
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Definition<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        match self.expr.as_ref() {
            // If the value is a function we can skip it and come back later to only compile it
            // when it is actually used. This saves the optimizer some work since we won't ever
            // have to search for and remove unused functions.
            Ast::Lambda(_) => (),
            _ => {
                let value = self.expr.codegen(generator, cache);
                generator.bind_irrefutable_pattern(self.pattern.as_ref(), value, cache);
            },
        }
        generator.unit_value()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::If<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let condition = self.condition.codegen(generator, cache);

        let current_function = generator.current_function();
        let then_block = generator.context.append_basic_block(current_function, "then");
        let end_block = generator.context.append_basic_block(current_function, "end_if");

        if let Some(otherwise) = &self.otherwise {
            // Setup conditional jump
            let else_block = generator.context.append_basic_block(current_function, "else");
            generator.builder.build_conditional_branch(condition.into_int_value(), then_block, else_block);

            generator.builder.position_at_end(then_block);
            let then_option = generator.codegen_branch(&self.then, end_block, cache);

            generator.builder.position_at_end(else_block);
            let else_option = generator.codegen_branch(otherwise, end_block, cache);

            // Create phi at the end of the if beforehand
            generator.builder.position_at_end(end_block);

            // Some of the branches may have terminated early. We need to check each case to
            // determine which we should add to the phi or if we should even create a phi at all.
            match (then_option, else_option) {
                (Some((then_branch, then_value)), Some((else_branch, else_value))) => {
                    let phi = generator.builder.build_phi(then_value.get_type(), "if_result");
                    phi.add_incoming(&[(&then_value, then_branch), (&else_value, else_branch)]);
                    phi.as_basic_value()
                }
                (Some((_, then_value)), None) => then_value,
                (None, Some((_, else_value))) => else_value,
                (None, None) => {
                    generator.builder.build_unreachable();

                    // Block is unreachable but we still need to return an undef value.
                    // If we return None the compiler would crash while compiling
                    // `2 + if true return "uh" else return "oh"`
                    let if_result_type = generator.convert_type(self.get_type().unwrap(), cache);
                    Generator::undef_value(if_result_type)
                },
            }
        } else {
            generator.builder.build_conditional_branch(condition.into_int_value(), then_block, end_block);

            generator.builder.position_at_end(then_block);
            generator.codegen_branch(&self.then, end_block, cache);

            generator.builder.position_at_end(end_block);
            generator.unit_value()
        }
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Match<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        generator.codegen_tree(self.decision_tree.as_ref().unwrap(), self, cache)
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::TypeDefinition<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, _cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        generator.unit_value()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::TypeAnnotation<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        self.lhs.codegen(generator, cache)
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Import<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, _cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        generator.unit_value()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::TraitDefinition<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, _cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        generator.unit_value()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::TraitImpl<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, _cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        generator.unit_value()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Return<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let value = self.expression.codegen(generator, cache);
        generator.builder.build_return(Some(&value));
        value
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Sequence<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        assert!(!self.statements.is_empty());

        for statement in self.statements.iter().take(self.statements.len() - 1) {
            statement.codegen(generator, cache);
        }

        self.statements.last().unwrap().codegen(generator, cache)
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Extern<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, _cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        generator.unit_value()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::MemberAccess<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let lhs = self.lhs.codegen(generator, cache);
        let collection = lhs.into_struct_value();

        let index = generator.get_field_index(&self.field, self.lhs.get_type().unwrap(), cache);
        generator.builder.build_extract_value(collection, index, &self.field).unwrap()
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Tuple<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let mut elements = vec![];
        let mut element_types = vec![];

        for element in self.elements.iter() {
            let value = element.codegen(generator, cache);
            element_types.push(value.get_type());
            elements.push(value);
        }

        generator.tuple(elements, element_types)
    }
}

impl<'g, 'c> CodeGen<'g, 'c> for ast::Assignment<'c> {
    fn codegen(&self, generator: &mut Generator<'g>, cache: &mut ModuleCache<'c>) -> BasicValueEnum<'g> {
        let lhs = self.lhs.codegen(generator, cache);
        let lhs_instruction = lhs.as_instruction_value().unwrap();

        assert_eq!(lhs_instruction.get_opcode(), InstructionOpcode::Load);

        let lhs = lhs_instruction.get_operand(0).unwrap().left().unwrap().into_pointer_value();
        let rhs = self.rhs.codegen(generator, cache);
        generator.builder.build_store(lhs, rhs);
        generator.unit_value()
    }
}
