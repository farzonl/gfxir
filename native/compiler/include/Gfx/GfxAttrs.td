#ifndef GFX_ATTRS_TD
#define GFX_ATTRS_TD
include "Gfx/GfxDialect.td"
include "mlir/IR/EnumAttr.td"

def Gfx_Stage : I32EnumAttr<"Stage", "execution stage", [
  I32EnumAttrCase<"Vertex", 0, "vertex">,
  I32EnumAttrCase<"Fragment", 1, "fragment">,
  I32EnumAttrCase<"Compute", 2, "compute">,
  I32EnumAttrCase<"Task", 3, "task">,
  I32EnumAttrCase<"Mesh", 4, "mesh">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}
def Gfx_StageAttr : EnumAttr<Gfx_Dialect, Gfx_Stage, "stage">;

def Gfx_Interpolation : I32EnumAttr<"Interpolation", "interpolation rule", [
  I32EnumAttrCase<"Perspective", 0, "perspective">,
  I32EnumAttrCase<"Flat", 1, "flat">,
  I32EnumAttrCase<"NoPerspective", 2, "no_perspective">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}
def Gfx_InterpolationAttr : EnumAttr<Gfx_Dialect, Gfx_Interpolation, "interpolation">;

def Gfx_Builtin : I32EnumAttr<"Builtin", "device-provided input", [
  I32EnumAttrCase<"VertexIndex", 0, "vertex_index">,
  I32EnumAttrCase<"GlobalInvocationId", 1, "global_invocation_id">,
  I32EnumAttrCase<"LocalInvocationId", 2, "local_invocation_id">,
  I32EnumAttrCase<"WorkgroupId", 3, "workgroup_id">,
  I32EnumAttrCase<"FragmentCoord", 4, "fragment_coord">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}
def Gfx_BuiltinAttr : EnumAttr<Gfx_Dialect, Gfx_Builtin, "builtin">;

def Gfx_ParameterFieldAttr : AttrDef<Gfx_Dialect, "ParameterField"> {
  let mnemonic = "parameter_field";
  let parameters = (ins StringRefParameter<>:$name, "::mlir::Type":$type);
  let assemblyFormat = "`<` $name `,` $type `>`";
  let genVerifyDecl = 1;
}

def Gfx_ParameterBlockAttr : AttrDef<Gfx_Dialect, "ParameterBlock"> {
  let mnemonic = "parameter_block";
  let parameters = (ins StringRefParameter<>:$name, "::mlir::ArrayAttr":$fields);
  let assemblyFormat = "`<` $name `,` $fields `>`";
  let genVerifyDecl = 1;
}

def Gfx_StorageClass : I32EnumAttr<"StorageClass", "device memory domain", [
  I32EnumAttrCase<"Global", 0, "global">,
  I32EnumAttrCase<"Workgroup", 1, "workgroup">,
  I32EnumAttrCase<"Private", 2, "private">,
  I32EnumAttrCase<"PushConstant", 3, "push_constant">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}

def Gfx_ResourceKind : I32EnumAttr<"ResourceKind", "opaque bindless resource kind", [
  I32EnumAttrCase<"Buffer", 0, "buffer">,
  I32EnumAttrCase<"Texture2D", 1, "texture_2d">,
  I32EnumAttrCase<"Sampler", 2, "sampler">,
  I32EnumAttrCase<"StorageImage2D", 3, "storage_image_2d">,
  I32EnumAttrCase<"AccelerationStructure", 4, "acceleration_structure">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}

def Gfx_AccessMode : I32EnumAttr<"AccessMode", "buffer access permissions", [
  I32EnumAttrCase<"ReadOnly", 0, "read_only">,
  I32EnumAttrCase<"WriteOnly", 1, "write_only">,
  I32EnumAttrCase<"ReadWrite", 2, "read_write">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}

def Gfx_AccessModeAttr : AttrDef<Gfx_Dialect, "AccessMode"> {
  let mnemonic = "access_mode";
  let parameters = (ins "AccessMode":$value, OptionalParameter<"::mlir::Type">:$type);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    operator AccessMode() const { return getValue(); }
  }];
}

def Gfx_BufferAccessAttr : AttrDef<Gfx_Dialect, "BufferAccess"> {
  let mnemonic = "buffer_access";
  let parameters = (ins EnumParameter<Gfx_AccessMode>:$value);
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    operator AccessMode() const { return getValue(); }
  }];
}

def Gfx_MemoryScope : I32EnumAttr<"MemoryScope", "synchronization memory scope", [
  I32EnumAttrCase<"Workgroup", 0, "workgroup">,
  I32EnumAttrCase<"Device", 1, "device">,
  I32EnumAttrCase<"All", 2, "all">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}
def Gfx_MemoryScopeAttr : EnumAttr<Gfx_Dialect, Gfx_MemoryScope, "memory_scope">;

def Gfx_AtomicOp : I32EnumAttr<"AtomicOp", "integer atomic read-modify-write operation", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Min", 2, "min">,
  I32EnumAttrCase<"Max", 3, "max">,
  I32EnumAttrCase<"UMin", 4, "umin">,
  I32EnumAttrCase<"UMax", 5, "umax">,
  I32EnumAttrCase<"And", 6, "and">,
  I32EnumAttrCase<"Or", 7, "or">,
  I32EnumAttrCase<"Xor", 8, "xor">,
  I32EnumAttrCase<"Xchg", 9, "xchg">
]> {
  let cppNamespace = "::mlir::gfx";
  let genSpecializedAttr = 0;
}
def Gfx_AtomicOpAttr : EnumAttr<Gfx_Dialect, Gfx_AtomicOp, "atomic_op">;
#endif
