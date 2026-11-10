#ifndef GFX_TYPES_TD
#define GFX_TYPES_TD
include "Gfx/GfxAttrs.td"
include "mlir/IR/AttrTypeBase.td"

def Gfx_PtrType : TypeDef<Gfx_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let summary = "!gfx.ptr";
  let description = "Typed device pointer; representation is selected by target legalization";
  let parameters = (ins "::mlir::Type":$elementType,
                        EnumParameter<Gfx_StorageClass>:$storageClass,
                        DefaultValuedParameter<"uint32_t", "0">:$alignment);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    uint32_t getEffectiveAlignment() const;
  }];
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType,
                     "::mlir::gfx::StorageClass":$storageClass), [{
      return $_get($_ctxt, elementType, storageClass, /*alignment=*/0);
    }]>
  ];
}

def Gfx_HandleType : TypeDef<Gfx_Dialect, "Handle"> {
  let mnemonic = "handle";
  let summary = "Opaque typed bindless token, not a descriptor binding slot";
  let parameters = (ins EnumParameter<Gfx_ResourceKind>:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

def Gfx_ArrayType : TypeDef<Gfx_Dialect, "Array"> {
  let mnemonic = "array";
  let summary = "Fixed-size array type with std430 stride and alignment";
  let parameters = (ins "::mlir::Type":$elementType, "uint32_t":$count);
  let assemblyFormat = "`<` $count `x` $elementType `>`";
}

def Gfx_StructType : TypeDef<Gfx_Dialect, "Struct"> {
  let mnemonic = "struct";
  let summary = "Aggregate struct type";
  let parameters = (ins StringRefParameter<>:$name, "::mlir::ArrayAttr":$fields);
  let assemblyFormat = "`<` $name `,` $fields `>`";
}
#endif
