#ifndef GFX_OPS_TD
#define GFX_OPS_TD
include "Gfx/GfxTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Gfx_BuiltinOp : Gfx_Op<"builtin", [Pure]> {
  let summary = "Read a stage-specific device builtin";
  let arguments = (ins Gfx_BuiltinAttr:$kind);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$kind attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Gfx_InputOp : Gfx_Op<"input", [Pure]> {
  let summary = "Read a typed rasterization interface value";
  let arguments = (ins I32Attr:$location, StrAttr:$name,
                       Gfx_InterpolationAttr:$interpolation);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$location $name $interpolation attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Gfx_OutputOp : Gfx_Op<"output", [MemoryEffects<[MemWrite]>]> {
  let summary = "Write a varying or render target through the stage interface";
  let arguments = (ins AnyType:$value, I32Attr:$location, StrAttr:$name,
                       Gfx_InterpolationAttr:$interpolation);
  let assemblyFormat = "$value $location $name $interpolation attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Gfx_PositionOp : Gfx_Op<"position", [MemoryEffects<[MemWrite]>]> {
  let summary = "Write canonical X-right, Y-up, zero-to-one-depth clip position";
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}
#endif
