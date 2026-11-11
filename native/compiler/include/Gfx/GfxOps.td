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

def Gfx_ParameterOp : Gfx_Op<"parameter", [Pure]> {
  let summary = "Read an immutable field of a compiler-laid-out parameter block";
  let arguments = (ins StrAttr:$block, StrAttr:$field);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$block $field attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def Gfx_IntegerOrIndex : Type<CPred<"::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>($_self)">, "integer or index">;

def Gfx_BufferBindingOp : Gfx_Op<"buffer_binding", [Pure]> {
  let summary = "Bind an external descriptor set/binding slot to a typed global pointer";
  let description = [{
    Binds a buffer descriptor identified by descriptor set and binding slot
    with given symbolic name and access permissions to a typed device pointer
    in the global storage class (!gfx.ptr<T, global>).
  }];

  let arguments = (ins
    I32Attr:$set,
    I32Attr:$binding,
    StrAttr:$name,
    Gfx_AccessModeAttr:$access
  );
  let results = (outs Gfx_PtrType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Gfx_PtrOffsetOp : Gfx_Op<"ptr_offset", [Pure]> {
  let summary = "Compute typed pointer offset by element count";
  let description = [{
    Computes a new typed device pointer by offsetting ptr by offset elements.
  }];

  let arguments = (ins
    Gfx_PtrType:$ptr,
    Gfx_IntegerOrIndex:$offset
  );
  let results = (outs Gfx_PtrType:$result);
  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Gfx_LoadOp : Gfx_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a value from a typed device pointer";
  let description = [{
    Loads a value from the typed device pointer ptr.
  }];

  let arguments = (ins
    Gfx_PtrType:$ptr,
    OptionalAttr<I32Attr>:$alignment
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ptr (`,` `alignment` `=` $alignment^)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Gfx_StoreOp : Gfx_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a value to a typed device pointer";
  let description = [{
    Stores value to the typed device pointer ptr.
  }];

  let arguments = (ins
    AnyType:$value,
    Gfx_PtrType:$ptr,
    OptionalAttr<I32Attr>:$alignment
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Gfx_SharedAllocOp : Gfx_Op<"shared_alloc", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Allocate a static workgroup-shared array in compute shaders";
  let description = [{
    Declares a statically sized array of count elements in workgroup-local shared
    memory and returns a typed pointer (!gfx.ptr<T, workgroup>) to element 0.
  }];

  let arguments = (ins
    I32Attr:$count,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs Gfx_PtrType:$result);
  let assemblyFormat = [{
    $count (`name` `=` $name^)? attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Gfx_BarrierOp : Gfx_Op<"barrier", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Synchronize workgroup execution and memory accesses";
  let description = [{
    Synchronizes execution across all invocations in a compute workgroup and
    orders memory accesses within the specified memory scope (workgroup, device,
    or all). Must execute in workgroup-uniform control flow.
  }];

  let arguments = (ins
    DefaultValuedAttr<Gfx_MemoryScopeAttr, "MemoryScope::Workgroup">:$scope
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Gfx_AtomicRMWOp : Gfx_Op<"atomic_rmw", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Perform a 32-bit integer atomic read-modify-write operation";
  let description = [{
    Atomically loads a 32-bit integer from ptr, combines it with value according
    to kind, stores the result back to ptr, and returns the original loaded value.
  }];

  let arguments = (ins
    Gfx_AtomicOpAttr:$kind,
    Gfx_PtrType:$ptr,
    AnyType:$value
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $kind $ptr `,` $value attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Gfx_AtomicCASOp : Gfx_Op<"atomic_cas", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Perform a 32-bit integer atomic compare-and-exchange operation";
  let description = [{
    Atomically compares the 32-bit integer at ptr with cmp; if equal, replaces
    it with value. Always returns the original 32-bit integer loaded from ptr.
  }];

  let arguments = (ins
    Gfx_PtrType:$ptr,
    AnyType:$cmp,
    AnyType:$value
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ptr `,` $cmp `,` $value attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}
#endif
