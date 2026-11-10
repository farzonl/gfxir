// RUN: %split-file %s %t
// RUN: %gfx-opt %t/buffer_binding.mlir | %FileCheck %s --check-prefix=CHECK-BINDING
// RUN: %gfx-opt %t/ptr_offset.mlir | %FileCheck %s --check-prefix=CHECK-OFFSET
// RUN: %gfx-opt %t/load_store.mlir | %FileCheck %s --check-prefix=CHECK-LOADSTORE
// RUN: %gfx-opt %t/aligned_load_store.mlir | %FileCheck %s --check-prefix=CHECK-ALIGN
// RUN: not %gfx-opt %t/binding_bad_storage.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-BIND-STORAGE
// RUN: not %gfx-opt %t/binding_bad_return.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-BIND-RETURN
// RUN: not %gfx-opt %t/binding_negative_set.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-BIND-SET
// RUN: not %gfx-opt %t/binding_negative_binding.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-BIND-SLOT
// RUN: not %gfx-opt %t/offset_type_mismatch.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-OFFSET-ELEM
// RUN: not %gfx-opt %t/offset_storage_mismatch.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-OFFSET-STORAGE
// RUN: not %gfx-opt %t/offset_non_pointer.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-OFFSET-NOTPTR
// RUN: not %gfx-opt %t/offset_non_integer.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-OFFSET-NOTINT
// RUN: not %gfx-opt %t/load_type_mismatch.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-LOAD-TYPE
// RUN: not %gfx-opt %t/load_non_pointer.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-LOAD-NOTPTR
// RUN: not %gfx-opt %t/load_bad_align_npot.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-LOAD-ALIGN-NPOT
// RUN: not %gfx-opt %t/load_bad_align_subnatural.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-LOAD-ALIGN-SUBNAT
// RUN: not %gfx-opt %t/store_type_mismatch.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-STORE-TYPE
// RUN: not %gfx-opt %t/store_push_constant.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-STORE-PUSHCONST
// RUN: not %gfx-opt %t/store_bad_align_npot.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-STORE-ALIGN-NPOT
// RUN: not %gfx-opt %t/store_bad_align_subnatural.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-STORE-ALIGN-SUBNAT
// RUN: %gfx-opt %t/spirv_lowering.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %gfx-opt %t/directx_lowering.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-DIRECTX
// RUN: %gfx-opt %t/spirv_si32_offset.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-SPIRV-SI32
// RUN: %gfx-opt %t/directx_si32_offset.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-DIRECTX-SI32
// RUN: %gfx-opt %t/directx_i64_offset.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-DIRECTX-I64
// RUN: %gfx-opt %t/directx_si64_offset.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-DIRECTX-SI64
// RUN: %gfx-opt %t/directx_ui64_offset.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-DIRECTX-UI64
// RUN: %gfx-opt %t/vector_add.mlir | %FileCheck %s --check-prefix=CHECK-VECADD
// RUN: %gfx-opt %t/in_place_saxpy.mlir | %FileCheck %s --check-prefix=CHECK-SAXPY

// =============================================================================
// FILECHECK DIRECTIVES
// =============================================================================

// CHECK-BINDING: %0 = gfx.buffer_binding 0, 0, "bufA", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
// CHECK-BINDING: %1 = gfx.buffer_binding 0, 1, "bufB", #gfx.access_mode<write_only> : !gfx.ptr<vector<4xf32>, global>
// CHECK-BINDING: %2 = gfx.buffer_binding 1, 0, "bufC", #gfx.access_mode<read_write> : !gfx.ptr<i32, global>

// CHECK-OFFSET: %0 = gfx.ptr_offset %arg0, %arg1 : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
// CHECK-OFFSET: %1 = gfx.ptr_offset %arg2, %arg3 : (!gfx.ptr<vector<4xf32>, workgroup>, i64) -> !gfx.ptr<vector<4xf32>, workgroup>
// CHECK-OFFSET: %2 = gfx.ptr_offset %arg0, %arg4 : (!gfx.ptr<f32, global>, si32) -> !gfx.ptr<f32, global>
// CHECK-OFFSET: %3 = gfx.ptr_offset %arg2, %arg5 : (!gfx.ptr<vector<4xf32>, workgroup>, si64) -> !gfx.ptr<vector<4xf32>, workgroup>

// CHECK-LOADSTORE: %0 = gfx.load %arg0 : (!gfx.ptr<f32, global>) -> f32
// CHECK-LOADSTORE: %1 = gfx.load %arg1 : (!gfx.ptr<vector<4xf32>, workgroup>) -> vector<4xf32>
// CHECK-LOADSTORE: gfx.store %0, %arg0 : f32, !gfx.ptr<f32, global>
// CHECK-LOADSTORE: gfx.store %1, %arg1 : vector<4xf32>, !gfx.ptr<vector<4xf32>, workgroup>

// CHECK-ALIGN: %0 = gfx.load %arg0, alignment = 16 : (!gfx.ptr<f32, global, 16>) -> f32
// CHECK-ALIGN: gfx.store %arg1, %arg0, alignment = 16 : f32, !gfx.ptr<f32, global, 16>

// ERR-BIND-STORAGE: buffer binding must produce a pointer with 'global' storage class, got 'workgroup'
// ERR-BIND-RETURN: result #0 must be !gfx.ptr
// ERR-BIND-SET: buffer binding set index must be non-negative, got -1
// ERR-BIND-SLOT: buffer binding slot index must be non-negative, got -2

// ERR-OFFSET-ELEM: result pointer element type 'i32' does not match source pointer element type 'f32'
// ERR-OFFSET-STORAGE: result pointer storage class 'workgroup' does not match source pointer storage class 'global'
// ERR-OFFSET-NOTPTR: operand #0 must be !gfx.ptr
// ERR-OFFSET-NOTINT: operand #1 must be integer or index

// ERR-LOAD-TYPE: load result type 'i32' does not match pointer element type 'f32'
// ERR-LOAD-NOTPTR: operand #0 must be !gfx.ptr
// ERR-LOAD-ALIGN-NPOT: load alignment must be a power of two, got 3
// ERR-LOAD-ALIGN-SUBNAT: load alignment (2) is smaller than natural alignment of pointee type (4)

// ERR-STORE-TYPE: stored value type 'vector<4xf32>' does not match pointer element type 'f32'
// ERR-STORE-PUSHCONST: cannot store into read-only storage class 'push_constant'
// ERR-STORE-ALIGN-NPOT: store alignment must be a power of two, got 5
// ERR-STORE-ALIGN-SUBNAT: store alignment (8) is smaller than natural alignment of pointee type (16)

// CHECK-SPIRV: spirv.PtrAccessChain
// CHECK-SPIRV: spirv.Load
// CHECK-SPIRV: spirv.Store

// CHECK-SPIRV-SI32: spirv.func @compute_main(%[[ARG0:.*]]: !spirv.ptr<f32, StorageBuffer>, %[[ARG1:.*]]: !spirv.ptr<si32, StorageBuffer>)
// CHECK-SPIRV-SI32: %[[IDX:.*]] = spirv.Load "StorageBuffer" %[[ARG1]] : si32
// CHECK-SPIRV-SI32: %[[OFF:.*]] = spirv.PtrAccessChain %[[ARG0]][%[[IDX]]] : !spirv.ptr<f32, StorageBuffer>, si32 -> !spirv.ptr<f32, StorageBuffer>
// CHECK-SPIRV-SI32: %[[VAL:.*]] = spirv.Load "StorageBuffer" %[[OFF]] : f32
// CHECK-SPIRV-SI32: spirv.Store "StorageBuffer" %[[OFF]], %[[VAL]] : f32

// CHECK-DIRECTX: llvm.getelementptr
// CHECK-DIRECTX: llvm.load
// CHECK-DIRECTX: llvm.store

// CHECK-DIRECTX-SI32: llvm.func @compute_main(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: !llvm.ptr<1>)
// CHECK-DIRECTX-SI32: %[[IDX:.*]] = llvm.load %[[ARG1]] : !llvm.ptr<1> -> i32
// CHECK-DIRECTX-SI32: %[[OFF:.*]] = llvm.getelementptr %[[ARG0]][%[[IDX]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
// CHECK-DIRECTX-SI32: %[[VAL:.*]] = llvm.load %[[OFF]] : !llvm.ptr<1> -> f32
// CHECK-DIRECTX-SI32: llvm.store %[[VAL]], %[[OFF]] : f32, !llvm.ptr<1>

// CHECK-DIRECTX-I64: llvm.func @compute_main(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: !llvm.ptr<1>)
// CHECK-DIRECTX-I64: %[[IDX:.*]] = llvm.load %[[ARG1]] : !llvm.ptr<1> -> i64
// CHECK-DIRECTX-I64: %[[TRUNC:.*]] = llvm.trunc %[[IDX]] : i64 to i32
// CHECK-DIRECTX-I64: %[[OFF:.*]] = llvm.getelementptr %[[ARG0]][%[[TRUNC]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
// CHECK-DIRECTX-I64: %[[VAL:.*]] = llvm.load %[[OFF]] : !llvm.ptr<1> -> f32
// CHECK-DIRECTX-I64: llvm.store %[[VAL]], %[[OFF]] : f32, !llvm.ptr<1>

// CHECK-DIRECTX-SI64: llvm.func @compute_main(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: !llvm.ptr<1>)
// CHECK-DIRECTX-SI64: %[[IDX:.*]] = llvm.load %[[ARG1]] : !llvm.ptr<1> -> i64
// CHECK-DIRECTX-SI64: %[[TRUNC:.*]] = llvm.trunc %[[IDX]] : i64 to i32
// CHECK-DIRECTX-SI64: %[[OFF:.*]] = llvm.getelementptr %[[ARG0]][%[[TRUNC]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf32>
// CHECK-DIRECTX-SI64: %[[VAL:.*]] = llvm.load %[[OFF]] : !llvm.ptr<1> -> vector<4xf32>
// CHECK-DIRECTX-SI64: llvm.store %[[VAL]], %[[OFF]] : vector<4xf32>, !llvm.ptr<1>

// CHECK-DIRECTX-UI64: llvm.func @compute_main(%[[ARG0:.*]]: !llvm.ptr<1>, %[[ARG1:.*]]: !llvm.ptr<1>)
// CHECK-DIRECTX-UI64: %[[IDX:.*]] = llvm.load %[[ARG1]] : !llvm.ptr<1> -> i64
// CHECK-DIRECTX-UI64: %[[TRUNC:.*]] = llvm.trunc %[[IDX]] : i64 to i32
// CHECK-DIRECTX-UI64: %[[OFF:.*]] = llvm.getelementptr %[[ARG0]][%[[TRUNC]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK-DIRECTX-UI64: %[[VAL:.*]] = llvm.load %[[OFF]] : !llvm.ptr<1> -> i32
// CHECK-DIRECTX-UI64: llvm.store %[[VAL]], %[[OFF]] : i32, !llvm.ptr<1>

// CHECK-VECADD: func.func @vector_add
// CHECK-SAXPY: func.func @saxpy

// =============================================================================
// VALID SYNTAX, ATTRIBUTES & ROUND-TRIP
// =============================================================================

//--- buffer_binding.mlir
module {
  func.func @test_bindings() {
    %0 = gfx.buffer_binding 0, 0, "bufA", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
    %1 = gfx.buffer_binding 0, 1, "bufB", #gfx.access_mode<write_only> : !gfx.ptr<vector<4xf32>, global>
    %2 = gfx.buffer_binding 1, 0, "bufC", #gfx.access_mode<read_write> : !gfx.ptr<i32, global>
    return
  }
}

//--- ptr_offset.mlir
module {
  func.func @test_offset(%arg0: !gfx.ptr<f32, global>, %arg1: i32,
                         %arg2: !gfx.ptr<vector<4xf32>, workgroup>, %arg3: i64,
                         %arg4: si32, %arg5: si64) {
    %0 = gfx.ptr_offset %arg0, %arg1 : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %1 = gfx.ptr_offset %arg2, %arg3 : (!gfx.ptr<vector<4xf32>, workgroup>, i64) -> !gfx.ptr<vector<4xf32>, workgroup>
    %2 = gfx.ptr_offset %arg0, %arg4 : (!gfx.ptr<f32, global>, si32) -> !gfx.ptr<f32, global>
    %3 = gfx.ptr_offset %arg2, %arg5 : (!gfx.ptr<vector<4xf32>, workgroup>, si64) -> !gfx.ptr<vector<4xf32>, workgroup>
    return
  }
}

//--- load_store.mlir
module {
  func.func @test_load_store(%arg0: !gfx.ptr<f32, global>, %arg1: !gfx.ptr<vector<4xf32>, workgroup>) {
    %0 = gfx.load %arg0 : (!gfx.ptr<f32, global>) -> f32
    %1 = gfx.load %arg1 : (!gfx.ptr<vector<4xf32>, workgroup>) -> vector<4xf32>
    gfx.store %0, %arg0 : f32, !gfx.ptr<f32, global>
    gfx.store %1, %arg1 : vector<4xf32>, !gfx.ptr<vector<4xf32>, workgroup>
    return
  }
}

//--- aligned_load_store.mlir
module {
  func.func @test_alignments(%arg0: !gfx.ptr<f32, global, 16>, %arg1: f32) {
    %0 = gfx.load %arg0, alignment = 16 : (!gfx.ptr<f32, global, 16>) -> f32
    gfx.store %arg1, %arg0, alignment = 16 : f32, !gfx.ptr<f32, global, 16>
    return
  }
}

// =============================================================================
// VERIFIER NEGATIVE TESTS
// =============================================================================

//--- binding_bad_storage.mlir
module {
  func.func @bad_binding_storage() {
    %0 = gfx.buffer_binding 0, 0, "bufA", #gfx.access_mode<read_only> : !gfx.ptr<f32, workgroup>
    return
  }
}

//--- binding_bad_return.mlir
module {
  func.func @bad_binding_return() {
    %0 = gfx.buffer_binding 0, 0, "bufA", #gfx.access_mode<read_only> : f32
    return
  }
}

//--- binding_negative_set.mlir
module {
  func.func @bad_binding_set() {
    %0 = gfx.buffer_binding -1, 0, "bufA", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
    return
  }
}

//--- binding_negative_binding.mlir
module {
  func.func @bad_binding_slot() {
    %0 = gfx.buffer_binding 0, -2, "bufA", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
    return
  }
}

//--- offset_type_mismatch.mlir
module {
  func.func @bad_offset_elem(%p: !gfx.ptr<f32, global>, %idx: i32) {
    %0 = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<i32, global>
    return
  }
}

//--- offset_storage_mismatch.mlir
module {
  func.func @bad_offset_storage(%p: !gfx.ptr<f32, global>, %idx: i32) {
    %0 = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, workgroup>
    return
  }
}

//--- offset_non_pointer.mlir
module {
  func.func @bad_offset_nonptr(%p: f32, %idx: i32) {
    %0 = gfx.ptr_offset %p, %idx : (f32, i32) -> f32
    return
  }
}

//--- offset_non_integer.mlir
module {
  func.func @bad_offset_nonint(%p: !gfx.ptr<f32, global>, %idx: f32) {
    %0 = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, f32) -> !gfx.ptr<f32, global>
    return
  }
}

//--- load_type_mismatch.mlir
module {
  func.func @bad_load_type(%p: !gfx.ptr<f32, global>) {
    %0 = gfx.load %p : (!gfx.ptr<f32, global>) -> i32
    return
  }
}

//--- load_non_pointer.mlir
module {
  func.func @bad_load_nonptr(%p: f32) {
    %0 = gfx.load %p : (f32) -> f32
    return
  }
}

//--- load_bad_align_npot.mlir
module {
  func.func @bad_load_align_npot(%p: !gfx.ptr<f32, global>) {
    %0 = gfx.load %p, alignment = 3 : (!gfx.ptr<f32, global>) -> f32
    return
  }
}

//--- load_bad_align_subnatural.mlir
module {
  func.func @bad_load_align_subnat(%p: !gfx.ptr<f32, global>) {
    %0 = gfx.load %p, alignment = 2 : (!gfx.ptr<f32, global>) -> f32
    return
  }
}

//--- store_type_mismatch.mlir
module {
  func.func @bad_store_type(%val: vector<4xf32>, %p: !gfx.ptr<f32, global>) {
    gfx.store %val, %p : vector<4xf32>, !gfx.ptr<f32, global>
    return
  }
}

//--- store_push_constant.mlir
module {
  func.func @bad_store_push_const(%val: f32, %p: !gfx.ptr<f32, push_constant>) {
    gfx.store %val, %p : f32, !gfx.ptr<f32, push_constant>
    return
  }
}

//--- store_bad_align_npot.mlir
module {
  func.func @bad_store_align_npot(%val: f32, %p: !gfx.ptr<f32, global>) {
    gfx.store %val, %p, alignment = 5 : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- store_bad_align_subnatural.mlir
module {
  func.func @bad_store_align_subnat(%val: vector<4xf32>, %p: !gfx.ptr<vector<4xf32>, global>) {
    gfx.store %val, %p, alignment = 8 : vector<4xf32>, !gfx.ptr<vector<4xf32>, global>
    return
  }
}

// =============================================================================
// COMPILER LOWERING PASSES
// =============================================================================

//--- spirv_lowering.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %id = gfx.builtin <global_invocation_id> : vector<3xi32>
    %idx = vector.extract %id[0] : i32 from vector<3xi32>
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %val = gfx.load %off : (!gfx.ptr<f32, global>) -> f32
    gfx.store %val, %off : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- directx_lowering.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %id = gfx.builtin <global_invocation_id> : vector<3xi32>
    %idx = vector.extract %id[0] : i32 from vector<3xi32>
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %val = gfx.load %off : (!gfx.ptr<f32, global>) -> f32
    gfx.store %val, %off : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- spirv_si32_offset.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<f32, global>, %indices: !gfx.ptr<si32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %idx = gfx.load %indices : (!gfx.ptr<si32, global>) -> si32
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, si32) -> !gfx.ptr<f32, global>
    %val = gfx.load %off : (!gfx.ptr<f32, global>) -> f32
    gfx.store %val, %off : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- directx_si32_offset.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<f32, global>, %indices: !gfx.ptr<si32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %idx = gfx.load %indices : (!gfx.ptr<si32, global>) -> si32
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, si32) -> !gfx.ptr<f32, global>
    %val = gfx.load %off : (!gfx.ptr<f32, global>) -> f32
    gfx.store %val, %off : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- directx_i64_offset.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<f32, global>, %indices: !gfx.ptr<i64, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %idx = gfx.load %indices : (!gfx.ptr<i64, global>) -> i64
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<f32, global>, i64) -> !gfx.ptr<f32, global>
    %val = gfx.load %off : (!gfx.ptr<f32, global>) -> f32
    gfx.store %val, %off : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- directx_si64_offset.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<vector<4xf32>, global>, %indices: !gfx.ptr<si64, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %idx = gfx.load %indices : (!gfx.ptr<si64, global>) -> si64
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<vector<4xf32>, global>, si64) -> !gfx.ptr<vector<4xf32>, global>
    %val = gfx.load %off : (!gfx.ptr<vector<4xf32>, global>) -> vector<4xf32>
    gfx.store %val, %off : vector<4xf32>, !gfx.ptr<vector<4xf32>, global>
    return
  }
}

//--- directx_ui64_offset.mlir
module {
  func.func @compute_main(%p: !gfx.ptr<i32, global>, %indices: !gfx.ptr<ui64, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %idx = gfx.load %indices : (!gfx.ptr<ui64, global>) -> ui64
    %off = gfx.ptr_offset %p, %idx : (!gfx.ptr<i32, global>, ui64) -> !gfx.ptr<i32, global>
    %val = gfx.load %off : (!gfx.ptr<i32, global>) -> i32
    gfx.store %val, %off : i32, !gfx.ptr<i32, global>
    return
  }
}

// =============================================================================
// COMPUTE KERNEL OPERATIONS
// =============================================================================

//--- vector_add.mlir
module {
  func.func @vector_add(%a: !gfx.ptr<f32, global>, %b: !gfx.ptr<f32, global>, %c: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %id = gfx.builtin <global_invocation_id> : vector<3xi32>
    %idx = vector.extract %id[0] : i32 from vector<3xi32>
    %a_ptr = gfx.ptr_offset %a, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %b_ptr = gfx.ptr_offset %b, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %c_ptr = gfx.ptr_offset %c, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %val_a = gfx.load %a_ptr : (!gfx.ptr<f32, global>) -> f32
    %val_b = gfx.load %b_ptr : (!gfx.ptr<f32, global>) -> f32
    %val_c = arith.addf %val_a, %val_b : f32
    gfx.store %val_c, %c_ptr : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- in_place_saxpy.mlir
module {
  func.func @saxpy(%x: !gfx.ptr<f32, global>, %y: !gfx.ptr<f32, global>, %alpha: !gfx.ptr<f32, push_constant>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %id = gfx.builtin <global_invocation_id> : vector<3xi32>
    %idx = vector.extract %id[0] : i32 from vector<3xi32>
    %x_ptr = gfx.ptr_offset %x, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %y_ptr = gfx.ptr_offset %y, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %val_x = gfx.load %x_ptr : (!gfx.ptr<f32, global>) -> f32
    %val_y = gfx.load %y_ptr : (!gfx.ptr<f32, global>) -> f32
    %alpha_val = gfx.load %alpha : (!gfx.ptr<f32, push_constant>) -> f32
    %scaled = arith.mulf %val_x, %alpha_val : f32
    %res = arith.addf %scaled, %val_y : f32
    gfx.store %res, %y_ptr : f32, !gfx.ptr<f32, global>
    return
  }
}
