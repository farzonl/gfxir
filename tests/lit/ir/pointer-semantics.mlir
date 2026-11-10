// RUN: %split-file %s %t
// RUN: %gfx-opt %t/tier1_scalars.mlir | %FileCheck %s --check-prefix=CHECK-TIER1-SCALARS
// RUN: %gfx-opt %t/tier1_vectors.mlir | %FileCheck %s --check-prefix=CHECK-TIER1-VECTORS
// RUN: %gfx-opt %t/tier1_storage_classes.mlir | %FileCheck %s --check-prefix=CHECK-TIER1-STORAGE
// RUN: %gfx-opt %t/tier1_alignments.mlir | %FileCheck %s --check-prefix=CHECK-TIER1-ALIGN
// RUN: %gfx-opt %t/tier1_structs.mlir | %FileCheck %s --check-prefix=CHECK-TIER1-STRUCTS
// RUN: not %gfx-opt %t/tier2_align_npot_3.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-ALIGN-NPOT
// RUN: not %gfx-opt %t/tier2_align_npot_5.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-ALIGN-NPOT
// RUN: not %gfx-opt %t/tier2_align_npot_6.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-ALIGN-NPOT
// RUN: not %gfx-opt %t/tier2_subnatural_f32.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-SUBNATURAL-F32
// RUN: not %gfx-opt %t/tier2_subnatural_vec2.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-SUBNATURAL-VEC2
// RUN: not %gfx-opt %t/tier2_subnatural_vec4.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-SUBNATURAL-VEC4
// RUN: not %gfx-opt %t/tier2_subnatural_f64.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-SUBNATURAL-F64
// RUN: not %gfx-opt %t/tier2_nested_pointer.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-NESTED-PTR
// RUN: not %gfx-opt %t/tier2_domain_mismatch.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-DOMAIN-MISMATCH
// RUN: not %gfx-opt %t/tier2_stage_workgroup_vertex.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-STAGE-WORKGROUP
// RUN: %gfx-opt %t/tier3_spirv_default.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-SPIRV-DEFAULT
// RUN: %gfx-compile %t/tier3_spirv_bda.mlir --input-format=mlir --target vulkan --required-capability PhysicalStorageBuffer64 --output %t/bda.spv --reflection %t/bda.refl
// RUN: not %gfx-compile %t/tier3_spirv_bda.mlir --input-format=mlir --target vulkan --output %t/bda_fail.spv --reflection %t/bda_fail.refl 2>&1 | %FileCheck %s --check-prefix=ERR-BDA-NOCAP
// RUN: %gfx-opt %t/tier3_directx_default.mlir --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-DX-DEFAULT
// RUN: %gfx-opt %t/tier4_vector_add.mlir | %FileCheck %s --check-prefix=CHECK-TIER4-VECADD
// RUN: %gfx-opt %t/tier4_shared_tiled.mlir | %FileCheck %s --check-prefix=CHECK-TIER4-TILED

// =============================================================================
// FILECHECK DIRECTIVES
// =============================================================================

// CHECK-TIER1-SCALARS: func.func private @scalar_ptrs(!gfx.ptr<f32, global>, !gfx.ptr<i32, global>, !gfx.ptr<ui32, global>, !gfx.ptr<f16, global>, !gfx.ptr<i16, global>, !gfx.ptr<i8, global>, !gfx.ptr<f64, global>, !gfx.ptr<i64, global>)
// CHECK-TIER1-VECTORS: func.func private @vector_ptrs(!gfx.ptr<vector<2xf32>, global>, !gfx.ptr<vector<3xf32>, global>, !gfx.ptr<vector<4xf32>, global>, !gfx.ptr<vector<2xi32>, global>, !gfx.ptr<vector<4xi32>, global>, !gfx.ptr<vector<4xf16>, global>)
// CHECK-TIER1-STORAGE: func.func private @storage_classes(!gfx.ptr<f32, global>, !gfx.ptr<f32, workgroup>, !gfx.ptr<f32, private>, !gfx.ptr<f32, push_constant>)
// CHECK-TIER1-ALIGN: func.func private @aligned_ptrs(!gfx.ptr<f32, global, 4>, !gfx.ptr<f32, global, 16>, !gfx.ptr<f32, global, 64>, !gfx.ptr<vector<4xf32>, global, 32>)
// CHECK-TIER1-STRUCTS: func.func private @struct_ptrs(!gfx.ptr<!llvm.struct<(f32, i32)>, global>, !gfx.ptr<!llvm.struct<(f32, vector<4xf32>)>, global, 16>)

// ERR-ALIGN-NPOT: pointer alignment must be a power of two
// ERR-SUBNATURAL-F32: pointer alignment (2) is smaller than the natural alignment of pointee type (4)
// ERR-SUBNATURAL-VEC2: pointer alignment (4) is smaller than the natural alignment of pointee type (8)
// ERR-SUBNATURAL-VEC4: pointer alignment (8) is smaller than the natural alignment of pointee type (16)
// ERR-SUBNATURAL-F64: pointer alignment (4) is smaller than the natural alignment of pointee type (8)
// ERR-NESTED-PTR: pointer element type must be a valid shader data type
// ERR-DOMAIN-MISMATCH: doesn't match function result type
// ERR-STAGE-WORKGROUP: workgroup pointers are only allowed in compute entry points

// CHECK-SPIRV-DEFAULT: !spirv.ptr<f32, StorageBuffer>
// CHECK-SPIRV-DEFAULT: !spirv.ptr<vector<4xf32>, Workgroup>
// CHECK-SPIRV-DEFAULT: spirv.ExecutionMode @compute_main "LocalSize", 8, 8, 1
// ERR-BDA-NOCAP: target capability profile does not support required capability 'PhysicalStorageBuffer64'
// CHECK-DX-DEFAULT: module attributes {hlsl.numthreads = "8,8,1"

// CHECK-TIER4-VECADD: func.func @vector_add(%arg0: !gfx.ptr<f32, global>, %arg1: !gfx.ptr<f32, global>, %arg2: !gfx.ptr<f32, global>)
// CHECK-TIER4-TILED: func.func @tiled_reduction(%arg0: !gfx.ptr<vector<4xf32>, global>, %arg1: !gfx.ptr<f32, workgroup>)

// =============================================================================
// TIER 1: FEATURE COVERAGE (VALID POINTER TYPES)
// =============================================================================

//--- tier1_scalars.mlir
module {
  func.func private @scalar_ptrs(
      !gfx.ptr<f32, global>,
      !gfx.ptr<i32, global>,
      !gfx.ptr<ui32, global>,
      !gfx.ptr<f16, global>,
      !gfx.ptr<i16, global>,
      !gfx.ptr<i8, global>,
      !gfx.ptr<f64, global>,
      !gfx.ptr<i64, global>)
}

//--- tier1_vectors.mlir
module {
  func.func private @vector_ptrs(
      !gfx.ptr<vector<2xf32>, global>,
      !gfx.ptr<vector<3xf32>, global>,
      !gfx.ptr<vector<4xf32>, global>,
      !gfx.ptr<vector<2xi32>, global>,
      !gfx.ptr<vector<4xi32>, global>,
      !gfx.ptr<vector<4xf16>, global>)
}

//--- tier1_storage_classes.mlir
module {
  func.func private @storage_classes(
      !gfx.ptr<f32, global>,
      !gfx.ptr<f32, workgroup>,
      !gfx.ptr<f32, private>,
      !gfx.ptr<f32, push_constant>)
}

//--- tier1_alignments.mlir
module {
  func.func private @aligned_ptrs(
      !gfx.ptr<f32, global, 4>,
      !gfx.ptr<f32, global, 16>,
      !gfx.ptr<f32, global, 64>,
      !gfx.ptr<vector<4xf32>, global, 32>)
}

//--- tier1_structs.mlir
module {
  func.func private @struct_ptrs(
      !gfx.ptr<!llvm.struct<(f32, i32)>, global>,
      !gfx.ptr<!llvm.struct<(f32, vector<4xf32>)>, global, 16>)
}

// =============================================================================
// TIER 2: BOUNDARY & CORNER CASES (VERIFIER NEGATIVE TESTS)
// =============================================================================

//--- tier2_align_npot_3.mlir
module {
  func.func private @bad_align_3(!gfx.ptr<f32, global, 3>)
}

//--- tier2_align_npot_5.mlir
module {
  func.func private @bad_align_5(!gfx.ptr<f32, global, 5>)
}

//--- tier2_align_npot_6.mlir
module {
  func.func private @bad_align_6(!gfx.ptr<vector<2xf32>, global, 6>)
}

//--- tier2_subnatural_f32.mlir
module {
  func.func private @subnatural_f32(!gfx.ptr<f32, global, 2>)
}

//--- tier2_subnatural_vec2.mlir
module {
  func.func private @subnatural_vec2(!gfx.ptr<vector<2xf32>, global, 4>)
}

//--- tier2_subnatural_vec4.mlir
module {
  func.func private @subnatural_vec4(!gfx.ptr<vector<4xf32>, global, 8>)
}

//--- tier2_subnatural_f64.mlir
module {
  func.func private @subnatural_f64(!gfx.ptr<f64, global, 4>)
}

//--- tier2_nested_pointer.mlir
module {
  func.func private @nested_ptr(!gfx.ptr<!gfx.ptr<f32, global>, global>)
}

//--- tier2_domain_mismatch.mlir
module {
  func.func private @domain_mismatch(%arg: !gfx.ptr<f32, private>) -> !gfx.ptr<f32, global> {
    return %arg : !gfx.ptr<f32, private>
  }
}

//--- tier2_stage_workgroup_vertex.mlir
module {
  func.func @vertex_main(%arg: !gfx.ptr<f32, workgroup>) attributes {
    gfx.stage = #gfx.stage<vertex>
  } {
    %pos = arith.constant dense<0.0> : vector<4xf32>
    gfx.position %pos : vector<4xf32>
    return
  }
}

// =============================================================================
// TIER 3: CROSS-TARGET ABI LOWERING & CAPABILITY GATING
// =============================================================================

//--- tier3_spirv_default.mlir
module {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>, %wg: !gfx.ptr<vector<4xf32>, workgroup>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- tier3_spirv_bda.mlir
module attributes {gfx.required_capabilities = ["PhysicalStorageBuffer64"]} {
  func.func @compute_main(%bda_ptr: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- tier3_directx_default.mlir
module {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

// =============================================================================
// TIER 4: REAL-WORLD COMPUTE KERNEL DECLARATIONS
// =============================================================================

//--- tier4_vector_add.mlir
module {
  func.func @vector_add(%a: !gfx.ptr<f32, global>, %b: !gfx.ptr<f32, global>, %c: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %id = gfx.builtin <global_invocation_id> : vector<3xi32>
    return
  }
}

//--- tier4_shared_tiled.mlir
module {
  func.func @tiled_reduction(%global_in: !gfx.ptr<vector<4xf32>, global>, %tile_shared: !gfx.ptr<f32, workgroup>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 16, 16, 1>,
    gfx.shared_memory_bytes = 4096 : i32
  } {
    %id = gfx.builtin <local_invocation_id> : vector<3xi32>
    return
  }
}
