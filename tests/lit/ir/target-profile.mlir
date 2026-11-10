// RUN: %split-file %s %t
// RUN: not %gfx-opt %t/invalid_cap_type.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-CAP-TYPE
// RUN: not %gfx-opt %t/invalid_cap_elem.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-CAP-ELEM
// RUN: not %gfx-opt %t/invalid_shmem_neg.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-NEG
// RUN: not %gfx-opt %t/invalid_shmem_stage.mlir 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-STAGE
// RUN: %gfx-opt %t/valid_dialect_attrs.mlir | %FileCheck %s --check-prefix=CHECK-VALID-DIALECT
// RUN: not %gfx-compile %t/missing_module_cap.mlir --input-format=mlir --target vulkan --output %t/out.spv --reflection %t/out.refl 2>&1 | %FileCheck %s --check-prefix=ERR-CAP-ABSENT
// RUN: not %gfx-compile %t/missing_func_cap.mlir --input-format=mlir --target vulkan --output %t/out.spv --reflection %t/out.refl 2>&1 | %FileCheck %s --check-prefix=ERR-FUNC-CAP-ABSENT
// RUN: %gfx-compile %t/valid_cap.mlir --input-format=mlir --target vulkan --required-capability Float64 --output %t/out.spv --reflection %t/out.refl
// RUN: not %gfx-compile %t/exceed_shmem.mlir --input-format=mlir --target vulkan --max-shared-memory-bytes 16384 --output %t/out.spv --reflection %t/out.refl 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-EXCEED
// RUN: %gfx-compile %t/valid_shmem.mlir --input-format=mlir --target vulkan --max-shared-memory-bytes 65536 --output %t/out.spv --reflection %t/out.refl
// RUN: not %gfx-compile %t/exceed_invocations.mlir --input-format=mlir --target vulkan --max-workgroup-invocations 64 --output %t/out.spv --reflection %t/out.refl 2>&1 | %FileCheck %s --check-prefix=ERR-INV-EXCEED
// RUN: %gfx-compile %t/valid_invocations.mlir --input-format=mlir --target vulkan --max-workgroup-invocations 64 --output %t/out.spv --reflection %t/out.refl

// ERR-CAP-TYPE: gfx.required_capabilities requires an array of string attributes on a module or function
// ERR-CAP-ELEM: gfx.required_capabilities elements must be string attributes
// ERR-SHMEM-NEG: gfx.shared_memory_bytes requires a non-negative integer on a compute entry point
// ERR-SHMEM-STAGE: gfx.shared_memory_bytes requires a non-negative integer on a compute entry point
// CHECK-VALID-DIALECT: gfx.required_capabilities = ["Float64"]
// CHECK-VALID-DIALECT: gfx.shared_memory_bytes = 1024 : i32
// ERR-CAP-ABSENT: target capability profile does not support required capability 'Float64'
// ERR-FUNC-CAP-ABSENT: target capability profile does not support required capability 'ShaderFloat64'
// ERR-SHMEM-EXCEED: shared memory allocation (32768 bytes) exceeds target limit of 16384 bytes
// ERR-INV-EXCEED: total workgroup invocations (128) exceeds target profile limit (64)

//--- invalid_cap_type.mlir
module attributes {gfx.required_capabilities = "Float64"} {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- invalid_cap_elem.mlir
module attributes {gfx.required_capabilities = [123]} {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- invalid_shmem_neg.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>,
    gfx.shared_memory_bytes = -1 : i32
  } {
    return
  }
}

//--- invalid_shmem_stage.mlir
module {
  func.func @vertex_main() attributes {
    gfx.stage = #gfx.stage<vertex>,
    gfx.shared_memory_bytes = 1024 : i32
  } {
    return
  }
}

//--- valid_dialect_attrs.mlir
module attributes {gfx.required_capabilities = ["Float64"]} {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>,
    gfx.shared_memory_bytes = 1024 : i32
  } {
    return
  }
}

//--- missing_module_cap.mlir
module attributes {gfx.required_capabilities = ["Float64"]} {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- missing_func_cap.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>,
    gfx.required_capabilities = ["ShaderFloat64"]
  } {
    return
  }
}

//--- valid_cap.mlir
module attributes {gfx.required_capabilities = ["Float64"]} {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- exceed_shmem.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>,
    gfx.shared_memory_bytes = 32768 : i32
  } {
    return
  }
}

//--- valid_shmem.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>,
    gfx.shared_memory_bytes = 49152 : i32
  } {
    return
  }
}

//--- exceed_invocations.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 16, 1>
  } {
    return
  }
}

//--- valid_invocations.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
