// RUN: %split-file %s %t
// RUN: %gfx-opt %t/valid_min.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-MIN
// RUN: %gfx-opt %t/valid_warp.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-WARP
// RUN: %gfx-opt %t/valid_max_x.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-MAX-X
// RUN: %gfx-opt %t/valid_max_y.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-MAX-Y
// RUN: %gfx-opt %t/valid_max_z.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-MAX-Z
// RUN: %gfx-opt %t/valid_max_invocations_2d.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-INV-2D
// RUN: %gfx-opt %t/valid_max_invocations_3d.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-VALID-INV-3D
// RUN: not %gfx-opt %t/invalid_zero_x.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-POSITIVE
// RUN: not %gfx-opt %t/invalid_zero_y.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-POSITIVE
// RUN: not %gfx-opt %t/invalid_zero_z.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-POSITIVE
// RUN: not %gfx-opt %t/invalid_negative.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-POSITIVE
// RUN: not %gfx-opt %t/invalid_exceed_x.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-BOUNDS-X
// RUN: not %gfx-opt %t/invalid_exceed_y.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-BOUNDS-Y
// RUN: not %gfx-opt %t/invalid_exceed_z.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-BOUNDS-Z
// RUN: not %gfx-opt %t/invalid_exceed_invocations.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-INVOCATIONS
// RUN: not %gfx-opt %t/invalid_rank_2.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-POSITIVE
// RUN: not %gfx-opt %t/invalid_stage_vertex.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-STAGE
// RUN: not %gfx-opt %t/missing_workgroup_size.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-MISSING
// RUN: %gfx-compile %t/valid_invocations_profile.mlir --input-format mlir --target vulkan --max-workgroup-invocations 64 --output %t/valid_inv.spv --reflection %t/valid_inv.refl
// RUN: not %gfx-compile %t/invalid_invocations_profile.mlir --input-format mlir --target vulkan --max-workgroup-invocations 64 --output %t/invalid_inv.spv --reflection %t/invalid_inv.refl 2>&1 | %FileCheck %s --check-prefix=ERR-INV-OVERRIDE

// CHECK-VALID-MIN: gfx.workgroup_size = array<i32: 1, 1, 1>
// CHECK-VALID-WARP: gfx.workgroup_size = array<i32: 64, 1, 1>
// CHECK-VALID-MAX-X: gfx.workgroup_size = array<i32: 1024, 1, 1>
// CHECK-VALID-MAX-Y: gfx.workgroup_size = array<i32: 1, 1024, 1>
// CHECK-VALID-MAX-Z: gfx.workgroup_size = array<i32: 1, 1, 64>
// CHECK-VALID-INV-2D: gfx.workgroup_size = array<i32: 32, 32, 1>
// CHECK-VALID-INV-3D: gfx.workgroup_size = array<i32: 16, 16, 4>

// ERR-POSITIVE: gfx.workgroup_size requires three positive dimensions on a compute entry point
// ERR-BOUNDS-X: workgroup X dimension (1025) exceeds target profile limit (1024)
// ERR-BOUNDS-Y: workgroup Y dimension (1025) exceeds target profile limit (1024)
// ERR-BOUNDS-Z: workgroup Z dimension (65) exceeds target profile limit (64)
// ERR-INVOCATIONS: total workgroup invocations (1056) exceeds target profile limit (1024)
// ERR-STAGE: gfx.workgroup_size requires three positive dimensions on a compute entry point
// ERR-MISSING: compute entry points require gfx.workgroup_size
// ERR-INV-OVERRIDE: total workgroup invocations (128) exceeds target profile limit (64)

//--- valid_min.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}

//--- valid_warp.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    return
  }
}

//--- valid_max_x.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1024, 1, 1>
  } {
    return
  }
}

//--- valid_max_y.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1024, 1>
  } {
    return
  }
}

//--- valid_max_z.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 64>
  } {
    return
  }
}

//--- valid_max_invocations_2d.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 32, 32, 1>
  } {
    return
  }
}

//--- valid_max_invocations_3d.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 16, 16, 4>
  } {
    return
  }
}

//--- invalid_zero_x.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 0, 8, 1>
  } {
    return
  }
}

//--- invalid_zero_y.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 0, 1>
  } {
    return
  }
}

//--- invalid_zero_z.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 0>
  } {
    return
  }
}

//--- invalid_negative.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: -1, 1, 1>
  } {
    return
  }
}

//--- invalid_exceed_x.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1025, 1, 1>
  } {
    return
  }
}

//--- invalid_exceed_y.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1025, 1>
  } {
    return
  }
}

//--- invalid_exceed_z.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 65>
  } {
    return
  }
}

//--- invalid_exceed_invocations.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 32, 33, 1>
  } {
    return
  }
}

//--- invalid_rank_2.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8>
  } {
    return
  }
}

//--- invalid_stage_vertex.mlir
module {
  func.func @vertex_main() attributes {
    gfx.stage = #gfx.stage<vertex>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- missing_workgroup_size.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>
  } {
    return
  }
}

//--- valid_invocations_profile.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}

//--- invalid_invocations_profile.mlir
module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 16, 1>
  } {
    return
  }
}
