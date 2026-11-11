// RUN: %split-file %s %t
// RUN: %gfx-opt %t/valid_shared_and_barriers.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-SHARED
// RUN: %gfx-opt %t/valid_atomics.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-ATOMICS
// RUN: %gfx-opt %t/valid_uniform_loops.mlir --gfx-verify | %FileCheck %s --check-prefix=CHECK-UNIFORM
// RUN: %gfx-compile %t/valid_shared_and_barriers.mlir --input-format mlir --target vulkan --output %t/shared.spv --reflection %t/shared.json
// RUN: %FileCheck %s --input-file=%t/shared.json --check-prefix=CHECK-REFL
// RUN: not %gfx-opt %t/err_shared_in_fragment.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-STAGE
// RUN: not %gfx-opt %t/err_shared_in_helper_from_fragment.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-HELPER
// RUN: not %gfx-opt %t/err_barrier_in_vertex.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-BARRIER-STAGE
// RUN: not %gfx-opt %t/err_shared_zero_count.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-COUNT
// RUN: not %gfx-opt %t/err_shared_wrong_storage.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-STORAGE
// RUN: not %gfx-opt %t/err_shared_exceed_budget.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-SHMEM-BUDGET
// RUN: %gfx-compile %t/high_end_shared_budget.mlir --input-format mlir --target vulkan --max-shared-memory-bytes 65536 --output %t/high_end.spv
// RUN: not %gfx-opt %t/err_divergent_if_barrier.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-DIV-IF
// RUN: not %gfx-opt %t/err_divergent_for_barrier.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-DIV-FOR
// RUN: not %gfx-opt %t/err_divergent_atomic_barrier.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-DIV-ATOMIC
// RUN: not %gfx-opt %t/err_atomic_float.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-ATOMIC-FLOAT
// RUN: not %gfx-opt %t/err_atomic_readonly_binding.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-ATOMIC-RO
// RUN: not %gfx-opt %t/err_atomic_writeonly_arg.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-ATOMIC-WO
// RUN: not %gfx-opt %t/err_atomic_push_constant.mlir --gfx-verify 2>&1 | %FileCheck %s --check-prefix=ERR-ATOMIC-SC

// CHECK-SHARED-LABEL: func.func @shared_and_barriers
// CHECK-SHARED: gfx.shared_alloc 128 name = "tile_f32" : !gfx.ptr<f32, workgroup>
// CHECK-SHARED: gfx.shared_alloc 64 : !gfx.ptr<vector<2xf32>, workgroup>
// CHECK-SHARED: gfx.barrier
// CHECK-SHARED: gfx.barrier <device>
// CHECK-SHARED: gfx.barrier <all>

// CHECK-REFL: "shared_memory_bytes":1024

// CHECK-ATOMICS-LABEL: func.func @atomic_operations
// CHECK-ATOMICS: gfx.atomic_rmw <add> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, workgroup>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <sub> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, workgroup>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <min> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, workgroup>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <max> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, workgroup>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <umin> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <umax> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <and> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <or> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <xor> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_rmw <xchg> %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_cas %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.ptr<i32, workgroup>, i32, i32) -> i32
// CHECK-ATOMICS: gfx.atomic_cas %{{.*}}, %{{.*}}, %{{.*}} : (!gfx.ptr<i32, global>, i32, i32) -> i32

// CHECK-UNIFORM-LABEL: func.func @uniform_control_flow_barriers
// CHECK-UNIFORM: scf.if
// CHECK-UNIFORM: gfx.barrier
// CHECK-UNIFORM: scf.while
// CHECK-UNIFORM: gfx.barrier

// ERR-SHMEM-STAGE: shared memory allocations are only allowed in compute shaders
// ERR-SHMEM-HELPER: shared memory allocations are only allowed in compute shaders
// ERR-BARRIER-STAGE: barriers are only allowed in compute shaders
// ERR-SHMEM-COUNT: shared memory element count must be positive, got 0
// ERR-SHMEM-STORAGE: shared_alloc must produce a pointer with 'workgroup' storage class, got 'global'
// ERR-SHMEM-BUDGET: shared memory allocation (32772 bytes) exceeds target limit of 32768 bytes
// ERR-DIV-IF: barrier cannot be placed inside thread-divergent control flow
// ERR-DIV-FOR: barrier cannot be placed inside thread-divergent control flow
// ERR-DIV-ATOMIC: barrier cannot be placed inside thread-divergent control flow
// ERR-ATOMIC-FLOAT: atomic operations require a 32-bit integer pointee type, got 'f32'
// ERR-ATOMIC-RO: atomic operations require read_write buffer binding 'input_buf', got 'read_only'
// ERR-ATOMIC-WO: atomic operations require read_write buffer argument '{{.*}}', got 'write_only'
// ERR-ATOMIC-SC: atomic operations require 'global' or 'workgroup' storage class, got 'push_constant'

//--- valid_shared_and_barriers.mlir
module {
  func.func @shared_and_barriers(
      %out: !gfx.ptr<f32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_write>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %tile_f = gfx.shared_alloc 128 name = "tile_f32" : !gfx.ptr<f32, workgroup>
    %tile_v = gfx.shared_alloc 64 : !gfx.ptr<vector<2xf32>, workgroup>
    %f_ptr = gfx.ptr_offset %tile_f, %lx : (!gfx.ptr<f32, workgroup>, i32) -> !gfx.ptr<f32, workgroup>
    %v_ptr = gfx.ptr_offset %tile_v, %lx : (!gfx.ptr<vector<2xf32>, workgroup>, i32) -> !gfx.ptr<vector<2xf32>, workgroup>
    %c1 = arith.constant 1.0 : f32
    %cv = arith.constant dense<2.0> : vector<2xf32>
    gfx.store %c1, %f_ptr : f32, !gfx.ptr<f32, workgroup>
    gfx.store %cv, %v_ptr : vector<2xf32>, !gfx.ptr<vector<2xf32>, workgroup>
    gfx.barrier
    gfx.barrier <workgroup>
    gfx.barrier <device>
    gfx.barrier <all>
    %loaded = gfx.load %f_ptr : (!gfx.ptr<f32, workgroup>) -> f32
    %out_ptr = gfx.ptr_offset %out, %lx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    gfx.store %loaded, %out_ptr : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- valid_atomics.mlir
module {
  func.func @atomic_operations(
      %buf: !gfx.ptr<i32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_write>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %smem = gfx.shared_alloc 64 name = "bins" : !gfx.ptr<i32, workgroup>
    %sptr = gfx.ptr_offset %smem, %lx : (!gfx.ptr<i32, workgroup>, i32) -> !gfx.ptr<i32, workgroup>
    %gptr = gfx.ptr_offset %buf, %lx : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %r0 = gfx.atomic_rmw <add> %sptr, %c1 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %r1 = gfx.atomic_rmw <sub> %sptr, %c1 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %r2 = gfx.atomic_rmw <min> %sptr, %r0 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %r3 = gfx.atomic_rmw <max> %sptr, %r1 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %r4 = gfx.atomic_rmw <umin> %gptr, %r2 : (!gfx.ptr<i32, global>, i32) -> i32
    %r5 = gfx.atomic_rmw <umax> %gptr, %r3 : (!gfx.ptr<i32, global>, i32) -> i32
    %r6 = gfx.atomic_rmw <and> %gptr, %r4 : (!gfx.ptr<i32, global>, i32) -> i32
    %r7 = gfx.atomic_rmw <or> %gptr, %r5 : (!gfx.ptr<i32, global>, i32) -> i32
    %r8 = gfx.atomic_rmw <xor> %gptr, %r6 : (!gfx.ptr<i32, global>, i32) -> i32
    %r9 = gfx.atomic_rmw <xchg> %gptr, %r8 : (!gfx.ptr<i32, global>, i32) -> i32
    %c_wg = gfx.atomic_cas %sptr, %c0, %r9 : (!gfx.ptr<i32, workgroup>, i32, i32) -> i32
    %c_gl = gfx.atomic_cas %gptr, %c0, %c_wg : (!gfx.ptr<i32, global>, i32, i32) -> i32
    return
  }
}

//--- valid_uniform_loops.mlir
module attributes {
  gfx.parameters = #gfx.parameter_block<"params", [
    #gfx.parameter_field<"num_tiles", i32>
  ]>
} {
  func.func @uniform_control_flow_barriers(
      %out: !gfx.ptr<f32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_write>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %wid = gfx.builtin <workgroup_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %wx = vector.extract %wid[0] : i32 from vector<3xi32>
    %num_tiles = gfx.parameter "params" "num_tiles" : i32
    %smem = gfx.shared_alloc 64 name = "tile" : !gfx.ptr<f32, workgroup>
    %sptr = gfx.ptr_offset %smem, %lx : (!gfx.ptr<f32, workgroup>, i32) -> !gfx.ptr<f32, workgroup>

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero_f = arith.constant 0.0 : f32
    %wx_is_zero = arith.cmpi eq, %wx, %c0 : i32
    scf.if %wx_is_zero {
      gfx.barrier
    }

    // Uniform loop trip count (%num_tiles) carrying a thread-divergent accumulator (%acc).
    %res:2 = scf.while (%t = %c0, %acc = %zero_f) : (i32, f32) -> (i32, f32) {
      %cond = arith.cmpi slt, %t, %num_tiles : i32
      scf.condition(%cond) %t, %acc : i32, f32
    } do {
    ^bb0(%t: i32, %acc: f32):
      %lx_f = arith.sitofp %lx : i32 to f32
      gfx.store %lx_f, %sptr : f32, !gfx.ptr<f32, workgroup>
      gfx.barrier
      %v = gfx.load %sptr : (!gfx.ptr<f32, workgroup>) -> f32
      %next_acc = arith.addf %acc, %v : f32
      %next_t = arith.addi %t, %c1 : i32
      scf.yield %next_t, %next_acc : i32, f32
    }

    %out_ptr = gfx.ptr_offset %out, %lx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    gfx.store %res#1, %out_ptr : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- err_shared_in_fragment.mlir
module {
  func.func @frag_main() attributes {gfx.stage = #gfx.stage<fragment>} {
    %s = gfx.shared_alloc 16 : !gfx.ptr<f32, workgroup>
    %color = arith.constant dense<1.0> : vector<4xf32>
    gfx.output %color 0 "out_color" <perspective> : vector<4xf32>
    return
  }
}

//--- err_shared_in_helper_from_fragment.mlir
module {
  func.func private @helper() {
    %s = gfx.shared_alloc 16 : !gfx.ptr<f32, workgroup>
    return
  }
  func.func @frag_main() attributes {gfx.stage = #gfx.stage<fragment>} {
    func.call @helper() : () -> ()
    %color = arith.constant dense<1.0> : vector<4xf32>
    gfx.output %color 0 "out_color" <perspective> : vector<4xf32>
    return
  }
}

//--- err_barrier_in_vertex.mlir
module {
  func.func @vert_main() attributes {gfx.stage = #gfx.stage<vertex>} {
    gfx.barrier
    %pos = arith.constant dense<0.0> : vector<4xf32>
    gfx.position %pos : vector<4xf32>
    return
  }
}

//--- err_shared_zero_count.mlir
module {
  func.func @zero_count() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %s = gfx.shared_alloc 0 : !gfx.ptr<f32, workgroup>
    return
  }
}

//--- err_shared_wrong_storage.mlir
module {
  func.func @wrong_storage() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %s = gfx.shared_alloc 16 : !gfx.ptr<f32, global>
    return
  }
}

//--- err_shared_exceed_budget.mlir
module {
  func.func @exceed_budget() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %s = gfx.shared_alloc 8193 : !gfx.ptr<f32, workgroup>
    return
  }
}

//--- high_end_shared_budget.mlir
module {
  func.func @high_end_budget(
      %out: !gfx.ptr<f32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_write>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %s = gfx.shared_alloc 12288 : !gfx.ptr<f32, workgroup>
    %sptr = gfx.ptr_offset %s, %lx : (!gfx.ptr<f32, workgroup>, i32) -> !gfx.ptr<f32, workgroup>
    %c1 = arith.constant 1.0 : f32
    gfx.store %c1, %sptr : f32, !gfx.ptr<f32, workgroup>
    gfx.barrier
    %v = gfx.load %sptr : (!gfx.ptr<f32, workgroup>) -> f32
    %optr = gfx.ptr_offset %out, %lx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    gfx.store %v, %optr : f32, !gfx.ptr<f32, global>
    return
  }
}

//--- err_divergent_if_barrier.mlir
module {
  func.func @divergent_if() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %c0 = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %lx, %c0 : i32
    scf.if %is_zero {
      gfx.barrier
    }
    return
  }
}

//--- err_divergent_for_barrier.mlir
module {
  func.func @divergent_for() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %gid = gfx.builtin <global_invocation_id> : vector<3xi32>
    %gx = vector.extract %gid[0] : i32 from vector<3xi32>
    %ub = arith.index_cast %gx : i32 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %ub step %c1 {
      gfx.barrier
    }
    return
  }
}

//--- err_divergent_atomic_barrier.mlir
module {
  func.func @divergent_atomic_if() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %s = gfx.shared_alloc 1 : !gfx.ptr<i32, workgroup>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %ptr = gfx.ptr_offset %s, %c0 : (!gfx.ptr<i32, workgroup>, i32) -> !gfx.ptr<i32, workgroup>
    %old = gfx.atomic_rmw <add> %ptr, %c1 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %first = arith.cmpi eq, %old, %c0 : i32
    scf.if %first {
      gfx.barrier
    }
    return
  }
}

//--- err_atomic_float.mlir
module {
  func.func @atomic_float() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %s = gfx.shared_alloc 16 : !gfx.ptr<f32, workgroup>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1.0 : f32
    %ptr = gfx.ptr_offset %s, %c0 : (!gfx.ptr<f32, workgroup>, i32) -> !gfx.ptr<f32, workgroup>
    %old = gfx.atomic_rmw <add> %ptr, %c1 : (!gfx.ptr<f32, workgroup>, f32) -> f32
    return
  }
}

//--- err_atomic_readonly_binding.mlir
module {
  func.func @atomic_ro() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %buf = gfx.buffer_binding @input_buf binding(0) access(read_only) : !gfx.ptr<i32, global>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %ptr = gfx.ptr_offset %buf, %c0 : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>
    %old = gfx.atomic_rmw <add> %ptr, %c1 : (!gfx.ptr<i32, global>, i32) -> i32
    return
  }
}

//--- err_atomic_writeonly_arg.mlir
module {
  func.func @atomic_wo(
      %buf: !gfx.ptr<i32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %ptr = gfx.ptr_offset %buf, %c0 : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>
    %old = gfx.atomic_cas %ptr, %c0, %c1 : (!gfx.ptr<i32, global>, i32, i32) -> i32
    return
  }
}

//--- err_atomic_push_constant.mlir
module {
  func.func @atomic_pc(%ptr: !gfx.ptr<i32, push_constant>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %c1 = arith.constant 1 : i32
    %old = gfx.atomic_rmw <add> %ptr, %c1 : (!gfx.ptr<i32, push_constant>, i32) -> i32
    return
  }
}
