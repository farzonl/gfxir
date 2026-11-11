// REQUIRES: llc, dxv
// RUN: %gfx-opt %s --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-MLIR-LLVM
// RUN: %gfx-compile %s --input-format mlir --emit-llvm -S | %FileCheck %s --check-prefix=CHECK-LLVM-IR
// RUN: %gfx-compile %s --input-format mlir --target directx --output %t.ll
// RUN: %llc -O0 -filetype=obj %t.ll -o %t.unsigned.dxil
// RUN: %dxv %t.unsigned.dxil -o %t.dxil

// CHECK-MLIR-LLVM-DAG: llvm.mlir.global internal @shared_bins_0() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<64 x i32>
// CHECK-MLIR-LLVM-DAG: llvm.mlir.global internal @shared_positions_1() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<64 x vector<2xf32>>
// CHECK-MLIR-LLVM-LABEL: llvm.func @shared_atomics_dxil()
// CHECK-MLIR-LLVM-SAME: "hlsl.shader", "compute"
// CHECK-MLIR-LLVM-SAME: "hlsl.numthreads", "64,1,1"
// CHECK-MLIR-LLVM: llvm.call_intrinsic "llvm.dx.group.memory.barrier.with.group.sync"() : () -> ()
// CHECK-MLIR-LLVM: llvm.atomicrmw add
// CHECK-MLIR-LLVM: llvm.sub
// CHECK-MLIR-LLVM: llvm.atomicrmw add
// CHECK-MLIR-LLVM: llvm.atomicrmw min
// CHECK-MLIR-LLVM: llvm.atomicrmw max
// CHECK-MLIR-LLVM: llvm.atomicrmw umin
// CHECK-MLIR-LLVM: llvm.atomicrmw umax
// CHECK-MLIR-LLVM: llvm.atomicrmw _and
// CHECK-MLIR-LLVM: llvm.atomicrmw _or
// CHECK-MLIR-LLVM: llvm.atomicrmw _xor
// CHECK-MLIR-LLVM: llvm.atomicrmw xchg
// CHECK-MLIR-LLVM: llvm.cmpxchg
// CHECK-MLIR-LLVM: llvm.call_intrinsic "llvm.dx.device.memory.barrier.with.group.sync"() : () -> ()
// CHECK-MLIR-LLVM: llvm.call_intrinsic "llvm.dx.all.memory.barrier.with.group.sync"() : () -> ()

// CHECK-LLVM-IR: target triple = "dxil-pc-shadermodel6.6-compute"
// CHECK-LLVM-IR-DAG: @shared_bins_0 = internal addrspace(3) global [64 x i32] undef, align 4
// CHECK-LLVM-IR-DAG: @shared_positions_1 = internal addrspace(3) global [64 x <2 x float>] undef, align 8
// CHECK-LLVM-IR-LABEL: define void @shared_atomics_dxil()
// CHECK-LLVM-IR: call void @llvm.dx.group.memory.barrier.with.group.sync()
// CHECK-LLVM-IR: atomicrmw add ptr addrspace(3)
// CHECK-LLVM-IR: atomicrmw umin ptr addrspace(1)
// CHECK-LLVM-IR: cmpxchg ptr addrspace(3)
// CHECK-LLVM-IR: cmpxchg ptr addrspace(1)
// CHECK-LLVM-IR: call void @llvm.dx.device.memory.barrier.with.group.sync()
// CHECK-LLVM-IR: call void @llvm.dx.all.memory.barrier.with.group.sync()

module {
  func.func @shared_atomics_dxil(
      %buf: !gfx.ptr<i32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_write>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %wid = gfx.builtin <workgroup_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %wx = vector.extract %wid[0] : i32 from vector<3xi32>
    %c64 = arith.constant 64 : i32
    %base = arith.muli %wx, %c64 : i32
    %gx = arith.addi %base, %lx : i32

    %bins = gfx.shared_alloc 64 name = "bins" : !gfx.ptr<i32, workgroup>
    %positions = gfx.shared_alloc 64 name = "positions" : !gfx.ptr<vector<2xf32>, workgroup>

    %sptr = gfx.ptr_offset %bins, %lx : (!gfx.ptr<i32, workgroup>, i32) -> !gfx.ptr<i32, workgroup>
    %pptr = gfx.ptr_offset %positions, %lx : (!gfx.ptr<vector<2xf32>, workgroup>, i32) -> !gfx.ptr<vector<2xf32>, workgroup>
    %gptr = gfx.ptr_offset %buf, %gx : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cv = arith.constant dense<1.5> : vector<2xf32>
    gfx.store %c0, %sptr : i32, !gfx.ptr<i32, workgroup>
    gfx.store %cv, %pptr : vector<2xf32>, !gfx.ptr<vector<2xf32>, workgroup>
    gfx.barrier

    %v = gfx.load %pptr : (!gfx.ptr<vector<2xf32>, workgroup>) -> vector<2xf32>
    %vx = vector.extract %v[0] : f32 from vector<2xf32>
    %vi = arith.fptosi %vx : f32 to i32

    %a0 = gfx.atomic_rmw <add> %sptr, %vi : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %a1 = gfx.atomic_rmw <sub> %sptr, %c1 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %a2 = gfx.atomic_rmw <min> %sptr, %a0 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %a3 = gfx.atomic_rmw <max> %sptr, %a1 : (!gfx.ptr<i32, workgroup>, i32) -> i32
    %a4 = gfx.atomic_rmw <umin> %gptr, %a2 : (!gfx.ptr<i32, global>, i32) -> i32
    %a5 = gfx.atomic_rmw <umax> %gptr, %a3 : (!gfx.ptr<i32, global>, i32) -> i32
    %a6 = gfx.atomic_rmw <and> %gptr, %a4 : (!gfx.ptr<i32, global>, i32) -> i32
    %a7 = gfx.atomic_rmw <or> %gptr, %a5 : (!gfx.ptr<i32, global>, i32) -> i32
    %a8 = gfx.atomic_rmw <xor> %gptr, %a6 : (!gfx.ptr<i32, global>, i32) -> i32
    %a9 = gfx.atomic_rmw <xchg> %gptr, %a8 : (!gfx.ptr<i32, global>, i32) -> i32
    %cas_wg = gfx.atomic_cas %sptr, %c0, %a9 : (!gfx.ptr<i32, workgroup>, i32, i32) -> i32
    %cas_gl = gfx.atomic_cas %gptr, %c0, %cas_wg : (!gfx.ptr<i32, global>, i32, i32) -> i32

    gfx.barrier <device>
    gfx.barrier <all>
    return
  }
}
