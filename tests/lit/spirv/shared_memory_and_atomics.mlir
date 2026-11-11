// REQUIRES: spirv-val
// RUN: %gfx-opt %s --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %gfx-compile %s --input-format mlir --target vulkan --output %t.spv
// RUN: %spirv-val --target-env vulkan1.0 %t.spv

// CHECK-SPIRV-LABEL: spirv.module Logical GLSL450
// CHECK-SPIRV-DAG: spirv.GlobalVariable @shared_bins_0 : !spirv.ptr<!spirv.array<64 x i32>, Workgroup>
// CHECK-SPIRV-DAG: spirv.GlobalVariable @shared_positions_1 : !spirv.ptr<!spirv.array<64 x vector<2xf32>>, Workgroup>
// CHECK-SPIRV-DAG: spirv.GlobalVariable @{{.*}} bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0]), Block>, StorageBuffer>
// CHECK-SPIRV-LABEL: spirv.func @shared_atomics_spirv() "None"
// CHECK-SPIRV: spirv.AccessChain %{{.*}}[%{{.*}}] : !spirv.ptr<!spirv.array<64 x i32>, Workgroup>, i32 -> !spirv.ptr<i32, Workgroup>
// CHECK-SPIRV: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK-SPIRV: spirv.AtomicIAdd <Workgroup> <None>
// CHECK-SPIRV: spirv.AtomicISub <Workgroup> <None>
// CHECK-SPIRV: spirv.AtomicSMin <Workgroup> <None>
// CHECK-SPIRV: spirv.AtomicSMax <Workgroup> <None>
// CHECK-SPIRV: spirv.AtomicUMin <Device> <None>
// CHECK-SPIRV: spirv.AtomicUMax <Device> <None>
// CHECK-SPIRV: spirv.AtomicAnd <Device> <None>
// CHECK-SPIRV: spirv.AtomicOr <Device> <None>
// CHECK-SPIRV: spirv.AtomicXor <Device> <None>
// CHECK-SPIRV: spirv.AtomicExchange <Device> <None>
// CHECK-SPIRV: spirv.AtomicCompareExchange <Workgroup> <None> <None>
// CHECK-SPIRV: spirv.AtomicCompareExchange <Device> <None> <None>
// CHECK-SPIRV: spirv.ControlBarrier <Workgroup>, <Device>, <AcquireRelease|UniformMemory>
// CHECK-SPIRV: spirv.ControlBarrier <Workgroup>, <Device>, <AcquireRelease|UniformMemory|WorkgroupMemory>
// CHECK-SPIRV: spirv.EntryPoint "GLCompute" @shared_atomics_spirv
// CHECK-SPIRV: spirv.ExecutionMode @shared_atomics_spirv "LocalSize", 64, 1, 1

module {
  func.func @shared_atomics_spirv(
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
