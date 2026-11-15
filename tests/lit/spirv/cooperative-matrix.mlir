// RUN: %gfx-opt --gfx-to-vulkan %s | %FileCheck %s

module {
  // CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.6, [Shader, CooperativeMatrixKHR, VulkanMemoryModel], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_cooperative_matrix]>
  func.func @coop_matmul() attributes {gfx.stage = #gfx.stage<compute>, gfx.workgroup_size = array<i32: 32, 1, 1>} {
    %a_ptr = gfx.buffer_binding 0, 0, "a", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
    %b_ptr = gfx.buffer_binding 0, 1, "b", #gfx.access_mode<read_only> : !gfx.ptr<f32, global>
    %c_ptr = gfx.buffer_binding 0, 2, "c", #gfx.access_mode<write_only> : !gfx.ptr<f32, global>
    %stride = arith.constant 16 : i32
    // CHECK: spirv.CompositeConstruct {{%.*}} : (f32) -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %acc0 = gfx.coop_matrix_zero : !gfx.coop_matrix<16 x 16 x f32, acc>
    // CHECK: spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixA>
    %a = gfx.coop_matrix_load %a_ptr, %stride, true : (!gfx.ptr<f32, global>, i32) -> !gfx.coop_matrix<16 x 16 x f32, a>
    // CHECK: spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixB>
    %b = gfx.coop_matrix_load %b_ptr, %stride, true : (!gfx.ptr<f32, global>, i32) -> !gfx.coop_matrix<16 x 16 x f32, b>
    // CHECK: spirv.KHR.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}} : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixA>, !spirv.coopmatrix<16x16xf32, Subgroup, MatrixB> -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %acc1 = gfx.coop_matrix_mul_add %a, %b, %acc0 : (!gfx.coop_matrix<16 x 16 x f32, a>, !gfx.coop_matrix<16 x 16 x f32, b>, !gfx.coop_matrix<16 x 16 x f32, acc>) -> !gfx.coop_matrix<16 x 16 x f32, acc>
    // CHECK: spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, i32
    gfx.coop_matrix_store %c_ptr, %acc1, %stride, true : !gfx.ptr<f32, global>, !gfx.coop_matrix<16 x 16 x f32, acc>, i32
    return
  }
}
