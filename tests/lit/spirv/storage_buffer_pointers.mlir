// REQUIRES: spirv-val
// RUN: %split-file %s %t
// RUN: %gfx-opt %t/inline_binding_syntax.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-INLINE
// RUN: %gfx-compile %t/inline_binding_syntax.mlir --input-format mlir --target vulkan --output %t/inline.spv
// RUN: %spirv-val --target-env vulkan1.0 %t/inline.spv
// RUN: %gfx-opt %t/physical_storage_buffer.mlir --convert-gfx-to-spirv | %FileCheck %s --check-prefix=CHECK-BDA
// RUN: not %gfx-compile %t/physical_storage_buffer.mlir --input-format mlir --target vulkan --output %t/bda_fail.spv 2>&1 | %FileCheck %s --check-prefix=ERR-BDA-CAP
// RUN: %gfx-compile %t/physical_storage_buffer.mlir --input-format mlir --target vulkan --required-capability buffer_device_address --output %t/bda.spv
// RUN: %spirv-val --target-env vulkan1.2 %t/bda.spv

// CHECK-INLINE-LABEL: spirv.module Logical GLSL450
// CHECK-INLINE-DAG: spirv.GlobalVariable @{{.*}} built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-INLINE-DAG: spirv.GlobalVariable @{{.*}} built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
// CHECK-INLINE-DAG: spirv.GlobalVariable @{{.*}} bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0, NonWritable]), Block>, StorageBuffer>
// CHECK-INLINE-DAG: spirv.GlobalVariable @{{.*}} bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
// CHECK-INLINE-LABEL: spirv.func @saxpy_inline() "None"
// CHECK-INLINE: spirv.AccessChain
// CHECK-INLINE: spirv.Load "StorageBuffer"
// CHECK-INLINE: spirv.Store "StorageBuffer"
// CHECK-INLINE: spirv.EntryPoint "GLCompute" @saxpy_inline
// CHECK-INLINE: spirv.ExecutionMode @saxpy_inline "LocalSize", 64, 1, 1

// CHECK-BDA-LABEL: spirv.module PhysicalStorageBuffer64 GLSL450
// CHECK-BDA-SAME: PhysicalStorageBufferAddresses
// CHECK-BDA-SAME: SPV_KHR_physical_storage_buffer
// CHECK-BDA: spirv.GlobalVariable @parameters : !spirv.ptr<!spirv.struct<(!spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0, NonWritable]), Block>, PhysicalStorageBuffer> [0], !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, PhysicalStorageBuffer> [8]), Block>, PushConstant>
// CHECK-BDA-LABEL: spirv.func @copy_bda() "None"
// CHECK-BDA: spirv.AccessChain
// CHECK-BDA: spirv.Load "PhysicalStorageBuffer" %{{.*}} ["Aligned", 4] : f32
// CHECK-BDA: spirv.Store "PhysicalStorageBuffer" %{{.*}}, %{{.*}} ["Aligned", 4] : f32

// ERR-BDA-CAP: buffer_device_address

//--- inline_binding_syntax.mlir
module {
  func.func @saxpy_inline() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %lid = gfx.builtin <local_invocation_id> : vector<3xi32>
    %wid = gfx.builtin <workgroup_id> : vector<3xi32>
    %lx = vector.extract %lid[0] : i32 from vector<3xi32>
    %wx = vector.extract %wid[0] : i32 from vector<3xi32>
    %c64 = arith.constant 64 : i32
    %base = arith.muli %wx, %c64 : i32
    %idx = arith.addi %base, %lx : i32

    %src = gfx.buffer_binding @input_buffer binding(0) access(read_only) : !gfx.ptr<f32, global>
    %dst = gfx.buffer_binding @output_buffer binding(1) access(read_write) : !gfx.ptr<f32, global>

    %src_ptr = gfx.ptr_offset %src, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %dst_ptr = gfx.ptr_offset %dst, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %x = gfx.load %src_ptr : (!gfx.ptr<f32, global>) -> f32
    %y = gfx.load %dst_ptr : (!gfx.ptr<f32, global>) -> f32
    %sum = arith.addf %x, %y : f32
    gfx.store %sum, %dst_ptr : (f32, !gfx.ptr<f32, global>)
    return
  }
}

//--- physical_storage_buffer.mlir
module attributes {
  gfx.required_capabilities = ["buffer_device_address"]
} {
  func.func @copy_bda(
      %src: !gfx.ptr<f32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<read_only>},
      %dst: !gfx.ptr<f32, global> {gfx.binding = 1 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %gid = gfx.builtin <global_invocation_id> : vector<3xi32>
    %idx = vector.extract %gid[0] : i32 from vector<3xi32>
    %src_ptr = gfx.ptr_offset %src, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %dst_ptr = gfx.ptr_offset %dst, %idx : (!gfx.ptr<f32, global>, i32) -> !gfx.ptr<f32, global>
    %val = gfx.load %src_ptr : (!gfx.ptr<f32, global>) -> f32
    gfx.store %val, %dst_ptr : (f32, !gfx.ptr<f32, global>)
    return
  }
}
