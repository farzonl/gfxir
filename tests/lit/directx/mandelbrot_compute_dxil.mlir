// REQUIRES: llc, dxv
// RUN: %gfx-opt %s --gfx-to-directx | %FileCheck %s --check-prefix=CHECK-MLIR-LLVM
// RUN: %gfx-compile %s --input-format mlir --emit-llvm -S | %FileCheck %s --check-prefix=CHECK-LLVM-IR
// RUN: %gfx-compile %s --input-format mlir --target directx --output %t.ll
// RUN: %llc -O0 -filetype=obj %t.ll -o %t.unsigned.dxil
// RUN: %dxv %t.unsigned.dxil -o %t.dxil

// CHECK-MLIR-LLVM-LABEL: llvm.func @mandelbrot_kernel()
// CHECK-MLIR-LLVM-SAME: "hlsl.shader", "compute"
// CHECK-MLIR-LLVM-SAME: "hlsl.numthreads", "16,16,1"
// CHECK-MLIR-LLVM-DAG: llvm.call_intrinsic "llvm.dx.resource.handlefrombinding"
// CHECK-MLIR-LLVM-DAG: llvm.call_intrinsic "llvm.dx.resource.getpointer"
// CHECK-MLIR-LLVM-DAG: llvm.call_intrinsic "llvm.dx.thread.id"
// CHECK-MLIR-LLVM: llvm.store

// CHECK-LLVM-IR: target triple = "dxil-pc-shadermodel6.6-compute"
// CHECK-LLVM-IR-LABEL: define void @mandelbrot_kernel()
// CHECK-LLVM-IR-DAG: call target("dx.RawBuffer"
// CHECK-LLVM-IR-DAG: call ptr addrspace(1) @llvm.dx.resource.getpointer
// CHECK-LLVM-IR-DAG: call i32 @llvm.dx.thread.id(i32 0)
// CHECK-LLVM-IR-DAG: call i32 @llvm.dx.thread.id(i32 1)
// CHECK-LLVM-IR: store i32
// CHECK-LLVM-IR-NOT: !dx.semantic.signatures
// CHECK-LLVM-IR: !dx.rootsignatures
// CHECK-LLVM-IR-DAG: !"RootConstants"
// CHECK-LLVM-IR-DAG: !"UAV", i32 1, i32 0, i32 0
// CHECK-LLVM-IR-DAG: !"DescriptorTable"

module attributes {
  gfx.parameters = #gfx.parameter_block<"mandelbrot_params", [
    #gfx.parameter_field<"width", i32>,
    #gfx.parameter_field<"height", i32>,
    #gfx.parameter_field<"max_iterations", i32>,
    #gfx.parameter_field<"xmin", f32>,
    #gfx.parameter_field<"xmax", f32>,
    #gfx.parameter_field<"ymin", f32>,
    #gfx.parameter_field<"ymax", f32>,
    #gfx.parameter_field<"pitch", i32>
  ]>
} {
  func.func @mandelbrot_kernel(
      %output_buffer: !gfx.ptr<i32, global> {gfx.binding = 0 : i32, gfx.buffer_access = #gfx.buffer_access<write_only>}
  ) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 16, 16, 1>
  } {
    %gid = gfx.builtin <global_invocation_id> : vector<3xi32>
    %px = vector.extract %gid[0] : i32 from vector<3xi32>
    %py = vector.extract %gid[1] : i32 from vector<3xi32>

    %width = gfx.parameter "mandelbrot_params" "width" : i32
    %height = gfx.parameter "mandelbrot_params" "height" : i32
    %max_iter = gfx.parameter "mandelbrot_params" "max_iterations" : i32
    %xmin = gfx.parameter "mandelbrot_params" "xmin" : f32
    %xmax = gfx.parameter "mandelbrot_params" "xmax" : f32
    %ymin = gfx.parameter "mandelbrot_params" "ymin" : f32
    %ymax = gfx.parameter "mandelbrot_params" "ymax" : f32
    %pitch = gfx.parameter "mandelbrot_params" "pitch" : i32

    %in_bounds_x = arith.cmpi ult, %px, %width : i32
    %in_bounds_y = arith.cmpi ult, %py, %height : i32
    %in_bounds = arith.andi %in_bounds_x, %in_bounds_y : i1

    scf.if %in_bounds {
      %half = arith.constant 0.5 : f32
      %px_f = arith.uitofp %px : i32 to f32
      %py_f = arith.uitofp %py : i32 to f32
      %width_f = arith.uitofp %width : i32 to f32
      %height_f = arith.uitofp %height : i32 to f32

      %u_center = arith.addf %px_f, %half : f32
      %v_center = arith.addf %py_f, %half : f32
      %u = arith.divf %u_center, %width_f : f32
      %v = arith.divf %v_center, %height_f : f32

      %dx = arith.subf %xmax, %xmin : f32
      %dy = arith.subf %ymax, %ymin : f32
      %cr_offset = arith.mulf %u, %dx : f32
      %ci_offset = arith.mulf %v, %dy : f32
      %cr = arith.addf %xmin, %cr_offset : f32
      %ci = arith.addf %ymin, %ci_offset : f32

      %zero_i32 = arith.constant 0 : i32
      %one_i32 = arith.constant 1 : i32
      %zero_f32 = arith.constant 0.0 : f32
      %four_f32 = arith.constant 4.0 : f32
      %two_f32 = arith.constant 2.0 : f32
      %true_i1 = arith.constant true

      %result:3 = scf.while (%k = %zero_i32, %x = %zero_f32, %y = %zero_f32)
          : (i32, f32, f32) -> (i32, f32, f32) {
        %x2 = arith.mulf %x, %x : f32
        %y2 = arith.mulf %y, %y : f32
        %mag2 = arith.addf %x2, %y2 : f32
        %escaped = arith.cmpf ogt, %mag2, %four_f32 : f32
        %not_escaped = arith.xori %escaped, %true_i1 : i1
        %within_max = arith.cmpi slt, %k, %max_iter : i32
        %continue = arith.andi %not_escaped, %within_max : i1
        scf.condition(%continue) %k, %x, %y : i32, f32, f32
      } do {
      ^bb0(%k: i32, %x: f32, %y: f32):
        %x2 = arith.mulf %x, %x : f32
        %y2 = arith.mulf %y, %y : f32
        %diff = arith.subf %x2, %y2 : f32
        %next_x = arith.addf %diff, %cr : f32
        %xy = arith.mulf %x, %y : f32
        %two_xy = arith.mulf %two_f32, %xy : f32
        %next_y = arith.addf %two_xy, %ci : f32
        %next_k = arith.addi %k, %one_i32 : i32
        scf.yield %next_k, %next_x, %next_y : i32, f32, f32
      }

      %row_offset = arith.muli %py, %pitch : i32
      %linear_idx = arith.addi %row_offset, %px : i32
      %out_ptr = gfx.ptr_offset %output_buffer, %linear_idx : (!gfx.ptr<i32, global>, i32) -> !gfx.ptr<i32, global>
      gfx.store %result#0, %out_ptr : (i32, !gfx.ptr<i32, global>)
    }
    return
  }
}
