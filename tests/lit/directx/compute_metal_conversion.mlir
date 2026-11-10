// REQUIRES: llc, dxv, metal-shaderconverter
// RUN: %gfx-compile %s --input-format mlir --target directx --output %t.ll
// RUN: %llc -O0 -filetype=obj %t.ll -o %t.unsigned.dxil
// RUN: %dxv %t.unsigned.dxil -o %t.dxil
// RUN: %metal-shaderconverter %t.dxil --entry-point=saxpy_metal --rename-entry-point=saxpy_metal -o %t.metallib

module {
  func.func @saxpy_metal() attributes {
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
