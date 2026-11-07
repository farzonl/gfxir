module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    %id = gfx.builtin <global_invocation_id> : vector<3xi32>
    return
  }
}
