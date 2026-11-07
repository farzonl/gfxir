find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(GFX_LIT_EXEC_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
configure_file(lit.site.cfg.py.in lit.site.cfg.py.gen @ONLY)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py.gen")

if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "${LLVM_RUNTIME_OUTPUT_INTDIR}" REQUIRED)
endif()

set(GFX_TEST_DEPENDS gfx-opt gfx-compile)
foreach(tool FileCheck split-file)
  if(TARGET ${tool})
    list(APPEND GFX_TEST_DEPENDS ${tool})
  endif()
endforeach()
add_custom_target(check-gfx
  COMMAND "${CMAKE_COMMAND}" -E env
    "GFX_LIT_SITE_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
    "${Python3_EXECUTABLE}" "${LLVM_EXTERNAL_LIT}" -sv "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${GFX_TEST_DEPENDS}
  USES_TERMINAL)
