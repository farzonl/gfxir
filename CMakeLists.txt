cmake_minimum_required(VERSION 3.20)
project(mlir_gpu_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(DXC_INCLUDE_DIR NAMES dxc/dxcapi.h dxcapi.h REQUIRED)
find_library(DXC_LIBRARY dxcompiler REQUIRED)
get_filename_component(DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)

foreach(tool IN ITEMS dxv dxil-assemble)
  add_executable(${tool} native/dxil_tool.cpp)
  target_include_directories(${tool} PRIVATE ${DXC_INCLUDE_DIR})
  target_link_libraries(${tool} PRIVATE ${DXC_LIBRARY})
  if(WIN32)
    target_compile_definitions(${tool} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  set_target_properties(${tool} PROPERTIES
    BUILD_RPATH "${DXC_LIBRARY_DIR}"
    INSTALL_RPATH "${DXC_LIBRARY_DIR}")
  install(TARGETS ${tool} RUNTIME DESTINATION bin)
endforeach()
target_compile_definitions(dxil-assemble PRIVATE MLIR_GPU_ASSEMBLE=1)

set(NACELLE_ROOT "" CACHE PATH "Extracted Nacelle release SDK (enables DirectX on Unix)")
if(NACELLE_ROOT AND WIN32)
  message(FATAL_ERROR "NACELLE_ROOT is supported only on Linux and macOS")
endif()

if(WIN32 OR NACELLE_ROOT)
  include(FetchContent)
  FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG v9.1.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(glaze)

  if(NACELLE_ROOT)
    FetchContent_Declare(
      directx_headers
      GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers.git
      GIT_TAG 61c7718954ec73dbdd43c723240fe1a0452a3724
    )
    set(DXHEADERS_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(DXHEADERS_BUILD_GOOGLE_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(directx_headers)
    set_target_properties(DirectX-Headers DirectX-Guids PROPERTIES POSITION_INDEPENDENT_CODE ON)
    find_path(NACELLE_INCLUDE_DIR nacelle/nacelle.h HINTS "${NACELLE_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(NACELLE_LIBRARY nacelle_d3d12 HINTS "${NACELLE_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    add_library(gfx-nacelle INTERFACE)
    target_compile_definitions(gfx-nacelle INTERFACE GFX_NACELLE=1)
    target_include_directories(gfx-nacelle INTERFACE "${NACELLE_INCLUDE_DIR}")
    target_link_libraries(gfx-nacelle INTERFACE "${NACELLE_LIBRARY}" Microsoft::DirectX-Headers Microsoft::DirectX-Guids)
  endif()

  add_executable(mlir-gpu-directx-view native/directx_viewer.cpp)
  set_target_properties(mlir-gpu-directx-view PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  target_compile_definitions(mlir-gpu-directx-view PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
  target_link_libraries(mlir-gpu-directx-view PRIVATE glaze::glaze)
  install(TARGETS mlir-gpu-directx-view RUNTIME DESTINATION bin)

  add_library(mlir-gpu-directx-compute SHARED native/directx_compute_bridge.cpp)
  set_target_properties(mlir-gpu-directx-compute PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON PREFIX "")
  target_compile_definitions(mlir-gpu-directx-compute PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
  install(TARGETS mlir-gpu-directx-compute LIBRARY DESTINATION bin RUNTIME DESTINATION bin)

  if(NACELLE_ROOT)
    get_filename_component(NACELLE_LIBRARY_DIR "${NACELLE_LIBRARY}" DIRECTORY)
    add_library(minigfx-nacelle SHARED examples/minigfx/_directx_bridge.cpp)
    set_target_properties(minigfx-nacelle PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON PREFIX "")
    target_include_directories(minigfx-nacelle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/native")
    install(TARGETS minigfx-nacelle LIBRARY DESTINATION bin)
    foreach(tool IN ITEMS mlir-gpu-directx-view mlir-gpu-directx-compute minigfx-nacelle)
      target_link_libraries(${tool} PRIVATE gfx-nacelle)
      set_target_properties(${tool} PROPERTIES
        BUILD_RPATH "${NACELLE_LIBRARY_DIR}"
        INSTALL_RPATH "${NACELLE_LIBRARY_DIR}")
    endforeach()
  else()
    target_link_libraries(mlir-gpu-directx-view PRIVATE d3d12 dxgi user32)
    target_link_libraries(mlir-gpu-directx-compute PRIVATE d3d12 dxgi)
  endif()
endif()

if(WIN32)
  find_file(DXC_RUNTIME dxcompiler.dll HINTS
    "${DXC_LIBRARY_DIR}/../bin" "${DXC_LIBRARY_DIR}/../../bin/x64" REQUIRED)
  foreach(tool IN ITEMS dxv dxil-assemble)
    add_custom_command(TARGET ${tool} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DXC_RUNTIME}" "$<TARGET_FILE_DIR:${tool}>")
  endforeach()
  install(FILES "${DXC_RUNTIME}" DESTINATION bin)
  find_file(DXIL_RUNTIME dxil.dll HINTS
    "${DXC_LIBRARY_DIR}/../bin" "${DXC_LIBRARY_DIR}/../../bin/x64")
  if(DXIL_RUNTIME)
    install(FILES "${DXIL_RUNTIME}" DESTINATION bin)
    foreach(tool IN ITEMS dxv dxil-assemble)
      add_custom_command(TARGET ${tool} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${DXIL_RUNTIME}" "$<TARGET_FILE_DIR:${tool}>")
    endforeach()
  endif()
endif()

if(APPLE)
  enable_language(OBJCXX)
  find_path(METAL_CONVERTER_RUNTIME_INCLUDE_DIR
    metal_irconverter_runtime/metal_irconverter_runtime.h REQUIRED)
  add_executable(mlir-gpu-metal-view native/metal_viewer.mm)
  target_include_directories(mlir-gpu-metal-view PRIVATE ${METAL_CONVERTER_RUNTIME_INCLUDE_DIR})
  target_compile_options(mlir-gpu-metal-view PRIVATE -fobjc-arc)
  target_link_libraries(mlir-gpu-metal-view PRIVATE
    "-framework Cocoa" "-framework Metal" "-framework MetalKit")
  install(TARGETS mlir-gpu-metal-view RUNTIME DESTINATION bin)

  add_library(mlir-gpu-metal-compute SHARED native/metal_compute_bridge.mm)
  set_target_properties(mlir-gpu-metal-compute PROPERTIES PREFIX "")
  target_include_directories(mlir-gpu-metal-compute PRIVATE ${METAL_CONVERTER_RUNTIME_INCLUDE_DIR})
  target_compile_options(mlir-gpu-metal-compute PRIVATE -fobjc-arc)
  target_link_libraries(mlir-gpu-metal-compute PRIVATE
    "-framework Foundation" "-framework Metal")
  install(TARGETS mlir-gpu-metal-compute LIBRARY DESTINATION bin RUNTIME DESTINATION bin)
endif()
