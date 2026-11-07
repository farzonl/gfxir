cmake_minimum_required(VERSION 3.20)
project(mlir_gpu_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(DXC_INCLUDE_DIR dxc/dxcapi.h REQUIRED)
find_library(DXC_LIBRARY dxcompiler REQUIRED)
get_filename_component(DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)

foreach(tool IN ITEMS dxv dxil-assemble)
  add_executable(${tool} native/dxil_tool.cpp)
  target_include_directories(${tool} PRIVATE ${DXC_INCLUDE_DIR})
  target_link_libraries(${tool} PRIVATE ${DXC_LIBRARY})
  set_target_properties(${tool} PROPERTIES
    BUILD_RPATH "${DXC_LIBRARY_DIR}"
    INSTALL_RPATH "${DXC_LIBRARY_DIR}")
  install(TARGETS ${tool} RUNTIME DESTINATION bin)
endforeach()
target_compile_definitions(dxil-assemble PRIVATE MLIR_GPU_ASSEMBLE=1)

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
endif()
