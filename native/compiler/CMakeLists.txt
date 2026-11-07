cmake_minimum_required(VERSION 3.20)
project(gfx LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  set(MLIR_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}/../mlir")
  set(MLIR_INCLUDE_DIRS "${MLIR_MAIN_SRC_DIR}/include;${LLVM_BINARY_DIR}/tools/mlir/include")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  include(AddMLIR)
endif()

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
add_subdirectory(include/Gfx)

add_mlir_library(GfxCompiler
  lib/GfxDialect.cpp
  lib/Frontend.cpp
  lib/Lowering.cpp
  DEPENDS GfxIncGen
  LINK_COMPONENTS FrontendHLSL
  LINK_LIBS PUBLIC
  MLIRIR MLIRParser MLIRTransforms MLIRArithDialect MLIRFuncDialect MLIRVectorDialect
  MLIRSPIRVDialect MLIRLLVMDialect MLIRArithToSPIRV MLIRVectorToSPIRV
  MLIRSPIRVSerialization MLIRArithToLLVM MLIRFuncToLLVM MLIRVectorToLLVMPass
  MLIRReconcileUnrealizedCasts MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport)

add_llvm_executable(gfx-opt tools/gfx-opt.cpp)
target_link_libraries(gfx-opt PRIVATE GfxCompiler MLIROptLib)
add_llvm_executable(gfx-compile tools/gfx-compile.cpp)
target_link_libraries(gfx-compile PRIVATE GfxCompiler)
