#include "Gfx/Gfx.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    input(llvm::cl::Positional, llvm::cl::Required,
          llvm::cl::desc("Shader JSON or MLIR input"));
static llvm::cl::opt<std::string> format("input-format",
                                         llvm::cl::init("json"));
static llvm::cl::opt<std::string> target("target", llvm::cl::Required,
                                         llvm::cl::desc("vulkan or directx"));
static llvm::cl::opt<std::string> output("output", llvm::cl::Required);
static llvm::cl::opt<std::string> irOutput("ir-output");
static llvm::cl::opt<std::string> loweredOutput("lowered-output");
static llvm::cl::opt<std::string> reflectionOutput("reflection",
                                                   llvm::cl::Required);

static LogicalResult writeFile(StringRef path,
                               function_ref<void(llvm::raw_ostream &)> writer) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << path << ": " << error.message() << "\n";
    return failure();
  }
  writer(stream);
  stream.flush();
  if (stream.has_error()) {
    llvm::errs() << path << ": write failed\n";
    stream.clear_error();
    return failure();
  }
  return success();
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Typed gfx shader compiler\n");
  if ((target != "vulkan" && target != "directx") ||
      (format != "json" && format != "mlir")) {
    llvm::errs() << "gfx-compile: target must be vulkan/directx and "
                    "input-format must be json/mlir\n";
    return 1;
  }
  DialectRegistry registry;
  gfx::registerGfxDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module;
  if (format == "mlir") {
    module = parseSourceFile<ModuleOp>(input, &context);
  } else {
    auto buffer = llvm::MemoryBuffer::getFile(input);
    if (!buffer) {
      llvm::errs() << input << ": " << buffer.getError().message() << "\n";
      return 1;
    }
    auto json = llvm::json::parse((*buffer)->getBuffer());
    if (!json) {
      llvm::errs() << "gfx-compile: " << llvm::toString(json.takeError())
                   << "\n";
      return 1;
    }
    auto object = json->getAsObject();
    if (!object) {
      llvm::errs() << "gfx-compile: shader JSON must be an object\n";
      return 1;
    }
    module = gfx::importShader(context, *object);
  }
  if (!module || failed(gfx::verifyShaderModule(*module)))
    return 1;
  if (!irOutput.empty() && failed(writeFile(irOutput, [&](auto &stream) {
        module->print(stream);
        stream << "\n";
      })))
    return 1;
  PassManager manager(&context);
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  if (failed(manager.run(*module)))
    return 1;
  auto reflection = gfx::reflectShader(*module);
  manager.clear();
  if (target == "vulkan")
    manager.addPass(gfx::createGfxToSPIRVPass());
  else
    gfx::buildDirectXPipeline(manager);
  if (failed(manager.run(*module)) || failed(verify(*module)))
    return 1;
  if (!loweredOutput.empty() &&
      failed(writeFile(loweredOutput, [&](auto &stream) {
        module->print(stream);
        stream << "\n";
      })))
    return 1;
  if (target == "vulkan") {
    auto shader = *module->getOps<spirv::ModuleOp>().begin();
    SmallVector<uint32_t> binary;
    if (failed(spirv::serialize(shader, binary)))
      return 1;
    if (failed(writeFile(output, [&](auto &stream) {
          for (uint32_t word : binary)
            for (unsigned byte = 0; byte < 4; ++byte)
              stream << static_cast<char>(word >> (8 * byte));
        })))
      return 1;
  } else {
    llvm::LLVMContext llvmContext;
    auto translated = gfx::exportDirectX(*module, llvmContext, reflection);
    if (!translated || failed(writeFile(output, [&](auto &stream) {
          translated->print(stream, nullptr);
        })))
      return 1;
  }
  if (failed(writeFile(reflectionOutput, [&](auto &stream) {
        stream << llvm::json::Value(gfx::serializeReflection(reflection))
               << "\n";
      })))
    return 1;
  return 0;
}
