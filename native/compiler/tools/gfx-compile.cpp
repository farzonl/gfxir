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
static llvm::cl::opt<std::string>
    targetProfile("target-profile",
                  llvm::cl::desc("Path to target profile JSON file or inline JSON string"));
static llvm::cl::opt<std::string>
    spirvVersion("spirv-version",
                 llvm::cl::desc("Target SPIR-V version (e.g. 1.0, 1.3, 1.5)"));
static llvm::cl::opt<std::string>
    shaderModel("shader-model",
                llvm::cl::desc("Target DirectX shader model (e.g. 6.0, 6.5, 6.6)"));
static llvm::cl::list<std::string>
    requiredCapabilities("required-capability",
                         llvm::cl::desc("Target capability supported by profile"));
static llvm::cl::opt<uint32_t>
    maxWorkgroupInvocations("max-workgroup-invocations",
                            llvm::cl::desc("Override max workgroup invocations (default 1024)"));
static llvm::cl::opt<uint32_t>
    maxSharedMemoryBytes("max-shared-memory-bytes",
                         llvm::cl::desc("Override max shared memory bytes (default 32768)"));
static llvm::cl::opt<std::string> cHeaderOutput(
    "c-header",
    llvm::cl::desc("Output C header file containing reflected aggregate struct definitions and compile-time static assertions"));

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
  gfx::TargetProfile profile;
  if (!targetProfile.empty()) {
    std::string content;
    if (llvm::sys::fs::exists(targetProfile)) {
      auto buffer = llvm::MemoryBuffer::getFile(targetProfile);
      if (!buffer) {
        llvm::errs() << "gfx-compile: failed to read target profile: "
                     << buffer.getError().message() << "\n";
        return 1;
      }
      content = (*buffer)->getBuffer().str();
    } else {
      content = targetProfile;
    }
    auto parsed = llvm::json::parse(content);
    if (!parsed || !parsed->getAsObject()) {
      llvm::errs() << "gfx-compile: target-profile must be valid JSON object\n";
      return 1;
    }
    auto loaded = gfx::TargetProfile::fromJson(*parsed->getAsObject());
    if (!loaded) {
      llvm::errs() << "gfx-compile: " << llvm::toString(loaded.takeError())
                   << "\n";
      return 1;
    }
    profile = *loaded;
  } else {
    profile = (target == "vulkan") ? gfx::TargetProfile::getDefaultVulkan()
                                   : gfx::TargetProfile::getDefaultDirectX();
  }

  if (!spirvVersion.empty()) {
    auto parsed = gfx::TargetProfile::parseSpirvVersion(spirvVersion);
    if (!parsed) {
      llvm::errs() << "gfx-compile: unrecognized spirv-version: "
                   << spirvVersion << "\n";
      return 1;
    }
    profile.spirv_version = *parsed;
  }
  if (!shaderModel.empty()) {
    auto parsed = gfx::TargetProfile::parseShaderModel(shaderModel);
    if (!parsed) {
      llvm::errs() << "gfx-compile: unrecognized shader-model: " << shaderModel
                   << "\n";
      return 1;
    }
    profile.shader_model = *parsed;
  }
  // Add CLI required capabilities to active profile
  for (const auto &cap : requiredCapabilities) {
    if (!profile.supportsCapability(cap))
      profile.required_capabilities.push_back(cap);
  }
  if (maxWorkgroupInvocations.getNumOccurrences() > 0)
    profile.max_workgroup_invocations = maxWorkgroupInvocations;
  if (maxSharedMemoryBytes.getNumOccurrences() > 0)
    profile.max_shared_memory_bytes = maxSharedMemoryBytes;

  if (!module || failed(gfx::verifyShaderModule(*module, profile)))
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
    manager.addPass(gfx::createGfxToSPIRVPass(profile));
  else
    gfx::buildDirectXPipeline(manager, profile);
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
  if (!cHeaderOutput.empty() &&
      failed(writeFile(cHeaderOutput, [&](auto &stream) {
        (void)gfx::emitCHeader(reflection, stream);
      })))
    return 1;
  return 0;
}
