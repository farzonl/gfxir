#include <dxc/dxcapi.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T> struct Release {
    void operator()(T* object) const { if (object) object->Release(); }
};
template <typename T> using ComPtr = std::unique_ptr<T, Release<T>>;

static void check(HRESULT status, const char* operation) {
    if (FAILED(status)) {
        throw std::runtime_error(std::string(operation) + " failed (HRESULT " +
                                 std::to_string(static_cast<int32_t>(status)) + ")");
    }
}

template <typename T> static ComPtr<T> create(REFCLSID clsid) {
    T* object = nullptr;
    check(DxcCreateInstance(clsid, __uuidof(T), reinterpret_cast<void**>(&object)),
          "DxcCreateInstance");
    return ComPtr<T>(object);
}

static ComPtr<IDxcBlob> result(IDxcOperationResult* operation) {
    HRESULT status;
    check(operation->GetStatus(&status), "GetStatus");
    if (FAILED(status)) {
        IDxcBlobEncoding* raw = nullptr;
        check(operation->GetErrorBuffer(&raw), "GetErrorBuffer");
        ComPtr<IDxcBlobEncoding> diagnostics(raw);
        std::string message = diagnostics && diagnostics->GetBufferSize()
            ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),
                          diagnostics->GetBufferSize())
            : "DXC returned no diagnostic text";
        throw std::runtime_error(message);
    }
    IDxcBlob* blob = nullptr;
    check(operation->GetResult(&blob), "GetResult");
    if (!blob) throw std::runtime_error("DXC did not return an output blob");
    return ComPtr<IDxcBlob>(blob);
}

int main(int argc, char** argv) {
#ifdef MLIR_GPU_ASSEMBLE
    const char* name = "dxil-assemble";
#else
    const char* name = "dxv";
#endif
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << name << ": mlir-gpu frontend for Microsoft's native DXC API\n";
        return 0;
    }
    if (argc != 4 || std::string(argv[2]) != "-o") {
        std::cerr << "Usage: " << name << " input -o output\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open input file");
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
        if (input.bad() || bytes.empty() || bytes.size() > std::numeric_limits<UINT32>::max())
            throw std::runtime_error("Input must be nonempty and smaller than 4 GiB");
        auto utils = create<IDxcUtils>(CLSID_DxcUtils);
        IDxcBlobEncoding* rawSource = nullptr;
        check(utils->CreateBlob(bytes.data(), static_cast<UINT32>(bytes.size()),
                                DXC_CP_UTF8, &rawSource), "CreateBlob");
        ComPtr<IDxcBlobEncoding> source(rawSource);
        IDxcOperationResult* rawOperation = nullptr;
#ifdef MLIR_GPU_ASSEMBLE
        auto assembler = create<IDxcAssembler>(CLSID_DxcAssembler);
        check(assembler->AssembleToContainer(source.get(), &rawOperation), "AssembleToContainer");
#else
        auto validator = create<IDxcValidator>(CLSID_DxcValidator);
        check(validator->Validate(source.get(), DxcValidatorFlags_InPlaceEdit, &rawOperation), "Validate");
#endif
        ComPtr<IDxcOperationResult> operation(rawOperation);
        auto output = result(operation.get());
        std::ofstream file(argv[3], std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("Cannot open output file");
        file.write(static_cast<const char*>(output->GetBufferPointer()), output->GetBufferSize());
        file.close();
        if (!file) throw std::runtime_error("Cannot write output file");
        std::cout << name << ": succeeded\n";
    } catch (const std::exception& error) {
        std::cerr << name << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}
