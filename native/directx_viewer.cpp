#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <glaze/glaze.hpp>

using Microsoft::WRL::ComPtr;

static void check(HRESULT status, const char* operation) {
    if (FAILED(status))
        throw std::runtime_error(std::string(operation) + " failed (HRESULT " +
                                 std::to_string(static_cast<int32_t>(status)) + ")");
}

static UINT number(const std::wstring& text, UINT maximum) {
    if (text.empty() || text.find_first_not_of(L"0123456789") != std::wstring::npos)
        throw std::runtime_error("Expected a nonnegative integer");
    unsigned long value = std::stoul(text);
    if (value > maximum) throw std::runtime_error("Numeric argument is out of range");
    return static_cast<UINT>(value);
}

static double seconds(const std::wstring& text) {
    if (text.empty())
        throw std::runtime_error("Time must be finite, nonnegative f32 seconds");
    size_t consumed = 0;
    double result = 0;
    try {
        result = std::stod(text, &consumed);
    } catch (...) {
        throw std::runtime_error("Time must be finite, nonnegative f32 seconds");
    }
    if (consumed != text.size() || !std::isfinite(result) || result < 0 ||
        result > 3.4028234663852886e38)
        throw std::runtime_error("Time must be finite, nonnegative f32 seconds");
    return result;
}

static std::vector<char> shader(const std::wstring& path) {
    std::ifstream input(std::filesystem::path(path), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open DXIL shader");
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (input.bad() || bytes.size() < 32 || std::memcmp(bytes.data(), "DXBC", 4))
        throw std::runtime_error("Expected a compiled DXIL container");
    return bytes;
}

static std::string deviceName(const wchar_t* text) {
    int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (!size) throw std::runtime_error("Cannot encode the device name");
    std::vector<char> result(size);
    if (!WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr))
        throw std::runtime_error("Cannot encode the device name");
    return std::string(result.data());
}

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Event() {
        if (!handle) throw std::runtime_error("Cannot create GPU completion event");
    }
    ~Event() { CloseHandle(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct SelectedDevice {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 description{};
    std::string name;
};

static bool isSoftwareAdapter(const DXGI_ADAPTER_DESC1& desc) {
    return (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 || desc.VendorId == 0x1414;
}

static SelectedDevice createDirectXDevice(bool warp, bool validation) {
    if (validation) {
        ComPtr<ID3D12Debug1> debug;
        check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),
              "D3D12 debug layer (install Windows Graphics Tools)");
        debug->EnableDebugLayer();
        debug->SetEnableGPUBasedValidation(TRUE);
    }
    SelectedDevice result;
    check(CreateDXGIFactory2(validation ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&result.factory)),
          "CreateDXGIFactory2");

    if (warp) {
        HRESULT status = result.factory->EnumWarpAdapter(IID_PPV_ARGS(&result.adapter));
        if (FAILED(status)) {
            throw std::runtime_error("DirectX WARP software adapter is unavailable. Accidental hardware/driver fallback rejected.");
        }
        check(result.adapter->GetDesc1(&result.description), "GetDesc1");
        if (!isSoftwareAdapter(result.description)) {
            throw std::runtime_error("WARP adapter enumeration did not return a software adapter. Accidental hardware/driver fallback rejected.");
        }
        status = D3D12CreateDevice(result.adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&result.device));
        if (status == DXGI_ERROR_UNSUPPORTED || status == E_INVALIDARG || FAILED(status)) {
            throw std::runtime_error("Failed to create DirectX 12 device on WARP adapter. Accidental hardware/driver fallback rejected.");
        }
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
        check(result.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model)),
              "Check Shader Model support");
        if (model.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
            throw std::runtime_error("DirectX WARP adapter does not support Shader Model 6.6. Accidental hardware/driver fallback rejected.");
        }
        result.name = deviceName(result.description.Description);
    } else {
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT status = result.factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (status == DXGI_ERROR_NOT_FOUND) break;
            check(status, "EnumAdapterByGpuPreference");
            DXGI_ADAPTER_DESC1 candidateDesc{};
            check(candidate->GetDesc1(&candidateDesc), "GetDesc1");
            if (isSoftwareAdapter(candidateDesc)) continue;
            status = D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&result.device));
            if (status == DXGI_ERROR_UNSUPPORTED || status == E_INVALIDARG) continue;
            check(status, "D3D12CreateDevice");
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
            check(result.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model)),
                  "Check Shader Model support");
            if (model.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
                result.device.Reset();
                continue;
            }
            result.adapter = candidate;
            result.description = candidateDesc;
            result.name = deviceName(result.description.Description);
            break;
        }
        if (!result.device) throw std::runtime_error("No DirectX 12 GPU supporting Shader Model 6.6 is available");
    }
    return result;
}

static std::string queryFeatureLevel(ID3D12Device* device) {
    const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_12_2,
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    D3D12_FEATURE_DATA_FEATURE_LEVELS data{};
    data.NumFeatureLevels = static_cast<UINT>(sizeof(levels) / sizeof(levels[0]));
    data.pFeatureLevelsRequested = levels;
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &data, sizeof(data)))) {
        switch (data.MaxSupportedFeatureLevel) {
            case D3D_FEATURE_LEVEL_12_2: return "12_2";
            case D3D_FEATURE_LEVEL_12_1: return "12_1";
            case D3D_FEATURE_LEVEL_12_0: return "12_0";
            case D3D_FEATURE_LEVEL_11_1: return "11_1";
            case D3D_FEATURE_LEVEL_11_0: return "11_0";
            default: break;
        }
    }
    return "unknown";
}

static std::string queryShaderModel(ID3D12Device* device) {
    const D3D_SHADER_MODEL candidateModels[] = {
        D3D_SHADER_MODEL_6_9, D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7,
        D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_4,
        D3D_SHADER_MODEL_6_3, D3D_SHADER_MODEL_6_2, D3D_SHADER_MODEL_6_1,
        D3D_SHADER_MODEL_6_0, D3D_SHADER_MODEL_5_1
    };
    for (D3D_SHADER_MODEL model : candidateModels) {
        D3D12_FEATURE_DATA_SHADER_MODEL data{model};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &data, sizeof(data)))) {
            UINT highest = static_cast<UINT>(data.HighestShaderModel);
            if (highest >= 0x51) {
                UINT major = highest >> 4;
                UINT minor = highest & 0xF;
                return std::to_string(major) + "_" + std::to_string(minor);
            }
        }
    }
    return "unknown";
}

struct CapabilitiesReport {
    std::string provider;
    std::string adapter_name;
    std::string device_name;
    std::string device;
    UINT vendor_id;
    UINT device_id;
    UINT subsystem_id;
    UINT revision;
    uint64_t dedicated_memory;
    uint64_t dedicated_video_memory;
    uint64_t dedicated_system_memory;
    uint64_t shared_system_memory;
    bool software_adapter;
    bool is_software;
    std::string feature_level;
    std::string d3d12_feature_level;
    std::string shader_model;
    std::string highest_shader_model;
    std::array<uint32_t, 3> max_workgroup_size = {1024, 1024, 64};
    uint32_t max_workgroup_invocations = 1024;
    uint32_t max_shared_memory_bytes = 32768;
};

struct ReadbackReport {
    std::string device;
    UINT width;
    UINT height;
};

struct ComputeBufferSpec {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    std::string access;
    uint64_t size = 0;
    std::string input_path;
    std::string output_path;
};

struct ComputeDispatchSpec {
    std::array<uint32_t, 3> workgroup_size = {1, 1, 1};
    std::array<uint32_t, 3> grid = {1, 1, 1};
    std::string root_constants_path;
    std::vector<ComputeBufferSpec> buffers;
};

struct ComputeReport {
    std::string device;
};

static std::vector<char> binaryFile(const std::filesystem::path& path, const char* description) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("Cannot open ") + description);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw std::runtime_error(std::string("Cannot read ") + description);
    return bytes;
}

static void checkValidationQueue(ID3D12InfoQueue* diagnostics) {
    if (!diagnostics) return;
    std::string errors;
    UINT64 count = diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 index = 0; index < count; ++index) {
        SIZE_T size = 0;
        check(diagnostics->GetMessage(index, nullptr, &size), "GetMessage size");
        std::vector<char> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        check(diagnostics->GetMessage(index, message, &size), "GetMessage");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
            errors += std::string(message->pDescription) + "\n";
    }
    diagnostics->ClearStoredMessages();
    if (!errors.empty()) throw std::runtime_error("D3D12 validation failed:\n" + errors);
}

static void runComputeDispatch(const std::wstring& computePath, const std::wstring& specPath,
                               bool warp, bool validation) {
    auto specBytes = binaryFile(std::filesystem::path(specPath), "compute dispatch spec");
    std::string specJson(specBytes.begin(), specBytes.end());
    ComputeDispatchSpec spec{};
    auto parseErr = glz::read<glz::opts{.error_on_unknown_keys = false}>(spec, specJson);
    if (parseErr) {
        throw std::runtime_error("Invalid compute dispatch spec JSON");
    }
    for (uint32_t dim : spec.grid) {
        if (dim == 0 || dim > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) {
            throw std::runtime_error("Compute dispatch grid dimension is out of D3D12 bounds");
        }
    }

    auto dev = createDirectXDevice(warp, validation);
    ComPtr<ID3D12InfoQueue> diagnostics;
    if (validation) {
        check(dev.device.As(&diagnostics), "Get D3D12 validation queue");
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    check(dev.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    ComPtr<ID3D12CommandAllocator> allocator;
    check(dev.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
          "CreateCommandAllocator");

    auto cs = shader(computePath);

    bool hasRootConstants = false;
    UINT rootConstantsParamIndex = 0;
    UINT rootConstantsNumWords = 0;
    bool hasDescriptorTable = false;
    UINT descriptorTableParamIndex = 0;

    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12VersionedRootSignatureDeserializer> deserializer;
    if (SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(
            cs.data(), cs.size(), IID_PPV_ARGS(&deserializer)))) {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc =
            deserializer->GetUnconvertedRootSignatureDesc();
        if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
            for (UINT i = 0; i < desc->Desc_1_1.NumParameters; ++i) {
                const auto& param = desc->Desc_1_1.pParameters[i];
                if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    hasRootConstants = true;
                    rootConstantsParamIndex = i;
                    rootConstantsNumWords = param.Constants.Num32BitValues;
                } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
                    hasDescriptorTable = true;
                    descriptorTableParamIndex = i;
                }
            }
        } else {
            for (UINT i = 0; i < desc->Desc_1_0.NumParameters; ++i) {
                const auto& param = desc->Desc_1_0.pParameters[i];
                if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    hasRootConstants = true;
                    rootConstantsParamIndex = i;
                    rootConstantsNumWords = param.Constants.Num32BitValues;
                } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
                    hasDescriptorTable = true;
                    descriptorTableParamIndex = i;
                }
            }
        }
        HRESULT rsStatus = dev.device->CreateRootSignature(
            0, cs.data(), cs.size(), IID_PPV_ARGS(&root));
        if (FAILED(rsStatus) && diagnostics) {
            diagnostics->ClearStoredMessages();
        }
    }

    if (!root) {
        D3D12_ROOT_SIGNATURE_DESC emptyDesc{};
        ComPtr<ID3DBlob> serialized, errors;
        HRESULT serStatus = D3D12SerializeRootSignature(
            &emptyDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
        if (FAILED(serStatus) && errors) {
            throw std::runtime_error(std::string(
                static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize()));
        }
        check(serStatus, "D3D12SerializeRootSignature");
        check(dev.device->CreateRootSignature(
                  0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)),
              "CreateRootSignature");
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = root.Get();
    psoDesc.CS = {cs.data(), cs.size()};
    ComPtr<ID3D12PipelineState> pipeline;
    HRESULT psoStatus = dev.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline));
    checkValidationQueue(diagnostics.Get());
    check(psoStatus, "CreateComputePipelineState (compiled DXIL)");

    std::vector<uint32_t> rootConstantWords;
    if (hasRootConstants) {
        rootConstantWords.assign(rootConstantsNumWords, 0);
        if (!spec.root_constants_path.empty()) {
            auto rcBytes = binaryFile(std::filesystem::path(spec.root_constants_path), "root constants");
            size_t requiredBytes = static_cast<size_t>(rootConstantsNumWords) * sizeof(uint32_t);
            if (rcBytes.size() < requiredBytes) {
                throw std::runtime_error("Root constants file is smaller than the root signature parameter block");
            }
            std::memcpy(rootConstantWords.data(), rcBytes.data(), requiredBytes);
        } else {
            throw std::runtime_error("Compute root signature requires root constants, but none were provided");
        }
    }

    std::stable_sort(spec.buffers.begin(), spec.buffers.end(),
                     [](const ComputeBufferSpec& a, const ComputeBufferSpec& b) {
                         if (a.set != b.set) return a.set < b.set;
                         return a.binding < b.binding;
                     });

    struct AllocatedComputeBuffer {
        ComputeBufferSpec spec;
        bool isUAV = false;
        uint64_t alignedSize = 0;
        ComPtr<ID3D12Resource> defaultBuf;
        ComPtr<ID3D12Resource> uploadBuf;
        ComPtr<ID3D12Resource> readbackBuf;
    };

    std::vector<AllocatedComputeBuffer> buffers;
    buffers.reserve(spec.buffers.size());
    std::vector<size_t> uavIndices;
    std::vector<size_t> srvIndices;

    for (const auto& bufSpec : spec.buffers) {
        if (bufSpec.access != "read_only" && bufSpec.access != "write_only" &&
            bufSpec.access != "read_write") {
            throw std::runtime_error("Unsupported compute buffer access mode: " + bufSpec.access);
        }
        if (bufSpec.size == 0) {
            throw std::runtime_error("Compute buffer byte size must be positive: " + bufSpec.name);
        }

        AllocatedComputeBuffer entry;
        entry.spec = bufSpec;
        entry.isUAV = (bufSpec.access != "read_only");
        entry.alignedSize = std::max<uint64_t>(4, (bufSpec.size + 3ULL) & ~3ULL);

        D3D12_HEAP_PROPERTIES defaultHeap{};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC defaultDesc{};
        defaultDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        defaultDesc.Width = entry.alignedSize;
        defaultDesc.Height = 1;
        defaultDesc.DepthOrArraySize = 1;
        defaultDesc.MipLevels = 1;
        defaultDesc.SampleDesc.Count = 1;
        defaultDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        defaultDesc.Flags = entry.isUAV ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                        : D3D12_RESOURCE_FLAG_NONE;
        check(dev.device->CreateCommittedResource(
                  &defaultHeap, D3D12_HEAP_FLAG_NONE, &defaultDesc,
                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&entry.defaultBuf)),
              "Create compute default buffer");

        D3D12_HEAP_PROPERTIES uploadHeap{};
        uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC stagingDesc = defaultDesc;
        stagingDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        check(dev.device->CreateCommittedResource(
                  &uploadHeap, D3D12_HEAP_FLAG_NONE, &stagingDesc,
                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&entry.uploadBuf)),
              "Create compute upload buffer");

        void* uploadMapped = nullptr;
        D3D12_RANGE emptyReadRange{0, 0};
        check(entry.uploadBuf->Map(0, &emptyReadRange, &uploadMapped), "Map compute upload buffer");
        std::memset(uploadMapped, 0, static_cast<size_t>(entry.alignedSize));
        if (!bufSpec.input_path.empty()) {
            auto inputBytes = binaryFile(std::filesystem::path(bufSpec.input_path), "compute input buffer");
            if (inputBytes.size() != bufSpec.size) {
                entry.uploadBuf->Unmap(0, nullptr);
                throw std::runtime_error("Compute input buffer file size does not match spec size: " + bufSpec.name);
            }
            std::memcpy(uploadMapped, inputBytes.data(), inputBytes.size());
        }
        entry.uploadBuf->Unmap(0, nullptr);

        if (entry.isUAV && !bufSpec.output_path.empty()) {
            D3D12_HEAP_PROPERTIES readbackHeap{};
            readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;
            check(dev.device->CreateCommittedResource(
                      &readbackHeap, D3D12_HEAP_FLAG_NONE, &stagingDesc,
                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&entry.readbackBuf)),
                  "Create compute readback buffer");
        }

        size_t idx = buffers.size();
        if (entry.isUAV) {
            uavIndices.push_back(idx);
        } else {
            srvIndices.push_back(idx);
        }
        buffers.push_back(std::move(entry));
    }

    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    size_t totalDescriptors = uavIndices.size() + srvIndices.size();
    if (hasDescriptorTable || totalDescriptors > 0) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = static_cast<UINT>(std::max<size_t>(1, totalDescriptors));
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(dev.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descriptorHeap)),
              "Create compute CBV_SRV_UAV descriptor heap");

        UINT increment = dev.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE handle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();

        for (size_t idx : uavIndices) {
            const auto& buf = buffers[idx];
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
            uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Buffer.FirstElement = 0;
            uavDesc.Buffer.NumElements = static_cast<UINT>(buf.alignedSize / 4);
            uavDesc.Buffer.StructureByteStride = 0;
            uavDesc.Buffer.CounterOffsetInBytes = 0;
            uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            dev.device->CreateUnorderedAccessView(buf.defaultBuf.Get(), nullptr, &uavDesc, handle);
            handle.ptr += increment;
        }
        for (size_t idx : srvIndices) {
            const auto& buf = buffers[idx];
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
            srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Buffer.FirstElement = 0;
            srvDesc.Buffer.NumElements = static_cast<UINT>(buf.alignedSize / 4);
            srvDesc.Buffer.StructureByteStride = 0;
            srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            dev.device->CreateShaderResourceView(buf.defaultBuf.Get(), &srvDesc, handle);
            handle.ptr += increment;
        }
    }

    ComPtr<ID3D12GraphicsCommandList> commands;
    check(dev.device->CreateCommandList(
              0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
              IID_PPV_ARGS(&commands)),
          "Create compute command list");

    if (!buffers.empty()) {
        std::vector<D3D12_RESOURCE_BARRIER> toCopyDest;
        toCopyDest.reserve(buffers.size());
        for (const auto& buf : buffers) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = buf.defaultBuf.Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            toCopyDest.push_back(b);
        }
        commands->ResourceBarrier(static_cast<UINT>(toCopyDest.size()), toCopyDest.data());

        for (const auto& buf : buffers) {
            commands->CopyBufferRegion(
                buf.defaultBuf.Get(), 0, buf.uploadBuf.Get(), 0, buf.alignedSize);
        }

        std::vector<D3D12_RESOURCE_BARRIER> toShaderState;
        toShaderState.reserve(buffers.size());
        for (const auto& buf : buffers) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = buf.defaultBuf.Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = buf.isUAV ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            toShaderState.push_back(b);
        }
        commands->ResourceBarrier(static_cast<UINT>(toShaderState.size()), toShaderState.data());
    }

    commands->SetComputeRootSignature(root.Get());
    if (descriptorHeap) {
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap.Get()};
        commands->SetDescriptorHeaps(1, heaps);
    }
    if (hasRootConstants) {
        commands->SetComputeRoot32BitConstants(
            rootConstantsParamIndex, rootConstantsNumWords, rootConstantWords.data(), 0);
    }
    if (hasDescriptorTable && descriptorHeap) {
        commands->SetComputeRootDescriptorTable(
            descriptorTableParamIndex, descriptorHeap->GetGPUDescriptorHandleForHeapStart());
    }

    commands->Dispatch(spec.grid[0], spec.grid[1], spec.grid[2]);

    std::vector<D3D12_RESOURCE_BARRIER> postDispatchBarriers;
    for (const auto& buf : buffers) {
        if (buf.readbackBuf) {
            D3D12_RESOURCE_BARRIER uavBarrier{};
            uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uavBarrier.UAV.pResource = buf.defaultBuf.Get();
            postDispatchBarriers.push_back(uavBarrier);

            D3D12_RESOURCE_BARRIER toCopySource{};
            toCopySource.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            toCopySource.Transition.pResource = buf.defaultBuf.Get();
            toCopySource.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            toCopySource.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            toCopySource.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            postDispatchBarriers.push_back(toCopySource);
        }
    }
    if (!postDispatchBarriers.empty()) {
        commands->ResourceBarrier(
            static_cast<UINT>(postDispatchBarriers.size()), postDispatchBarriers.data());
    }
    for (const auto& buf : buffers) {
        if (buf.readbackBuf) {
            commands->CopyBufferRegion(
                buf.readbackBuf.Get(), 0, buf.defaultBuf.Get(), 0, buf.alignedSize);
        }
    }

    check(commands->Close(), "Close compute command list");
    ID3D12CommandList* lists[] = {commands.Get()};
    queue->ExecuteCommandLists(1, lists);

    ComPtr<ID3D12Fence> fence;
    check(dev.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    Event completion;
    check(queue->Signal(fence.Get(), 1), "Signal compute fence");
    check(fence->SetEventOnCompletion(1, completion.handle), "SetEventOnCompletion");
    DWORD waitStatus = WaitForSingleObject(completion.handle, 30000);
    if (waitStatus != WAIT_OBJECT_0) {
        check(dev.device->GetDeviceRemovedReason(), "GPU device removed");
        throw std::runtime_error("Compute GPU completion wait failed or timed out");
    }
    check(dev.device->GetDeviceRemovedReason(), "GPU device removed");
    checkValidationQueue(diagnostics.Get());

    for (const auto& buf : buffers) {
        if (!buf.readbackBuf) continue;
        void* mapped = nullptr;
        D3D12_RANGE readRange{0, static_cast<SIZE_T>(buf.spec.size)};
        check(buf.readbackBuf->Map(0, &readRange, &mapped), "Map compute readback buffer");
        std::ofstream output(std::filesystem::path(buf.spec.output_path),
                             std::ios::binary | std::ios::trunc);
        if (!output) {
            D3D12_RANGE emptyWritten{0, 0};
            buf.readbackBuf->Unmap(0, &emptyWritten);
            throw std::runtime_error("Cannot open compute buffer readback output: " + buf.spec.name);
        }
        output.write(static_cast<const char*>(mapped), static_cast<std::streamsize>(buf.spec.size));
        D3D12_RANGE emptyWritten{0, 0};
        buf.readbackBuf->Unmap(0, &emptyWritten);
        output.close();
        if (!output) {
            throw std::runtime_error("Cannot write compute buffer readback output: " + buf.spec.name);
        }
    }

    std::cout << glz::write_json(ComputeReport{dev.name}).value_or("{}") << "\n" << std::flush;
}

static void reportDeviceCapabilities(bool warp, bool validation) {
    SelectedDevice dev = createDirectXDevice(warp, validation);
    std::string featureLevel = queryFeatureLevel(dev.device.Get());
    std::string shaderModel = queryShaderModel(dev.device.Get());
    std::string featureLevelDot = featureLevel;
    size_t underscore = featureLevelDot.find('_');
    if (underscore != std::string::npos) featureLevelDot[underscore] = '.';
    std::string shaderModelDot = shaderModel;
    underscore = shaderModelDot.find('_');
    if (underscore != std::string::npos) shaderModelDot[underscore] = '.';
    bool isSoftware = isSoftwareAdapter(dev.description) || warp;

    CapabilitiesReport report{
        warp ? "warp" : "hardware",
        dev.name,
        dev.name,
        dev.name,
        dev.description.VendorId,
        dev.description.DeviceId,
        dev.description.SubSysId,
        dev.description.Revision,
        static_cast<uint64_t>(dev.description.DedicatedVideoMemory),
        static_cast<uint64_t>(dev.description.DedicatedVideoMemory),
        static_cast<uint64_t>(dev.description.DedicatedSystemMemory),
        static_cast<uint64_t>(dev.description.SharedSystemMemory),
        isSoftware,
        isSoftware,
        featureLevelDot,
        featureLevel,
        shaderModel,
        shaderModelDot,
        std::array<uint32_t, 3>{1024, 1024, 64},
        1024,
        32768,
    };
    std::cout << glz::write_json(report).value_or("{}") << "\n" << std::flush;
}

struct DirectXDrawCall {
    std::string pipelineId;
    std::string type = "standard"; // "standard" or "indexed"
    UINT vertexCount = 0;
    UINT indexCount = 0;
    UINT instanceCount = 1;
    UINT firstVertex = 0;
    UINT firstIndex = 0;
    INT vertexOffset = 0;
    UINT firstInstance = 0;
    UINT constantOffset = 0;
    UINT constantSize = 0;
    std::vector<uint32_t> rootConstants;
};

class Renderer {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> diagnostics;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12Fence> fence;
    Event completion;
    UINT64 fenceValue = 0;
    UINT vertexCount;
    std::string name;
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> pipelines;
    std::vector<DirectXDrawCall> draws;
    bool hasFrameConstants = false;
    bool hasFixedTime = false;
    double fixedTime = 0.0;
    std::chrono::steady_clock::time_point started;

    void validationErrors() {
        if (!diagnostics) return;
        std::string errors;
        UINT64 count = diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter();
        for (UINT64 index = 0; index < count; ++index) {
            SIZE_T size = 0;
            check(diagnostics->GetMessage(index, nullptr, &size), "GetMessage size");
            std::vector<char> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(diagnostics->GetMessage(index, message, &size), "GetMessage");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
                errors += std::string(message->pDescription) + "\n";
        }
        diagnostics->ClearStoredMessages();
        if (!errors.empty()) throw std::runtime_error("D3D12 validation failed:\n" + errors);
    }

    void wait() {
        check(queue->Signal(fence.Get(), ++fenceValue), "Signal");
        check(fence->SetEventOnCompletion(fenceValue, completion.handle), "SetEventOnCompletion");
        DWORD status = WaitForSingleObject(completion.handle, 30000);
        if (status != WAIT_OBJECT_0) {
            check(device->GetDeviceRemovedReason(), "GPU device removed");
            throw std::runtime_error("GPU completion wait failed or timed out");
        }
        check(device->GetDeviceRemovedReason(), "GPU device removed");
        validationErrors();
    }

    void begin() {
        check(allocator->Reset(), "Reset command allocator");
        check(commands->Reset(allocator.Get(), pipeline.Get()), "Reset command list");
    }

    void barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER transition{};
        transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition.pResource = resource;
        transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        transition.Transition.StateBefore = before;
        transition.Transition.StateAfter = after;
        commands->ResourceBarrier(1, &transition);
    }

    void draw(ID3D12Resource* target, UINT width, UINT height, double elapsed) {
        auto handle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target, nullptr, handle);
        commands->SetGraphicsRootSignature(root.Get());
        if (hasFrameConstants) {
            if (!std::isfinite(elapsed) || elapsed < 0 || elapsed > 3.4028234663852886e38)
                throw std::runtime_error("Invalid elapsed frame time");
            const float constants[4] = {
                static_cast<float>(elapsed),
                0.0f,
                static_cast<float>(width),
                static_cast<float>(height),
            };
            commands->SetGraphicsRoot32BitConstants(0, 4, constants, 0);
        }
        D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
        D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        commands->RSSetViewports(1, &viewport);
        commands->RSSetScissorRects(1, &scissor);
        commands->OMSetRenderTargets(1, &handle, FALSE, nullptr);
        const float background[]{0.02f, 0.02f, 0.04f, 1.0f};
        commands->ClearRenderTargetView(handle, background, 0, nullptr);
        commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        if (draws.empty()) {
            commands->DrawInstanced(vertexCount, 1, 0, 0);
        } else {
            ID3D12PipelineState* currentPso = nullptr;
            for (const auto& d : draws) {
                ID3D12PipelineState* nextPso = pipeline.Get();
                auto it = pipelines.find(d.pipelineId);
                if (it != pipelines.end() && it->second) {
                    nextPso = it->second.Get();
                }
                if (nextPso != currentPso) {
                    commands->SetPipelineState(nextPso);
                    currentPso = nextPso;
                }
                if (!d.rootConstants.empty()) {
                    commands->SetGraphicsRoot32BitConstants(
                        1, static_cast<UINT>(d.rootConstants.size()), d.rootConstants.data(), 0);
                }
                if (d.type == "indexed") {
                    commands->DrawIndexedInstanced(
                        d.indexCount, d.instanceCount, d.firstIndex, d.vertexOffset, d.firstInstance);
                } else {
                    commands->DrawInstanced(
                        d.vertexCount ? d.vertexCount : vertexCount,
                        d.instanceCount, d.firstVertex, d.firstInstance);
                }
            }
        }
    }

    void submit() {
        check(commands->Close(), "Close command list");
        ID3D12CommandList* lists[]{commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        wait();
    }

public:
    void setDraws(std::vector<DirectXDrawCall> drawList) {
        draws = std::move(drawList);
    }
    void addPipeline(const std::string& id, ComPtr<ID3D12PipelineState> pso) {
        pipelines[id] = std::move(pso);
    }

    Renderer(const std::wstring& vertex, const std::wstring& fragment, UINT count, bool validation,
             bool warp = false, bool hasTime = false, double time = 0.0)
        : vertexCount(count), hasFixedTime(hasTime), fixedTime(time),
          started(std::chrono::steady_clock::now()) {
        auto dev = createDirectXDevice(warp, validation);
        factory = std::move(dev.factory);
        device = std::move(dev.device);
        name = std::move(dev.name);
        if (validation) check(device.As(&diagnostics), "Get D3D12 validation queue");
        D3D12_COMMAND_QUEUE_DESC queueDescription{};
        queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
              "CreateCommandAllocator");

        auto vs = shader(vertex), ps = shader(fragment);

        for (const auto* bytecode : {&vs, &ps}) {
            ComPtr<ID3D12VersionedRootSignatureDeserializer> deserializer;
            if (SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(
                    bytecode->data(), bytecode->size(), IID_PPV_ARGS(&deserializer)))) {
                const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc =
                    deserializer->GetUnconvertedRootSignatureDesc();
                UINT numParams = (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_1)
                                     ? desc->Desc_1_1.NumParameters
                                     : desc->Desc_1_0.NumParameters;
                HRESULT status = device->CreateRootSignature(
                    0, bytecode->data(), bytecode->size(), IID_PPV_ARGS(&root));
                if (SUCCEEDED(status)) {
                    hasFrameConstants = numParams > 0;
                    break;
                }
                if (diagnostics) diagnostics->ClearStoredMessages();
            }
        }

        if (!root) {
            D3D12_ROOT_SIGNATURE_DESC rootDescription{};
            rootDescription.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

            ComPtr<ID3DBlob> serialized, errors;
            HRESULT status = D3D12SerializeRootSignature(
                &rootDescription, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
            if (FAILED(status) && errors)
                throw std::runtime_error(std::string(static_cast<const char*>(errors->GetBufferPointer()),
                                                    errors->GetBufferSize()));
            check(status, "D3D12SerializeRootSignature");
            check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&root)), "CreateRootSignature");
        }

        D3D12_GRAPHICS_PIPELINE_STATE_DESC description{};
        description.pRootSignature = root.Get();
        description.VS = {vs.data(), vs.size()};
        description.PS = {ps.data(), ps.size()};
        auto& blend = description.BlendState.RenderTarget[0];
        blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
        blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
        blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        blend.LogicOp = D3D12_LOGIC_OP_NOOP;
        blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        description.SampleMask = UINT_MAX;
        description.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        description.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        description.RasterizerState.DepthClipEnable = TRUE;
        description.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        description.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
        description.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
        auto& stencil = description.DepthStencilState.FrontFace;
        stencil.StencilFailOp = stencil.StencilDepthFailOp = stencil.StencilPassOp = D3D12_STENCIL_OP_KEEP;
        stencil.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        description.DepthStencilState.BackFace = stencil;
        description.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        description.NumRenderTargets = 1;
        description.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.SampleDesc.Count = 1;
        HRESULT status = device->CreateGraphicsPipelineState(&description, IID_PPV_ARGS(&pipeline));
        validationErrors();
        check(status, "CreateGraphicsPipelineState (compiled DXIL)");
        D3D12_DESCRIPTOR_HEAP_DESC heap{};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap.NumDescriptors = 1;
        check(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&rtvHeap)), "CreateDescriptorHeap");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                       IID_PPV_ARGS(&commands)), "CreateCommandList");
        check(commands->Close(), "Close initial command list");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        validationErrors();
    }

    void readback(UINT width, UINT height, const std::wstring& path) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = width;
        texture.Height = height;
        texture.DepthOrArraySize = texture.MipLevels = 1;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        ComPtr<ID3D12Resource> target, buffer;
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&target)), "Create offscreen target");
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 size = 0;
        device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, nullptr, nullptr, &size);
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC storage{};
        storage.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        storage.Width = size;
        storage.Height = storage.DepthOrArraySize = storage.MipLevels = 1;
        storage.SampleDesc.Count = 1;
        storage.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &storage,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)), "Create readback buffer");
        double elapsed = hasFixedTime ? fixedTime : 0.0;
        begin();
        draw(target.Get(), width, height, elapsed);
        barrier(target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
        source.pResource = target.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource = buffer.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprint;
        commands->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        submit();
        std::vector<char> rgba(static_cast<size_t>(width) * height * 4);
        void* mapped = nullptr;
        D3D12_RANGE range{0, static_cast<SIZE_T>(size)};
        check(buffer->Map(0, &range, &mapped), "Map readback");
        for (UINT row = 0; row < height; ++row)
            std::memcpy(rgba.data() + static_cast<size_t>(row) * width * 4,
                        static_cast<char*>(mapped) + footprint.Offset +
                            static_cast<size_t>(row) * footprint.Footprint.RowPitch,
                        static_cast<size_t>(width) * 4);
        D3D12_RANGE written{0, 0};
        buffer->Unmap(0, &written);
        std::ofstream output(std::filesystem::path(path), std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open RGBA readback output");
        output.write(rgba.data(), rgba.size());
        output.close();
        if (!output) throw std::runtime_error("Cannot write RGBA readback output");
        std::cout << glz::write_json(ReadbackReport{name, width, height}).value_or("{}") << "\n";
    }

    void window(HWND window, UINT width, UINT height, UINT limit) {
        DXGI_SWAP_CHAIN_DESC1 description{};
        description.Width = width;
        description.Height = height;
        description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.SampleDesc.Count = 1;
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = 2;
        description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<IDXGISwapChain1> initial;
        ComPtr<IDXGISwapChain3> swapchain;
        HRESULT creation = factory->CreateSwapChainForHwnd(
            queue.Get(), window, &description, nullptr, nullptr, &initial);
        if (creation == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)
            throw std::runtime_error(
                "DXGI presentation is unavailable. Run the live viewer in an interactive Windows "
                "desktop session (not Session 0), or use --output for offscreen rendering.");
        check(creation, "CreateSwapChainForHwnd");
        check(initial.As(&swapchain), "Get swapchain");
        check(factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER), "MakeWindowAssociation");
        UINT support = 0;
        check(swapchain->CheckColorSpaceSupport(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, &support),
              "CheckColorSpaceSupport");
        if (!(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
            throw std::runtime_error("The display does not support sRGB presentation");
        check(swapchain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709), "SetColorSpace1");
        std::cout << "DirectX 12 viewer ready\n" << std::flush;
        UINT frames = 0;
        bool running = true;
        while (running && (!limit || frames < limit)) {
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
                if (message.message == WM_QUIT) running = false;
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            if (!running) break;
            RECT area{};
            if (!GetClientRect(window, &area)) throw std::runtime_error("Cannot query window dimensions");
            UINT currentWidth = area.right - area.left, currentHeight = area.bottom - area.top;
            if (IsIconic(window) || !currentWidth || !currentHeight) {
                MsgWaitForMultipleObjects(0, nullptr, FALSE, 100, QS_ALLINPUT);
                continue;
            }
            if (width != currentWidth || height != currentHeight) {
                wait();
                check(swapchain->ResizeBuffers(2, currentWidth, currentHeight, description.Format, 0),
                      "ResizeBuffers");
                width = currentWidth;
                height = currentHeight;
            }
            double elapsed = hasFixedTime ? fixedTime :
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            ComPtr<ID3D12Resource> target;
            check(swapchain->GetBuffer(swapchain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&target)),
                  "Get swapchain buffer");
            begin();
            barrier(target.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
            draw(target.Get(), width, height, elapsed);
            barrier(target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
            submit();
            HRESULT status = swapchain->Present(1, 0);
            check(status, "Present");
            if (status == DXGI_STATUS_OCCLUDED) {
                MsgWaitForMultipleObjects(0, nullptr, FALSE, 100, QS_ALLINPUT);
                continue;
            }
            ++frames;
        }
        wait();
        std::cout << "Presented " << frames << " DirectX 12 frames.\n";
    }
};

static LRESULT CALLBACK windowProcedure(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_CLOSE || (message == WM_KEYDOWN && wparam == VK_ESCAPE)) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wparam, lparam);
}

struct Window {
    HWND handle = nullptr;
    ~Window() { if (handle) DestroyWindow(handle); }
};

int wmain(int argc, wchar_t** argv) {
    try {
        std::map<std::wstring, std::wstring> options;
        bool validation = false;
        bool warp = false;
        bool reportCapabilities = false;
        for (int index = 1; index < argc; ++index) {
            std::wstring key = argv[index];
            if (key == L"--validation") {
                if (validation) throw std::runtime_error("Missing or duplicate native viewer argument");
                validation = true;
                continue;
            }
            if (key == L"--warp") {
                if (warp) throw std::runtime_error("Missing or duplicate native viewer argument");
                warp = true;
                continue;
            }
            if (key == L"--report-capabilities") {
                if (reportCapabilities) throw std::runtime_error("Missing or duplicate native viewer argument");
                reportCapabilities = true;
                continue;
            }
            if (key == L"-frames") key = L"--frames";
            if (key != L"--vertex" && key != L"--fragment" && key != L"--width" &&
                key != L"--height" && key != L"--frames" && key != L"--vertex-count" &&
                key != L"--readback" && key != L"--time" &&
                key != L"--compute" && key != L"--compute-spec")
                throw std::runtime_error("Unknown native DirectX viewer argument");
            if (++index == argc || !options.emplace(key, argv[index]).second)
                throw std::runtime_error("Missing or duplicate native viewer argument");
        }
        if (reportCapabilities) {
            reportDeviceCapabilities(warp, validation);
            return 0;
        }
        if (options.count(L"--compute") || options.count(L"--compute-spec")) {
            if (!options.count(L"--compute") || !options.count(L"--compute-spec"))
                throw std::runtime_error("Both --compute and --compute-spec are required for compute dispatch");
            if (options.count(L"--vertex") || options.count(L"--fragment") ||
                options.count(L"--width") || options.count(L"--height") ||
                options.count(L"--frames") || options.count(L"--vertex-count") ||
                options.count(L"--readback") || options.count(L"--time"))
                throw std::runtime_error("Compute dispatch arguments cannot be combined with graphics arguments");
            runComputeDispatch(options.at(L"--compute"), options.at(L"--compute-spec"), warp, validation);
            return 0;
        }
        if (!options.count(L"--vertex") || !options.count(L"--fragment"))
            throw std::runtime_error("--vertex and --fragment DXIL paths are required");
        UINT width = options.count(L"--width") ? number(options.at(L"--width"), 4096) : 800;
        UINT height = options.count(L"--height") ? number(options.at(L"--height"), 4096) : 600;
        UINT count = options.count(L"--vertex-count") ? number(options.at(L"--vertex-count"), UINT_MAX) : 3;
        UINT frames = options.count(L"--frames") ? number(options.at(L"--frames"), UINT_MAX) : 0;
        if (!width || !height || !count || count % 3)
            throw std::runtime_error("Dimensions must be positive; vertex count must be a positive multiple of three");
        if (options.count(L"--readback") && options.count(L"--frames"))
            throw std::runtime_error("--readback and --frames are mutually exclusive");

        bool hasTime = options.count(L"--time") != 0;
        double fixedTime = hasTime ? seconds(options.at(L"--time")) : 0.0;

        Renderer renderer(options.at(L"--vertex"), options.at(L"--fragment"), count, validation,
                          warp, hasTime, fixedTime);
        if (options.count(L"--readback")) {
            renderer.readback(width, height, options.at(L"--readback"));
        } else {
            HINSTANCE instance = GetModuleHandleW(nullptr);
            WNDCLASSW description{};
            description.lpfnWndProc = windowProcedure;
            description.hInstance = instance;
            description.lpszClassName = L"MLIRGPUDirectXPreview";
            description.hCursor = LoadCursorW(nullptr, IDC_ARROW);
            if (!RegisterClassW(&description)) throw std::runtime_error("Cannot register preview window class");
            RECT area{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
            if (!AdjustWindowRect(&area, WS_OVERLAPPEDWINDOW, FALSE))
                throw std::runtime_error("Cannot size preview window");
            Window window;
            window.handle = CreateWindowExW(0, description.lpszClassName, L"GfxIR - DirectX 12",
                WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, area.right - area.left,
                area.bottom - area.top, nullptr, nullptr, instance, nullptr);
            if (!window.handle) throw std::runtime_error("Cannot create DirectX preview window");
            ShowWindow(window.handle, SW_SHOW);
            renderer.window(window.handle, width, height, frames);
        }
    } catch (const std::exception& error) {
        std::cerr << "mlir-gpu-directx-view: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
