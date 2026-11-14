#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

static std::string hresultToString(HRESULT hr) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << static_cast<uint32_t>(hr);
    return out.str();
}

static std::string wideToUtf8(const wchar_t* wide) {
    if (!wide || !*wide) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) return {};
    std::string out(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), len, nullptr, nullptr);
    return out;
}

static void writeErrorBuffer(char* errBuf, size_t errBufLen, const std::string& msg) {
    if (!errBuf || errBufLen == 0) return;
    size_t copyLen = std::min(msg.size(), errBufLen - 1);
    if (copyLen > 0) {
        std::memcpy(errBuf, msg.data(), copyLen);
    }
    errBuf[copyLen] = '\0';
}

struct EventHandle {
    HANDLE handle = nullptr;

    EventHandle() = default;
    ~EventHandle() {
        if (handle) {
            CloseHandle(handle);
            handle = nullptr;
        }
    }
    EventHandle(const EventHandle&) = delete;
    EventHandle& operator=(const EventHandle&) = delete;

    bool init(std::string& err) {
        handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!handle) {
            err = "CreateEventW failed";
            return false;
        }
        return true;
    }
};

struct GfxDx12Device {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> diagnostics;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12Fence> fence;
    EventHandle fenceEvent;
    uint64_t fenceValue = 0;

    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    UINT descriptorHeapCapacity = 0;
    UINT descriptorIncrement = 0;

    std::string name;
    bool isSoftware = false;
    bool validationEnabled = false;
    std::string lastError;

    bool setError(const std::string& msg) {
        lastError = msg;
        return false;
    }

    bool checkHr(HRESULT hr, const char* what) {
        if (SUCCEEDED(hr)) return true;
        return setError(std::string(what) + " failed (HRESULT " + hresultToString(hr) + ")");
    }

    bool drainValidationQueue() {
        if (!diagnostics) return true;
        std::string errors;
        UINT64 count = diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter();
        for (UINT64 index = 0; index < count; ++index) {
            SIZE_T size = 0;
            if (FAILED(diagnostics->GetMessage(index, nullptr, &size)) || size == 0) {
                continue;
            }
            std::vector<char> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (SUCCEEDED(diagnostics->GetMessage(index, message, &size))) {
                if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR && message->pDescription) {
                    errors += message->pDescription;
                    errors += "\n";
                }
            }
        }
        diagnostics->ClearStoredMessages();
        if (!errors.empty()) {
            return setError("D3D12 validation failed:\n" + errors);
        }
        return true;
    }

    bool ensureDescriptorHeap(UINT requiredCount) {
        UINT needed = std::max<UINT>(1, requiredCount);
        if (descriptorHeap && descriptorHeapCapacity >= needed) {
            return true;
        }
        UINT newCapacity = std::max<UINT>(needed, std::max<UINT>(16, descriptorHeapCapacity * 2));
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = newCapacity;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> newHeap;
        if (!checkHr(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&newHeap)),
                     "CreateDescriptorHeap")) {
            return false;
        }
        descriptorHeap = std::move(newHeap);
        descriptorHeapCapacity = newCapacity;
        return true;
    }

    bool executeAndWait() {
        if (!checkHr(commandList->Close(), "Close command list")) {
            return false;
        }
        ID3D12CommandList* lists[] = {commandList.Get()};
        queue->ExecuteCommandLists(1, lists);

        ++fenceValue;
        if (!checkHr(queue->Signal(fence.Get(), fenceValue), "Signal fence")) {
            return false;
        }
        if (fence->GetCompletedValue() < fenceValue) {
            if (!checkHr(fence->SetEventOnCompletion(fenceValue, fenceEvent.handle),
                         "SetEventOnCompletion")) {
                return false;
            }
            DWORD waitStatus = WaitForSingleObject(fenceEvent.handle, 30000);
            if (waitStatus != WAIT_OBJECT_0) {
                HRESULT removed = device->GetDeviceRemovedReason();
                return setError("Compute GPU completion wait failed or timed out (device removed: " +
                                hresultToString(removed) + ")");
            }
        }
        HRESULT removed = device->GetDeviceRemovedReason();
        if (FAILED(removed)) {
            return setError("GPU device removed (HRESULT " + hresultToString(removed) + ")");
        }
        return drainValidationQueue();
    }

    bool resetCommandList(ID3D12PipelineState* initialPso) {
        if (!checkHr(allocator->Reset(), "Reset command allocator")) {
            return false;
        }
        if (!checkHr(commandList->Reset(allocator.Get(), initialPso), "Reset command list")) {
            return false;
        }
        return true;
    }
};

struct GfxDx12Buffer {
    GfxDx12Device* owner = nullptr;
    uint64_t sizeBytes = 0;
    uint64_t alignedSize = 0;
    D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON;
    ComPtr<ID3D12Resource> defaultBuf;
    ComPtr<ID3D12Resource> uploadBuf;
    ComPtr<ID3D12Resource> readbackBuf;

    bool ensureUploadBuffer() {
        if (uploadBuf) return true;
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = alignedSize;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        return owner->checkHr(
            owner->device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf)),
            "Create upload buffer");
    }

    bool ensureReadbackBuffer() {
        if (readbackBuf) return true;
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = alignedSize;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        return owner->checkHr(
            owner->device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuf)),
            "Create readback buffer");
    }
};

struct GfxDx12Pipeline {
    GfxDx12Device* owner = nullptr;
    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;
    bool hasRootConstants = false;
    UINT rootConstantsParamIndex = 0;
    UINT rootConstantsNumWords = 0;
    bool hasDescriptorTable = false;
    UINT descriptorTableParamIndex = 0;
};

}  // namespace

extern "C" {

struct GfxDx12BufferBinding {
    void* buffer;
    int is_uav;
};

__declspec(dllexport) void* gfx_dx12_create_device(
    int use_warp, int enable_validation, char* err_buf, size_t err_buf_len) {
    auto dev = std::make_unique<GfxDx12Device>();
    dev->validationEnabled = (enable_validation != 0);

    if (dev->validationEnabled) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
        }
    }

    ComPtr<IDXGIFactory4> factory4;
    HRESULT hr = CreateDXGIFactory2(
        dev->validationEnabled ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&factory4));
    if (FAILED(hr)) {
        hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory4));
    }
    if (FAILED(hr)) {
        writeErrorBuffer(err_buf, err_buf_len,
                         "CreateDXGIFactory failed (HRESULT " + hresultToString(hr) + ")");
        return nullptr;
    }

    ComPtr<IDXGIAdapter1> chosenAdapter;
    DXGI_ADAPTER_DESC1 chosenDesc{};

    if (use_warp != 0) {
        hr = factory4->EnumWarpAdapter(IID_PPV_ARGS(&chosenAdapter));
        if (FAILED(hr) || !chosenAdapter) {
            writeErrorBuffer(err_buf, err_buf_len,
                             "EnumWarpAdapter failed (HRESULT " + hresultToString(hr) + ")");
            return nullptr;
        }
        chosenAdapter->GetDesc1(&chosenDesc);
        if ((chosenDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            writeErrorBuffer(err_buf, err_buf_len,
                             "WARP requested but EnumWarpAdapter returned a non-software adapter");
            return nullptr;
        }
    } else {
        ComPtr<IDXGIFactory6> factory6;
        if (SUCCEEDED(factory4.As(&factory6))) {
            for (UINT idx = 0;; ++idx) {
                ComPtr<IDXGIAdapter1> candidate;
                if (factory6->EnumAdapterByGpuPreference(
                        idx, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                        IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND) {
                    break;
                }
                DXGI_ADAPTER_DESC1 desc{};
                candidate->GetDesc1(&desc);
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0,
                                                __uuidof(ID3D12Device), nullptr))) {
                    chosenAdapter = candidate;
                    chosenDesc = desc;
                    break;
                }
            }
        }
        if (!chosenAdapter) {
            for (UINT idx = 0;; ++idx) {
                ComPtr<IDXGIAdapter1> candidate;
                if (factory4->EnumAdapters1(idx, &candidate) == DXGI_ERROR_NOT_FOUND) {
                    break;
                }
                DXGI_ADAPTER_DESC1 desc{};
                candidate->GetDesc1(&desc);
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0,
                                                __uuidof(ID3D12Device), nullptr))) {
                    chosenAdapter = candidate;
                    chosenDesc = desc;
                    break;
                }
            }
        }
        if (!chosenAdapter) {
            writeErrorBuffer(err_buf, err_buf_len,
                             "No DirectX 12 hardware adapter found");
            return nullptr;
        }
    }

    hr = D3D12CreateDevice(chosenAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                           IID_PPV_ARGS(&dev->device));
    if (FAILED(hr)) {
        writeErrorBuffer(err_buf, err_buf_len,
                         "D3D12CreateDevice failed (HRESULT " + hresultToString(hr) + ")");
        return nullptr;
    }

    dev->name = wideToUtf8(chosenDesc.Description);
    if (dev->name.empty()) {
        dev->name = (use_warp != 0) ? "Microsoft Basic Render Driver" : "DirectX 12 Adapter";
    }
    dev->isSoftware = (chosenDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

    if (dev->validationEnabled) {
        if (!dev->checkHr(dev->device.As(&dev->diagnostics), "Get ID3D12InfoQueue")) {
            writeErrorBuffer(err_buf, err_buf_len, dev->lastError);
            return nullptr;
        }
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!dev->checkHr(dev->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&dev->queue)),
                      "CreateCommandQueue")) {
        writeErrorBuffer(err_buf, err_buf_len, dev->lastError);
        return nullptr;
    }

    if (!dev->checkHr(dev->device->CreateCommandAllocator(
                          D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dev->allocator)),
                      "CreateCommandAllocator")) {
        writeErrorBuffer(err_buf, err_buf_len, dev->lastError);
        return nullptr;
    }

    if (!dev->checkHr(dev->device->CreateCommandList(
                          0, D3D12_COMMAND_LIST_TYPE_DIRECT, dev->allocator.Get(), nullptr,
                          IID_PPV_ARGS(&dev->commandList)),
                      "CreateCommandList")) {
        writeErrorBuffer(err_buf, err_buf_len, dev->lastError);
        return nullptr;
    }
    if (!dev->checkHr(dev->commandList->Close(), "Close initial command list")) {
        writeErrorBuffer(err_buf, err_buf_len, dev->lastError);
        return nullptr;
    }

    if (!dev->checkHr(dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&dev->fence)),
                      "CreateFence")) {
        writeErrorBuffer(err_buf, err_buf_len, dev->lastError);
        return nullptr;
    }

    std::string eventErr;
    if (!dev->fenceEvent.init(eventErr)) {
        writeErrorBuffer(err_buf, err_buf_len, eventErr);
        return nullptr;
    }

    dev->descriptorIncrement =
        dev->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    return dev.release();
}

__declspec(dllexport) const char* gfx_dx12_device_name(void* dev_ptr) {
    if (!dev_ptr) return "";
    return static_cast<GfxDx12Device*>(dev_ptr)->name.c_str();
}

__declspec(dllexport) int gfx_dx12_device_is_software(void* dev_ptr) {
    if (!dev_ptr) return 0;
    return static_cast<GfxDx12Device*>(dev_ptr)->isSoftware ? 1 : 0;
}

__declspec(dllexport) const char* gfx_dx12_last_error(void* dev_ptr) {
    if (!dev_ptr) return "Null DirectX 12 device";
    return static_cast<GfxDx12Device*>(dev_ptr)->lastError.c_str();
}

__declspec(dllexport) void gfx_dx12_destroy_device(void* dev_ptr) {
    if (!dev_ptr) return;
    std::unique_ptr<GfxDx12Device> dev(static_cast<GfxDx12Device*>(dev_ptr));
    if (dev->queue && dev->fence && dev->fenceEvent.handle) {
        ++dev->fenceValue;
        if (SUCCEEDED(dev->queue->Signal(dev->fence.Get(), dev->fenceValue)) &&
            dev->fence->GetCompletedValue() < dev->fenceValue) {
            if (SUCCEEDED(dev->fence->SetEventOnCompletion(dev->fenceValue, dev->fenceEvent.handle))) {
                WaitForSingleObject(dev->fenceEvent.handle, 5000);
            }
        }
    }
}

__declspec(dllexport) void* gfx_dx12_create_buffer(void* dev_ptr, uint64_t size_bytes) {
    if (!dev_ptr) return nullptr;
    auto* dev = static_cast<GfxDx12Device*>(dev_ptr);
    if (size_bytes == 0) {
        dev->setError("Compute buffer byte size must be positive");
        return nullptr;
    }

    auto buf = std::make_unique<GfxDx12Buffer>();
    buf->owner = dev;
    buf->sizeBytes = size_bytes;
    buf->alignedSize = std::max<uint64_t>(4, (size_bytes + 3ULL) & ~3ULL);
    buf->currentState = D3D12_RESOURCE_STATE_COMMON;

    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC defaultDesc{};
    defaultDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    defaultDesc.Width = buf->alignedSize;
    defaultDesc.Height = 1;
    defaultDesc.DepthOrArraySize = 1;
    defaultDesc.MipLevels = 1;
    defaultDesc.SampleDesc.Count = 1;
    defaultDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    defaultDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (!dev->checkHr(
            dev->device->CreateCommittedResource(
                &defaultHeap, D3D12_HEAP_FLAG_NONE, &defaultDesc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf->defaultBuf)),
            "Create compute default buffer")) {
        return nullptr;
    }

    if (!dev->drainValidationQueue()) {
        return nullptr;
    }
    return buf.release();
}

__declspec(dllexport) uint64_t gfx_dx12_buffer_size(void* buf_ptr) {
    if (!buf_ptr) return 0;
    return static_cast<GfxDx12Buffer*>(buf_ptr)->sizeBytes;
}

__declspec(dllexport) int gfx_dx12_upload_buffer(
    void* dev_ptr, void* buf_ptr, const void* data, uint64_t size_bytes, uint64_t offset) {
    if (!dev_ptr) return -1;
    auto* dev = static_cast<GfxDx12Device*>(dev_ptr);
    if (!buf_ptr) {
        dev->setError("Null buffer handle passed to upload");
        return -1;
    }
    auto* buf = static_cast<GfxDx12Buffer*>(buf_ptr);
    if (buf->owner != dev) {
        dev->setError("Buffer does not belong to this DirectX 12 device");
        return -1;
    }
    if (offset > buf->sizeBytes || size_bytes > buf->sizeBytes - offset) {
        dev->setError("Upload byte range exceeds buffer capacity");
        return -1;
    }
    if (size_bytes == 0) {
        return 0;
    }
    if (!data) {
        dev->setError("Null data pointer passed to upload");
        return -1;
    }
    if (!buf->ensureUploadBuffer()) {
        return -1;
    }

    void* mapped = nullptr;
    D3D12_RANGE emptyReadRange{0, 0};
    if (!dev->checkHr(buf->uploadBuf->Map(0, &emptyReadRange, &mapped), "Map upload buffer")) {
        return -1;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size_bytes));
    buf->uploadBuf->Unmap(0, nullptr);

    if (!dev->resetCommandList(nullptr)) {
        return -1;
    }
    if (buf->currentState != D3D12_RESOURCE_STATE_COPY_DEST) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = buf->defaultBuf.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = buf->currentState;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        dev->commandList->ResourceBarrier(1, &barrier);
        buf->currentState = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    dev->commandList->CopyBufferRegion(
        buf->defaultBuf.Get(), offset, buf->uploadBuf.Get(), 0, size_bytes);

    return dev->executeAndWait() ? 0 : -1;
}

__declspec(dllexport) int gfx_dx12_download_buffer(
    void* dev_ptr, void* buf_ptr, void* out_data, uint64_t size_bytes, uint64_t offset) {
    if (!dev_ptr) return -1;
    auto* dev = static_cast<GfxDx12Device*>(dev_ptr);
    if (!buf_ptr) {
        dev->setError("Null buffer handle passed to download");
        return -1;
    }
    auto* buf = static_cast<GfxDx12Buffer*>(buf_ptr);
    if (buf->owner != dev) {
        dev->setError("Buffer does not belong to this DirectX 12 device");
        return -1;
    }
    if (offset > buf->sizeBytes || size_bytes > buf->sizeBytes - offset) {
        dev->setError("Download byte range exceeds buffer capacity");
        return -1;
    }
    if (size_bytes == 0) {
        return 0;
    }
    if (!out_data) {
        dev->setError("Null output pointer passed to download");
        return -1;
    }
    if (!buf->ensureReadbackBuffer()) {
        return -1;
    }

    if (!dev->resetCommandList(nullptr)) {
        return -1;
    }
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    if (buf->currentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        D3D12_RESOURCE_BARRIER uavBarrier{};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.UAV.pResource = buf->defaultBuf.Get();
        barriers.push_back(uavBarrier);
    }
    if (buf->currentState != D3D12_RESOURCE_STATE_COPY_SOURCE) {
        D3D12_RESOURCE_BARRIER toCopySource{};
        toCopySource.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toCopySource.Transition.pResource = buf->defaultBuf.Get();
        toCopySource.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        toCopySource.Transition.StateBefore = buf->currentState;
        toCopySource.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barriers.push_back(toCopySource);
        buf->currentState = D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if (!barriers.empty()) {
        dev->commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    }
    dev->commandList->CopyBufferRegion(
        buf->readbackBuf.Get(), 0, buf->defaultBuf.Get(), offset, size_bytes);

    if (!dev->executeAndWait()) {
        return -1;
    }

    void* mapped = nullptr;
    D3D12_RANGE readRange{0, static_cast<SIZE_T>(size_bytes)};
    if (!dev->checkHr(buf->readbackBuf->Map(0, &readRange, &mapped), "Map readback buffer")) {
        return -1;
    }
    std::memcpy(out_data, mapped, static_cast<size_t>(size_bytes));
    D3D12_RANGE emptyWritten{0, 0};
    buf->readbackBuf->Unmap(0, &emptyWritten);
    return 0;
}

__declspec(dllexport) void gfx_dx12_destroy_buffer(void* buf_ptr) {
    if (!buf_ptr) return;
    delete static_cast<GfxDx12Buffer*>(buf_ptr);
}

__declspec(dllexport) void* gfx_dx12_create_pipeline(
    void* dev_ptr, const void* dxil_bytes, size_t dxil_size) {
    if (!dev_ptr) return nullptr;
    auto* dev = static_cast<GfxDx12Device*>(dev_ptr);
    if (!dxil_bytes || dxil_size == 0) {
        dev->setError("DXIL shader bytecode must be non-empty");
        return nullptr;
    }

    auto pipe = std::make_unique<GfxDx12Pipeline>();
    pipe->owner = dev;

    ComPtr<ID3D12VersionedRootSignatureDeserializer> deserializer;
    if (SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(
            dxil_bytes, dxil_size, IID_PPV_ARGS(&deserializer)))) {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc =
            deserializer->GetUnconvertedRootSignatureDesc();
        if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
            for (UINT i = 0; i < desc->Desc_1_1.NumParameters; ++i) {
                const auto& param = desc->Desc_1_1.pParameters[i];
                if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    pipe->hasRootConstants = true;
                    pipe->rootConstantsParamIndex = i;
                    pipe->rootConstantsNumWords = param.Constants.Num32BitValues;
                } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
                    pipe->hasDescriptorTable = true;
                    pipe->descriptorTableParamIndex = i;
                }
            }
        } else {
            for (UINT i = 0; i < desc->Desc_1_0.NumParameters; ++i) {
                const auto& param = desc->Desc_1_0.pParameters[i];
                if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    pipe->hasRootConstants = true;
                    pipe->rootConstantsParamIndex = i;
                    pipe->rootConstantsNumWords = param.Constants.Num32BitValues;
                } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
                    pipe->hasDescriptorTable = true;
                    pipe->descriptorTableParamIndex = i;
                }
            }
        }
        HRESULT rsStatus = dev->device->CreateRootSignature(
            0, dxil_bytes, dxil_size, IID_PPV_ARGS(&pipe->rootSignature));
        if (FAILED(rsStatus) && dev->diagnostics) {
            dev->diagnostics->ClearStoredMessages();
        }
    }

    if (!pipe->rootSignature) {
        D3D12_ROOT_SIGNATURE_DESC emptyDesc{};
        ComPtr<ID3DBlob> serialized, errors;
        HRESULT serStatus = D3D12SerializeRootSignature(
            &emptyDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
        if (FAILED(serStatus)) {
            if (errors) {
                dev->setError(std::string(static_cast<const char*>(errors->GetBufferPointer()),
                                          errors->GetBufferSize()));
            } else {
                dev->checkHr(serStatus, "D3D12SerializeRootSignature");
            }
            return nullptr;
        }
        if (!dev->checkHr(
                dev->device->CreateRootSignature(
                    0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                    IID_PPV_ARGS(&pipe->rootSignature)),
                "CreateRootSignature")) {
            return nullptr;
        }
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = pipe->rootSignature.Get();
    psoDesc.CS = {dxil_bytes, dxil_size};
    HRESULT psoStatus = dev->device->CreateComputePipelineState(
        &psoDesc, IID_PPV_ARGS(&pipe->pipelineState));
    if (!dev->drainValidationQueue()) {
        return nullptr;
    }
    if (!dev->checkHr(psoStatus, "CreateComputePipelineState")) {
        return nullptr;
    }
    return pipe.release();
}

__declspec(dllexport) uint32_t gfx_dx12_pipeline_root_constants_bytes(void* pipe_ptr) {
    if (!pipe_ptr) return 0;
    auto* pipe = static_cast<GfxDx12Pipeline*>(pipe_ptr);
    return pipe->hasRootConstants ? pipe->rootConstantsNumWords * 4U : 0U;
}

__declspec(dllexport) void gfx_dx12_destroy_pipeline(void* pipe_ptr) {
    if (!pipe_ptr) return;
    delete static_cast<GfxDx12Pipeline*>(pipe_ptr);
}

__declspec(dllexport) int gfx_dx12_dispatch(
    void* dev_ptr,
    void* pipe_ptr,
    uint32_t gx,
    uint32_t gy,
    uint32_t gz,
    const void* root_constants,
    size_t root_constants_bytes,
    const GfxDx12BufferBinding* bindings,
    size_t num_bindings) {
    if (!dev_ptr) return -1;
    auto* dev = static_cast<GfxDx12Device*>(dev_ptr);
    if (!pipe_ptr) {
        dev->setError("Null pipeline handle passed to dispatch");
        return -1;
    }
    auto* pipe = static_cast<GfxDx12Pipeline*>(pipe_ptr);
    if (pipe->owner != dev) {
        dev->setError("Pipeline does not belong to this DirectX 12 device");
        return -1;
    }
    if (gx == 0 || gy == 0 || gz == 0 ||
        gx > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
        gy > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
        gz > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) {
        dev->setError("Compute dispatch grid dimension is out of D3D12 bounds");
        return -1;
    }

    std::vector<uint32_t> rootConstantWords;
    if (pipe->hasRootConstants) {
        size_t requiredBytes = static_cast<size_t>(pipe->rootConstantsNumWords) * sizeof(uint32_t);
        if (!root_constants || root_constants_bytes < requiredBytes) {
            dev->setError("Compute root signature requires root constants, but insufficient bytes were provided");
            return -1;
        }
        rootConstantWords.resize(pipe->rootConstantsNumWords);
        std::memcpy(rootConstantWords.data(), root_constants, requiredBytes);
    }

    if (num_bindings > 0 && !bindings) {
        dev->setError("Null bindings array passed to dispatch");
        return -1;
    }

    std::vector<size_t> uavIndices;
    std::vector<size_t> srvIndices;
    struct UniqueBufferState {
        GfxDx12Buffer* buf;
        D3D12_RESOURCE_STATES targetState;
    };
    std::vector<UniqueBufferState> uniqueBuffers;
    uniqueBuffers.reserve(num_bindings);

    for (size_t i = 0; i < num_bindings; ++i) {
        if (!bindings[i].buffer) {
            dev->setError("Null buffer handle in dispatch bindings");
            return -1;
        }
        auto* buf = static_cast<GfxDx12Buffer*>(bindings[i].buffer);
        if (buf->owner != dev) {
            dev->setError("Bound buffer does not belong to this DirectX 12 device");
            return -1;
        }
        bool isUav = (bindings[i].is_uav != 0);
        D3D12_RESOURCE_STATES targetState = isUav
            ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        auto existing = std::find_if(
            uniqueBuffers.begin(), uniqueBuffers.end(),
            [buf](const UniqueBufferState& entry) { return entry.buf == buf; });
        if (existing != uniqueBuffers.end()) {
            if (existing->targetState != targetState) {
                dev->setError("Cannot bind the same buffer as both read_only and writable in a single dispatch");
                return -1;
            }
        } else {
            uniqueBuffers.push_back({buf, targetState});
        }

        if (isUav) {
            uavIndices.push_back(i);
        } else {
            srvIndices.push_back(i);
        }
    }

    if (pipe->hasDescriptorTable || num_bindings > 0) {
        if (!dev->ensureDescriptorHeap(static_cast<UINT>(num_bindings))) {
            return -1;
        }
        D3D12_CPU_DESCRIPTOR_HANDLE handle = dev->descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        for (size_t idx : uavIndices) {
            auto* buf = static_cast<GfxDx12Buffer*>(bindings[idx].buffer);
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
            uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Buffer.FirstElement = 0;
            uavDesc.Buffer.NumElements = static_cast<UINT>(buf->alignedSize / 4);
            uavDesc.Buffer.StructureByteStride = 0;
            uavDesc.Buffer.CounterOffsetInBytes = 0;
            uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            dev->device->CreateUnorderedAccessView(buf->defaultBuf.Get(), nullptr, &uavDesc, handle);
            handle.ptr += dev->descriptorIncrement;
        }
        for (size_t idx : srvIndices) {
            auto* buf = static_cast<GfxDx12Buffer*>(bindings[idx].buffer);
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
            srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Buffer.FirstElement = 0;
            srvDesc.Buffer.NumElements = static_cast<UINT>(buf->alignedSize / 4);
            srvDesc.Buffer.StructureByteStride = 0;
            srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            dev->device->CreateShaderResourceView(buf->defaultBuf.Get(), &srvDesc, handle);
            handle.ptr += dev->descriptorIncrement;
        }
    }

    if (!dev->resetCommandList(pipe->pipelineState.Get())) {
        return -1;
    }

    std::vector<D3D12_RESOURCE_BARRIER> preBarriers;
    preBarriers.reserve(uniqueBuffers.size());
    for (const auto& entry : uniqueBuffers) {
        if (entry.buf->currentState != entry.targetState) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = entry.buf->defaultBuf.Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = entry.buf->currentState;
            b.Transition.StateAfter = entry.targetState;
            preBarriers.push_back(b);
            entry.buf->currentState = entry.targetState;
        } else if (entry.targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = entry.buf->defaultBuf.Get();
            preBarriers.push_back(uav);
        }
    }
    if (!preBarriers.empty()) {
        dev->commandList->ResourceBarrier(static_cast<UINT>(preBarriers.size()), preBarriers.data());
    }

    dev->commandList->SetComputeRootSignature(pipe->rootSignature.Get());
    if (dev->descriptorHeap && (pipe->hasDescriptorTable || num_bindings > 0)) {
        ID3D12DescriptorHeap* heaps[] = {dev->descriptorHeap.Get()};
        dev->commandList->SetDescriptorHeaps(1, heaps);
    }
    if (pipe->hasRootConstants) {
        dev->commandList->SetComputeRoot32BitConstants(
            pipe->rootConstantsParamIndex, pipe->rootConstantsNumWords,
            rootConstantWords.data(), 0);
    }
    if (pipe->hasDescriptorTable && dev->descriptorHeap) {
        dev->commandList->SetComputeRootDescriptorTable(
            pipe->descriptorTableParamIndex,
            dev->descriptorHeap->GetGPUDescriptorHandleForHeapStart());
    }

    dev->commandList->Dispatch(gx, gy, gz);

    std::vector<D3D12_RESOURCE_BARRIER> postBarriers;
    for (const auto& entry : uniqueBuffers) {
        if (entry.targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = entry.buf->defaultBuf.Get();
            postBarriers.push_back(uav);
        }
    }
    if (!postBarriers.empty()) {
        dev->commandList->ResourceBarrier(static_cast<UINT>(postBarriers.size()), postBarriers.data());
    }

    return dev->executeAndWait() ? 0 : -1;
}

}  // extern "C"
