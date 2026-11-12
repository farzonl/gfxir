#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;

constexpr UINT kSwapchainBufferCount = 2;
constexpr UINT kMaxSrvDescriptors = 4096;
constexpr UINT kMaxSamplerDescriptors = 512;

struct WinEvent {
    HANDLE handle = nullptr;

    WinEvent() : handle(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
    ~WinEvent() {
        if (handle != nullptr) {
            CloseHandle(handle);
        }
    }
    WinEvent(const WinEvent&) = delete;
    WinEvent& operator=(const WinEvent&) = delete;
    WinEvent(WinEvent&& other) noexcept : handle(std::exchange(other.handle, nullptr)) {}
    WinEvent& operator=(WinEvent&& other) noexcept {
        if (this != &other) {
            if (handle != nullptr) {
                CloseHandle(handle);
            }
            handle = std::exchange(other.handle, nullptr);
        }
        return *this;
    }
};

struct RootConstantSlot {
    UINT rootParamIndex = 0;
    UINT registerIndex = 0;
    UINT num32BitValues = 0;
};

struct MgfxBufferEntry {
    ComPtr<ID3D12Resource> uploadBuffer;
    std::byte* mappedPtr = nullptr;
    size_t capacity = 0;
    UINT srvSlot = UINT_MAX;
};

struct MgfxTextureSamplerEntry {
    ComPtr<ID3D12Resource> texture;
    UINT srvSlot = UINT_MAX;
    UINT samplerSlot = UINT_MAX;
};

struct MgfxPipelineEntry {
    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> state;
    bool depthEnabled = true;
    std::vector<RootConstantSlot> constantSlots;
    bool hasVsSrvTable = false;
    UINT vsSrvParamIndex = 0;
    bool hasPsSrvTable = false;
    UINT psSrvParamIndex = 0;
    bool hasPsSamplerTable = false;
    UINT psSamplerParamIndex = 0;
};

struct MgfxDirectXContext {
    HWND hwnd = nullptr;
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> infoQueue;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<IDXGISwapChain3> swapchain;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    ComPtr<ID3D12DescriptorHeap> srvHeap;
    ComPtr<ID3D12DescriptorHeap> samplerHeap;
    UINT rtvIncrement = 0;
    UINT srvIncrement = 0;
    UINT samplerIncrement = 0;
    UINT nextSrvSlot = 0;
    UINT nextSamplerSlot = 0;

    std::array<ComPtr<ID3D12Resource>, kSwapchainBufferCount> backBuffers;
    ComPtr<ID3D12Resource> depthTexture;
    int scWidth = 0;
    int scHeight = 0;
    UINT currentBackBuffer = 0;
    bool frameActive = false;
    bool currentDepthBound = false;
    int activePipelineId = -1;

    ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue = 0;
    WinEvent fenceEvent;

    std::vector<MgfxPipelineEntry> pipelines;
    std::unordered_map<int, MgfxBufferEntry> buffers;
    std::unordered_map<int, MgfxTextureSamplerEntry> textureSamplers;

    std::string deviceNameStr;
    std::string lastError;
};

std::wstring utf8ToWide(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    int needed = MultiByteToWideChar(
        CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (needed <= 0) {
        return {};
    }
    std::wstring result(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), needed);
    return result;
}

std::string wideToUtf8(const wchar_t* text) {
    if (text == nullptr || *text == L'\0') {
        return {};
    }
    int needed = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) {
        return {};
    }
    std::string result(static_cast<size_t>(needed - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), needed - 1, nullptr, nullptr);
    return result;
}

void setError(MgfxDirectXContext* ctx, std::string_view message, HRESULT hr = S_OK) {
    if (ctx == nullptr) {
        return;
    }
    ctx->lastError = std::string(message);
    if (FAILED(hr)) {
        ctx->lastError += " (HRESULT " + std::to_string(static_cast<int32_t>(hr)) + ")";
    }
}

bool collectValidationErrors(MgfxDirectXContext* ctx) {
    if (ctx == nullptr || !ctx->infoQueue) {
        return true;
    }
    std::string errors;
    UINT64 count = ctx->infoQueue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        if (FAILED(ctx->infoQueue->GetMessage(i, nullptr, &size)) || size == 0) {
            continue;
        }
        std::vector<std::byte> storage(size);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(ctx->infoQueue->GetMessage(i, msg, &size))) {
            if (msg->Severity <= D3D12_MESSAGE_SEVERITY_ERROR && msg->pDescription != nullptr) {
                errors += msg->pDescription;
                errors += '\n';
            }
        }
    }
    ctx->infoQueue->ClearStoredMessages();
    if (!errors.empty()) {
        ctx->lastError = "D3D12 validation error:\n" + errors;
        return false;
    }
    return true;
}

bool waitForGpu(MgfxDirectXContext* ctx) {
    if (ctx == nullptr || !ctx->queue || !ctx->fence || ctx->fenceEvent.handle == nullptr) {
        return false;
    }
    UINT64 signalVal = ++ctx->fenceValue;
    HRESULT hr = ctx->queue->Signal(ctx->fence.Get(), signalVal);
    if (FAILED(hr)) {
        setError(ctx, "Failed to signal D3D12 command queue fence", hr);
        return false;
    }
    if (ctx->fence->GetCompletedValue() < signalVal) {
        hr = ctx->fence->SetEventOnCompletion(signalVal, ctx->fenceEvent.handle);
        if (FAILED(hr)) {
            setError(ctx, "Failed to set D3D12 fence completion event", hr);
            return false;
        }
        DWORD waitRes = WaitForSingleObject(ctx->fenceEvent.handle, 30000);
        if (waitRes != WAIT_OBJECT_0) {
            HRESULT removed = ctx->device ? ctx->device->GetDeviceRemovedReason() : E_FAIL;
            setError(ctx, "Timed out waiting for DirectX 12 GPU fence", removed);
            return false;
        }
    }
    if (ctx->device) {
        HRESULT removed = ctx->device->GetDeviceRemovedReason();
        if (FAILED(removed)) {
            setError(ctx, "DirectX 12 GPU device was removed", removed);
            return false;
        }
    }
    return collectValidationErrors(ctx);
}

bool isSoftwareAdapter(const DXGI_ADAPTER_DESC1& desc) {
    return (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 || desc.VendorId == 0x1414;
}

bool createSwapchainTargets(MgfxDirectXContext* ctx, int width, int height) {
    int w = std::max(1, width);
    int h = std::max(1, height);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = ctx->rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kSwapchainBufferCount; ++i) {
        HRESULT hr = ctx->swapchain->GetBuffer(i, IID_PPV_ARGS(&ctx->backBuffers[i]));
        if (FAILED(hr)) {
            setError(ctx, "Failed to acquire DXGI swapchain backbuffer", hr);
            return false;
        }
        ctx->device->CreateRenderTargetView(ctx->backBuffers[i].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += ctx->rtvIncrement;
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC depthDesc{};
    depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDesc.Width = static_cast<UINT64>(w);
    depthDesc.Height = static_cast<UINT>(h);
    depthDesc.DepthOrArraySize = 1;
    depthDesc.MipLevels = 1;
    depthDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE clearVal{};
    clearVal.Format = DXGI_FORMAT_D32_FLOAT;
    clearVal.DepthStencil.Depth = 1.0f;
    clearVal.DepthStencil.Stencil = 0;

    ctx->depthTexture.Reset();
    HRESULT hr = ctx->device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &depthDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &clearVal,
        IID_PPV_ARGS(&ctx->depthTexture));
    if (FAILED(hr)) {
        setError(ctx, "Failed to allocate DirectX 12 depth texture", hr);
        return false;
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
    ctx->device->CreateDepthStencilView(
        ctx->depthTexture.Get(),
        &dsvDesc,
        ctx->dsvHeap->GetCPUDescriptorHandleForHeapStart());

    ctx->scWidth = w;
    ctx->scHeight = h;
    return true;
}

bool resizeSwapchain(MgfxDirectXContext* ctx, int width, int height) {
    if (!waitForGpu(ctx)) {
        return false;
    }
    for (auto& buf : ctx->backBuffers) {
        buf.Reset();
    }
    ctx->depthTexture.Reset();

    int w = std::max(1, width);
    int h = std::max(1, height);
    HRESULT hr = ctx->swapchain->ResizeBuffers(
        kSwapchainBufferCount,
        static_cast<UINT>(w),
        static_cast<UINT>(h),
        DXGI_FORMAT_R8G8B8A8_UNORM,
        0);
    if (FAILED(hr)) {
        setError(ctx, "DXGI SwapChain ResizeBuffers failed", hr);
        return false;
    }
    return createSwapchainTargets(ctx, w, h);
}

bool readBinaryFile(std::string_view utf8Path, std::vector<std::byte>& outBytes) {
    std::filesystem::path filePath(utf8ToWide(utf8Path));
    std::ifstream input(filePath, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    std::streamsize sz = input.tellg();
    if (sz < 32) {
        return false;
    }
    input.seekg(0, std::ios::beg);
    outBytes.resize(static_cast<size_t>(sz));
    if (!input.read(reinterpret_cast<char*>(outBytes.data()), sz)) {
        return false;
    }
    return std::memcmp(outBytes.data(), "DXBC", 4) == 0;
}

D3D12_TEXTURE_ADDRESS_MODE toD3D12AddressMode(int mode) {
    if (mode == 1) {
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
    if (mode == 2) {
        return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    }
    return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
}

D3D12_FILTER toD3D12Filter(int minLinear, int magLinear) {
    if (minLinear != 0 && magLinear != 0) {
        return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    }
    if (minLinear != 0 && magLinear == 0) {
        return D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
    }
    if (minLinear == 0 && magLinear != 0) {
        return D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
    }
    return D3D12_FILTER_MIN_MAG_MIP_POINT;
}

void recordTransitionBarrier(
    ID3D12GraphicsCommandList* cmd,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES stateBefore,
    D3D12_RESOURCE_STATES stateAfter
) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = stateBefore;
    barrier.Transition.StateAfter = stateAfter;
    cmd->ResourceBarrier(1, &barrier);
}

int finishFrameInternal(MgfxDirectXContext* ctx, std::span<std::byte> captureOut) {
    if (ctx == nullptr) {
        return -1;
    }
    if (!ctx->frameActive) {
        return 0;
    }
    ctx->frameActive = false;
    ctx->activePipelineId = -1;

    ID3D12Resource* backBuf = ctx->backBuffers[ctx->currentBackBuffer].Get();

    ComPtr<ID3D12Resource> readbackBuf;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 readbackTotalBytes = 0;
    const bool doCapture = !captureOut.empty();

    if (doCapture) {
        size_t expectedBytes = static_cast<size_t>(ctx->scWidth) * static_cast<size_t>(ctx->scHeight) * 4;
        if (captureOut.size() < expectedBytes) {
            setError(ctx, "Readback output buffer is smaller than framebuffer RGBA byte size");
            return -1;
        }
        D3D12_RESOURCE_DESC rtDesc = backBuf->GetDesc();
        ctx->device->GetCopyableFootprints(
            &rtDesc, 0, 1, 0, &footprint, nullptr, nullptr, &readbackTotalBytes);

        D3D12_HEAP_PROPERTIES rbHeap{};
        rbHeap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rbDesc{};
        rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rbDesc.Width = readbackTotalBytes;
        rbDesc.Height = 1;
        rbDesc.DepthOrArraySize = 1;
        rbDesc.MipLevels = 1;
        rbDesc.SampleDesc.Count = 1;
        rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        HRESULT hr = ctx->device->CreateCommittedResource(
            &rbHeap,
            D3D12_HEAP_FLAG_NONE,
            &rbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&readbackBuf));
        if (FAILED(hr)) {
            setError(ctx, "Failed to create DirectX 12 frame capture readback buffer", hr);
            return -1;
        }

        recordTransitionBarrier(
            ctx->cmdList.Get(), backBuf, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);

        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = backBuf;
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = readbackBuf.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.PlacedFootprint = footprint;

        ctx->cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        recordTransitionBarrier(
            ctx->cmdList.Get(), backBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    } else {
        recordTransitionBarrier(
            ctx->cmdList.Get(), backBuf, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    }

    HRESULT hr = ctx->cmdList->Close();
    if (FAILED(hr)) {
        setError(ctx, "Failed to close DirectX 12 graphics command list", hr);
        return -1;
    }

    ID3D12CommandList* lists[] = {ctx->cmdList.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);

    hr = ctx->swapchain->Present(1, 0);
    if (FAILED(hr)) {
        setError(ctx, "DXGI SwapChain Present failed", hr);
        return -1;
    }

    if (!waitForGpu(ctx)) {
        return -1;
    }

    if (doCapture && readbackBuf) {
        void* mapped = nullptr;
        D3D12_RANGE readRange{0, static_cast<SIZE_T>(readbackTotalBytes)};
        hr = readbackBuf->Map(0, &readRange, &mapped);
        if (FAILED(hr)) {
            setError(ctx, "Failed to map DirectX 12 frame capture readback buffer", hr);
            return -1;
        }
        const auto* srcBase = static_cast<const std::byte*>(mapped) + footprint.Offset;
        size_t rowBytes = static_cast<size_t>(ctx->scWidth) * 4;
        for (int row = 0; row < ctx->scHeight; ++row) {
            std::memcpy(
                captureOut.data() + static_cast<size_t>(row) * rowBytes,
                srcBase + static_cast<size_t>(row) * footprint.Footprint.RowPitch,
                rowBytes);
        }
        D3D12_RANGE writtenRange{0, 0};
        readbackBuf->Unmap(0, &writtenRange);
    }

    return 0;
}

} // namespace

extern "C" {

__declspec(dllexport) void* mgfx_dx12_create(
    void* win32_hwnd,
    int fb_width,
    int fb_height,
    int enable_validation,
    int use_warp,
    char* err_buf,
    size_t err_cap
) {
    auto writeErr = [&](std::string_view msg) {
        if (err_buf != nullptr && err_cap > 0) {
            size_t copyLen = std::min(msg.size(), err_cap - 1);
            std::memcpy(err_buf, msg.data(), copyLen);
            err_buf[copyLen] = '\0';
        }
    };

    if (win32_hwnd == nullptr) {
        writeErr("Null Win32 HWND handle from GLFW");
        return nullptr;
    }

    auto ctx = std::make_unique<MgfxDirectXContext>();
    ctx->hwnd = static_cast<HWND>(win32_hwnd);
    if (ctx->fenceEvent.handle == nullptr) {
        writeErr("Failed to create Win32 GPU completion event");
        return nullptr;
    }

    UINT factoryFlags = 0;
    if (enable_validation != 0) {
        ComPtr<ID3D12Debug1> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
        }
    }

    HRESULT hr = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&ctx->factory));
    if (FAILED(hr)) {
        writeErr("CreateDXGIFactory2 failed");
        return nullptr;
    }

    auto trySelectWarp = [&]() -> bool {
        ComPtr<IDXGIAdapter1> warpAdapter;
        if (FAILED(ctx->factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)))) {
            return false;
        }
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(warpAdapter->GetDesc1(&desc))) {
            return false;
        }
        ComPtr<ID3D12Device> dev;
        if (FAILED(D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
            return false;
        }
        ctx->adapter = std::move(warpAdapter);
        ctx->device = std::move(dev);
        ctx->deviceNameStr = wideToUtf8(desc.Description);
        return true;
    };

    bool created = false;
    if (use_warp != 0) {
        created = trySelectWarp();
    } else {
        for (UINT idx = 0;; ++idx) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT enumHr = ctx->factory->EnumAdapterByGpuPreference(
                idx, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (enumHr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (FAILED(enumHr)) {
                continue;
            }
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc)) || isSoftwareAdapter(desc)) {
                continue;
            }
            ComPtr<ID3D12Device> dev;
            if (FAILED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
                continue;
            }
            ctx->adapter = std::move(candidate);
            ctx->device = std::move(dev);
            ctx->deviceNameStr = wideToUtf8(desc.Description);
            created = true;
            break;
        }
        if (!created) {
            created = trySelectWarp();
        }
    }

    if (!created || !ctx->device) {
        writeErr("No DirectX 12 capable GPU or WARP adapter available");
        return nullptr;
    }

    if (ctx->deviceNameStr.empty()) {
        ctx->deviceNameStr = "DirectX 12";
    }

    if (enable_validation != 0) {
        ctx->device.As(&ctx->infoQueue);
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = ctx->device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&ctx->queue));
    if (FAILED(hr)) {
        writeErr("Failed to create DirectX 12 command queue");
        return nullptr;
    }

    hr = ctx->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx->allocator));
    if (FAILED(hr)) {
        writeErr("Failed to create DirectX 12 command allocator");
        return nullptr;
    }

    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        ctx->allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&ctx->cmdList));
    if (FAILED(hr)) {
        writeErr("Failed to create DirectX 12 graphics command list");
        return nullptr;
    }
    ctx->cmdList->Close();

    hr = ctx->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence));
    if (FAILED(hr)) {
        writeErr("Failed to create DirectX 12 fence");
        return nullptr;
    }

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.NumDescriptors = kSwapchainBufferCount;
    if (FAILED(ctx->device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&ctx->rtvHeap)))) {
        writeErr("Failed to create RTV descriptor heap");
        return nullptr;
    }
    ctx->rtvIncrement = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.NumDescriptors = 1;
    if (FAILED(ctx->device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&ctx->dsvHeap)))) {
        writeErr("Failed to create DSV descriptor heap");
        return nullptr;
    }

    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.NumDescriptors = kMaxSrvDescriptors;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(ctx->device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&ctx->srvHeap)))) {
        writeErr("Failed to create shader-visible CBV_SRV_UAV descriptor heap");
        return nullptr;
    }
    ctx->srvIncrement =
        ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_DESCRIPTOR_HEAP_DESC sampHeapDesc{};
    sampHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampHeapDesc.NumDescriptors = kMaxSamplerDescriptors;
    sampHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(ctx->device->CreateDescriptorHeap(&sampHeapDesc, IID_PPV_ARGS(&ctx->samplerHeap)))) {
        writeErr("Failed to create shader-visible SAMPLER descriptor heap");
        return nullptr;
    }
    ctx->samplerIncrement =
        ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    int initW = std::max(1, fb_width);
    int initH = std::max(1, fb_height);
    DXGI_SWAP_CHAIN_DESC1 scDesc{};
    scDesc.Width = static_cast<UINT>(initW);
    scDesc.Height = static_cast<UINT>(initH);
    scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scDesc.SampleDesc.Count = 1;
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = kSwapchainBufferCount;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    ComPtr<IDXGISwapChain1> sc1;
    hr = ctx->factory->CreateSwapChainForHwnd(
        ctx->queue.Get(), ctx->hwnd, &scDesc, nullptr, nullptr, &sc1);
    if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE) {
        writeErr(
            "DXGI presentation is unavailable in this session. Run in an interactive desktop "
            "session or pass --offscreen / --output.");
        return nullptr;
    }
    if (FAILED(hr) || FAILED(sc1.As(&ctx->swapchain))) {
        writeErr("Failed to create DXGI SwapChain for window");
        return nullptr;
    }

    ctx->factory->MakeWindowAssociation(ctx->hwnd, DXGI_MWA_NO_ALT_ENTER);

    UINT colorSpaceSupport = 0;
    if (SUCCEEDED(ctx->swapchain->CheckColorSpaceSupport(
            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, &colorSpaceSupport)) &&
        (colorSpaceSupport & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) != 0) {
        ctx->swapchain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
    }

    if (!createSwapchainTargets(ctx.get(), initW, initH)) {
        writeErr(ctx->lastError);
        return nullptr;
    }

    return ctx.release();
}

__declspec(dllexport) const char* mgfx_dx12_last_error(void* handle) {
    if (handle == nullptr) {
        return "Null DirectX 12 context";
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);
    return ctx->lastError.c_str();
}

__declspec(dllexport) const char* mgfx_dx12_device_name(void* handle) {
    if (handle == nullptr) {
        return "DirectX 12";
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);
    return ctx->deviceNameStr.c_str();
}

__declspec(dllexport) int mgfx_dx12_create_pipeline(
    void* handle,
    const char* vs_dxil_path,
    const char* ps_dxil_path,
    int depth_enabled,
    int alpha_blend
) {
    if (handle == nullptr || vs_dxil_path == nullptr || ps_dxil_path == nullptr) {
        return -1;
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);

    std::vector<std::byte> vsBytes;
    std::vector<std::byte> psBytes;
    if (!readBinaryFile(vs_dxil_path, vsBytes)) {
        setError(ctx, "Cannot load valid vertex DXIL container");
        return -1;
    }
    if (!readBinaryFile(ps_dxil_path, psBytes)) {
        setError(ctx, "Cannot load valid fragment DXIL container");
        return -1;
    }

    struct ExtractedTable {
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL;
        bool isVertexStage = false;
        bool isSampler = false;
        std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
    };

    std::map<std::pair<UINT, UINT>, D3D12_ROOT_CONSTANTS> mergedConstants;
    std::vector<ExtractedTable> extractedTables;

    auto inspectStage = [&](std::span<const std::byte> bytecode, bool isVertexStage) {
        ComPtr<ID3D12VersionedRootSignatureDeserializer> deserializer;
        if (FAILED(D3D12CreateVersionedRootSignatureDeserializer(
                bytecode.data(), bytecode.size(), IID_PPV_ARGS(&deserializer)))) {
            if (ctx->infoQueue) {
                ctx->infoQueue->ClearStoredMessages();
            }
            return;
        }
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc =
            deserializer->GetUnconvertedRootSignatureDesc();
        D3D12_SHADER_VISIBILITY stageVis =
            isVertexStage ? D3D12_SHADER_VISIBILITY_VERTEX : D3D12_SHADER_VISIBILITY_PIXEL;

        if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
            for (UINT i = 0; i < desc->Desc_1_1.NumParameters; ++i) {
                const auto& param = desc->Desc_1_1.pParameters[i];
                if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    auto key = std::make_pair(
                        param.Constants.RegisterSpace, param.Constants.ShaderRegister);
                    auto it = mergedConstants.find(key);
                    if (it == mergedConstants.end() ||
                        param.Constants.Num32BitValues > it->second.Num32BitValues) {
                        mergedConstants[key] = param.Constants;
                    }
                } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
                           param.DescriptorTable.NumDescriptorRanges > 0) {
                    ExtractedTable tbl;
                    tbl.visibility = stageVis;
                    tbl.isVertexStage = isVertexStage;
                    tbl.isSampler = (param.DescriptorTable.pDescriptorRanges[0].RangeType ==
                                     D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER);
                    tbl.ranges.assign(
                        param.DescriptorTable.pDescriptorRanges,
                        param.DescriptorTable.pDescriptorRanges +
                            param.DescriptorTable.NumDescriptorRanges);
                    extractedTables.push_back(std::move(tbl));
                }
            }
        } else {
            for (UINT i = 0; i < desc->Desc_1_0.NumParameters; ++i) {
                const auto& param = desc->Desc_1_0.pParameters[i];
                if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    auto key = std::make_pair(
                        param.Constants.RegisterSpace, param.Constants.ShaderRegister);
                    auto it = mergedConstants.find(key);
                    if (it == mergedConstants.end() ||
                        param.Constants.Num32BitValues > it->second.Num32BitValues) {
                        mergedConstants[key] = param.Constants;
                    }
                } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
                           param.DescriptorTable.NumDescriptorRanges > 0) {
                    ExtractedTable tbl;
                    tbl.visibility = stageVis;
                    tbl.isVertexStage = isVertexStage;
                    tbl.isSampler = (param.DescriptorTable.pDescriptorRanges[0].RangeType ==
                                     D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER);
                    tbl.ranges.reserve(param.DescriptorTable.NumDescriptorRanges);
                    for (UINT r = 0; r < param.DescriptorTable.NumDescriptorRanges; ++r) {
                        const auto& r0 = param.DescriptorTable.pDescriptorRanges[r];
                        D3D12_DESCRIPTOR_RANGE1 r1{};
                        r1.RangeType = r0.RangeType;
                        r1.NumDescriptors = r0.NumDescriptors;
                        r1.BaseShaderRegister = r0.BaseShaderRegister;
                        r1.RegisterSpace = r0.RegisterSpace;
                        r1.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
                        r1.OffsetInDescriptorsFromTableStart = r0.OffsetInDescriptorsFromTableStart;
                        tbl.ranges.push_back(r1);
                    }
                    extractedTables.push_back(std::move(tbl));
                }
            }
        }
    };

    inspectStage(vsBytes, true);
    inspectStage(psBytes, false);

    MgfxPipelineEntry entry;
    entry.depthEnabled = (depth_enabled != 0);

    std::vector<D3D12_ROOT_PARAMETER1> rootParams;
    rootParams.reserve(mergedConstants.size() + extractedTables.size());

    for (const auto& [key, constants] : mergedConstants) {
        UINT paramIdx = static_cast<UINT>(rootParams.size());
        if (constants.RegisterSpace == 0 && constants.ShaderRegister < 4) {
            entry.constantSlots.push_back(
                RootConstantSlot{paramIdx, constants.ShaderRegister, constants.Num32BitValues});
        }
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p.Constants = constants;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams.push_back(p);
    }

    for (const auto& tbl : extractedTables) {
        UINT paramIdx = static_cast<UINT>(rootParams.size());
        if (tbl.isVertexStage && !tbl.isSampler) {
            entry.hasVsSrvTable = true;
            entry.vsSrvParamIndex = paramIdx;
        } else if (!tbl.isVertexStage) {
            if (tbl.isSampler) {
                entry.hasPsSamplerTable = true;
                entry.psSamplerParamIndex = paramIdx;
            } else {
                entry.hasPsSrvTable = true;
                entry.psSrvParamIndex = paramIdx;
            }
        }
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(tbl.ranges.size());
        p.DescriptorTable.pDescriptorRanges = tbl.ranges.data();
        p.ShaderVisibility = tbl.visibility;
        rootParams.push_back(p);
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rsDesc.Desc_1_1.NumParameters = static_cast<UINT>(rootParams.size());
    rsDesc.Desc_1_1.pParameters = rootParams.empty() ? nullptr : rootParams.data();
    rsDesc.Desc_1_1.NumStaticSamplers = 0;
    rsDesc.Desc_1_1.pStaticSamplers = nullptr;
    rsDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serializedRs;
    ComPtr<ID3DBlob> rsErrors;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rsDesc, &serializedRs, &rsErrors);
    if (FAILED(hr)) {
        if (rsErrors && rsErrors->GetBufferSize() > 0) {
            std::string_view errText(
                static_cast<const char*>(rsErrors->GetBufferPointer()),
                rsErrors->GetBufferSize());
            setError(ctx, errText, hr);
        } else {
            setError(ctx, "D3D12SerializeVersionedRootSignature failed", hr);
        }
        return -1;
    }

    hr = ctx->device->CreateRootSignature(
        0,
        serializedRs->GetBufferPointer(),
        serializedRs->GetBufferSize(),
        IID_PPV_ARGS(&entry.rootSignature));
    if (FAILED(hr)) {
        setError(ctx, "CreateRootSignature failed", hr);
        return -1;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = entry.rootSignature.Get();
    psoDesc.VS = {vsBytes.data(), vsBytes.size()};
    psoDesc.PS = {psBytes.data(), psBytes.size()};

    auto& blend = psoDesc.BlendState.RenderTarget[0];
    if (alpha_blend != 0) {
        blend.BlendEnable = TRUE;
        blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        blend.BlendOp = D3D12_BLEND_OP_ADD;
        blend.SrcBlendAlpha = D3D12_BLEND_ONE;
        blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    } else {
        blend.BlendEnable = FALSE;
        blend.SrcBlend = D3D12_BLEND_ONE;
        blend.DestBlend = D3D12_BLEND_ZERO;
        blend.BlendOp = D3D12_BLEND_OP_ADD;
        blend.SrcBlendAlpha = D3D12_BLEND_ONE;
        blend.DestBlendAlpha = D3D12_BLEND_ZERO;
        blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    }
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;

    if (entry.depthEnabled) {
        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    } else {
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    }
    psoDesc.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    psoDesc.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    auto& front = psoDesc.DepthStencilState.FrontFace;
    front.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    front.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    front.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    front.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    psoDesc.DepthStencilState.BackFace = front;

    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;

    hr = ctx->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&entry.state));
    if (!collectValidationErrors(ctx) || FAILED(hr)) {
        if (ctx->lastError.empty()) {
            setError(ctx, "CreateGraphicsPipelineState failed", hr);
        }
        return -1;
    }

    int idOut = static_cast<int>(ctx->pipelines.size());
    ctx->pipelines.push_back(std::move(entry));
    return idOut;
}

__declspec(dllexport) int mgfx_dx12_upload_buffer(
    void* handle,
    int buf_id,
    const void* data,
    size_t byte_size
) {
    if (handle == nullptr || data == nullptr || byte_size == 0) {
        return -1;
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);
    auto& slot = ctx->buffers[buf_id];

    size_t aligned = (std::max<size_t>(byte_size, 64) + 15) & ~static_cast<size_t>(15);
    if (!slot.uploadBuffer || aligned > slot.capacity) {
        if (slot.srvSlot == UINT_MAX) {
            if (ctx->nextSrvSlot >= kMaxSrvDescriptors) {
                setError(ctx, "DirectX 12 SRV descriptor heap capacity exceeded");
                return -1;
            }
            slot.srvSlot = ctx->nextSrvSlot++;
        }

        if (slot.uploadBuffer && slot.mappedPtr != nullptr) {
            slot.uploadBuffer->Unmap(0, nullptr);
            slot.mappedPtr = nullptr;
        }

        size_t allocSize = std::max(aligned, slot.capacity * 2);
        D3D12_HEAP_PROPERTIES uploadHeap{};
        uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC bufDesc{};
        bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufDesc.Width = static_cast<UINT64>(allocSize);
        bufDesc.Height = 1;
        bufDesc.DepthOrArraySize = 1;
        bufDesc.MipLevels = 1;
        bufDesc.SampleDesc.Count = 1;
        bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        ComPtr<ID3D12Resource> newBuf;
        HRESULT hr = ctx->device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &bufDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&newBuf));
        if (FAILED(hr)) {
            setError(ctx, "Failed to allocate DirectX 12 storage buffer", hr);
            return -1;
        }

        void* rawMapped = nullptr;
        D3D12_RANGE emptyRead{0, 0};
        hr = newBuf->Map(0, &emptyRead, &rawMapped);
        if (FAILED(hr) || rawMapped == nullptr) {
            setError(ctx, "Failed to map DirectX 12 storage buffer", hr);
            return -1;
        }

        slot.uploadBuffer = std::move(newBuf);
        slot.mappedPtr = static_cast<std::byte*>(rawMapped);
        slot.capacity = allocSize;
        std::memset(slot.mappedPtr, 0, allocSize);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = static_cast<UINT>(allocSize / 4);
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = ctx->srvHeap->GetCPUDescriptorHandleForHeapStart();
        cpuHandle.ptr += static_cast<SIZE_T>(slot.srvSlot) * ctx->srvIncrement;
        ctx->device->CreateShaderResourceView(slot.uploadBuffer.Get(), &srvDesc, cpuHandle);
    }

    std::memcpy(slot.mappedPtr, data, byte_size);
    return 0;
}

__declspec(dllexport) int mgfx_dx12_upload_texture_sampler(
    void* handle,
    int tex_id,
    int width,
    int height,
    int is_float32,
    const void* pixels,
    size_t byte_size,
    int min_linear,
    int mag_linear,
    int addr_u,
    int addr_v
) {
    if (handle == nullptr || pixels == nullptr || width <= 0 || height <= 0) {
        return -1;
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);

    size_t w = static_cast<size_t>(width);
    size_t h = static_cast<size_t>(height);
    size_t bpp = (is_float32 != 0) ? 16 : 4;
    size_t expected = w * h * bpp;
    if (byte_size != expected) {
        setError(ctx, "Texture payload byte size does not match dimensions");
        return -1;
    }

    auto& slot = ctx->textureSamplers[tex_id];
    if (slot.srvSlot == UINT_MAX) {
        if (ctx->nextSrvSlot >= kMaxSrvDescriptors) {
            setError(ctx, "DirectX 12 SRV descriptor heap capacity exceeded");
            return -1;
        }
        slot.srvSlot = ctx->nextSrvSlot++;
    }
    if (slot.samplerSlot == UINT_MAX) {
        if (ctx->nextSamplerSlot >= kMaxSamplerDescriptors) {
            setError(ctx, "DirectX 12 Sampler descriptor heap capacity exceeded");
            return -1;
        }
        slot.samplerSlot = ctx->nextSamplerSlot++;
    }

    DXGI_FORMAT fmt = (is_float32 != 0) ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC texDesc{};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = static_cast<UINT64>(w);
    texDesc.Height = static_cast<UINT>(h);
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = fmt;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    ComPtr<ID3D12Resource> tex;
    HRESULT hr = ctx->device->CreateCommittedResource(
        &defaultHeap,
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&tex));
    if (FAILED(hr)) {
        setError(ctx, "Failed to allocate DirectX 12 2D texture", hr);
        return -1;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT numRows = 0;
    UINT64 rowSizeInBytes = 0;
    UINT64 totalBytes = 0;
    ctx->device->GetCopyableFootprints(
        &texDesc, 0, 1, 0, &footprint, &numRows, &rowSizeInBytes, &totalBytes);

    D3D12_HEAP_PROPERTIES uploadHeap{};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC stagingDesc{};
    stagingDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    stagingDesc.Width = totalBytes;
    stagingDesc.Height = 1;
    stagingDesc.DepthOrArraySize = 1;
    stagingDesc.MipLevels = 1;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> staging;
    hr = ctx->device->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &stagingDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&staging));
    if (FAILED(hr)) {
        setError(ctx, "Failed to allocate DirectX 12 texture upload buffer", hr);
        return -1;
    }

    void* mapped = nullptr;
    D3D12_RANGE emptyRead{0, 0};
    hr = staging->Map(0, &emptyRead, &mapped);
    if (FAILED(hr) || mapped == nullptr) {
        setError(ctx, "Failed to map DirectX 12 texture upload buffer", hr);
        return -1;
    }
    std::memset(mapped, 0, static_cast<size_t>(totalBytes));
    auto* dstBase = static_cast<std::byte*>(mapped) + footprint.Offset;
    const auto* srcBase = static_cast<const std::byte*>(pixels);
    size_t srcRowBytes = w * bpp;
    for (size_t row = 0; row < h; ++row) {
        std::memcpy(
            dstBase + row * footprint.Footprint.RowPitch,
            srcBase + row * srcRowBytes,
            srcRowBytes);
    }
    staging->Unmap(0, nullptr);

    ComPtr<ID3D12CommandAllocator> uploadAlloc;
    hr = ctx->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadAlloc));
    if (FAILED(hr)) {
        setError(ctx, "Failed to create command allocator for texture upload", hr);
        return -1;
    }
    ComPtr<ID3D12GraphicsCommandList> uploadCmd;
    hr = ctx->device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, uploadAlloc.Get(), nullptr, IID_PPV_ARGS(&uploadCmd));
    if (FAILED(hr)) {
        setError(ctx, "Failed to create command list for texture upload", hr);
        return -1;
    }

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = tex.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = staging.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = footprint;

    uploadCmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    recordTransitionBarrier(
        uploadCmd.Get(),
        tex.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    hr = uploadCmd->Close();
    if (FAILED(hr)) {
        setError(ctx, "Failed to close texture upload command list", hr);
        return -1;
    }
    ID3D12CommandList* lists[] = {uploadCmd.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    if (!waitForGpu(ctx)) {
        return -1;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = fmt;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.PlaneSlice = 0;
    srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu = ctx->srvHeap->GetCPUDescriptorHandleForHeapStart();
    srvCpu.ptr += static_cast<SIZE_T>(slot.srvSlot) * ctx->srvIncrement;
    ctx->device->CreateShaderResourceView(tex.Get(), &srvDesc, srvCpu);

    D3D12_SAMPLER_DESC sampDesc{};
    sampDesc.Filter = toD3D12Filter(min_linear, mag_linear);
    sampDesc.AddressU = toD3D12AddressMode(addr_u);
    sampDesc.AddressV = toD3D12AddressMode(addr_v);
    sampDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampDesc.MipLODBias = 0.0f;
    sampDesc.MaxAnisotropy = 1;
    sampDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampDesc.MinLOD = 0.0f;
    sampDesc.MaxLOD = 0.0f;

    D3D12_CPU_DESCRIPTOR_HANDLE sampCpu = ctx->samplerHeap->GetCPUDescriptorHandleForHeapStart();
    sampCpu.ptr += static_cast<SIZE_T>(slot.samplerSlot) * ctx->samplerIncrement;
    ctx->device->CreateSampler(&sampDesc, sampCpu);

    slot.texture = std::move(tex);
    return 0;
}

__declspec(dllexport) int mgfx_dx12_begin_frame(
    void* handle,
    int fb_width,
    int fb_height,
    float clear_r,
    float clear_g,
    float clear_b,
    float clear_a
) {
    if (handle == nullptr) {
        return -1;
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);
    if (fb_width <= 0 || fb_height <= 0 || (ctx->hwnd != nullptr && IsIconic(ctx->hwnd) != 0)) {
        return 1; // Minimized; skip frame gracefully
    }

    if (fb_width != ctx->scWidth || fb_height != ctx->scHeight) {
        if (!resizeSwapchain(ctx, fb_width, fb_height)) {
            return -1;
        }
    }

    ctx->currentBackBuffer = ctx->swapchain->GetCurrentBackBufferIndex();
    HRESULT hr = ctx->allocator->Reset();
    if (FAILED(hr)) {
        setError(ctx, "Failed to reset DirectX 12 command allocator", hr);
        return -1;
    }
    hr = ctx->cmdList->Reset(ctx->allocator.Get(), nullptr);
    if (FAILED(hr)) {
        setError(ctx, "Failed to reset DirectX 12 graphics command list", hr);
        return -1;
    }

    ID3D12Resource* backBuf = ctx->backBuffers[ctx->currentBackBuffer].Get();
    recordTransitionBarrier(
        ctx->cmdList.Get(), backBuf, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = ctx->rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += static_cast<SIZE_T>(ctx->currentBackBuffer) * ctx->rtvIncrement;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = ctx->dsvHeap->GetCPUDescriptorHandleForHeapStart();

    const float clearColor[4] = {clear_r, clear_g, clear_b, clear_a};
    ctx->cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    ctx->cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    ctx->cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    ctx->currentDepthBound = true;
    ctx->activePipelineId = -1;

    D3D12_VIEWPORT viewport{
        0.0f,
        0.0f,
        static_cast<float>(ctx->scWidth),
        static_cast<float>(ctx->scHeight),
        0.0f,
        1.0f};
    D3D12_RECT scissor{0, 0, static_cast<LONG>(ctx->scWidth), static_cast<LONG>(ctx->scHeight)};
    ctx->cmdList->RSSetViewports(1, &viewport);
    ctx->cmdList->RSSetScissorRects(1, &scissor);
    ctx->cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    ID3D12DescriptorHeap* heaps[] = {ctx->srvHeap.Get(), ctx->samplerHeap.Get()};
    ctx->cmdList->SetDescriptorHeaps(2, heaps);

    ctx->frameActive = true;
    return 0;
}

__declspec(dllexport) int mgfx_dx12_draw(
    void* handle,
    int pipeline_id,
    int buf_id,
    uint32_t vertex_count,
    const void* b0_data,
    size_t b0_size,
    const void* b1_data,
    size_t b1_size,
    const void* b2_data,
    size_t b2_size,
    const void* b3_data,
    size_t b3_size,
    int tex_id
) {
    if (handle == nullptr || vertex_count == 0) {
        return 0;
    }
    auto* ctx = static_cast<MgfxDirectXContext*>(handle);
    if (!ctx->frameActive) {
        return -1;
    }
    if (pipeline_id < 0 || static_cast<size_t>(pipeline_id) >= ctx->pipelines.size()) {
        return -1;
    }
    auto bufIt = ctx->buffers.find(buf_id);
    if (bufIt == ctx->buffers.end() || !bufIt->second.uploadBuffer) {
        return -1;
    }

    const auto& pipe = ctx->pipelines[static_cast<size_t>(pipeline_id)];
    const auto& buf = bufIt->second;
    ID3D12GraphicsCommandList* cmd = ctx->cmdList.Get();

    if (pipe.depthEnabled != ctx->currentDepthBound) {
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = ctx->rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtvHandle.ptr += static_cast<SIZE_T>(ctx->currentBackBuffer) * ctx->rtvIncrement;
        if (pipe.depthEnabled) {
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = ctx->dsvHeap->GetCPUDescriptorHandleForHeapStart();
            cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
        } else {
            cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        }
        ctx->currentDepthBound = pipe.depthEnabled;
    }

    if (ctx->activePipelineId != pipeline_id) {
        cmd->SetGraphicsRootSignature(pipe.rootSignature.Get());
        cmd->SetPipelineState(pipe.state.Get());
        ctx->activePipelineId = pipeline_id;
    }

    const std::array<std::span<const std::byte>, 4> blockSpans = {
        b0_data != nullptr ? std::span<const std::byte>(static_cast<const std::byte*>(b0_data), b0_size)
                           : std::span<const std::byte>{},
        b1_data != nullptr ? std::span<const std::byte>(static_cast<const std::byte*>(b1_data), b1_size)
                           : std::span<const std::byte>{},
        b2_data != nullptr ? std::span<const std::byte>(static_cast<const std::byte*>(b2_data), b2_size)
                           : std::span<const std::byte>{},
        b3_data != nullptr ? std::span<const std::byte>(static_cast<const std::byte*>(b3_data), b3_size)
                           : std::span<const std::byte>{},
    };

    for (const auto& cSlot : pipe.constantSlots) {
        if (cSlot.registerIndex >= blockSpans.size() || cSlot.num32BitValues == 0 ||
            cSlot.num32BitValues > 64) {
            continue;
        }
        std::array<uint32_t, 64> words{};
        const auto& span = blockSpans[cSlot.registerIndex];
        size_t copyBytes = std::min(span.size(), static_cast<size_t>(cSlot.num32BitValues) * 4);
        if (copyBytes > 0) {
            std::memcpy(words.data(), span.data(), copyBytes);
        }
        cmd->SetGraphicsRoot32BitConstants(
            cSlot.rootParamIndex, cSlot.num32BitValues, words.data(), 0);
    }

    if (pipe.hasVsSrvTable) {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuSrv = ctx->srvHeap->GetGPUDescriptorHandleForHeapStart();
        gpuSrv.ptr += static_cast<UINT64>(buf.srvSlot) * ctx->srvIncrement;
        cmd->SetGraphicsRootDescriptorTable(pipe.vsSrvParamIndex, gpuSrv);
    }

    if (pipe.hasPsSrvTable || pipe.hasPsSamplerTable) {
        if (tex_id < 0) {
            setError(ctx, "Textured DirectX 12 pipeline requires a valid texture/sampler ID");
            return -1;
        }
        auto texIt = ctx->textureSamplers.find(tex_id);
        if (texIt == ctx->textureSamplers.end() || !texIt->second.texture) {
            setError(ctx, "Invalid DirectX 12 texture/sampler ID");
            return -1;
        }
        const auto& ts = texIt->second;
        if (pipe.hasPsSrvTable) {
            D3D12_GPU_DESCRIPTOR_HANDLE gpuSrv = ctx->srvHeap->GetGPUDescriptorHandleForHeapStart();
            gpuSrv.ptr += static_cast<UINT64>(ts.srvSlot) * ctx->srvIncrement;
            cmd->SetGraphicsRootDescriptorTable(pipe.psSrvParamIndex, gpuSrv);
        }
        if (pipe.hasPsSamplerTable) {
            D3D12_GPU_DESCRIPTOR_HANDLE gpuSamp =
                ctx->samplerHeap->GetGPUDescriptorHandleForHeapStart();
            gpuSamp.ptr += static_cast<UINT64>(ts.samplerSlot) * ctx->samplerIncrement;
            cmd->SetGraphicsRootDescriptorTable(pipe.psSamplerParamIndex, gpuSamp);
        }
    }

    cmd->DrawInstanced(vertex_count, 1, 0, 0);
    return 0;
}

__declspec(dllexport) int mgfx_dx12_end_frame(void* handle) {
    return finishFrameInternal(static_cast<MgfxDirectXContext*>(handle), {});
}

__declspec(dllexport) int mgfx_dx12_end_frame_capture(
    void* handle,
    void* out_rgba,
    size_t byte_size
) {
    if (out_rgba == nullptr || byte_size == 0) {
        return -1;
    }
    std::span<std::byte> outSpan(static_cast<std::byte*>(out_rgba), byte_size);
    return finishFrameInternal(static_cast<MgfxDirectXContext*>(handle), outSpan);
}

__declspec(dllexport) void mgfx_dx12_destroy(void* handle) {
    if (handle == nullptr) {
        return;
    }
    std::unique_ptr<MgfxDirectXContext> ctx(static_cast<MgfxDirectXContext*>(handle));
    waitForGpu(ctx.get());
    for (auto& [id, slot] : ctx->buffers) {
        if (slot.uploadBuffer && slot.mappedPtr != nullptr) {
            slot.uploadBuffer->Unmap(0, nullptr);
            slot.mappedPtr = nullptr;
        }
    }
}

} // extern "C"
