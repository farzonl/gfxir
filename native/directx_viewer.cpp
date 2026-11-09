#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
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

struct ReadbackReport {
    std::string device;
    UINT width;
    UINT height;
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Event() {
        if (!handle) throw std::runtime_error("Cannot create GPU completion event");
    }
    ~Event() { CloseHandle(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
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
        commands->DrawInstanced(vertexCount, 1, 0, 0);
    }

    void submit() {
        check(commands->Close(), "Close command list");
        ID3D12CommandList* lists[]{commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        wait();
    }

public:
    Renderer(const std::wstring& vertex, const std::wstring& fragment, UINT count, bool validation,
             bool hasTime = false, double time = 0.0)
        : vertexCount(count), hasFixedTime(hasTime), fixedTime(time),
          started(std::chrono::steady_clock::now()) {
        if (validation) {
            ComPtr<ID3D12Debug1> debug;
            check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),
                  "D3D12 debug layer (install Windows Graphics Tools)");
            debug->EnableDebugLayer();
            debug->SetEnableGPUBasedValidation(TRUE);
        }
        check(CreateDXGIFactory2(validation ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&factory)),
              "CreateDXGIFactory2");
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            HRESULT status = factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
            if (status == DXGI_ERROR_NOT_FOUND) break;
            check(status, "EnumAdapterByGpuPreference");
            DXGI_ADAPTER_DESC1 description{};
            check(adapter->GetDesc1(&description), "GetDesc1");
            if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || description.VendorId == 0x1414) continue;
            status = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
            if (status == DXGI_ERROR_UNSUPPORTED || status == E_INVALIDARG) continue;
            check(status, "D3D12CreateDevice");
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
            check(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model)),
                  "Check Shader Model support");
            if (model.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
                device.Reset();
                continue;
            }
            name = deviceName(description.Description);
            break;
        }
        if (!device) throw std::runtime_error("No DirectX 12 GPU supporting Shader Model 6.6 is available");
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
        for (int index = 1; index < argc; ++index) {
            std::wstring key = argv[index];
            if (key == L"--validation") {
                if (validation) throw std::runtime_error("Missing or duplicate native viewer argument");
                validation = true;
                continue;
            }
            if (key == L"-frames") key = L"--frames";
            if (key != L"--vertex" && key != L"--fragment" && key != L"--width" &&
                key != L"--height" && key != L"--frames" && key != L"--vertex-count" &&
                key != L"--readback" && key != L"--time")
                throw std::runtime_error("Unknown native DirectX viewer argument");
            if (++index == argc || !options.emplace(key, argv[index]).second)
                throw std::runtime_error("Missing or duplicate native viewer argument");
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
                          hasTime, fixedTime);
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
