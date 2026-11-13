#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct MgfxBufferEntry {
    id<MTLBuffer> dataBuffer = nil;
    id<MTLBuffer> descriptorTable = nil;
    size_t capacity = 0;
};

struct MgfxTextureSamplerEntry {
    id<MTLTexture> texture = nil;
    id<MTLSamplerState> sampler = nil;
    id<MTLBuffer> srvTable = nil;
    id<MTLBuffer> samplerTable = nil;
};

struct MgfxPipelineEntry {
    id<MTLRenderPipelineState> state = nil;
    bool depthEnabled = true;
    std::vector<size_t> fragmentPointerOffsets;
};

struct MgfxMetalContext {
    NSWindow* __unsafe_unretained window = nil;
    CAMetalLayer* layer = nil;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLDepthStencilState> depthEnabledState = nil;
    id<MTLDepthStencilState> depthDisabledState = nil;
    id<MTLTexture> depthTexture = nil;
    int depthWidth = 0;
    int depthHeight = 0;

    std::vector<MgfxPipelineEntry> pipelines;
    std::unordered_map<int, MgfxBufferEntry> buffers;
    std::unordered_map<int, MgfxTextureSamplerEntry> textureSamplers;

    id<CAMetalDrawable> currentDrawable = nil;
    id<MTLCommandBuffer> currentCommandBuffer = nil;
    id<MTLRenderCommandEncoder> currentEncoder = nil;
    std::string lastError;
};

static MTLSamplerAddressMode toMetalAddressMode(int mode) {
    if (mode == 1) return MTLSamplerAddressModeClampToEdge;
    if (mode == 2) return MTLSamplerAddressModeMirrorRepeat;
    return MTLSamplerAddressModeRepeat;
}

static void setError(MgfxMetalContext* ctx, NSString* msg, NSError* err = nil) {
    if (!ctx) return;
    if (err) {
        NSString* full = [NSString stringWithFormat:@"%@: %@", msg, err.localizedDescription];
        ctx->lastError = full.UTF8String ? full.UTF8String : "Metal error";
    } else {
        ctx->lastError = msg.UTF8String ? msg.UTF8String : "Metal error";
    }
}

} // namespace

extern "C" {

void* mgfx_metal_create(void* cocoa_nswindow, int fb_width, int fb_height, char* err_buf, size_t err_cap) {
    @autoreleasepool {
        auto write_err = [&](const char* msg) {
            if (err_buf && err_cap > 0) {
                std::strncpy(err_buf, msg, err_cap - 1);
                err_buf[err_cap - 1] = '\0';
            }
        };

        if (!cocoa_nswindow) {
            write_err("Null Cocoa NSWindow handle from GLFW");
            return nullptr;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            write_err("No Metal device available on this system");
            return nullptr;
        }
        if (device.argumentBuffersSupport != MTLArgumentBuffersTier2) {
            write_err("Metal Argument Buffers Tier 2 is required for GfxIR DXIL-to-Metal execution");
            return nullptr;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            write_err("Failed to create Metal command queue");
            return nullptr;
        }

        NSWindow* window = (__bridge NSWindow*)cocoa_nswindow;
        NSView* contentView = [window contentView];
        if (!contentView) {
            write_err("NSWindow has no contentView");
            return nullptr;
        }

        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.device = device;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = YES;
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        if (cs) {
            layer.colorspace = cs;
            CGColorSpaceRelease(cs);
        }
        CGFloat scale = [window backingScaleFactor];
        layer.contentsScale = scale > 0.0 ? scale : 1.0;
        layer.drawableSize = CGSizeMake(std::max(1, fb_width), std::max(1, fb_height));

        [contentView setWantsLayer:YES];
        [contentView setLayer:layer];

        MTLDepthStencilDescriptor* depthOnDesc = [MTLDepthStencilDescriptor new];
        depthOnDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
        depthOnDesc.depthWriteEnabled = YES;
        id<MTLDepthStencilState> depthOn = [device newDepthStencilStateWithDescriptor:depthOnDesc];

        MTLDepthStencilDescriptor* depthOffDesc = [MTLDepthStencilDescriptor new];
        depthOffDesc.depthCompareFunction = MTLCompareFunctionAlways;
        depthOffDesc.depthWriteEnabled = NO;
        id<MTLDepthStencilState> depthOff = [device newDepthStencilStateWithDescriptor:depthOffDesc];

        if (!depthOn || !depthOff) {
            write_err("Failed to create Metal depth-stencil states");
            return nullptr;
        }

        auto ctx = std::make_unique<MgfxMetalContext>();
        ctx->window = window;
        ctx->layer = layer;
        ctx->device = device;
        ctx->queue = queue;
        ctx->depthEnabledState = depthOn;
        ctx->depthDisabledState = depthOff;
        return ctx.release();
    }
}

const char* mgfx_metal_last_error(void* handle) {
    if (!handle) return "Null Metal context";
    auto* ctx = static_cast<MgfxMetalContext*>(handle);
    return ctx->lastError.c_str();
}

const char* mgfx_metal_device_name(void* handle) {
    if (!handle) return "Metal";
    auto* ctx = static_cast<MgfxMetalContext*>(handle);
    return ctx->device.name.UTF8String;
}

int mgfx_metal_create_pipeline(
    void* handle,
    const char* vs_metallib_path,
    const char* vs_entry,
    const char* fs_metallib_path,
    const char* fs_entry,
    int depth_enabled,
    int alpha_blend
) {
    @autoreleasepool {
        if (!handle) return -1;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        NSError* error = nil;

        NSURL* vsURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:vs_metallib_path]];
        id<MTLLibrary> vsLib = [ctx->device newLibraryWithURL:vsURL error:&error];
        if (!vsLib) {
            setError(ctx, @"Cannot load vertex metallib", error);
            return -1;
        }

        NSURL* fsURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:fs_metallib_path]];
        id<MTLLibrary> fsLib = [ctx->device newLibraryWithURL:fsURL error:&error];
        if (!fsLib) {
            setError(ctx, @"Cannot load fragment metallib", error);
            return -1;
        }

        id<MTLFunction> vsFn = [vsLib newFunctionWithName:[NSString stringWithUTF8String:vs_entry]];
        id<MTLFunction> fsFn = [fsLib newFunctionWithName:[NSString stringWithUTF8String:fs_entry]];
        if (!vsFn || !fsFn) {
            setError(ctx, @"Entry point missing in converted metallib");
            return -1;
        }

        MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
        desc.vertexFunction = vsFn;
        desc.fragmentFunction = fsFn;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

        if (alpha_blend) {
            desc.colorAttachments[0].blendingEnabled = YES;
            desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        }

        MTLAutoreleasedRenderPipelineReflection reflection = nil;
        MTLPipelineOption options = MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
        id<MTLRenderPipelineState> pso = [ctx->device newRenderPipelineStateWithDescriptor:desc
                                                                                   options:options
                                                                                reflection:&reflection
                                                                                     error:&error];
        if (!pso) {
            setError(ctx, @"Cannot create Metal render pipeline state", error);
            return -1;
        }

        MgfxPipelineEntry entry;
        entry.state = pso;
        entry.depthEnabled = (depth_enabled != 0);
        if (reflection != nil) {
            for (id<MTLBinding> binding in reflection.fragmentBindings) {
                if (binding.type == MTLBindingTypeBuffer && binding.index == kIRArgumentBufferBindPoint) {
                    id untyped = binding;
                    id<MTLBufferBinding> bufBinding = untyped;
                    MTLStructType* structType = bufBinding.bufferStructType;
                    if (structType != nil) {
                        for (MTLStructMember* member in structType.members) {
                            if (member.dataType == MTLDataTypePointer) {
                                entry.fragmentPointerOffsets.push_back(member.offset);
                            }
                        }
                        std::sort(entry.fragmentPointerOffsets.begin(), entry.fragmentPointerOffsets.end());
                    }
                    break;
                }
            }
        }
        int id_out = static_cast<int>(ctx->pipelines.size());
        ctx->pipelines.push_back(std::move(entry));
        return id_out;
    }
}

int mgfx_metal_upload_buffer(void* handle, int buf_id, const void* data, size_t byte_size) {
    @autoreleasepool {
        if (!handle || !data || byte_size == 0) return -1;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        auto& slot = ctx->buffers[buf_id];

        size_t aligned = (std::max<size_t>(byte_size, 64) + 15) & ~static_cast<size_t>(15);
        if (!slot.dataBuffer || aligned > slot.capacity) {
            size_t allocSize = std::max(aligned, slot.capacity * 2);
            id<MTLBuffer> newBuf = [ctx->device newBufferWithLength:allocSize options:MTLResourceStorageModeShared];
            if (!newBuf) {
                setError(ctx, @"Failed to allocate Metal vertex storage buffer");
                return -1;
            }
            if (!slot.descriptorTable) {
                slot.descriptorTable = [ctx->device newBufferWithLength:sizeof(IRDescriptorTableEntry)
                                                                options:MTLResourceStorageModeShared];
                if (!slot.descriptorTable) {
                    setError(ctx, @"Failed to allocate Metal IRDescriptorTableEntry");
                    return -1;
                }
            }
            slot.dataBuffer = newBuf;
            slot.capacity = allocSize;
            auto* entries = static_cast<IRDescriptorTableEntry*>(slot.descriptorTable.contents);
            std::memset(entries, 0, sizeof(IRDescriptorTableEntry));
            IRDescriptorTableSetBuffer(&entries[0], slot.dataBuffer.gpuAddress, 0);
        }

        std::memcpy(slot.dataBuffer.contents, data, byte_size);
        return 0;
    }
}

int mgfx_metal_begin_frame(
    void* handle,
    int fb_width,
    int fb_height,
    float clear_r,
    float clear_g,
    float clear_b,
    float clear_a
) {
    @autoreleasepool {
        if (!handle) return -1;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        int w = std::max(1, fb_width);
        int h = std::max(1, fb_height);

        if (ctx->window) {
            CGFloat scale = [ctx->window backingScaleFactor];
            if (scale > 0.0 && ctx->layer.contentsScale != scale) {
                ctx->layer.contentsScale = scale;
            }
        }
        CGSize curSize = ctx->layer.drawableSize;
        if (static_cast<int>(curSize.width) != w || static_cast<int>(curSize.height) != h) {
            ctx->layer.drawableSize = CGSizeMake(w, h);
        }

        if (!ctx->depthTexture || ctx->depthWidth != w || ctx->depthHeight != h) {
            MTLTextureDescriptor* depthDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                             width:static_cast<NSUInteger>(w)
                                            height:static_cast<NSUInteger>(h)
                                         mipmapped:NO];
            depthDesc.storageMode = MTLStorageModePrivate;
            depthDesc.usage = MTLTextureUsageRenderTarget;
            ctx->depthTexture = [ctx->device newTextureWithDescriptor:depthDesc];
            if (!ctx->depthTexture) {
                setError(ctx, @"Failed to allocate Metal depth texture");
                return -1;
            }
            ctx->depthWidth = w;
            ctx->depthHeight = h;
        }

        id<CAMetalDrawable> drawable = [ctx->layer nextDrawable];
        if (!drawable) {
            return 1; // Minimized or momentarily unavailable; skip frame gracefully
        }

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) {
            setError(ctx, @"Failed to allocate Metal command buffer");
            return -1;
        }

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(clear_r, clear_g, clear_b, clear_a);

        pass.depthAttachment.texture = ctx->depthTexture;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
        pass.depthAttachment.clearDepth = 1.0;

        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
        if (!enc) {
            setError(ctx, @"Failed to create Metal render command encoder");
            return -1;
        }

        [enc setCullMode:MTLCullModeNone];
        [enc setDepthClipMode:MTLDepthClipModeClip];
        MTLViewport viewport{
            0.0, 0.0,
            static_cast<double>(drawable.texture.width),
            static_cast<double>(drawable.texture.height),
            0.0, 1.0
        };
        [enc setViewport:viewport];

        ctx->currentDrawable = drawable;
        ctx->currentCommandBuffer = cmd;
        ctx->currentEncoder = enc;
        return 0;
    }
}

int mgfx_metal_upload_texture_sampler(
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
    @autoreleasepool {
        if (!handle || !pixels || width <= 0 || height <= 0) return -1;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        NSUInteger w = static_cast<NSUInteger>(width);
        NSUInteger h = static_cast<NSUInteger>(height);
        NSUInteger bpp = is_float32 ? 16 : 4;
        NSUInteger expected = w * h * bpp;
        if (byte_size != expected) {
            setError(ctx, @"Texture payload byte size does not match dimensions");
            return -1;
        }

        MTLPixelFormat fmt = is_float32 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA8Unorm;
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
        texDesc.storageMode = MTLStorageModePrivate;
        texDesc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> tex = [ctx->device newTextureWithDescriptor:texDesc];
        if (!tex) {
            setError(ctx, @"Failed to allocate Metal 2D texture");
            return -1;
        }

        id<MTLBuffer> staging = [ctx->device newBufferWithBytes:pixels
                                                         length:byte_size
                                                        options:MTLResourceStorageModeShared];
        if (!staging) {
            setError(ctx, @"Failed to allocate Metal texture staging buffer");
            return -1;
        }

        id<MTLCommandBuffer> uploadCmd = [ctx->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [uploadCmd blitCommandEncoder];
        if (!uploadCmd || !blit) {
            setError(ctx, @"Failed to create Metal texture upload blit encoder");
            return -1;
        }
        [blit copyFromBuffer:staging
                sourceOffset:0
           sourceBytesPerRow:w * bpp
         sourceBytesPerImage:expected
                  sourceSize:MTLSizeMake(w, h, 1)
                   toTexture:tex
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [uploadCmd commit];
        [uploadCmd waitUntilCompleted];
        if (uploadCmd.status == MTLCommandBufferStatusError) {
            setError(ctx, @"Metal texture upload failed", uploadCmd.error);
            return -1;
        }

        MTLSamplerDescriptor* sampDesc = [MTLSamplerDescriptor new];
        sampDesc.minFilter = min_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sampDesc.magFilter = mag_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sampDesc.mipFilter = MTLSamplerMipFilterNotMipmapped;
        sampDesc.sAddressMode = toMetalAddressMode(addr_u);
        sampDesc.tAddressMode = toMetalAddressMode(addr_v);
        sampDesc.lodMinClamp = 0.0f;
        sampDesc.lodMaxClamp = 0.0f;
        sampDesc.supportArgumentBuffers = YES;
        id<MTLSamplerState> samp = [ctx->device newSamplerStateWithDescriptor:sampDesc];
        if (!samp) {
            setError(ctx, @"Failed to create Metal sampler state");
            return -1;
        }

        id<MTLBuffer> srvTable = [ctx->device newBufferWithLength:sizeof(IRDescriptorTableEntry)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> samplerTable = [ctx->device newBufferWithLength:sizeof(IRDescriptorTableEntry)
                                                              options:MTLResourceStorageModeShared];
        if (!srvTable || !samplerTable) {
            setError(ctx, @"Failed to allocate Metal texture/sampler descriptor tables");
            return -1;
        }

        auto* srvEntry = static_cast<IRDescriptorTableEntry*>(srvTable.contents);
        std::memset(srvEntry, 0, sizeof(IRDescriptorTableEntry));
        IRDescriptorTableSetTexture(srvEntry, tex, 0.0f, 0);

        auto* sampEntry = static_cast<IRDescriptorTableEntry*>(samplerTable.contents);
        std::memset(sampEntry, 0, sizeof(IRDescriptorTableEntry));
        IRDescriptorTableSetSampler(sampEntry, samp, 0.0f);

        MgfxTextureSamplerEntry slot;
        slot.texture = tex;
        slot.sampler = samp;
        slot.srvTable = srvTable;
        slot.samplerTable = samplerTable;
        ctx->textureSamplers[tex_id] = std::move(slot);
        return 0;
    }
}

int mgfx_metal_draw_textured(
    void* handle,
    int pipeline_id,
    int buf_id,
    uint32_t vertex_count,
    const void* vs_consts,
    size_t vs_const_size,
    const void* fs_consts,
    size_t fs_const_size,
    int tex_id
) {
    @autoreleasepool {
        if (!handle || vertex_count == 0) return 0;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        if (!ctx->currentEncoder) return -1;
        if (pipeline_id < 0 || static_cast<size_t>(pipeline_id) >= ctx->pipelines.size()) return -1;
        auto bufIt = ctx->buffers.find(buf_id);
        if (bufIt == ctx->buffers.end() || !bufIt->second.dataBuffer || !bufIt->second.descriptorTable) return -1;
        if (vs_const_size > 512 || fs_const_size > 512) return -1;

        const auto& pipe = ctx->pipelines[static_cast<size_t>(pipeline_id)];
        const auto& buf = bufIt->second;
        id<MTLRenderCommandEncoder> enc = ctx->currentEncoder;

        [enc setRenderPipelineState:pipe.state];
        [enc setDepthStencilState:(pipe.depthEnabled ? ctx->depthEnabledState : ctx->depthDisabledState)];

        uint8_t vsArgBuf[528];
        std::memset(vsArgBuf, 0, sizeof(vsArgBuf));
        if (vs_consts && vs_const_size > 0) {
            std::memcpy(vsArgBuf, vs_consts, vs_const_size);
        }
        uint64_t tableGpuAddr = buf.descriptorTable.gpuAddress;
        std::memcpy(vsArgBuf + vs_const_size, &tableGpuAddr, sizeof(uint64_t));
        size_t totalVsArg = std::max<size_t>(vs_const_size + sizeof(uint64_t), 16);
        [enc setVertexBytes:vsArgBuf length:totalVsArg atIndex:kIRArgumentBufferBindPoint];

        if (tex_id >= 0) {
            auto texIt = ctx->textureSamplers.find(tex_id);
            if (texIt == ctx->textureSamplers.end()) return -1;
            const auto& ts = texIt->second;
            size_t srvOff = !pipe.fragmentPointerOffsets.empty()
                ? pipe.fragmentPointerOffsets[0]
                : ((fs_const_size + 7) & ~static_cast<size_t>(7));
            size_t sampOff = pipe.fragmentPointerOffsets.size() >= 2
                ? pipe.fragmentPointerOffsets[1]
                : (srvOff + sizeof(uint64_t));
            size_t totalFsArg = std::max<size_t>(sampOff + sizeof(uint64_t), 16);
            std::vector<uint8_t> fsArgBuf(totalFsArg, 0);
            if (fs_consts && fs_const_size > 0) {
                std::memcpy(fsArgBuf.data(), fs_consts, fs_const_size);
            }
            uint64_t srvAddr = ts.srvTable.gpuAddress;
            uint64_t sampAddr = ts.samplerTable.gpuAddress;
            std::memcpy(fsArgBuf.data() + srvOff, &srvAddr, sizeof(uint64_t));
            std::memcpy(fsArgBuf.data() + sampOff, &sampAddr, sizeof(uint64_t));
            [enc setFragmentBytes:fsArgBuf.data() length:fsArgBuf.size() atIndex:kIRArgumentBufferBindPoint];
            [enc useResource:ts.srvTable usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
            [enc useResource:ts.samplerTable usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
            [enc useResource:ts.texture usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        } else if (fs_consts && fs_const_size > 0) {
            [enc setFragmentBytes:fs_consts length:fs_const_size atIndex:kIRArgumentBufferBindPoint];
        }

        [enc useResource:buf.descriptorTable usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [enc useResource:buf.dataBuffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];

        IRRuntimeDrawPrimitives(enc, MTLPrimitiveTypeTriangle, uint64_t{0}, static_cast<uint64_t>(vertex_count));
        return 0;
    }
}

int mgfx_metal_draw(
    void* handle,
    int pipeline_id,
    int buf_id,
    uint32_t vertex_count,
    const void* vs_consts,
    size_t vs_const_size,
    const void* fs_consts,
    size_t fs_const_size
) {
    @autoreleasepool {
        if (!handle || vertex_count == 0) return 0;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        if (!ctx->currentEncoder) return -1;
        if (pipeline_id < 0 || static_cast<size_t>(pipeline_id) >= ctx->pipelines.size()) return -1;
        auto bufIt = ctx->buffers.find(buf_id);
        if (bufIt == ctx->buffers.end() || !bufIt->second.dataBuffer || !bufIt->second.descriptorTable) return -1;
        if (vs_const_size > 512 || fs_const_size > 512) return -1;

        const auto& pipe = ctx->pipelines[static_cast<size_t>(pipeline_id)];
        const auto& buf = bufIt->second;
        id<MTLRenderCommandEncoder> enc = ctx->currentEncoder;

        [enc setRenderPipelineState:pipe.state];
        [enc setDepthStencilState:(pipe.depthEnabled ? ctx->depthEnabledState : ctx->depthDisabledState)];

        uint8_t vsArgBuf[528];
        std::memset(vsArgBuf, 0, sizeof(vsArgBuf));
        if (vs_consts && vs_const_size > 0) {
            std::memcpy(vsArgBuf, vs_consts, vs_const_size);
        }
        uint64_t tableGpuAddr = buf.descriptorTable.gpuAddress;
        std::memcpy(vsArgBuf + vs_const_size, &tableGpuAddr, sizeof(uint64_t));
        size_t totalVsArg = std::max<size_t>(vs_const_size + sizeof(uint64_t), 16);
        [enc setVertexBytes:vsArgBuf length:totalVsArg atIndex:kIRArgumentBufferBindPoint];

        if (fs_consts && fs_const_size > 0) {
            [enc setFragmentBytes:fs_consts length:fs_const_size atIndex:kIRArgumentBufferBindPoint];
        }

        [enc useResource:buf.descriptorTable usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [enc useResource:buf.dataBuffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];

        IRRuntimeDrawPrimitives(enc, MTLPrimitiveTypeTriangle, uint64_t{0}, static_cast<uint64_t>(vertex_count));
        return 0;
    }
}

int mgfx_metal_end_frame(void* handle) {
    @autoreleasepool {
        if (!handle) return -1;
        auto* ctx = static_cast<MgfxMetalContext*>(handle);
        if (!ctx->currentEncoder || !ctx->currentCommandBuffer || !ctx->currentDrawable) {
            ctx->currentEncoder = nil;
            ctx->currentCommandBuffer = nil;
            ctx->currentDrawable = nil;
            return 0;
        }

        id<MTLRenderCommandEncoder> enc = ctx->currentEncoder;
        id<MTLCommandBuffer> cmd = ctx->currentCommandBuffer;
        id<CAMetalDrawable> drawable = ctx->currentDrawable;
        ctx->currentEncoder = nil;
        ctx->currentCommandBuffer = nil;
        ctx->currentDrawable = nil;

        [enc endEncoding];
        [cmd presentDrawable:drawable];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusError) {
            setError(ctx, @"Metal command buffer execution failed", cmd.error);
            return -1;
        }
        return 0;
    }
}

void mgfx_metal_destroy(void* handle) {
    @autoreleasepool {
        if (!handle) return;
        std::unique_ptr<MgfxMetalContext> ctx(static_cast<MgfxMetalContext*>(handle));
        ctx->currentEncoder = nil;
        ctx->currentCommandBuffer = nil;
        ctx->currentDrawable = nil;
        ctx->textureSamplers.clear();
        ctx->buffers.clear();
        ctx->pipelines.clear();
        ctx->depthTexture = nil;
        ctx->depthEnabledState = nil;
        ctx->depthDisabledState = nil;
        ctx->queue = nil;
        ctx->device = nil;
        ctx->layer = nil;
    }
}

} // extern "C"
