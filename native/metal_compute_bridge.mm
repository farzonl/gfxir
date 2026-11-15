#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void set_error(char* err_buf, size_t err_cap, const std::string& msg) {
    if (!err_buf || err_cap == 0) {
        return;
    }
    const size_t n = std::min(msg.size(), err_cap - 1);
    std::memcpy(err_buf, msg.data(), n);
    err_buf[n] = '\0';
}

struct GfxMtlBufferBinding {
    void* buffer;
    int is_uav;
    uint32_t set;
    uint32_t binding;
};

struct GfxMtlContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> last_command = nil;
    std::string device_name;
};

struct GfxMtlBuffer {
    GfxMtlContext* ctx = nullptr;
    id<MTLBuffer> buffer = nil;
    uint64_t size_bytes = 0;
    uint64_t aligned_size = 0;
};

struct GfxMtlPipeline {
    GfxMtlContext* ctx = nullptr;
    id<MTLComputePipelineState> pipeline = nil;
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;
    uint32_t wg_z = 1;
    size_t reflected_arg_size = 0;
    size_t descriptor_table_offset = 0;
    bool has_descriptor_table_offset = false;
};

int sync_context(GfxMtlContext* ctx, char* err_buf, size_t err_cap) {
    if (!ctx) {
        set_error(err_buf, err_cap, "Metal compute context is null");
        return -1;
    }
    if (ctx->last_command != nil) {
        [ctx->last_command waitUntilCompleted];
        if (ctx->last_command.status == MTLCommandBufferStatusError) {
            NSString* desc = ctx->last_command.error ? ctx->last_command.error.localizedDescription
                                                     : @"unknown Metal command buffer error";
            ctx->last_command = nil;
            set_error(err_buf, err_cap, std::string("Metal compute dispatch failed: ") + [desc UTF8String]);
            return -1;
        }
        ctx->last_command = nil;
    }
    return 0;
}

}  // namespace

extern "C" {

void* gfx_mtl_create_context(char* err_buf, size_t err_cap) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            set_error(err_buf, err_cap, "No Metal device is available");
            return nullptr;
        }
        if (device.argumentBuffersSupport != MTLArgumentBuffersTier2) {
            set_error(
                err_buf,
                err_cap,
                "Converted DXIL Metal compute execution requires Argument Buffers Tier 2"
            );
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            set_error(err_buf, err_cap, "Cannot create a Metal command queue");
            return nullptr;
        }
        auto* ctx = new GfxMtlContext();
        ctx->device = device;
        ctx->queue = queue;
        ctx->device_name = device.name ? [device.name UTF8String] : "Metal Device";
        return ctx;
    }
}

void gfx_mtl_destroy_context(void* ctx_ptr) {
    @autoreleasepool {
        auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
        if (!ctx) {
            return;
        }
        if (ctx->last_command != nil) {
            [ctx->last_command waitUntilCompleted];
            ctx->last_command = nil;
        }
        ctx->queue = nil;
        ctx->device = nil;
        delete ctx;
    }
}

const char* gfx_mtl_context_device_name(void* ctx_ptr) {
    auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
    if (!ctx) {
        return "";
    }
    return ctx->device_name.c_str();
}

void* gfx_mtl_create_buffer(
    void* ctx_ptr,
    uint64_t size_bytes,
    const void* initial_data,
    uint64_t initial_size,
    char* err_buf,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
        if (!ctx || !ctx->device) {
            set_error(err_buf, err_cap, "Metal compute context is null");
            return nullptr;
        }
        if (size_bytes == 0) {
            set_error(err_buf, err_cap, "Metal compute buffer size must be positive");
            return nullptr;
        }
        if (initial_data && initial_size > size_bytes) {
            set_error(err_buf, err_cap, "initial_size exceeds Metal buffer size");
            return nullptr;
        }
        const uint64_t aligned_size = std::max<uint64_t>(4, (size_bytes + 3) & ~static_cast<uint64_t>(3));
        id<MTLBuffer> mtl_buf = [ctx->device newBufferWithLength:static_cast<NSUInteger>(aligned_size)
                                                         options:MTLResourceStorageModeShared];
        if (!mtl_buf || !mtl_buf.contents) {
            set_error(err_buf, err_cap, "Cannot allocate Metal shared storage buffer");
            return nullptr;
        }
        std::memset(mtl_buf.contents, 0, static_cast<size_t>(aligned_size));
        if (initial_data && initial_size > 0) {
            std::memcpy(mtl_buf.contents, initial_data, static_cast<size_t>(initial_size));
        }
        auto* buf = new GfxMtlBuffer();
        buf->ctx = ctx;
        buf->buffer = mtl_buf;
        buf->size_bytes = size_bytes;
        buf->aligned_size = aligned_size;
        return buf;
    }
}

void gfx_mtl_destroy_buffer(void* buf_ptr) {
    @autoreleasepool {
        auto* buf = static_cast<GfxMtlBuffer*>(buf_ptr);
        if (!buf) {
            return;
        }
        buf->buffer = nil;
        delete buf;
    }
}

int gfx_mtl_upload_buffer(
    void* ctx_ptr,
    void* buf_ptr,
    uint64_t offset,
    const void* data,
    uint64_t data_size,
    char* err_buf,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
        auto* buf = static_cast<GfxMtlBuffer*>(buf_ptr);
        if (!ctx || !buf || !buf->buffer || buf->ctx != ctx) {
            set_error(err_buf, err_cap, "Invalid Metal buffer or context");
            return -1;
        }
        if (offset + data_size > buf->size_bytes) {
            set_error(err_buf, err_cap, "Upload range exceeds Metal buffer size");
            return -1;
        }
        if (sync_context(ctx, err_buf, err_cap) != 0) {
            return -1;
        }
        if (data_size > 0) {
            auto* dst = static_cast<uint8_t*>(buf->buffer.contents);
            std::memcpy(dst + offset, data, static_cast<size_t>(data_size));
        }
        return 0;
    }
}

int gfx_mtl_download_buffer(
    void* ctx_ptr,
    void* buf_ptr,
    uint64_t offset,
    void* out_data,
    uint64_t read_size,
    char* err_buf,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
        auto* buf = static_cast<GfxMtlBuffer*>(buf_ptr);
        if (!ctx || !buf || !buf->buffer || buf->ctx != ctx) {
            set_error(err_buf, err_cap, "Invalid Metal buffer or context");
            return -1;
        }
        if (offset + read_size > buf->size_bytes) {
            set_error(err_buf, err_cap, "Download range exceeds Metal buffer size");
            return -1;
        }
        if (sync_context(ctx, err_buf, err_cap) != 0) {
            return -1;
        }
        if (read_size > 0) {
            const auto* src = static_cast<const uint8_t*>(buf->buffer.contents);
            std::memcpy(out_data, src + offset, static_cast<size_t>(read_size));
        }
        return 0;
    }
}

void* gfx_mtl_create_pipeline(
    void* ctx_ptr,
    const char* metallib_path,
    const char* entry_point,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    char* err_buf,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
        if (!ctx || !ctx->device) {
            set_error(err_buf, err_cap, "Metal compute context is null");
            return nullptr;
        }
        if (!metallib_path || !entry_point) {
            set_error(err_buf, err_cap, "metallib_path and entry_point are required");
            return nullptr;
        }
        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:@(metallib_path)];
        id<MTLLibrary> library = [ctx->device newLibraryWithURL:url error:&error];
        if (!library) {
            NSString* desc = error ? error.localizedDescription : @"unknown error";
            set_error(err_buf, err_cap, std::string("Cannot load converted compute metallib: ") + [desc UTF8String]);
            return nullptr;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@(entry_point)];
        if (!fn) {
            set_error(err_buf, err_cap, std::string("Entry point not found in metallib: ") + entry_point);
            return nullptr;
        }

        MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = fn;
        MTLAutoreleasedComputePipelineReflection reflection = nil;
        MTLPipelineOption options = MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithDescriptor:desc
                                                                                    options:options
                                                                                 reflection:&reflection
                                                                                      error:&error];
        if (!pso) {
            NSString* msg = error ? error.localizedDescription : @"unknown error";
            set_error(err_buf, err_cap, std::string("Cannot create Metal compute pipeline state: ") + [msg UTF8String]);
            return nullptr;
        }
        const NSUInteger threads = static_cast<NSUInteger>(wg_x) * wg_y * wg_z;
        if (threads > pso.maxTotalThreadsPerThreadgroup) {
            set_error(err_buf, err_cap, "Compute workgroup_size exceeds Metal maxTotalThreadsPerThreadgroup");
            return nullptr;
        }

        auto* pipe = new GfxMtlPipeline();
        pipe->ctx = ctx;
        pipe->pipeline = pso;
        pipe->wg_x = wg_x;
        pipe->wg_y = wg_y;
        pipe->wg_z = wg_z;

        if (reflection != nil) {
            for (id<MTLBinding> binding in reflection.bindings) {
                if (binding.type == MTLBindingTypeBuffer && binding.index == kIRArgumentBufferBindPoint) {
                    id untyped = binding;
                    id<MTLBufferBinding> bufferBinding = untyped;
                    pipe->reflected_arg_size = bufferBinding.bufferDataSize;
                    MTLStructType* structType = bufferBinding.bufferStructType;
                    if (structType != nil) {
                        for (MTLStructMember* member in structType.members) {
                            if (member.dataType == MTLDataTypePointer) {
                                pipe->descriptor_table_offset = member.offset;
                                pipe->has_descriptor_table_offset = true;
                            }
                        }
                    }
                    break;
                }
            }
        }
        return pipe;
    }
}

void gfx_mtl_destroy_pipeline(void* pipe_ptr) {
    @autoreleasepool {
        auto* pipe = static_cast<GfxMtlPipeline*>(pipe_ptr);
        if (!pipe) {
            return;
        }
        pipe->pipeline = nil;
        delete pipe;
    }
}

int gfx_mtl_dispatch(
    void* ctx_ptr,
    void* pipe_ptr,
    const GfxMtlBufferBinding* bindings,
    uint32_t num_bindings,
    const void* root_constants,
    uint32_t root_constants_size,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    char* err_buf,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<GfxMtlContext*>(ctx_ptr);
        auto* pipe = static_cast<GfxMtlPipeline*>(pipe_ptr);
        if (!ctx || !pipe || pipe->ctx != ctx) {
            set_error(err_buf, err_cap, "Invalid Metal pipeline or context");
            return -1;
        }

        std::vector<const GfxMtlBufferBinding*> ordered;
        ordered.reserve(num_bindings);
        for (uint32_t i = 0; i < num_bindings; ++i) {
            auto* buf = static_cast<GfxMtlBuffer*>(bindings[i].buffer);
            if (!buf || !buf->buffer || buf->ctx != ctx) {
                set_error(err_buf, err_cap, "Invalid or cross-device Metal buffer binding");
                return -1;
            }
            ordered.push_back(&bindings[i]);
        }

        std::stable_sort(
            ordered.begin(),
            ordered.end(),
            [](const GfxMtlBufferBinding* a, const GfxMtlBufferBinding* b) {
                if (a->set != b->set) return a->set < b->set;
                return a->binding < b->binding;
            }
        );

        std::vector<const GfxMtlBufferBinding*> uavs;
        std::vector<const GfxMtlBufferBinding*> srvs;
        for (const auto* b : ordered) {
            if (b->is_uav) uavs.push_back(b);
            else srvs.push_back(b);
        }

        const size_t total_descriptors = uavs.size() + srvs.size();
        id<MTLBuffer> descriptor_table = nil;
        if (total_descriptors > 0) {
            const size_t table_bytes = total_descriptors * sizeof(IRDescriptorTableEntry);
            descriptor_table = [ctx->device newBufferWithLength:table_bytes options:MTLResourceStorageModeShared];
            if (!descriptor_table) {
                set_error(err_buf, err_cap, "Cannot allocate Metal IRDescriptorTableEntry buffer");
                return -1;
            }
            auto* entries = static_cast<IRDescriptorTableEntry*>(descriptor_table.contents);
            std::memset(entries, 0, table_bytes);
            size_t slot = 0;
            for (const auto* b : uavs) {
                auto* buf = static_cast<GfxMtlBuffer*>(b->buffer);
                IRDescriptorTableSetBuffer(&entries[slot++], buf->buffer.gpuAddress, 0);
            }
            for (const auto* b : srvs) {
                auto* buf = static_cast<GfxMtlBuffer*>(b->buffer);
                IRDescriptorTableSetBuffer(&entries[slot++], buf->buffer.gpuAddress, 0);
            }
        }

        const bool needs_arg_buf = (root_constants_size > 0) || (total_descriptors > 0);
        id<MTLBuffer> top_level_arg_buf = nil;
        if (needs_arg_buf) {
            const size_t fallback_offset = (static_cast<size_t>(root_constants_size) + 7) & ~static_cast<size_t>(7);
            const size_t table_offset =
                pipe->has_descriptor_table_offset ? pipe->descriptor_table_offset : fallback_offset;
            size_t min_size = root_constants_size;
            if (total_descriptors > 0) {
                min_size = std::max<size_t>(min_size, table_offset + sizeof(uint64_t));
            }
            const size_t arg_buf_size = std::max<size_t>(std::max<size_t>(pipe->reflected_arg_size, min_size), 16);
            top_level_arg_buf = [ctx->device newBufferWithLength:arg_buf_size options:MTLResourceStorageModeShared];
            if (!top_level_arg_buf) {
                set_error(err_buf, err_cap, "Cannot allocate Metal top-level argument buffer");
                return -1;
            }
            auto* arg_bytes = static_cast<uint8_t*>(top_level_arg_buf.contents);
            std::memset(arg_bytes, 0, arg_buf_size);
            if (root_constants && root_constants_size > 0) {
                std::memcpy(arg_bytes, root_constants, root_constants_size);
            }
            if (total_descriptors > 0 && descriptor_table != nil) {
                uint64_t table_addr = descriptor_table.gpuAddress;
                std::memcpy(arg_bytes + table_offset, &table_addr, sizeof(table_addr));
            }
        }

        id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
        if (!command) {
            set_error(err_buf, err_cap, "Cannot allocate Metal compute command buffer");
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            set_error(err_buf, err_cap, "Cannot create Metal compute command encoder");
            return -1;
        }

        [encoder setComputePipelineState:pipe->pipeline];
        if (top_level_arg_buf != nil) {
            [encoder setBuffer:top_level_arg_buf offset:0 atIndex:kIRArgumentBufferBindPoint];
        }
        if (descriptor_table != nil) {
            [encoder useResource:descriptor_table usage:MTLResourceUsageRead];
        }
        for (const auto* b : ordered) {
            auto* buf = static_cast<GfxMtlBuffer*>(b->buffer);
            const MTLResourceUsage usage = b->is_uav ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                                                     : MTLResourceUsageRead;
            [encoder useResource:buf->buffer usage:usage];
        }

        [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, grid_z)
                threadsPerThreadgroup:MTLSizeMake(pipe->wg_x, pipe->wg_y, pipe->wg_z)];
        [encoder endEncoding];
        [command commit];
        ctx->last_command = command;
        return 0;
    }
}

int gfx_mtl_synchronize(void* ctx_ptr, char* err_buf, size_t err_cap) {
    @autoreleasepool {
        return sync_context(static_cast<GfxMtlContext*>(ctx_ptr), err_buf, err_cap);
    }
}

}  // extern "C"
