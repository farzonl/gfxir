#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static int exitStatus = 0;

static void stopPreview() {
    [NSApp stop:nil];
    [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
        location:NSZeroPoint modifierFlags:0 timestamp:0 windowNumber:0
        context:nil subtype:0 data1:0 data2:0] atStart:NO];
}

static std::runtime_error failure(NSString* context, NSError* error = nil) {
    NSString* message = error ? [NSString stringWithFormat:@"%@: %@", context, error.localizedDescription] : context;
    return std::runtime_error(message.UTF8String);
}

static NSUInteger number(const std::string& value, NSUInteger maximum) {
    size_t consumed = 0;
    if (value.empty() || value[0] == '-') throw std::runtime_error("Invalid nonnegative integer");
    unsigned long result = std::stoul(value, &consumed);
    if (consumed != value.size() || result > maximum) throw std::runtime_error("Numeric argument is out of range");
    return result;
}

static double seconds(const std::string& value) {
    size_t consumed = 0;
    double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result) || result < 0 ||
        result > 3.4028234663852886e38)
        throw std::runtime_error("Time must be finite, nonnegative f32 seconds");
    return result;
}

static bool matchesJSON(id actual, id expected) {
    if ([expected isKindOfClass:[NSDictionary class]]) {
        if (![actual isKindOfClass:[NSDictionary class]] || [actual count] != [expected count]) return false;
        for (id key in expected) if (!matchesJSON(actual[key], expected[key])) return false;
        return true;
    }
    if ([expected isKindOfClass:[NSArray class]]) {
        if (![actual isKindOfClass:[NSArray class]] || [actual count] != [expected count]) return false;
        for (NSUInteger i = 0; i < [expected count]; ++i)
            if (!matchesJSON(actual[i], expected[i])) return false;
        return true;
    }
    if ([expected isKindOfClass:[NSNumber class]]) {
        if (![actual isKindOfClass:[NSNumber class]]) return false;
        bool boolean = CFGetTypeID((__bridge CFTypeRef)expected) == CFBooleanGetTypeID();
        if (boolean != (CFGetTypeID((__bridge CFTypeRef)actual) == CFBooleanGetTypeID())) return false;
        if (!boolean && CFNumberIsFloatType((__bridge CFNumberRef)actual) !=
                        CFNumberIsFloatType((__bridge CFNumberRef)expected)) return false;
    }
    return [actual isEqual:expected];
}

static NSDictionary* frameLayout(const std::string& path) {
    NSError* error = nil;
    NSData* data = [NSData dataWithContentsOfFile:@(path.c_str()) options:0 error:&error];
    if (!data) throw failure(@"Cannot read frame layout", error);
    id layout = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![layout isKindOfClass:[NSDictionary class]]) throw failure(@"Invalid frame layout JSON", error);
    id stages = layout[@"stages"];
    if (![stages isKindOfClass:[NSArray class]] || ![stages count] || [stages count] > 2 ||
        [[NSSet setWithArray:stages] count] != [stages count])
        throw failure(@"Invalid frame stage visibility");
    for (id stage in stages)
        if (![stage isEqual:@"vertex"] && ![stage isEqual:@"fragment"])
            throw failure(@"Unsupported frame stage");
    NSDictionary* expected = @{
        @"schema_version": @1, @"name": @"frame", @"layout": @"gfx-root-f32-v1",
        @"size": @16, @"alignment": @16, @"ownership": @"host", @"access": @"read-only",
        @"addressable": @NO, @"stages": stages,
        @"fields": @[
            @{@"name": @"time", @"type": @"f32", @"offset": @0, @"size": @4, @"alignment": @4},
            @{@"name": @"resolution", @"type": @"f32x2", @"offset": @8, @"size": @8, @"alignment": @8}],
        @"bindings": @{
            @"vulkan": @{@"storage": @"push_constant", @"offset": @0},
            @"directx": @{@"storage": @"root_constants", @"parameter_index": @0, @"register": @0,
                @"space": @0, @"num32bit_constants": @4, @"visibility": @"all", @"root_signature_version": @"1.1"},
            @"metal": @{@"storage": @"converter_argument_buffer", @"buffer_index": @(kIRArgumentBufferBindPoint),
                @"offset": @0, @"size": @16}}};
    if (!matchesJSON(layout, expected))
        throw failure(@"Unsupported or mismatched frame layout/bindings");
    return layout;
}

struct MetalDepthSettings {
    bool enabled = false;
    bool writeEnabled = true;
    MTLCompareFunction compareFunction = MTLCompareFunctionLessEqual;
};

static MTLCompareFunction parseMetalCompareFunction(const std::string& op) {
    if (op == "less_equal") return MTLCompareFunctionLessEqual;
    if (op == "less") return MTLCompareFunctionLess;
    if (op == "equal") return MTLCompareFunctionEqual;
    if (op == "greater") return MTLCompareFunctionGreater;
    if (op == "greater_equal") return MTLCompareFunctionGreaterEqual;
    if (op == "not_equal") return MTLCompareFunctionNotEqual;
    if (op == "always") return MTLCompareFunctionAlways;
    if (op == "never") return MTLCompareFunctionNever;
    throw std::runtime_error("Unsupported depth compare op: " + op);
}

struct MetalBufferResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    std::string access;
    size_t size = 0;
    size_t alignedSize = 0;
    bool isUAV = false;
    std::string outputPath;
    id<MTLBuffer> buffer = nil;
};

struct MetalTextureResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    NSUInteger width = 0;
    NSUInteger height = 0;
    id<MTLTexture> texture = nil;
};

struct MetalSamplerResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    id<MTLSamplerState> sampler = nil;
};

struct MetalStageResources {
    std::vector<MetalBufferResource> buffers;
    std::vector<MetalTextureResource> textures;
    std::vector<MetalSamplerResource> samplers;
    id<MTLBuffer> srvUavTable = nil;
    id<MTLBuffer> samplerTable = nil;
};

static MTLSamplerMinMagFilter parseMetalFilter(const std::string& mode) {
    if (mode == "nearest") return MTLSamplerMinMagFilterNearest;
    if (mode == "linear") return MTLSamplerMinMagFilterLinear;
    throw std::runtime_error("Unsupported sampler filter: " + mode);
}

static MTLSamplerAddressMode parseMetalAddressMode(const std::string& mode) {
    if (mode == "repeat") return MTLSamplerAddressModeRepeat;
    if (mode == "clamp_to_edge") return MTLSamplerAddressModeClampToEdge;
    if (mode == "mirror_repeat") return MTLSamplerAddressModeMirrorRepeat;
    throw std::runtime_error("Unsupported sampler address mode: " + mode);
}

static MetalStageResources loadStageResources(id<MTLDevice> device, id<MTLCommandQueue> queue, NSDictionary* spec) {
    MetalStageResources stage;
    if (!spec) return stage;
    NSError* error = nil;

    id rawBuffers = spec[@"buffers"];
    if (rawBuffers != nil) {
        if (![rawBuffers isKindOfClass:[NSArray class]])
            throw failure(@"Resource spec 'buffers' must be an array");
        NSArray* bufferSpecs = rawBuffers;
        stage.buffers.reserve(bufferSpecs.count);
        for (id item in bufferSpecs) {
            if (![item isKindOfClass:[NSDictionary class]])
                throw failure(@"Invalid buffer entry in resource spec");
            NSDictionary* dict = item;
            NSString* nameNs = dict[@"name"];
            NSString* accessNs = dict[@"access"];
            NSNumber* sizeNs = dict[@"size"];
            if (![nameNs isKindOfClass:[NSString class]] || ![accessNs isKindOfClass:[NSString class]] ||
                ![sizeNs isKindOfClass:[NSNumber class]])
                throw failure(@"Buffer entry is missing required fields");

            MetalBufferResource entry;
            entry.name = nameNs.UTF8String;
            entry.access = accessNs.UTF8String;
            if (entry.access != "read_only" && entry.access != "write_only" && entry.access != "read_write")
                throw std::runtime_error("Unsupported buffer access mode: " + entry.access);
            long long rawSize = [sizeNs longLongValue];
            if (rawSize <= 0)
                throw std::runtime_error("Buffer byte size must be positive: " + entry.name);
            entry.size = static_cast<size_t>(rawSize);
            entry.alignedSize = std::max<size_t>(4, (entry.size + 3) & ~static_cast<size_t>(3));
            entry.isUAV = (entry.access != "read_only");
            if ([dict[@"set"] isKindOfClass:[NSNumber class]])
                entry.set = static_cast<uint32_t>([dict[@"set"] unsignedIntValue]);
            if ([dict[@"binding"] isKindOfClass:[NSNumber class]])
                entry.binding = static_cast<uint32_t>([dict[@"binding"] unsignedIntValue]);
            if ([dict[@"output_path"] isKindOfClass:[NSString class]])
                entry.outputPath = [dict[@"output_path"] UTF8String];

            entry.buffer = [device newBufferWithLength:entry.alignedSize options:MTLResourceStorageModeShared];
            if (!entry.buffer) throw failure(@"Cannot allocate Metal storage buffer");
            std::memset(entry.buffer.contents, 0, entry.alignedSize);

            id inputPathObj = dict[@"input_path"];
            if ([inputPathObj isKindOfClass:[NSString class]] && [inputPathObj length] > 0) {
                NSData* inData = [NSData dataWithContentsOfFile:inputPathObj options:0 error:&error];
                if (!inData) throw failure(@"Cannot read input buffer file", error);
                if (inData.length != entry.size)
                    throw std::runtime_error("Input buffer file size does not match spec size: " + entry.name);
                std::memcpy(entry.buffer.contents, inData.bytes, entry.size);
            }
            stage.buffers.push_back(std::move(entry));
        }
    }

    id rawTextures = spec[@"textures"];
    if (rawTextures != nil) {
        if (![rawTextures isKindOfClass:[NSArray class]])
            throw failure(@"Resource spec 'textures' must be an array");
        NSArray* textureSpecs = rawTextures;
        stage.textures.reserve(textureSpecs.count);

        id<MTLCommandBuffer> uploadCmd = nil;
        id<MTLBlitCommandEncoder> blit = nil;
        std::vector<id<MTLBuffer>> stagingBuffers;

        for (id item in textureSpecs) {
            if (![item isKindOfClass:[NSDictionary class]])
                throw failure(@"Invalid texture entry in resource spec");
            NSDictionary* dict = item;
            NSString* nameNs = dict[@"name"];
            NSString* fmtNs = dict[@"format"];
            NSString* pathNs = dict[@"path"];
            NSNumber* widthNs = dict[@"width"];
            NSNumber* heightNs = dict[@"height"];
            if (![nameNs isKindOfClass:[NSString class]] || ![fmtNs isKindOfClass:[NSString class]] ||
                ![pathNs isKindOfClass:[NSString class]] || ![widthNs isKindOfClass:[NSNumber class]] ||
                ![heightNs isKindOfClass:[NSNumber class]])
                throw failure(@"Texture entry is missing required fields");

            MetalTextureResource entry;
            entry.name = nameNs.UTF8String;
            long long rawW = [widthNs longLongValue];
            long long rawH = [heightNs longLongValue];
            if (rawW <= 0 || rawH <= 0)
                throw std::runtime_error("Texture dimensions must be positive: " + entry.name);
            entry.width = static_cast<NSUInteger>(rawW);
            entry.height = static_cast<NSUInteger>(rawH);
            if ([dict[@"set"] isKindOfClass:[NSNumber class]])
                entry.set = static_cast<uint32_t>([dict[@"set"] unsignedIntValue]);
            if ([dict[@"binding"] isKindOfClass:[NSNumber class]])
                entry.binding = static_cast<uint32_t>([dict[@"binding"] unsignedIntValue]);

            std::string fmtStr = fmtNs.UTF8String;
            MTLPixelFormat pixelFmt = MTLPixelFormatRGBA8Unorm;
            NSUInteger bytesPerPixel = 4;
            if (fmtStr == "rgba8_unorm") {
                pixelFmt = MTLPixelFormatRGBA8Unorm;
                bytesPerPixel = 4;
            } else if (fmtStr == "rgba32_sfloat") {
                pixelFmt = MTLPixelFormatRGBA32Float;
                bytesPerPixel = 16;
            } else {
                throw std::runtime_error("Unsupported texture format: " + fmtStr);
            }

            NSData* texData = [NSData dataWithContentsOfFile:pathNs options:0 error:&error];
            if (!texData) throw failure(@"Cannot read texture payload file", error);
            NSUInteger expectedBytes = entry.width * entry.height * bytesPerPixel;
            if (texData.length != expectedBytes)
                throw std::runtime_error("Texture payload byte size does not match dimensions: " + entry.name);

            MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:pixelFmt width:entry.width height:entry.height mipmapped:NO];
            texDesc.storageMode = MTLStorageModePrivate;
            texDesc.usage = MTLTextureUsageShaderRead;
            entry.texture = [device newTextureWithDescriptor:texDesc];
            if (!entry.texture) throw failure(@"Cannot allocate Metal 2D texture");

            id<MTLBuffer> staging = [device newBufferWithBytes:texData.bytes
                                                        length:texData.length
                                                       options:MTLResourceStorageModeShared];
            if (!staging) throw failure(@"Cannot allocate Metal texture upload staging buffer");
            stagingBuffers.push_back(staging);

            if (!uploadCmd) {
                uploadCmd = [queue commandBuffer];
                if (!uploadCmd) throw failure(@"Cannot allocate texture upload command buffer");
                blit = [uploadCmd blitCommandEncoder];
                if (!blit) throw failure(@"Cannot allocate texture upload blit encoder");
            }
            NSUInteger rowBytes = entry.width * bytesPerPixel;
            [blit copyFromBuffer:staging
                    sourceOffset:0
               sourceBytesPerRow:rowBytes
             sourceBytesPerImage:expectedBytes
                      sourceSize:MTLSizeMake(entry.width, entry.height, 1)
                       toTexture:entry.texture
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];

            stage.textures.push_back(std::move(entry));
        }

        if (uploadCmd) {
            [blit endEncoding];
            [uploadCmd commit];
            [uploadCmd waitUntilCompleted];
            if (uploadCmd.status == MTLCommandBufferStatusError)
                throw failure(@"Metal texture upload failed", uploadCmd.error);
        }
    }

    id rawSamplers = spec[@"samplers"];
    if (rawSamplers != nil) {
        if (![rawSamplers isKindOfClass:[NSArray class]])
            throw failure(@"Resource spec 'samplers' must be an array");
        NSArray* samplerSpecs = rawSamplers;
        stage.samplers.reserve(samplerSpecs.count);
        for (id item in samplerSpecs) {
            if (![item isKindOfClass:[NSDictionary class]])
                throw failure(@"Invalid sampler entry in resource spec");
            NSDictionary* dict = item;
            NSString* nameNs = dict[@"name"];
            if (![nameNs isKindOfClass:[NSString class]])
                throw failure(@"Sampler entry is missing name");

            MetalSamplerResource entry;
            entry.name = nameNs.UTF8String;
            if ([dict[@"set"] isKindOfClass:[NSNumber class]])
                entry.set = static_cast<uint32_t>([dict[@"set"] unsignedIntValue]);
            if ([dict[@"binding"] isKindOfClass:[NSNumber class]])
                entry.binding = static_cast<uint32_t>([dict[@"binding"] unsignedIntValue]);

            std::string minF = [dict[@"min_filter"] isKindOfClass:[NSString class]] ? [dict[@"min_filter"] UTF8String] : "linear";
            std::string magF = [dict[@"mag_filter"] isKindOfClass:[NSString class]] ? [dict[@"mag_filter"] UTF8String] : "linear";
            std::string addrU = [dict[@"address_mode_u"] isKindOfClass:[NSString class]] ? [dict[@"address_mode_u"] UTF8String] : "repeat";
            std::string addrV = [dict[@"address_mode_v"] isKindOfClass:[NSString class]] ? [dict[@"address_mode_v"] UTF8String] : "repeat";

            MTLSamplerDescriptor* sampDesc = [MTLSamplerDescriptor new];
            sampDesc.minFilter = parseMetalFilter(minF);
            sampDesc.magFilter = parseMetalFilter(magF);
            sampDesc.mipFilter = MTLSamplerMipFilterNotMipmapped;
            sampDesc.sAddressMode = parseMetalAddressMode(addrU);
            sampDesc.tAddressMode = parseMetalAddressMode(addrV);
            sampDesc.lodMinClamp = 0.0f;
            sampDesc.lodMaxClamp = 0.0f;
            sampDesc.supportArgumentBuffers = YES;
            entry.sampler = [device newSamplerStateWithDescriptor:sampDesc];
            if (!entry.sampler) throw failure(@"Cannot create Metal sampler state");
            stage.samplers.push_back(std::move(entry));
        }
    }

    // Build SRV/UAV descriptor table matching GfxToLLVMPass register ordering:
    // UAV buffers sorted by (set, binding), then read-only buffers + 2D textures sorted together by (set, binding).
    struct OrderedSlot {
        uint32_t set = 0;
        uint32_t binding = 0;
        bool isTexture = false;
        size_t index = 0;
    };
    std::vector<OrderedSlot> uavSlots;
    std::vector<OrderedSlot> srvSlots;
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        OrderedSlot slot{stage.buffers[i].set, stage.buffers[i].binding, false, i};
        if (stage.buffers[i].isUAV) uavSlots.push_back(slot);
        else srvSlots.push_back(slot);
    }
    for (size_t i = 0; i < stage.textures.size(); ++i) {
        srvSlots.push_back(OrderedSlot{stage.textures[i].set, stage.textures[i].binding, true, i});
    }
    auto slotLess = [](const OrderedSlot& a, const OrderedSlot& b) {
        if (a.set != b.set) return a.set < b.set;
        return a.binding < b.binding;
    };
    std::stable_sort(uavSlots.begin(), uavSlots.end(), slotLess);
    std::stable_sort(srvSlots.begin(), srvSlots.end(), slotLess);

    size_t totalSrvUav = uavSlots.size() + srvSlots.size();
    if (totalSrvUav > 0) {
        size_t tableBytes = totalSrvUav * sizeof(IRDescriptorTableEntry);
        stage.srvUavTable = [device newBufferWithLength:tableBytes options:MTLResourceStorageModeShared];
        if (!stage.srvUavTable) throw failure(@"Cannot allocate Metal SRV/UAV descriptor table buffer");
        auto* entries = static_cast<IRDescriptorTableEntry*>(stage.srvUavTable.contents);
        std::memset(entries, 0, tableBytes);
        size_t slotIdx = 0;
        for (const auto& uav : uavSlots) {
            IRDescriptorTableSetBuffer(&entries[slotIdx++], stage.buffers[uav.index].buffer.gpuAddress, 0);
        }
        for (const auto& srv : srvSlots) {
            if (srv.isTexture) {
                IRDescriptorTableSetTexture(&entries[slotIdx++], stage.textures[srv.index].texture, 0.0f, 0);
            } else {
                IRDescriptorTableSetBuffer(&entries[slotIdx++], stage.buffers[srv.index].buffer.gpuAddress, 0);
            }
        }
    }

    if (!stage.samplers.empty()) {
        std::stable_sort(stage.samplers.begin(), stage.samplers.end(),
                         [](const MetalSamplerResource& a, const MetalSamplerResource& b) {
                             if (a.set != b.set) return a.set < b.set;
                             return a.binding < b.binding;
                         });
        size_t sampBytes = stage.samplers.size() * sizeof(IRDescriptorTableEntry);
        stage.samplerTable = [device newBufferWithLength:sampBytes options:MTLResourceStorageModeShared];
        if (!stage.samplerTable) throw failure(@"Cannot allocate Metal sampler descriptor table buffer");
        auto* entries = static_cast<IRDescriptorTableEntry*>(stage.samplerTable.contents);
        std::memset(entries, 0, sampBytes);
        for (size_t i = 0; i < stage.samplers.size(); ++i) {
            IRDescriptorTableSetSampler(&entries[i], stage.samplers[i].sampler, 0.0f);
        }
    }

    return stage;
}

struct ReflectedArgBufferLayout {
    size_t bufferSize = 0;
    std::vector<size_t> pointerOffsets;
};

static ReflectedArgBufferLayout extractReflectedArgBufferLayout(NSArray<id<MTLBinding>>* bindings) {
    ReflectedArgBufferLayout info;
    if (!bindings) return info;
    for (id<MTLBinding> binding in bindings) {
        if (binding.type == MTLBindingTypeBuffer && binding.index == kIRArgumentBufferBindPoint) {
            id untyped = binding;
            id<MTLBufferBinding> bufferBinding = untyped;
            info.bufferSize = bufferBinding.bufferDataSize;
            MTLStructType* structType = bufferBinding.bufferStructType;
            if (structType != nil) {
                for (MTLStructMember* member in structType.members) {
                    if (member.dataType == MTLDataTypePointer) {
                        info.pointerOffsets.push_back(member.offset);
                    }
                }
                std::sort(info.pointerOffsets.begin(), info.pointerOffsets.end());
            }
            break;
        }
    }
    return info;
}

static std::vector<uint8_t> packTopLevelArgBuffer(const uint8_t* rootConstantData, size_t rootConstantSize,
                                                  const MetalStageResources& stage,
                                                  const ReflectedArgBufferLayout& reflected) {
    std::vector<uint64_t> tableAddresses;
    if (stage.srvUavTable != nil) tableAddresses.push_back(stage.srvUavTable.gpuAddress);
    if (stage.samplerTable != nil) tableAddresses.push_back(stage.samplerTable.gpuAddress);
    if (rootConstantSize == 0 && tableAddresses.empty()) return {};

    std::vector<size_t> offsets = reflected.pointerOffsets;
    size_t nextFallbackOffset = (rootConstantSize + 7) & ~static_cast<size_t>(7);
    if (!offsets.empty()) {
        nextFallbackOffset = offsets.back() + sizeof(uint64_t);
    }
    while (offsets.size() < tableAddresses.size()) {
        offsets.push_back(nextFallbackOffset);
        nextFallbackOffset += sizeof(uint64_t);
    }

    size_t minSize = rootConstantSize;
    for (size_t i = 0; i < tableAddresses.size(); ++i) {
        minSize = std::max<size_t>(minSize, offsets[i] + sizeof(uint64_t));
    }
    size_t totalSize = std::max<size_t>(std::max<size_t>(reflected.bufferSize, minSize), 16);
    std::vector<uint8_t> bytes(totalSize, 0);
    if (rootConstantData != nullptr && rootConstantSize > 0) {
        std::memcpy(bytes.data(), rootConstantData, rootConstantSize);
    }
    for (size_t i = 0; i < tableAddresses.size(); ++i) {
        uint64_t addr = tableAddresses[i];
        std::memcpy(bytes.data() + offsets[i], &addr, sizeof(addr));
    }
    return bytes;
}

@interface GraphicsRenderer : NSObject <MTKViewDelegate, NSWindowDelegate>
{
    std::chrono::steady_clock::time_point _started;
    ReflectedArgBufferLayout _vertexArgLayout;
    ReflectedArgBufferLayout _fragmentArgLayout;
    std::map<std::string, ReflectedArgBufferLayout> _pipelineVertexArgLayouts;
    std::map<std::string, ReflectedArgBufferLayout> _pipelineFragmentArgLayouts;
    MetalStageResources _vertexResources;
    MetalStageResources _fragmentResources;
}
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLRenderPipelineState> pipeline;
@property(nonatomic) NSUInteger frames;
@property(nonatomic) NSUInteger limit;
@property(nonatomic) NSUInteger vertexCount;
@property(nonatomic, copy) NSDictionary* frameLayout;
@property(nonatomic) BOOL live;
@property(nonatomic) BOOL hasFixedTime;
@property(nonatomic) double fixedTime;
@property(nonatomic, strong) NSDictionary<NSString*, id<MTLRenderPipelineState>>* pipelines;
@property(nonatomic, strong) NSDictionary<NSString*, id<MTLDepthStencilState>>* pipelineDepthStates;
@property(nonatomic, strong) NSDictionary<NSString*, id>* pipelineParameters;
@property(nonatomic, copy) NSArray<NSDictionary*>* draws;
@property(nonatomic, strong) id<MTLDepthStencilState> depthState;
@property(nonatomic) BOOL hasDepth;
@property(nonatomic) BOOL isIndexed;
@property(nonatomic) NSUInteger indexCount;
@property(nonatomic, copy) NSString* indexType;
@property(nonatomic, strong) NSDictionary* inputLayout;
@property(nonatomic, strong) id<MTLBuffer> vertexBuffer;
@property(nonatomic, strong) id<MTLBuffer> indexBuffer;
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry
                    inputLayout:(NSDictionary*)inputLayout
                  depthSettings:(MetalDepthSettings)depthSettings;
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry
                    inputLayout:(NSDictionary*)inputLayout;
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry;
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry
                  depthSettings:(MetalDepthSettings)depthSettings;
- (void)loadResourcesFromPath:(NSString*)path;
- (ReflectedArgBufferLayout)vertexArgLayout;
- (ReflectedArgBufferLayout)fragmentArgLayout;
- (void)setArgLayoutsForPipeline:(NSString*)pipeId
                          vertex:(ReflectedArgBufferLayout)vertexLayout
                        fragment:(ReflectedArgBufferLayout)fragmentLayout;
- (void)encode:(id<MTLCommandBuffer>)command pass:(MTLRenderPassDescriptor*)pass;
@end

@implementation GraphicsRenderer
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry {
    return [self initWithVertex:vertex vertexEntry:vertexEntry
                        fragment:fragment fragmentEntry:fragmentEntry
                     inputLayout:nil
                   depthSettings:{}];
}

- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry
                    inputLayout:(NSDictionary*)inputLayout {
    return [self initWithVertex:vertex vertexEntry:vertexEntry
                        fragment:fragment fragmentEntry:fragmentEntry
                     inputLayout:inputLayout
                   depthSettings:{}];
}

- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry
                  depthSettings:(MetalDepthSettings)depthSettings {
    return [self initWithVertex:vertex vertexEntry:vertexEntry
                        fragment:fragment fragmentEntry:fragmentEntry
                     inputLayout:nil
                   depthSettings:depthSettings];
}

- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry
                    inputLayout:(NSDictionary*)inputLayout
                  depthSettings:(MetalDepthSettings)depthSettings {
    self = [super init];
    if (!self) throw failure(@"Cannot allocate Metal renderer");
    _inputLayout = inputLayout;
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) throw failure(@"No Metal device is available");
    if (_device.argumentBuffersSupport != MTLArgumentBuffersTier2)
        throw failure(@"Converted DXIL requires Argument Buffers Tier 2");
    NSError* error = nil;
    id<MTLLibrary> vs = [_device newLibraryWithURL:[NSURL fileURLWithPath:vertex] error:&error];
    if (!vs) throw failure(@"Cannot load converted vertex metallib", error);
    id<MTLLibrary> fs = [_device newLibraryWithURL:[NSURL fileURLWithPath:fragment] error:&error];
    if (!fs) throw failure(@"Cannot load converted fragment metallib", error);
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [vs newFunctionWithName:vertexEntry];
    descriptor.fragmentFunction = [fs newFunctionWithName:fragmentEntry];
    if (!descriptor.vertexFunction || !descriptor.fragmentFunction)
        throw failure(@"The converted library does not contain the manifest's entry point");
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    if (depthSettings.enabled) {
        descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    }
    if (inputLayout) {
        MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
        NSArray* attrs = inputLayout[@"attributes"];
        NSUInteger maxEndOffset = 0;
        for (NSDictionary* attr in attrs) {
            NSUInteger loc = [attr[@"location"] unsignedIntegerValue];
            NSUInteger offset = [attr[@"offset"] unsignedIntegerValue];
            NSUInteger binding = [attr[@"binding"] unsignedIntegerValue];
            NSString* format = attr[@"format"];
            MTLVertexFormat mtlFmt = MTLVertexFormatFloat3;
            NSUInteger attrSize = 12;
            if ([format isEqualToString:@"f32x3"] || [format isEqualToString:@"vec3"] || [format isEqualToString:@"R32G32B32_FLOAT"]) {
                mtlFmt = MTLVertexFormatFloat3; attrSize = 12;
            } else if ([format isEqualToString:@"f32x2"] || [format isEqualToString:@"vec2"] || [format isEqualToString:@"R32G32_FLOAT"]) {
                mtlFmt = MTLVertexFormatFloat2; attrSize = 8;
            } else if ([format isEqualToString:@"f32x4"] || [format isEqualToString:@"vec4"] || [format isEqualToString:@"R32G32B32A32_FLOAT"]) {
                mtlFmt = MTLVertexFormatFloat4; attrSize = 16;
            } else if ([format isEqualToString:@"f32"] || [format isEqualToString:@"R32_FLOAT"]) {
                mtlFmt = MTLVertexFormatFloat; attrSize = 4;
            } else if ([format isEqualToString:@"u32"] || [format isEqualToString:@"R32_UINT"]) {
                mtlFmt = MTLVertexFormatUInt; attrSize = 4;
            } else if ([format isEqualToString:@"i32"] || [format isEqualToString:@"R32_SINT"]) {
                mtlFmt = MTLVertexFormatInt; attrSize = 4;
            } else if ([format isEqualToString:@"u16"] || [format isEqualToString:@"R16_UINT"]) {
                mtlFmt = MTLVertexFormatUShort; attrSize = 2;
            } else if ([format isEqualToString:@"i16"] || [format isEqualToString:@"R16_SINT"]) {
                mtlFmt = MTLVertexFormatShort; attrSize = 2;
            }
            vd.attributes[kIRStageInAttributeStartIndex + loc].format = mtlFmt;
            vd.attributes[kIRStageInAttributeStartIndex + loc].offset = offset;
            vd.attributes[kIRStageInAttributeStartIndex + loc].bufferIndex = kIRVertexBufferBindPoint + binding;
            if (offset + attrSize > maxEndOffset) {
                maxEndOffset = offset + attrSize;
            }
        }
        NSArray* bindings = inputLayout[@"bindings"];
        if (bindings && bindings.count > 0) {
            for (NSDictionary* b in bindings) {
                NSUInteger binding = [b[@"binding"] unsignedIntegerValue];
                NSUInteger stride = [b[@"stride"] unsignedIntegerValue];
                vd.layouts[kIRVertexBufferBindPoint + binding].stride = stride;
                vd.layouts[kIRVertexBufferBindPoint + binding].stepFunction = MTLVertexStepFunctionPerVertex;
                vd.layouts[kIRVertexBufferBindPoint + binding].stepRate = 1;
            }
        } else if (attrs && attrs.count > 0) {
            vd.layouts[kIRVertexBufferBindPoint].stride = maxEndOffset;
            vd.layouts[kIRVertexBufferBindPoint].stepFunction = MTLVertexStepFunctionPerVertex;
            vd.layouts[kIRVertexBufferBindPoint].stepRate = 1;
        }
        descriptor.vertexDescriptor = vd;
    }
    MTLAutoreleasedRenderPipelineReflection renderReflection = nil;
    MTLPipelineOption pipelineOptions = MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
    _pipeline = [_device newRenderPipelineStateWithDescriptor:descriptor
                                                      options:pipelineOptions
                                                   reflection:&renderReflection
                                                        error:&error];
    if (!_pipeline) throw failure(@"Cannot create the converted DXIL render pipeline", error);
    if (renderReflection != nil) {
        _vertexArgLayout = extractReflectedArgBufferLayout(renderReflection.vertexBindings);
        _fragmentArgLayout = extractReflectedArgBufferLayout(renderReflection.fragmentBindings);
    }

    if (depthSettings.enabled) {
        MTLDepthStencilDescriptor* depthDesc = [MTLDepthStencilDescriptor new];
        depthDesc.depthCompareFunction = depthSettings.compareFunction;
        depthDesc.depthWriteEnabled = depthSettings.writeEnabled ? YES : NO;
        _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
        if (!_depthState) throw failure(@"Cannot create Metal depth-stencil state");
        _hasDepth = YES;
    }

    _queue = [_device newCommandQueue];
    if (!_queue) throw failure(@"Cannot create a Metal command queue");
    _started = std::chrono::steady_clock::now();
    return self;
}

- (void)loadResourcesFromPath:(NSString*)path {
    NSError* error = nil;
    NSData* data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    if (!data) throw failure(@"Cannot read graphics resources JSON", error);
    id raw = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![raw isKindOfClass:[NSDictionary class]])
        throw failure(@"Invalid graphics resources JSON", error);
    NSDictionary* dict = raw;
    if ([dict[@"vertex"] isKindOfClass:[NSDictionary class]]) {
        _vertexResources = loadStageResources(_device, _queue, dict[@"vertex"]);
    }
    if ([dict[@"fragment"] isKindOfClass:[NSDictionary class]]) {
        _fragmentResources = loadStageResources(_device, _queue, dict[@"fragment"]);
    }
}

- (ReflectedArgBufferLayout)vertexArgLayout {
    return _vertexArgLayout;
}

- (ReflectedArgBufferLayout)fragmentArgLayout {
    return _fragmentArgLayout;
}

- (void)setArgLayoutsForPipeline:(NSString*)pipeId
                          vertex:(ReflectedArgBufferLayout)vertexLayout
                        fragment:(ReflectedArgBufferLayout)fragmentLayout {
    if (!pipeId) return;
    std::string key(pipeId.UTF8String);
    _pipelineVertexArgLayouts[key] = std::move(vertexLayout);
    _pipelineFragmentArgLayouts[key] = std::move(fragmentLayout);
}

- (void)encode:(id<MTLCommandBuffer>)command pass:(MTLRenderPassDescriptor*)pass {
    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw failure(@"Cannot create a Metal render encoder");
    if (self.depthState) {
        [encoder setDepthStencilState:self.depthState];
    }
    [encoder setCullMode:MTLCullModeNone];
    [encoder setDepthClipMode:MTLDepthClipModeClip];
    id<MTLTexture> texture = pass.colorAttachments[0].texture;
    if (!texture || !texture.width || !texture.height) throw failure(@"Framebuffer dimensions must be positive");
    std::vector<uint8_t> frameBytes;
    bool vertexHasFrame = false;
    bool fragmentHasFrame = false;
    NSUInteger argBufBindIndex = kIRArgumentBufferBindPoint;
    if (self.frameLayout) {
        double elapsed = self.hasFixedTime ? self.fixedTime :
            (self.live ? std::chrono::duration<double>(std::chrono::steady_clock::now() - _started).count() : 0);
        if (!std::isfinite(elapsed) || elapsed < 0 || elapsed > 3.4028234663852886e38)
            throw failure(@"Invalid elapsed frame time");
        frameBytes.assign([self.frameLayout[@"size"] unsignedIntegerValue], 0);
        NSArray* fields = self.frameLayout[@"fields"];
        float time = static_cast<float>(elapsed);
        float resolution[] = {static_cast<float>(texture.width), static_cast<float>(texture.height)};
        std::memcpy(frameBytes.data() + [fields[0][@"offset"] unsignedIntegerValue], &time, sizeof(time));
        std::memcpy(frameBytes.data() + [fields[1][@"offset"] unsignedIntegerValue], resolution, sizeof(resolution));
        argBufBindIndex = [self.frameLayout[@"bindings"][@"metal"][@"buffer_index"] unsignedIntegerValue];
        NSArray* stages = self.frameLayout[@"stages"];
        vertexHasFrame = [stages containsObject:@"vertex"];
        fragmentHasFrame = [stages containsObject:@"fragment"];
    }

    std::vector<uint8_t> vsArgBytes = packTopLevelArgBuffer(
        vertexHasFrame ? frameBytes.data() : nullptr,
        vertexHasFrame ? frameBytes.size() : 0,
        _vertexResources,
        _vertexArgLayout);
    if (!vsArgBytes.empty()) {
        [encoder setVertexBytes:vsArgBytes.data() length:vsArgBytes.size() atIndex:argBufBindIndex];
    }
    if (_vertexResources.srvUavTable != nil) {
        [encoder useResource:_vertexResources.srvUavTable usage:MTLResourceUsageRead stage:MTLRenderStageVertex];
    }
    if (_vertexResources.samplerTable != nil) {
        [encoder useResource:_vertexResources.samplerTable usage:MTLResourceUsageRead stage:MTLRenderStageVertex];
    }
    for (const auto& buf : _vertexResources.buffers) {
        MTLResourceUsage usage = buf.isUAV ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
        [encoder useResource:buf.buffer usage:usage stage:MTLRenderStageVertex];
    }
    for (const auto& tex : _vertexResources.textures) {
        [encoder useResource:tex.texture usage:MTLResourceUsageRead stage:MTLRenderStageVertex];
    }

    std::vector<uint8_t> fsArgBytes = packTopLevelArgBuffer(
        fragmentHasFrame ? frameBytes.data() : nullptr,
        fragmentHasFrame ? frameBytes.size() : 0,
        _fragmentResources,
        _fragmentArgLayout);
    if (!fsArgBytes.empty()) {
        [encoder setFragmentBytes:fsArgBytes.data() length:fsArgBytes.size() atIndex:argBufBindIndex];
    }
    if (_fragmentResources.srvUavTable != nil) {
        [encoder useResource:_fragmentResources.srvUavTable usage:MTLResourceUsageRead stage:MTLRenderStageFragment];
    }
    if (_fragmentResources.samplerTable != nil) {
        [encoder useResource:_fragmentResources.samplerTable usage:MTLResourceUsageRead stage:MTLRenderStageFragment];
    }
    for (const auto& buf : _fragmentResources.buffers) {
        MTLResourceUsage usage = buf.isUAV ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
        [encoder useResource:buf.buffer usage:usage stage:MTLRenderStageFragment];
    }
    for (const auto& tex : _fragmentResources.textures) {
        [encoder useResource:tex.texture usage:MTLResourceUsageRead stage:MTLRenderStageFragment];
    }
    [encoder setViewport:(MTLViewport){0, 0, static_cast<double>(texture.width),
        static_cast<double>(texture.height), 0, 1}];
    if (self.vertexBuffer) {
        [encoder setVertexBuffer:self.vertexBuffer offset:0 atIndex:kIRVertexBufferBindPoint];
    }

    if (!self.draws || self.draws.count == 0) {
        [encoder setRenderPipelineState:self.pipeline];
        if (self.isIndexed && self.indexBuffer) {
            MTLIndexType type = [self.indexType isEqualToString:@"uint16"] ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
            IRRuntimeDrawIndexedPrimitives(encoder, MTLPrimitiveTypeTriangle, self.indexCount,
                                           type, self.indexBuffer, 0);
        } else {
            IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle, uint64_t{0},
                                   static_cast<uint64_t>(self.vertexCount));
        }
    } else {
        id<MTLRenderPipelineState> currentPipeline = nil;
        id<MTLDepthStencilState> currentDepthState = nil;
        for (NSDictionary* draw in self.draws) {
            NSString* pipeId = draw[@"pipeline"];
            id<MTLRenderPipelineState> nextPipeline = (self.pipelines && pipeId) ? self.pipelines[pipeId] : self.pipeline;
            if (!nextPipeline) nextPipeline = self.pipeline;
            if (nextPipeline != currentPipeline) {
                [encoder setRenderPipelineState:nextPipeline];
                currentPipeline = nextPipeline;
            }
            id<MTLDepthStencilState> nextDepth = (self.pipelineDepthStates && pipeId) ? self.pipelineDepthStates[pipeId] : self.depthState;
            if (!nextDepth) nextDepth = self.depthState;
            if (nextDepth && nextDepth != currentDepthState) {
                [encoder setDepthStencilState:nextDepth];
                currentDepthState = nextDepth;
            }

            std::vector<uint8_t> rawBytes;
            id rawParams = draw[@"parameters"];
            if ([rawParams isKindOfClass:[NSData class]]) {
                NSData* d = rawParams;
                const auto* ptr = static_cast<const uint8_t*>(d.bytes);
                rawBytes.assign(ptr, ptr + d.length);
            } else if ([rawParams isKindOfClass:[NSDictionary class]] && [rawParams[@"data"] isKindOfClass:[NSArray class]]) {
                NSArray* dataArr = rawParams[@"data"];
                rawBytes.reserve(dataArr.count);
                for (NSNumber* num in dataArr) {
                    rawBytes.push_back(static_cast<uint8_t>([num unsignedIntValue] & 0xFF));
                }
            } else if ([draw[@"constant_size"] unsignedIntegerValue] > 0) {
                rawBytes.assign([draw[@"constant_size"] unsignedIntegerValue], 0);
            }

            if (!rawBytes.empty()) {
                NSUInteger cOff = [draw[@"constant_offset"] unsignedIntegerValue];
                std::vector<uint8_t> fullConst(cOff + rawBytes.size(), 0);
                std::memcpy(fullConst.data() + cOff, rawBytes.data(), rawBytes.size());

                const ReflectedArgBufferLayout* activeVsLayout = &_vertexArgLayout;
                const ReflectedArgBufferLayout* activeFsLayout = &_fragmentArgLayout;
                if (pipeId) {
                    std::string key(pipeId.UTF8String);
                    auto vsIt = _pipelineVertexArgLayouts.find(key);
                    if (vsIt != _pipelineVertexArgLayouts.end()) activeVsLayout = &vsIt->second;
                    auto fsIt = _pipelineFragmentArgLayouts.find(key);
                    if (fsIt != _pipelineFragmentArgLayouts.end()) activeFsLayout = &fsIt->second;
                }
                auto bindVsTopLevel = [&](const uint8_t* ptr, size_t len) {
                    auto packed = packTopLevelArgBuffer(ptr, len, _vertexResources, *activeVsLayout);
                    if (!packed.empty())
                        [encoder setVertexBytes:packed.data() length:packed.size() atIndex:kIRArgumentBufferBindPoint];
                };
                auto bindFsTopLevel = [&](const uint8_t* ptr, size_t len) {
                    auto packed = packTopLevelArgBuffer(ptr, len, _fragmentResources, *activeFsLayout);
                    if (!packed.empty())
                        [encoder setFragmentBytes:packed.data() length:packed.size() atIndex:kIRArgumentBufferBindPoint];
                };

                id pipeParams = (self.pipelineParameters && pipeId) ? self.pipelineParameters[pipeId] : nil;
                if ([pipeParams isKindOfClass:[NSArray class]]) {
                    NSArray* paramBlocks = pipeParams;
                    std::vector<uint8_t> vsArg, fsArg;
                    for (NSDictionary* block in paramBlocks) {
                        NSUInteger bOff = [block[@"bindings"][@"vulkan"][@"offset"] unsignedIntegerValue];
                        NSUInteger bSize = [block[@"size"] unsignedIntegerValue];
                        NSArray* stages = block[@"stages"];
                        auto appendSlice = [&](std::vector<uint8_t>& dst) {
                            size_t prev = dst.size();
                            dst.resize(prev + bSize, 0);
                            if (bOff < fullConst.size()) {
                                size_t avail = std::min<size_t>(bSize, fullConst.size() - bOff);
                                std::memcpy(dst.data() + prev, fullConst.data() + bOff, avail);
                            }
                        };
                        if ([stages containsObject:@"vertex"]) appendSlice(vsArg);
                        if ([stages containsObject:@"fragment"]) appendSlice(fsArg);
                    }
                    if (!vsArg.empty()) bindVsTopLevel(vsArg.data(), vsArg.size());
                    if (!fsArg.empty()) bindFsTopLevel(fsArg.data(), fsArg.size());
                } else if ([pipeParams isKindOfClass:[NSDictionary class]]) {
                    NSDictionary* block = pipeParams;
                    NSUInteger bOff = [block[@"bindings"][@"vulkan"][@"offset"] unsignedIntegerValue];
                    NSUInteger bSize = [block[@"size"] unsignedIntegerValue] ?: rawBytes.size();
                    std::vector<uint8_t> slice(bSize, 0);
                    if (bOff < fullConst.size()) {
                        size_t avail = std::min<size_t>(bSize, fullConst.size() - bOff);
                        std::memcpy(slice.data(), fullConst.data() + bOff, avail);
                    }
                    NSArray* stages = block[@"stages"];
                    if (!stages || [stages containsObject:@"vertex"])
                        bindVsTopLevel(slice.data(), slice.size());
                    if (!stages || [stages containsObject:@"fragment"])
                        bindFsTopLevel(slice.data(), slice.size());
                } else {
                    bindVsTopLevel(rawBytes.data(), rawBytes.size());
                    bindFsTopLevel(rawBytes.data(), rawBytes.size());
                }
            }
            NSString* drawType = draw[@"type"] ?: @"standard";
            if ([drawType isEqualToString:@"indexed"]) {
                uint64_t indexCount = [draw[@"index_count"] unsignedLongLongValue];
                uint64_t instanceCount = [draw[@"instance_count"] unsignedLongLongValue] ?: 1;
                uint64_t firstIndex = [draw[@"first_index"] unsignedLongLongValue];
                int64_t vertexOffset = [draw[@"vertex_offset"] longLongValue];
                uint64_t firstInstance = [draw[@"first_instance"] unsignedLongLongValue];
                id<MTLBuffer> indexBuffer = draw[@"index_buffer"] ?: self.indexBuffer;
                if (indexBuffer) {
                    NSString* idxTypeStr = draw[@"index_type"] ?: self.indexType ?: @"uint16";
                    bool isU16 = [idxTypeStr isEqualToString:@"uint16"] || [idxTypeStr isEqualToString:@"u16"];
                    MTLIndexType idxType = isU16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
                    NSUInteger elemBytes = isU16 ? 2 : 4;
                    IRRuntimeDrawIndexedPrimitives(encoder, MTLPrimitiveTypeTriangle, indexCount,
                                                   idxType, indexBuffer, firstIndex * elemBytes,
                                                   instanceCount, vertexOffset, firstInstance);
                } else {
                    IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle,
                                           firstIndex, indexCount, instanceCount, firstInstance);
                }
            } else {
                uint64_t vCount = [draw[@"vertex_count"] unsignedLongLongValue] ?: self.vertexCount;
                uint64_t firstVertex = [draw[@"first_vertex"] unsignedLongLongValue];
                uint64_t instanceCount = [draw[@"instance_count"] unsignedLongLongValue] ?: 1;
                uint64_t firstInstance = [draw[@"first_instance"] unsignedLongLongValue];
                IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle,
                                       firstVertex, vCount, instanceCount, firstInstance);
            }
        }
    }
    [encoder endEncoding];
}

- (void)drawInMTKView:(MTKView*)view {
    @autoreleasepool {
        MTLRenderPassDescriptor* pass = view.currentRenderPassDescriptor;
        id<CAMetalDrawable> drawable = view.currentDrawable;
        if (!pass || !drawable) return;
        try {
            id<MTLCommandBuffer> command = [self.queue commandBuffer];
            if (!command) throw failure(@"Cannot allocate a Metal command buffer");
            [self encode:command pass:pass];
            [command presentDrawable:drawable];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw failure(@"Metal rendering failed", command.error);
            self.frames += 1;
            if (self.limit && self.frames >= self.limit) stopPreview();
        } catch (const std::exception& error) {
            std::cerr << "mlir-gpu-metal-view: " << error.what() << "\n";
            exitStatus = 1;
            stopPreview();
        }
    }
}
- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {}
- (void)windowWillClose:(NSNotification*)notification { stopPreview(); }
@end

@interface PreviewView : MTKView
@end
@implementation PreviewView
- (BOOL)acceptsFirstResponder { return YES; }
- (void)keyDown:(NSEvent*)event {
    if (event.keyCode == 53) stopPreview();
    else [super keyDown:event];
}
@end

static void readback(GraphicsRenderer* renderer, NSUInteger width, NSUInteger height, const std::string& output) {
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [renderer.device newTextureWithDescriptor:descriptor];
    NSUInteger stride = (width * 4 + 255) & ~static_cast<NSUInteger>(255);
    id<MTLBuffer> buffer = [renderer.device newBufferWithLength:stride * height options:MTLResourceStorageModeShared];
    if (!texture || !buffer) throw failure(@"Cannot allocate offscreen Metal resources");
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.02, 0.02, 0.04, 1.0);

    if (renderer.hasDepth) {
        MTLTextureDescriptor* depthDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:width height:height mipmapped:NO];
        depthDesc.storageMode = MTLStorageModePrivate;
        depthDesc.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> depthTexture = [renderer.device newTextureWithDescriptor:depthDesc];
        if (!depthTexture) throw failure(@"Cannot allocate offscreen depth texture");

        pass.depthAttachment.texture = depthTexture;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
        pass.depthAttachment.clearDepth = 1.0;
    }

    id<MTLCommandBuffer> command = [renderer.queue commandBuffer];
    if (!command) throw failure(@"Cannot allocate an offscreen command buffer");
    [renderer encode:command pass:pass];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!blit) throw failure(@"Cannot allocate a readback encoder");
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
        sourceSize:MTLSizeMake(width, height, 1) toBuffer:buffer destinationOffset:0
        destinationBytesPerRow:stride destinationBytesPerImage:stride * height];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) throw failure(@"Metal readback failed", command.error);
    const uint8_t* source = static_cast<const uint8_t*>(buffer.contents);
    if (!source) throw failure(@"Metal readback buffer is not CPU accessible");
    std::vector<uint8_t> rgba(width * height * 4);
    for (NSUInteger y = 0; y < height; ++y) {
        for (NSUInteger x = 0; x < width; ++x) {
            NSUInteger target = (y * width + x) * 4, pixel = y * stride + x * 4;
            rgba[target] = source[pixel + 2];
            rgba[target + 1] = source[pixel + 1];
            rgba[target + 2] = source[pixel];
            rgba[target + 3] = source[pixel + 3];
        }
    }
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open readback output");
    file.write(reinterpret_cast<const char*>(rgba.data()), rgba.size());
    file.close();
    if (!file) throw std::runtime_error("Cannot write readback output");
    NSError* error = nil;
    NSData* report = [NSJSONSerialization dataWithJSONObject:@{
        @"device": renderer.device.name, @"width": @(width), @"height": @(height)
    } options:0 error:&error];
    if (!report) throw failure(@"Cannot encode render report", error);
    std::cout.write(static_cast<const char*>(report.bytes), report.length);
    std::cout << "\n";
}

static std::array<NSUInteger, 3> parseDimTriple(id value, NSString* label) {
    if (![value isKindOfClass:[NSArray class]] || [value count] != 3) {
        throw failure([NSString stringWithFormat:@"%@ must be an array of 3 positive integers", label]);
    }
    NSArray* array = value;
    std::array<NSUInteger, 3> dims{};
    for (NSUInteger i = 0; i < 3; ++i) {
        id item = array[i];
        if (![item isKindOfClass:[NSNumber class]]) {
            throw failure([NSString stringWithFormat:@"%@ dimensions must be integers", label]);
        }
        NSInteger signedVal = [item integerValue];
        if (signedVal < 1) {
            throw failure([NSString stringWithFormat:@"%@ dimensions must be positive", label]);
        }
        dims[i] = static_cast<NSUInteger>(signedVal);
    }
    return dims;
}

static void runComputeDispatch(const std::string& metallibPath, const std::string& entryPoint,
                               const std::string& specPath) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw failure(@"No Metal device is available");
    if (device.argumentBuffersSupport != MTLArgumentBuffersTier2)
        throw failure(@"Converted DXIL compute execution requires Argument Buffers Tier 2");

    NSError* error = nil;
    NSData* specData = [NSData dataWithContentsOfFile:@(specPath.c_str()) options:0 error:&error];
    if (!specData) throw failure(@"Cannot read compute dispatch spec", error);
    id rawSpec = [NSJSONSerialization JSONObjectWithData:specData options:0 error:&error];
    if (![rawSpec isKindOfClass:[NSDictionary class]])
        throw failure(@"Invalid compute dispatch spec JSON", error);
    NSDictionary* spec = rawSpec;

    std::array<NSUInteger, 3> workgroupSize = parseDimTriple(spec[@"workgroup_size"], @"workgroup_size");
    std::array<NSUInteger, 3> grid = parseDimTriple(spec[@"grid"], @"grid");

    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(metallibPath.c_str())] error:&error];
    if (!library) throw failure(@"Cannot load converted compute metallib", error);
    id<MTLFunction> computeFunction = [library newFunctionWithName:@(entryPoint.c_str())];
    if (!computeFunction)
        throw failure(@"The converted compute library does not contain the requested entry point");

    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = computeFunction;
    MTLAutoreleasedComputePipelineReflection reflection = nil;
    MTLPipelineOption options = MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithDescriptor:descriptor
                                              options:options
                                           reflection:&reflection
                                                error:&error];
    if (!pipeline) throw failure(@"Cannot create Metal compute pipeline state", error);

    NSUInteger threadsPerGroup = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    if (threadsPerGroup > pipeline.maxTotalThreadsPerThreadgroup) {
        throw failure(@"Compute workgroup_size exceeds Metal pipeline maxTotalThreadsPerThreadgroup");
    }

    std::vector<uint8_t> rootConstants;
    id rootConstantsPathObj = spec[@"root_constants_path"];
    if ([rootConstantsPathObj isKindOfClass:[NSString class]] && [rootConstantsPathObj length] > 0) {
        NSString* rcPath = rootConstantsPathObj;
        NSData* rcData = [NSData dataWithContentsOfFile:rcPath options:0 error:&error];
        if (!rcData) throw failure(@"Cannot read compute root constants file", error);
        const auto* rcPtr = static_cast<const uint8_t*>(rcData.bytes);
        rootConstants.assign(rcPtr, rcPtr + rcData.length);
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) throw failure(@"Cannot create a Metal command queue");

    if (![spec[@"buffers"] isKindOfClass:[NSArray class]])
        throw failure(@"Compute dispatch spec 'buffers' must be an array");
    MetalStageResources stageResources = loadStageResources(device, queue, spec);

    ReflectedArgBufferLayout reflectedLayout;
    if (reflection != nil) {
        reflectedLayout = extractReflectedArgBufferLayout(reflection.bindings);
    }
    std::vector<uint8_t> topLevelArgBytes = packTopLevelArgBuffer(
        rootConstants.empty() ? nullptr : rootConstants.data(),
        rootConstants.size(),
        stageResources,
        reflectedLayout);
    id<MTLBuffer> topLevelArgBuffer = nil;
    if (!topLevelArgBytes.empty()) {
        topLevelArgBuffer = [device newBufferWithBytes:topLevelArgBytes.data()
                                                length:topLevelArgBytes.size()
                                               options:MTLResourceStorageModeShared];
        if (!topLevelArgBuffer)
            throw failure(@"Cannot allocate Metal top-level argument buffer");
    }

    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) throw failure(@"Cannot allocate a Metal compute command buffer");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) throw failure(@"Cannot create a Metal compute command encoder");

    [encoder setComputePipelineState:pipeline];
    if (topLevelArgBuffer != nil) {
        [encoder setBuffer:topLevelArgBuffer offset:0 atIndex:kIRArgumentBufferBindPoint];
    }
    if (stageResources.srvUavTable != nil) {
        [encoder useResource:stageResources.srvUavTable usage:MTLResourceUsageRead];
    }
    if (stageResources.samplerTable != nil) {
        [encoder useResource:stageResources.samplerTable usage:MTLResourceUsageRead];
    }
    for (const auto& buf : stageResources.buffers) {
        MTLResourceUsage usage = buf.isUAV ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                                           : MTLResourceUsageRead;
        [encoder useResource:buf.buffer usage:usage];
    }
    for (const auto& tex : stageResources.textures) {
        [encoder useResource:tex.texture usage:MTLResourceUsageRead];
    }

    [encoder dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
            threadsPerThreadgroup:MTLSizeMake(workgroupSize[0], workgroupSize[1], workgroupSize[2])];
    [encoder endEncoding];

    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
        throw failure(@"Metal compute dispatch failed", command.error);

    for (const auto& buf : stageResources.buffers) {
        if (!buf.isUAV || buf.outputPath.empty()) continue;
        const auto* source = static_cast<const char*>(buf.buffer.contents);
        if (!source) throw failure(@"Metal compute buffer is not CPU accessible");
        std::ofstream out(buf.outputPath, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot open compute readback output: " + buf.name);
        out.write(source, static_cast<std::streamsize>(buf.size));
        out.close();
        if (!out) throw std::runtime_error("Cannot write compute readback output: " + buf.name);
    }

    NSData* report = [NSJSONSerialization dataWithJSONObject:@{
        @"device": device.name
    } options:0 error:&error];
    if (!report) throw failure(@"Cannot encode compute dispatch report", error);
    std::cout.write(static_cast<const char*>(report.bytes), static_cast<std::streamsize>(report.length));
    std::cout << "\n" << std::flush;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        @try {
            try {
                std::map<std::string, std::string> arguments;
                MetalDepthSettings depthSettings;
                for (int i = 1; i < argc; ++i) {
                    std::string option = argv[i];
                    if (option == "-frames") option = "--frames";
                    if (option == "--depth") {
                        depthSettings.enabled = true;
                        if (i + 1 < argc && (std::string(argv[i + 1]) == "1" || std::string(argv[i + 1]) == "true")) {
                            ++i;
                        }
                        continue;
                    }
                    if (i + 1 >= argc) throw std::runtime_error("Expected unique --option value pairs");
                    std::string val = argv[++i];
                    if (!arguments.emplace(option, val).second)
                        throw std::runtime_error("Expected unique --option value pairs");
                }
                for (const auto& pair : arguments) {
                    if (pair.first != "--vertex" && pair.first != "--fragment" &&
                        pair.first != "--vertex-entry" && pair.first != "--fragment-entry" &&
                        pair.first != "--width" && pair.first != "--height" &&
                        pair.first != "--frames" && pair.first != "--readback" &&
                        pair.first != "--vertex-count" && pair.first != "--frame-layout" &&
                        pair.first != "--time" &&
                        pair.first != "--input-layout" && pair.first != "--vertex-buffer" &&
                        pair.first != "--index-buffer" && pair.first != "--index-type" &&
                        pair.first != "--index-count" &&
                        pair.first != "--compute" && pair.first != "--compute-entry" &&
                        pair.first != "--compute-spec" && pair.first != "--multi-spec" &&
                        pair.first != "--resources" &&
                        pair.first != "--depth-compare" && pair.first != "--depth-write" &&
                        pair.first != "--depth-format")
                        throw std::runtime_error("Unknown option: " + pair.first);
                }
                if (arguments.count("--depth-compare")) {
                    depthSettings.compareFunction = parseMetalCompareFunction(arguments.at("--depth-compare"));
                }
                if (arguments.count("--depth-write")) {
                    std::string w = arguments.at("--depth-write");
                    depthSettings.writeEnabled = (w == "1" || w == "true");
                }
                if (arguments.count("--depth-format")) {
                    std::string fmt = arguments.at("--depth-format");
                    if (fmt != "d32_sfloat") throw std::runtime_error("Unsupported depth format: " + fmt);
                }
                auto required = [&](const char* key) -> const std::string& {
                    auto found = arguments.find(key);
                    if (found == arguments.end() || found->second.empty())
                        throw std::runtime_error(std::string("Missing ") + key);
                    return found->second;
                };
                if (arguments.count("--compute") || arguments.count("--compute-entry") ||
                    arguments.count("--compute-spec")) {
                    if (arguments.count("--vertex") || arguments.count("--fragment") ||
                        arguments.count("--vertex-entry") || arguments.count("--fragment-entry") ||
                        arguments.count("--width") || arguments.count("--height") ||
                        arguments.count("--frames") || arguments.count("--readback") ||
                        arguments.count("--vertex-count") || arguments.count("--frame-layout") ||
                        arguments.count("--time") || arguments.count("--input-layout") ||
                        arguments.count("--vertex-buffer") || arguments.count("--index-buffer") ||
                        arguments.count("--index-type") || arguments.count("--index-count") ||
                        arguments.count("--multi-spec") || arguments.count("--resources"))
                        throw std::runtime_error("Compute dispatch options cannot be combined with graphics options");
                    runComputeDispatch(
                        required("--compute"),
                        required("--compute-entry"),
                        required("--compute-spec"));
                    return 0;
                }
                NSUInteger width = number(required("--width"), 4096);
                NSUInteger height = number(required("--height"), 4096);
                if (!width || !height) throw std::runtime_error("Dimensions must be positive");
                NSDictionary* layout = arguments.count("--frame-layout") ? frameLayout(arguments.at("--frame-layout")) : nil;
                double fixedTime = arguments.count("--time") ? seconds(arguments.at("--time")) : 0;
                NSDictionary* inputLayoutDict = nil;
                if (arguments.count("--input-layout")) {
                    NSData* data = [NSData dataWithContentsOfFile:@(arguments.at("--input-layout").c_str())];
                    if (data) {
                        id parsedJson = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                        if ([parsedJson isKindOfClass:[NSDictionary class]]) {
                            NSDictionary* dict = parsedJson;
                            if (dict[@"input_layout"] && [dict[@"input_layout"] isKindOfClass:[NSDictionary class]]) {
                                inputLayoutDict = dict[@"input_layout"];
                            } else {
                                inputLayoutDict = dict;
                            }
                        }
                    }
                }
                GraphicsRenderer* renderer = [[GraphicsRenderer alloc]
                    initWithVertex:@(required("--vertex").c_str()) vertexEntry:@(required("--vertex-entry").c_str())
                    fragment:@(required("--fragment").c_str()) fragmentEntry:@(required("--fragment-entry").c_str())
                    inputLayout:inputLayoutDict
                  depthSettings:depthSettings];
                if (arguments.count("--index-count")) {
                    renderer.isIndexed = YES;
                    renderer.indexCount = number(arguments.at("--index-count"), UINT32_MAX);
                    renderer.indexType = arguments.count("--index-type") ? @(arguments.at("--index-type").c_str()) : @"uint16";
                    renderer.vertexCount = renderer.indexCount;
                } else {
                    renderer.vertexCount = number(required("--vertex-count"), UINT32_MAX);
                    if (renderer.vertexCount % 3) throw std::runtime_error("Triangle-list vertex count must be a multiple of three");
                }
                if (arguments.count("--vertex-buffer")) {
                    NSData* vbData = [NSData dataWithContentsOfFile:@(arguments.at("--vertex-buffer").c_str())];
                    if (vbData && vbData.length > 0) {
                        renderer.vertexBuffer = [renderer.device newBufferWithBytes:vbData.bytes
                                                                             length:vbData.length
                                                                            options:MTLResourceStorageModeShared];
                    }
                }
                if (arguments.count("--index-buffer")) {
                    NSData* ibData = [NSData dataWithContentsOfFile:@(arguments.at("--index-buffer").c_str())];
                    if (ibData && ibData.length > 0) {
                        renderer.indexBuffer = [renderer.device newBufferWithBytes:ibData.bytes
                                                                            length:ibData.length
                                                                           options:MTLResourceStorageModeShared];
                    }
                }
                if (arguments.count("--resources")) {
                    [renderer loadResourcesFromPath:@(arguments.at("--resources").c_str())];
                }
                renderer.limit = arguments.count("--frames") ? number(arguments.at("--frames"), NSUIntegerMax) : 0;
                renderer.frameLayout = layout;
                renderer.live = !arguments.count("--readback");
                renderer.hasFixedTime = arguments.count("--time");
                renderer.fixedTime = fixedTime;
                if (arguments.count("--multi-spec")) {
                    NSError* specErr = nil;
                    NSData* specData = [NSData dataWithContentsOfFile:@(arguments.at("--multi-spec").c_str()) options:0 error:&specErr];
                    if (!specData) throw failure(@"Cannot read Metal multi-spec file", specErr);
                    NSDictionary* specDict = [NSJSONSerialization JSONObjectWithData:specData options:0 error:&specErr];
                    if (![specDict isKindOfClass:[NSDictionary class]]) throw failure(@"Invalid Metal multi-spec JSON", specErr);
                    NSDictionary* pipesSpec = specDict[@"pipelines"];
                    NSMutableDictionary<NSString*, id<MTLRenderPipelineState>>* builtPipes = [NSMutableDictionary new];
                    NSMutableDictionary<NSString*, id<MTLDepthStencilState>>* builtDepths = [NSMutableDictionary new];
                    NSMutableDictionary<NSString*, id>* builtParams = [NSMutableDictionary new];
                    if ([pipesSpec isKindOfClass:[NSDictionary class]]) {
                        for (NSString* pId in pipesSpec) {
                            NSDictionary* pCfg = pipesSpec[pId];
                            NSDictionary* pLayout = ([pCfg[@"input_layout"] isKindOfClass:[NSDictionary class]]) ? pCfg[@"input_layout"] : inputLayoutDict;
                            MetalDepthSettings pDepth = depthSettings;
                            NSDictionary* dsCfg = pCfg[@"depth_stencil"];
                            if ([dsCfg isKindOfClass:[NSDictionary class]]) {
                                bool dsTest = dsCfg[@"depth_test_enable"] ? [dsCfg[@"depth_test_enable"] boolValue] : true;
                                bool dsWrite = dsCfg[@"depth_write_enable"] ? [dsCfg[@"depth_write_enable"] boolValue] : true;
                                std::string cmpStr = dsCfg[@"depth_compare_op"] ? [dsCfg[@"depth_compare_op"] UTF8String] : "less_equal";
                                pDepth.compareFunction = dsTest ? parseMetalCompareFunction(cmpStr) : MTLCompareFunctionAlways;
                                pDepth.writeEnabled = dsTest && dsWrite;
                            }
                            GraphicsRenderer* sub = [[GraphicsRenderer alloc]
                                initWithVertex:pCfg[@"vertex"] vertexEntry:pCfg[@"vertex_entry"]
                                      fragment:pCfg[@"fragment"] fragmentEntry:pCfg[@"fragment_entry"]
                                   inputLayout:pLayout
                                 depthSettings:pDepth];
                            builtPipes[pId] = sub.pipeline;
                            [renderer setArgLayoutsForPipeline:pId
                                                        vertex:[sub vertexArgLayout]
                                                      fragment:[sub fragmentArgLayout]];
                            if (sub.depthState) builtDepths[pId] = sub.depthState;
                            if (pCfg[@"parameters"] && pCfg[@"parameters"] != [NSNull null]) {
                                builtParams[pId] = pCfg[@"parameters"];
                            }
                        }
                    }
                    renderer.pipelines = builtPipes;
                    renderer.pipelineDepthStates = builtDepths;
                    renderer.pipelineParameters = builtParams;
                    if ([specDict[@"draws"] isKindOfClass:[NSArray class]]) {
                        renderer.draws = specDict[@"draws"];
                    }
                }
                if (arguments.count("--readback")) {
                    readback(renderer, width, height, arguments.at("--readback"));
                    return 0;
                }
                NSApplication* app = [NSApplication sharedApplication];
                [app setActivationPolicy:NSApplicationActivationPolicyRegular];
                NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, width, height)
                    styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                              NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                    backing:NSBackingStoreBuffered defer:NO];
                window.title = @"mlir-gpu | DXIL to Metal | Esc to close";
                window.releasedWhenClosed = NO;
                PreviewView* view = [[PreviewView alloc] initWithFrame:NSMakeRect(0, 0, width, height) device:renderer.device];
                view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
                view.autoResizeDrawable = YES;
                if (renderer.hasDepth) {
                    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
                    view.clearDepth = 1.0;
                }
                CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
                if (!colorSpace) throw failure(@"Cannot create the canonical presentation color space");
                view.colorspace = colorSpace;
                CGColorSpaceRelease(colorSpace);
                view.clearColor = MTLClearColorMake(0.02, 0.02, 0.04, 1.0);
                view.delegate = renderer;
                window.delegate = renderer;
                window.contentView = view;
                [window makeFirstResponder:view];
                [window center];
                [window makeKeyAndOrderFront:nil];
                [app activateIgnoringOtherApps:YES];
                std::cout << "DXIL-to-Metal viewer ready on " << renderer.device.name.UTF8String
                          << ". Close the window or press Esc to exit.\n" << std::flush;
                [app run];
                view.paused = YES;
                std::cout << "Presented " << renderer.frames << " Metal frames.\n";
                return exitStatus;
            } catch (const std::exception& error) {
                std::cerr << "mlir-gpu-metal-view: " << error.what() << "\n";
                return 1;
            }
        } @catch (NSException* exception) {
            std::cerr << "mlir-gpu-metal-view: " << exception.reason.UTF8String << "\n";
            return 1;
        }
    }
}
