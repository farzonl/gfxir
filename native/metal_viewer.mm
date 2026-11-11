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

@interface GraphicsRenderer : NSObject <MTKViewDelegate, NSWindowDelegate>
{
    std::chrono::steady_clock::time_point _started;
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
            vd.attributes[loc].format = mtlFmt;
            vd.attributes[loc].offset = offset;
            vd.attributes[loc].bufferIndex = binding;
            if (offset + attrSize > maxEndOffset) {
                maxEndOffset = offset + attrSize;
            }
        }
        NSArray* bindings = inputLayout[@"bindings"];
        if (bindings && bindings.count > 0) {
            for (NSDictionary* b in bindings) {
                NSUInteger binding = [b[@"binding"] unsignedIntegerValue];
                NSUInteger stride = [b[@"stride"] unsignedIntegerValue];
                vd.layouts[binding].stride = stride;
                vd.layouts[binding].stepFunction = MTLVertexStepFunctionPerVertex;
                vd.layouts[binding].stepRate = 1;
            }
        } else if (attrs && attrs.count > 0) {
            vd.layouts[0].stride = maxEndOffset;
            vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
            vd.layouts[0].stepRate = 1;
        }
        descriptor.vertexDescriptor = vd;
    }
    _pipeline = [_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!_pipeline) throw failure(@"Cannot create the converted DXIL render pipeline", error);

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
    if (self.frameLayout) {
        double elapsed = self.hasFixedTime ? self.fixedTime :
            (self.live ? std::chrono::duration<double>(std::chrono::steady_clock::now() - _started).count() : 0);
        if (!std::isfinite(elapsed) || elapsed < 0 || elapsed > 3.4028234663852886e38)
            throw failure(@"Invalid elapsed frame time");
        std::vector<uint8_t> bytes([self.frameLayout[@"size"] unsignedIntegerValue], 0);
        NSArray* fields = self.frameLayout[@"fields"];
        float time = static_cast<float>(elapsed);
        float resolution[] = {static_cast<float>(texture.width), static_cast<float>(texture.height)};
        std::memcpy(bytes.data() + [fields[0][@"offset"] unsignedIntegerValue], &time, sizeof(time));
        std::memcpy(bytes.data() + [fields[1][@"offset"] unsignedIntegerValue], resolution, sizeof(resolution));
        NSUInteger index = [self.frameLayout[@"bindings"][@"metal"][@"buffer_index"] unsignedIntegerValue];
        NSArray* stages = self.frameLayout[@"stages"];
        if ([stages containsObject:@"vertex"]) [encoder setVertexBytes:bytes.data() length:bytes.size() atIndex:index];
        if ([stages containsObject:@"fragment"]) [encoder setFragmentBytes:bytes.data() length:bytes.size() atIndex:index];
    }
    [encoder setViewport:(MTLViewport){0, 0, static_cast<double>(texture.width),
        static_cast<double>(texture.height), 0, 1}];
    if (self.vertexBuffer) {
        [encoder setVertexBuffer:self.vertexBuffer offset:0 atIndex:0];
    }

    if (!self.draws || self.draws.count == 0) {
        [encoder setRenderPipelineState:self.pipeline];
        if (self.isIndexed && self.indexBuffer) {
            MTLIndexType type = [self.indexType isEqualToString:@"uint16"] ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
            [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:self.indexCount
                                 indexType:type
                               indexBuffer:self.indexBuffer
                         indexBufferOffset:0];
        } else {
            IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle, uint64_t{0},
                                   static_cast<uint64_t>(self.vertexCount));
        }
    } else {
        id<MTLRenderPipelineState> currentPipeline = nil;
        for (NSDictionary* draw in self.draws) {
            NSString* pipeId = draw[@"pipeline"];
            id<MTLRenderPipelineState> nextPipeline = (self.pipelines && pipeId) ? self.pipelines[pipeId] : self.pipeline;
            if (!nextPipeline) nextPipeline = self.pipeline;
            if (nextPipeline != currentPipeline) {
                [encoder setRenderPipelineState:nextPipeline];
                currentPipeline = nextPipeline;
            }
            NSData* perDrawBytes = draw[@"parameters"];
            if (perDrawBytes && [perDrawBytes isKindOfClass:[NSData class]] && perDrawBytes.length > 0) {
                [encoder setVertexBytes:perDrawBytes.bytes length:perDrawBytes.length atIndex:3];
            } else if ([draw[@"constant_size"] unsignedIntegerValue] > 0) {
                NSUInteger sz = [draw[@"constant_size"] unsignedIntegerValue];
                std::vector<uint8_t> zeros(sz, 0);
                [encoder setVertexBytes:zeros.data() length:sz atIndex:3];
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

    struct MetalComputeBuffer {
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

    id rawBuffers = spec[@"buffers"];
    if (![rawBuffers isKindOfClass:[NSArray class]])
        throw failure(@"Compute dispatch spec 'buffers' must be an array");
    NSArray* bufferSpecs = rawBuffers;

    std::vector<MetalComputeBuffer> buffers;
    buffers.reserve(bufferSpecs.count);

    for (id item in bufferSpecs) {
        if (![item isKindOfClass:[NSDictionary class]])
            throw failure(@"Invalid buffer entry in compute dispatch spec");
        NSDictionary* dict = item;
        NSString* nameNs = dict[@"name"];
        NSString* accessNs = dict[@"access"];
        NSNumber* sizeNs = dict[@"size"];
        if (![nameNs isKindOfClass:[NSString class]] || ![accessNs isKindOfClass:[NSString class]] ||
            ![sizeNs isKindOfClass:[NSNumber class]])
            throw failure(@"Compute buffer entry is missing required fields");

        MetalComputeBuffer entry;
        entry.name = nameNs.UTF8String;
        entry.access = accessNs.UTF8String;
        if (entry.access != "read_only" && entry.access != "write_only" && entry.access != "read_write")
            throw std::runtime_error("Unsupported compute buffer access mode: " + entry.access);
        long long rawSize = [sizeNs longLongValue];
        if (rawSize <= 0)
            throw std::runtime_error("Compute buffer byte size must be positive: " + entry.name);
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
        if (!entry.buffer)
            throw failure(@"Cannot allocate Metal compute storage buffer");
        std::memset(entry.buffer.contents, 0, entry.alignedSize);

        id inputPathObj = dict[@"input_path"];
        if ([inputPathObj isKindOfClass:[NSString class]] && [inputPathObj length] > 0) {
            NSData* inData = [NSData dataWithContentsOfFile:inputPathObj options:0 error:&error];
            if (!inData) throw failure(@"Cannot read compute input buffer file", error);
            if (inData.length != entry.size)
                throw std::runtime_error("Compute input buffer file size does not match spec size: " + entry.name);
            std::memcpy(entry.buffer.contents, inData.bytes, entry.size);
        }

        buffers.push_back(std::move(entry));
    }

    std::stable_sort(buffers.begin(), buffers.end(),
                     [](const MetalComputeBuffer& a, const MetalComputeBuffer& b) {
                         if (a.set != b.set) return a.set < b.set;
                         return a.binding < b.binding;
                     });

    std::vector<size_t> uavIndices;
    std::vector<size_t> srvIndices;
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i].isUAV) uavIndices.push_back(i);
        else srvIndices.push_back(i);
    }

    size_t totalDescriptors = uavIndices.size() + srvIndices.size();
    id<MTLBuffer> descriptorTable = nil;
    if (totalDescriptors > 0) {
        size_t tableByteSize = totalDescriptors * sizeof(IRDescriptorTableEntry);
        descriptorTable = [device newBufferWithLength:tableByteSize options:MTLResourceStorageModeShared];
        if (!descriptorTable)
            throw failure(@"Cannot allocate Metal IRDescriptorTableEntry buffer");
        auto* entries = static_cast<IRDescriptorTableEntry*>(descriptorTable.contents);
        std::memset(entries, 0, tableByteSize);
        size_t slot = 0;
        for (size_t idx : uavIndices) {
            IRDescriptorTableSetBuffer(&entries[slot++], buffers[idx].buffer.gpuAddress, 0);
        }
        for (size_t idx : srvIndices) {
            IRDescriptorTableSetBuffer(&entries[slot++], buffers[idx].buffer.gpuAddress, 0);
        }
    }

    bool needsTopLevelArgBuffer = !rootConstants.empty() || totalDescriptors > 0;
    size_t reflectedSize = 0;
    size_t descriptorTableOffset = (rootConstants.size() + 7) & ~static_cast<size_t>(7);
    if (reflection != nil) {
        for (id<MTLBinding> binding in reflection.bindings) {
            if (binding.type == MTLBindingTypeBuffer && binding.index == kIRArgumentBufferBindPoint) {
                id untyped = binding;
                id<MTLBufferBinding> bufferBinding = untyped;
                reflectedSize = bufferBinding.bufferDataSize;
                MTLStructType* structType = bufferBinding.bufferStructType;
                if (structType != nil) {
                    for (MTLStructMember* member in structType.members) {
                        if (member.dataType == MTLDataTypePointer) {
                            descriptorTableOffset = member.offset;
                        }
                    }
                }
                break;
            }
        }
    }

    id<MTLBuffer> topLevelArgBuffer = nil;
    if (needsTopLevelArgBuffer) {
        size_t minSize = rootConstants.size();
        if (totalDescriptors > 0) {
            minSize = std::max<size_t>(minSize, descriptorTableOffset + sizeof(uint64_t));
        }
        size_t argBufSize = std::max<size_t>(std::max<size_t>(reflectedSize, minSize), 16);
        topLevelArgBuffer = [device newBufferWithLength:argBufSize options:MTLResourceStorageModeShared];
        if (!topLevelArgBuffer)
            throw failure(@"Cannot allocate Metal top-level argument buffer");
        auto* argBytes = static_cast<uint8_t*>(topLevelArgBuffer.contents);
        std::memset(argBytes, 0, argBufSize);
        if (!rootConstants.empty()) {
            std::memcpy(argBytes, rootConstants.data(), rootConstants.size());
        }
        if (totalDescriptors > 0 && descriptorTable != nil) {
            uint64_t tableAddress = descriptorTable.gpuAddress;
            std::memcpy(argBytes + descriptorTableOffset, &tableAddress, sizeof(tableAddress));
        }
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) throw failure(@"Cannot create a Metal command queue");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) throw failure(@"Cannot allocate a Metal compute command buffer");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) throw failure(@"Cannot create a Metal compute command encoder");

    [encoder setComputePipelineState:pipeline];
    if (topLevelArgBuffer != nil) {
        [encoder setBuffer:topLevelArgBuffer offset:0 atIndex:kIRArgumentBufferBindPoint];
    }
    if (descriptorTable != nil) {
        [encoder useResource:descriptorTable usage:MTLResourceUsageRead];
    }
    for (const auto& buf : buffers) {
        MTLResourceUsage usage = buf.isUAV ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                                           : MTLResourceUsageRead;
        [encoder useResource:buf.buffer usage:usage];
    }

    [encoder dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
            threadsPerThreadgroup:MTLSizeMake(workgroupSize[0], workgroupSize[1], workgroupSize[2])];
    [encoder endEncoding];

    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
        throw failure(@"Metal compute dispatch failed", command.error);

    for (const auto& buf : buffers) {
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
                        pair.first != "--compute-spec" &&
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
                        arguments.count("--index-type") || arguments.count("--index-count"))
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
                renderer.limit = arguments.count("--frames") ? number(arguments.at("--frames"), NSUIntegerMax) : 0;
                renderer.frameLayout = layout;
                renderer.live = !arguments.count("--readback");
                renderer.hasFixedTime = arguments.count("--time");
                renderer.fixedTime = fixedTime;
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
