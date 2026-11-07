#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
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
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry;
- (void)encode:(id<MTLCommandBuffer>)command pass:(MTLRenderPassDescriptor*)pass;
@end

@implementation GraphicsRenderer
- (instancetype)initWithVertex:(NSString*)vertex vertexEntry:(NSString*)vertexEntry
                       fragment:(NSString*)fragment fragmentEntry:(NSString*)fragmentEntry {
    self = [super init];
    if (!self) throw failure(@"Cannot allocate Metal renderer");
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
    _pipeline = [_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!_pipeline) throw failure(@"Cannot create the converted DXIL render pipeline", error);
    _queue = [_device newCommandQueue];
    if (!_queue) throw failure(@"Cannot create a Metal command queue");
    _started = std::chrono::steady_clock::now();
    return self;
}

- (void)encode:(id<MTLCommandBuffer>)command pass:(MTLRenderPassDescriptor*)pass {
    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
    if (!encoder) throw failure(@"Cannot create a Metal render encoder");
    [encoder setRenderPipelineState:self.pipeline];
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
    IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle, uint64_t{0},
                           static_cast<uint64_t>(self.vertexCount));
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

int main(int argc, char** argv) {
    @autoreleasepool {
        @try {
            try {
                std::map<std::string, std::string> arguments;
                for (int i = 1; i < argc; i += 2) {
                    std::string option = argv[i];
                    if (option == "-frames") option = "--frames";
                    if (i + 1 >= argc || !arguments.emplace(option, argv[i + 1]).second)
                        throw std::runtime_error("Expected unique --option value pairs");
                }
                for (const auto& pair : arguments) {
                    if (pair.first != "--vertex" && pair.first != "--fragment" &&
                        pair.first != "--vertex-entry" && pair.first != "--fragment-entry" &&
                        pair.first != "--width" && pair.first != "--height" &&
                        pair.first != "--frames" && pair.first != "--readback" &&
                        pair.first != "--vertex-count" && pair.first != "--frame-layout" && pair.first != "--time")
                        throw std::runtime_error("Unknown option: " + pair.first);
                }
                auto required = [&](const char* key) -> const std::string& {
                    auto found = arguments.find(key);
                    if (found == arguments.end() || found->second.empty())
                        throw std::runtime_error(std::string("Missing ") + key);
                    return found->second;
                };
                NSUInteger width = number(required("--width"), 4096);
                NSUInteger height = number(required("--height"), 4096);
                if (!width || !height) throw std::runtime_error("Dimensions must be positive");
                NSDictionary* layout = arguments.count("--frame-layout") ? frameLayout(arguments.at("--frame-layout")) : nil;
                double fixedTime = arguments.count("--time") ? seconds(arguments.at("--time")) : 0;
                GraphicsRenderer* renderer = [[GraphicsRenderer alloc]
                    initWithVertex:@(required("--vertex").c_str()) vertexEntry:@(required("--vertex-entry").c_str())
                    fragment:@(required("--fragment").c_str()) fragmentEntry:@(required("--fragment-entry").c_str())];
                renderer.vertexCount = number(required("--vertex-count"), UINT32_MAX);
                if (renderer.vertexCount % 3) throw std::runtime_error("Triangle-list vertex count must be a multiple of three");
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
