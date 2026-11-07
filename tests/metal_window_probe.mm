#define main gfxir_preview_main
#include "../native/metal_viewer.mm"
#undef main
#include <thread>

static void events() {
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
}

static NSDictionary* sample(GraphicsRenderer* renderer, MTKView* view) {
    MTLRenderPassDescriptor* pass = view.currentRenderPassDescriptor;
    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!pass || !drawable) throw failure(@"No live drawable for window probe");
    id<MTLTexture> texture = drawable.texture;
    NSUInteger stride = ((texture.width * 4 + 255) / 256) * 256;
    id<MTLBuffer> buffer = [renderer.device newBufferWithLength:stride * texture.height
                                                     options:MTLResourceStorageModeShared];
    if (!buffer) throw failure(@"Cannot allocate window probe readback");
    id<MTLCommandBuffer> command = [renderer.queue commandBuffer];
    if (!command) throw failure(@"Cannot allocate window probe command");
    [renderer encode:command pass:pass];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!blit) throw failure(@"Cannot allocate window probe blit");
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(texture.width, texture.height, 1) toBuffer:buffer destinationOffset:0
    destinationBytesPerRow:stride destinationBytesPerImage:stride * texture.height];
    [blit endEncoding];
    [command presentDrawable:drawable];
    [command commit];
    [command waitUntilCompleted];
    if (command.error) throw failure(@"Window probe command failed", command.error);
    const uint8_t* pixel = static_cast<const uint8_t*>(buffer.contents) +
                          (texture.height / 2) * stride + (texture.width / 2) * 4;
    return @{@"width": @(texture.width), @"height": @(texture.height),
             @"points_width": @(view.bounds.size.width), @"scale": @(view.window.backingScaleFactor),
             @"rgba": @[@(pixel[2]), @(pixel[1]), @(pixel[0]), @(pixel[3])]};
}

@interface WindowProbe : NSObject <MTKViewDelegate>
@property(nonatomic, strong) GraphicsRenderer* renderer;
@property(nonatomic, copy) NSDictionary* report;
@end

@implementation WindowProbe
- (void)drawInMTKView:(MTKView*)view {
    self.report = sample(self.renderer, view);
}
- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {}
@end

static NSDictionary* capture(WindowProbe* probe, MTKView* view) {
    probe.report = nil;
    [view draw];
    if (!probe.report) throw failure(@"Window probe did not draw");
    return probe.report;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 6) throw failure(@"Window probe needs two libraries/entries and a frame layout");
            NSApplication* app = [NSApplication sharedApplication];
            [app setActivationPolicy:NSApplicationActivationPolicyRegular];
            GraphicsRenderer* renderer = [[GraphicsRenderer alloc]
                initWithVertex:@(argv[1]) vertexEntry:@(argv[2]) fragment:@(argv[3]) fragmentEntry:@(argv[4])];
            renderer.vertexCount = 3;
            renderer.frameLayout = frameLayout(argv[5]);
            renderer.live = YES;
            NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 37, 23)
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskResizable
                backing:NSBackingStoreBuffered defer:NO];
            window.releasedWhenClosed = NO;
            MTKView* view = [[MTKView alloc] initWithFrame:NSMakeRect(0, 0, 37, 23) device:renderer.device];
            view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
            view.framebufferOnly = NO;
            view.autoResizeDrawable = YES;
            view.paused = YES;
            WindowProbe* probe = [[WindowProbe alloc] init];
            probe.renderer = renderer;
            view.delegate = probe;
            window.contentView = view;
            [window makeKeyAndOrderFront:nil];
            events();
            NSDictionary* first = capture(probe, view);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            events();
            NSDictionary* advanced = capture(probe, view);
            renderer.hasFixedTime = YES;
            renderer.fixedTime = 1.5;
            [window setContentSize:NSMakeSize(53, 31)];
            events();
            NSDictionary* resized = capture(probe, view);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            events();
            NSDictionary* frozen = capture(probe, view);
            [window close];
            NSError* error = nil;
            NSData* report = [NSJSONSerialization dataWithJSONObject:@[first, advanced, resized, frozen]
                                                             options:0 error:&error];
            if (!report) throw failure(@"Cannot serialize window probe report", error);
            std::cout.write(static_cast<const char*>(report.bytes), report.length);
            std::cout << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "Metal window probe: " << error.what() << "\n";
            return 1;
        }
    }
}
