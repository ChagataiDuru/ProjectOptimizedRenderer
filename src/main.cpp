#include "core/VulkanContext.h"
#include "core/Device.h"
#include "core/Window.h"
#include "core/Swapchain.h"
#include "core/Renderer.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>

int main()
{
    auto logger = spdlog::stdout_color_mt("POR");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    spdlog::info("ProjectOptimizedRenderer starting");

    try {
        Window window(1280, 720, "ProjectOptimizedRenderer");
        window.init();

        VulkanContext vulkanContext;
        vulkanContext.init();

        uint32_t w, h;
        window.getExtent(w, h);

        Swapchain swapchain(vulkanContext);
        swapchain.init(window.getHandle(), w, h);

        Device device(vulkanContext);

        Renderer renderer(vulkanContext, swapchain);
        renderer.init();

        spdlog::info("Entering render loop");

        while (!window.shouldClose()) {
            window.pollEvents();
            renderer.beginFrame();
            renderer.endFrame();
        }

        renderer.shutdown();
        swapchain.shutdown();
        vulkanContext.shutdown();
        window.shutdown();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }

    return 0;
}
