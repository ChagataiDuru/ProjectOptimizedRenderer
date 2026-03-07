#include "core/VulkanContext.h"
#include "core/Device.h"

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
        VulkanContext vulkanContext;
        vulkanContext.init();

        Device device(vulkanContext);

        spdlog::info("ProjectOptimizedRenderer initialized successfully");

        vulkanContext.shutdown();
    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }

    return 0;
}
