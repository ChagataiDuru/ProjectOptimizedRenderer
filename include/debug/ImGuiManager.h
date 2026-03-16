#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include <functional>
#include <string>
#include <vector>

// A named UI panel whose draw callback is invoked once per frame between
// ImGui::NewFrame() and ImGui::Render().
struct DebugPanel {
    std::string           name;
    std::function<void()> draw;
};

class ImGuiManager {
public:
    ImGuiManager(VulkanContext& ctx, Swapchain& swapchain);
    ~ImGuiManager();

    ImGuiManager(const ImGuiManager&)            = delete;
    ImGuiManager& operator=(const ImGuiManager&) = delete;

    // sdlWindow must be the SDL_Window* cast to void* (matches Window::getHandle()).
    void init(void* sdlWindow);
    void shutdown();

    void registerPanel(DebugPanel panel);

    // Call once per frame before renderer.beginFrame().
    // Starts a new ImGui frame and invokes all registered panel callbacks.
    void beginFrame();

    // Record the ImGui overlay pass into cmd using dynamic rendering (LOAD_OP_LOAD).
    // Call after the PBR vkCmdEndRendering and before the PRESENT_SRC transition.
    void recordRenderPass(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent);

    // Forward an SDL_Event to ImGui (pass pointer to the SDL_Event).
    void processEvent(const void* sdlEvent);

private:
    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;

    std::vector<DebugPanel> m_panels;
    bool                    m_initialized = false;
};
