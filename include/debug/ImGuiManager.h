#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include <functional>
#include <string>
#include <vector>

// Where a panel prefers to be docked in the default layout.
enum class DockLocation {
    Right,      // Right column (scene hierarchy, properties)
    Bottom,     // Bottom row (timing, stats, console)
    Floating,   // Not docked by default
};

// A named UI panel whose draw callback is invoked once per frame between
// ImGui::NewFrame() and ImGui::Render().
struct DebugPanel {
    std::string           name;
    std::function<void()> drawFn;
    bool                  visible      = true;
    DockLocation          dockLocation = DockLocation::Floating;
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

    void registerPanel(const std::string& name, std::function<void()> drawFn,
                       DockLocation dock = DockLocation::Floating);

    // Call once per frame before endFrame().
    // Starts a new ImGui frame (NewFrame). Widget drawing happens in endFrame().
    void beginFrame();

    // Builds the dockspace, draws all visible panels, and finalizes draw data
    // (ImGui::Render). Call after beginFrame() and before recordRenderPass().
    // When hidden, becomes a no-op except for ImGui::Render() to keep state consistent.
    void endFrame();

    // Record the ImGui overlay pass into cmd using dynamic rendering (LOAD_OP_LOAD).
    // Call after the PBR vkCmdEndRendering and before the PRESENT_SRC transition.
    // Skips all GPU commands when ImGui is hidden.
    void recordRenderPass(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent);

    // Forward an SDL_Event to ImGui (pass pointer to the SDL_Event).
    void processEvent(const void* sdlEvent);

    // Toggle entire ImGui visibility (F11 fullscreen mode).
    // When hidden, endFrame/recordRenderPass become no-ops for widget drawing.
    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const        { return m_visible; }
    void toggleVisible()          { m_visible = !m_visible; }

private:
    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;

    std::vector<DebugPanel> m_panels;
    bool                    m_initialized = false;
    bool                    m_visible     = true;
    bool                    m_firstFrame  = true;  // Applies default layout on first frame only

    void buildDefaultLayout(ImGuiID dockspaceId);
};
