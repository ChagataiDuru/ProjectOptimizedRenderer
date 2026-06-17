#pragma once

#include "core/RenderFramePacket.h"
#include "core/RenderScenePacket.h"
#include "core/Renderer.h"
#include "core/RendererOverlay.h"
#include "resource/Model.h"

#include <cstdint>
#include <memory>
#include <string>

class Swapchain;
class VulkanContext;
class Window;

enum class RendererResult {
    Success,
    Skipped,
    Error,
};

struct RendererCreateInfo {
    uint32_t width = 1280;
    uint32_t height = 720;
    const char* title = "ProjectOptimizedRenderer";
};

class RendererInstance {
public:
    RendererInstance();
    ~RendererInstance();

    RendererInstance(const RendererInstance&) = delete;
    RendererInstance& operator=(const RendererInstance&) = delete;

    static RendererResult create(const RendererCreateInfo& info, RendererInstance& outInstance);

    RendererResult create(const RendererCreateInfo& info);
    void shutdown();

    RendererResult attachOverlay(RendererOverlay* overlay);
    void detachOverlay();

    RendererResult resize(uint32_t width, uint32_t height);
    RendererResult uploadModelResources(const Model& model);
    RendererResult submitScene(RenderScenePacket scene);
    RendererResult renderFrame(const RenderFramePacket& packet);

    void requestScreenshot(const std::string& filename = "");
    RendererResult loadHdrPanorama(const std::string& path);
    RendererResult setAntiAliasingSettings(const AntiAliasingSettings& settings);

    const Renderer::RenderStats& getRenderStats() const;
    const GPUTimer& getGPUTimer() const;
    const ShadowDebugInfo& getShadowDebugInfo() const;
    const AntiAliasingStatus& getAntiAliasingStatus() const;
    const AntiAliasingSettings& getAntiAliasingSettings() const;

    void* getWindowHandle() const;
    void getWindowExtent(uint32_t& width, uint32_t& height) const;

    const std::string& getLastError() const { return m_lastError; }
    bool isCreated() const { return m_renderer != nullptr; }

private:
    template <typename Fn>
    RendererResult guarded(Fn&& fn);

    std::unique_ptr<Window> m_window;
    std::unique_ptr<VulkanContext> m_context;
    std::unique_ptr<Swapchain> m_swapchain;
    std::unique_ptr<Renderer> m_renderer;
    RendererOverlay* m_overlay = nullptr;
    std::string m_lastError;
    Renderer::RenderStats m_emptyStats{};
    GPUTimer* m_emptyTimer = nullptr;
};
