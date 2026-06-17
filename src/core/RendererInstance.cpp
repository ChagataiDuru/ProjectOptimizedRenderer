#include "core/RendererInstance.h"

#include "core/Swapchain.h"
#include "core/VulkanContext.h"
#include "core/Window.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

RendererInstance::RendererInstance() = default;

RendererInstance::~RendererInstance()
{
    shutdown();
}

RendererResult RendererInstance::create(const RendererCreateInfo& info,
                                        RendererInstance& outInstance)
{
    return outInstance.create(info);
}

RendererResult RendererInstance::create(const RendererCreateInfo& info)
{
    return guarded([&]() {
        shutdown();

        m_window = std::make_unique<Window>(info.width, info.height, info.title);
        m_window->init();

        m_context = std::make_unique<VulkanContext>();
        m_context->init();

        uint32_t width = 0;
        uint32_t height = 0;
        m_window->getExtent(width, height);

        m_swapchain = std::make_unique<Swapchain>(*m_context);
        m_swapchain->init(m_window->getHandle(), width, height);

        m_renderer = std::make_unique<Renderer>(*m_context, *m_swapchain);
        m_renderer->init();
    });
}

void RendererInstance::shutdown()
{
    if (m_overlay) {
        m_overlay->shutdown();
        m_overlay = nullptr;
    }
    if (m_renderer) {
        m_renderer->shutdown();
        m_renderer.reset();
    }
    if (m_swapchain) {
        m_swapchain->shutdown();
        m_swapchain.reset();
    }
    if (m_context) {
        m_context->shutdown();
        m_context.reset();
    }
    if (m_window) {
        m_window->shutdown();
        m_window.reset();
    }
}

RendererResult RendererInstance::attachOverlay(RendererOverlay* overlay)
{
    return guarded([&]() {
        if (m_overlay && m_overlay != overlay) {
            m_overlay->shutdown();
        }
        m_overlay = overlay;
        if (m_overlay) {
            if (!m_context || !m_swapchain || !m_window) {
                throw std::runtime_error("RendererInstance::attachOverlay called before create");
            }
            m_overlay->init(RendererOverlayInitInfo{
                .context = m_context.get(),
                .swapchain = m_swapchain.get(),
                .platformWindow = m_window->getHandle(),
            });
        }
    });
}

void RendererInstance::detachOverlay()
{
    if (m_overlay) {
        m_overlay->shutdown();
        m_overlay = nullptr;
    }
}

RendererResult RendererInstance::resize(uint32_t width, uint32_t height)
{
    return guarded([&]() {
        if (!m_renderer) {
            throw std::runtime_error("RendererInstance::resize called before create");
        }
        m_renderer->resize(width, height);
    });
}

RendererResult RendererInstance::uploadModelResources(const Model& model)
{
    return guarded([&]() {
        if (!m_renderer) {
            throw std::runtime_error("RendererInstance::uploadModelResources called before create");
        }
        m_renderer->uploadModelResources(model);
    });
}

RendererResult RendererInstance::submitScene(RenderScenePacket scene)
{
    return guarded([&]() {
        if (!m_renderer) {
            throw std::runtime_error("RendererInstance::submitScene called before create");
        }
        m_renderer->submitScene(std::move(scene));
    });
}

RendererResult RendererInstance::renderFrame(const RenderFramePacket& packet)
{
    return guarded([&]() -> RendererResult {
        if (!m_renderer) {
            throw std::runtime_error("RendererInstance::renderFrame called before create");
        }
        if (!m_renderer->beginFrame()) {
            return RendererResult::Skipped;
        }
        m_renderer->submitFrame(packet);
        m_renderer->endFrame(m_overlay);
        return RendererResult::Success;
    });
}

void RendererInstance::requestScreenshot(const std::string& filename)
{
    if (m_renderer) {
        m_renderer->requestScreenshot(filename);
    }
}

RendererResult RendererInstance::loadHdrPanorama(const std::string& path)
{
    return guarded([&]() {
        if (!m_renderer) {
            throw std::runtime_error("RendererInstance::loadHdrPanorama called before create");
        }
        m_renderer->loadHdrPanorama(path);
    });
}

RendererResult RendererInstance::setAntiAliasingSettings(const AntiAliasingSettings& settings)
{
    return guarded([&]() {
        if (!m_renderer) {
            throw std::runtime_error("RendererInstance::setAntiAliasingSettings called before create");
        }
        m_renderer->setAntiAliasingSettings(settings);
    });
}

const Renderer::RenderStats& RendererInstance::getRenderStats() const
{
    return m_renderer ? m_renderer->getRenderStats() : m_emptyStats;
}

const GPUTimer& RendererInstance::getGPUTimer() const
{
    if (!m_renderer) {
        throw std::runtime_error("RendererInstance::getGPUTimer called before create");
    }
    return m_renderer->getGPUTimer();
}

const ShadowDebugInfo& RendererInstance::getShadowDebugInfo() const
{
    if (!m_renderer) {
        throw std::runtime_error("RendererInstance::getShadowDebugInfo called before create");
    }
    return m_renderer->getShadowDebugInfo();
}

const AntiAliasingStatus& RendererInstance::getAntiAliasingStatus() const
{
    if (!m_renderer) {
        throw std::runtime_error("RendererInstance::getAntiAliasingStatus called before create");
    }
    return m_renderer->getAntiAliasingStatus();
}

const AntiAliasingSettings& RendererInstance::getAntiAliasingSettings() const
{
    if (!m_renderer) {
        throw std::runtime_error("RendererInstance::getAntiAliasingSettings called before create");
    }
    return m_renderer->getAntiAliasingSettings();
}

void* RendererInstance::getWindowHandle() const
{
    return m_window ? m_window->getHandle() : nullptr;
}

void RendererInstance::getWindowExtent(uint32_t& width, uint32_t& height) const
{
    if (!m_window) {
        width = 0;
        height = 0;
        return;
    }
    m_window->getExtent(width, height);
}

template <typename Fn>
RendererResult RendererInstance::guarded(Fn&& fn)
{
    try {
        m_lastError.clear();
        if constexpr (std::is_same_v<decltype(fn()), RendererResult>) {
            return fn();
        } else {
            fn();
            return RendererResult::Success;
        }
    } catch (const std::exception& e) {
        m_lastError = e.what();
    } catch (...) {
        m_lastError = "Unknown renderer error";
    }
    return RendererResult::Error;
}
