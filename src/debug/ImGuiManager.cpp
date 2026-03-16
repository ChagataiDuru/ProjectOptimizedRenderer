#include "debug/ImGuiManager.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

// volk.h is already included transitively (via VulkanContext.h).
// IMGUI_IMPL_VULKAN_USE_VOLK is defined project-wide via CMake so both
// imgui_impl_vulkan.h and .cpp resolve Vulkan symbols through volk.
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

// SDL3 event type for the processEvent cast.
#include <SDL3/SDL_events.h>

// ── Lifecycle ─────────────────────────────────────────────────────────────────

ImGuiManager::ImGuiManager(VulkanContext& ctx, Swapchain& swapchain)
    : m_ctx(ctx)
    , m_swapchain(swapchain)
{
}

ImGuiManager::~ImGuiManager()
{
    shutdown();
}

void ImGuiManager::init(void* sdlWindow)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // ── SDL3 platform backend ─────────────────────────────────────────────────
    ImGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(sdlWindow));

    // ── Vulkan renderer backend (dynamic rendering) ───────────────────────────
    const VkFormat colorFormat = m_swapchain.getFormat();

    // VkPipelineRenderingCreateInfoKHR tells ImGui which color format its
    // internal pipeline must be compatible with. Must match the swapchain format.
    const VkPipelineRenderingCreateInfoKHR pipelineRenderingCI{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &colorFormat,
    };

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance                = m_ctx.getInstance();
    initInfo.PhysicalDevice          = m_ctx.getPhysicalDevice();
    initInfo.Device                  = m_ctx.getDevice();
    initInfo.QueueFamily             = m_ctx.getGraphicsQueueFamily();
    initInfo.Queue                   = m_ctx.getGraphicsQueue();
    initInfo.RenderPass              = VK_NULL_HANDLE;   // dynamic rendering: no render pass
    initInfo.MinImageCount           = 2;
    initInfo.ImageCount              = m_swapchain.getImageCount();
    initInfo.MSAASamples             = VK_SAMPLE_COUNT_1_BIT;
    initInfo.DescriptorPoolSize      = 2;                // backend creates its own pool
    initInfo.UseDynamicRendering     = true;
    initInfo.PipelineRenderingCreateInfo = pipelineRenderingCI;

    if (!ImGui_ImplVulkan_Init(&initInfo))
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");

    m_initialized = true;
    spdlog::info("ImGuiManager initialized ({} panels registered so far)", m_panels.size());
}

void ImGuiManager::shutdown()
{
    if (!m_initialized) return;
    vkDeviceWaitIdle(m_ctx.getDevice());
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    m_initialized = false;
}

// ── Panel registry ────────────────────────────────────────────────────────────

void ImGuiManager::registerPanel(DebugPanel panel)
{
    m_panels.push_back(std::move(panel));
}

// ── Per-frame API ─────────────────────────────────────────────────────────────

void ImGuiManager::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    for (auto& panel : m_panels) {
        if (ImGui::Begin(panel.name.c_str()))
            panel.draw();
        ImGui::End();
    }
}

void ImGuiManager::recordRenderPass(VkCommandBuffer cmd,
                                     VkImageView     swapchainView,
                                     VkExtent2D      extent)
{
    // Finalize draw lists (must happen after all ImGui:: calls but before render).
    ImGui::Render();

    // Overlay pass: LOAD_OP_LOAD preserves the PBR scene beneath the UI.
    const VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = swapchainView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
    };

    const VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = extent },
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void ImGuiManager::processEvent(const void* sdlEvent)
{
    ImGui_ImplSDL3_ProcessEvent(static_cast<const SDL_Event*>(sdlEvent));
}
