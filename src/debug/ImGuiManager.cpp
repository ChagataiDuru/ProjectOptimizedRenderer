#include "debug/ImGuiManager.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

// volk.h is already included transitively (via VulkanContext.h).
// IMGUI_IMPL_VULKAN_USE_VOLK is defined project-wide via CMake so both
// imgui_impl_vulkan.h and .cpp resolve Vulkan symbols through volk.
#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder API (stable but not in public imgui.h)
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
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;    // Enable docking

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

void ImGuiManager::registerPanel(const std::string& name, std::function<void()> drawFn,
                                  DockLocation dock)
{
    m_panels.push_back({ name, std::move(drawFn), true, dock });
    spdlog::debug("ImGui panel registered: '{}' (dock: {})", name,
                  dock == DockLocation::Right  ? "right" :
                  dock == DockLocation::Bottom ? "bottom" : "floating");
}

// ── Per-frame API ─────────────────────────────────────────────────────────────

void ImGuiManager::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    // Widget drawing happens in endFrame(). All frames call NewFrame regardless
    // of visibility so ImGui's internal state stays consistent.
}

void ImGuiManager::endFrame()
{
    if (!m_visible) {
        // Still call Render() to keep ImGui state consistent, but no widgets
        // were drawn. recordRenderPass will also skip the GPU pass.
        ImGui::Render();
        return;
    }

    // ── Dockspace over entire viewport ──────────────────────────────────────
    // An invisible full-screen host window that panels dock into.
    // The 3D scene renders behind it via the swapchain (LOAD_OP_LOAD).
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags dockspaceFlags =
        ImGuiWindowFlags_NoDocking         |
        ImGuiWindowFlags_NoTitleBar        |
        ImGuiWindowFlags_NoCollapse        |
        ImGuiWindowFlags_NoResize          |
        ImGuiWindowFlags_NoMove            |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus        |
        ImGuiWindowFlags_NoBackground      |   // Transparent — scene shows through
        ImGuiWindowFlags_MenuBar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("DockSpaceWindow", nullptr, dockspaceFlags);
    ImGui::PopStyleVar(3);

    ImGuiID dockspaceId = ImGui::GetID("MainDockSpace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);  // Central area is transparent

    // ── Default layout (first frame only) ───────────────────────────────────
    if (m_firstFrame) {
        m_firstFrame = false;
        buildDefaultLayout(dockspaceId);
    }

    // ── Menu bar ────────────────────────────────────────────────────────────
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            for (auto& panel : m_panels) {
                ImGui::MenuItem(panel.name.c_str(), nullptr, &panel.visible);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Reset Layout")) {
                m_firstFrame = true;  // Rebuild layout next frame
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Debug")) {
            // Future: wireframe toggle, culling mode, SMAA toggle, etc.
            ImGui::MenuItem("(Future debug options)", nullptr, false, false);
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    ImGui::End(); // DockSpaceWindow

    // ── Draw panels ─────────────────────────────────────────────────────────
    for (auto& panel : m_panels) {
        if (panel.visible) {
            ImGui::Begin(panel.name.c_str(), &panel.visible);
            panel.drawFn();
            ImGui::End();
        }
    }

    ImGui::Render();
}

void ImGuiManager::recordRenderPass(VkCommandBuffer cmd,
                                     VkImageView     swapchainView,
                                     VkExtent2D      extent)
{
    // Skip GPU commands when ImGui is hidden (F11 fullscreen mode).
    // ImGui::Render() was already called in endFrame() to keep state consistent,
    // but the draw data is empty so there is nothing to record.
    if (!m_visible) return;

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

// ── Default layout builder ────────────────────────────────────────────────────

void ImGuiManager::buildDefaultLayout(ImGuiID dockspaceId)
{
    // Clear any existing layout and start fresh.
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    // Split: right column (25% width)
    ImGuiID dockRight;
    ImGuiID dockMainAndBottom = ImGui::DockBuilderSplitNode(
        dockspaceId, ImGuiDir_Right, 0.25f, &dockRight, nullptr);

    // Split remaining: bottom row (25% height of remaining area)
    ImGuiID dockBottom;
    ImGuiID dockCenter;
    dockCenter = ImGui::DockBuilderSplitNode(
        dockMainAndBottom, ImGuiDir_Down, 0.25f, &dockBottom, nullptr);

    // Split right column into top and bottom halves
    ImGuiID dockRightTop;
    ImGuiID dockRightBottom;
    ImGui::DockBuilderSplitNode(
        dockRight, ImGuiDir_Down, 0.5f, &dockRightBottom, &dockRightTop);

    // Assign panels to dock nodes based on their DockLocation.
    // Multiple panels in the same node will automatically tab together.
    bool rightTopUsed = false;
    for (const auto& panel : m_panels) {
        ImGuiID targetNode;
        switch (panel.dockLocation) {
            case DockLocation::Right:
                // First Right panel goes to top half; subsequent ones go bottom (tabbed).
                targetNode = rightTopUsed ? dockRightBottom : dockRightTop;
                rightTopUsed = true;
                break;
            case DockLocation::Bottom:
                targetNode = dockBottom;
                break;
            case DockLocation::Floating:
            default:
                continue;  // Don't dock floating panels
        }
        ImGui::DockBuilderDockWindow(panel.name.c_str(), targetNode);
    }

    ImGui::DockBuilderFinish(dockspaceId);

    spdlog::info("Default dock layout applied");
}
