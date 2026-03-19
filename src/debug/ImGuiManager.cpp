#include "debug/ImGuiManager.h"

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

// volk.h is already included transitively (via VulkanContext.h).
// IMGUI_IMPL_VULKAN_USE_VOLK is defined project-wide via CMake so both
// imgui_impl_vulkan.h and .cpp resolve Vulkan symbols through volk.
#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder API (stable but not in public imgui.h)
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <IconsFontAwesome6.h>

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
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Load fonts before the backends are initialised — the atlas is built once
    // during ImGui_ImplVulkan_Init and cannot be rebuilt without recreation.
    loadFonts();

    // ── SDL3 platform backend ─────────────────────────────────────────────────
    ImGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(sdlWindow));

    // ── Vulkan renderer backend (dynamic rendering) ───────────────────────────
    const VkFormat colorFormat = m_swapchain.getFormat();

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
    initInfo.RenderPass              = VK_NULL_HANDLE;
    initInfo.MinImageCount           = 2;
    initInfo.ImageCount              = m_swapchain.getImageCount();
    initInfo.MSAASamples             = VK_SAMPLE_COUNT_1_BIT;
    initInfo.DescriptorPoolSize      = 2;
    initInfo.UseDynamicRendering     = true;
    initInfo.PipelineRenderingCreateInfo = pipelineRenderingCI;

    if (!ImGui_ImplVulkan_Init(&initInfo))
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");

    // Apply initial theme after backends are ready.
    applyTheme();

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

// ── Font loading ──────────────────────────────────────────────────────────────

void ImGuiManager::loadFonts()
{
    ImGuiIO& io = ImGui::GetIO();

    // ASSET_DIR is defined project-wide via CMake target_compile_definitions.
    const std::string fontDir = std::string(ASSET_DIR) + "/fonts/";
    const float       fontSize = 18.0f;

    // Primary font: Roboto Regular.
    ImFont* roboto = io.Fonts->AddFontFromFileTTF(
        (fontDir + "Roboto-Regular.ttf").c_str(), fontSize);

    if (!roboto) {
        spdlog::warn("Failed to load Roboto-Regular.ttf — falling back to default font");
        io.Fonts->AddFontDefault();
    }

    // Merge FontAwesome 6 Solid icons into the Roboto atlas.
    // After this merge, any ImGui::Text() call can embed ICON_FA_* constants
    // directly: e.g. ImGui::Text(ICON_FA_SUN " Light").
    ImFontConfig iconConfig;
    iconConfig.MergeMode        = true;      // Append glyphs into the previous font
    iconConfig.PixelSnapH       = true;
    iconConfig.GlyphMinAdvanceX = fontSize;  // Keep icons monospace-width
    iconConfig.GlyphOffset      = ImVec2(0.0f, 2.0f);  // Align icon baseline with text

    // FA6 glyph range (matches ICON_MIN_FA / ICON_MAX_FA in IconsFontAwesome6.h)
    static const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };

    ImFont* icons = io.Fonts->AddFontFromFileTTF(
        (fontDir + "fa-solid-900.ttf").c_str(), fontSize, &iconConfig, iconRanges);

    if (!icons) {
        spdlog::warn("Failed to load fa-solid-900.ttf — icons will render as empty boxes");
    }

    spdlog::info("Fonts loaded: Roboto {}px + FontAwesome 6 icons", fontSize);
}

// ── Theme ─────────────────────────────────────────────────────────────────────

void ImGuiManager::setTheme(Theme theme)
{
    m_theme = theme;
    applyTheme();
}

void ImGuiManager::setBackgroundAlpha(float alpha)
{
    m_bgAlpha = glm::clamp(alpha, 0.0f, 1.0f);
    applyTheme();
}

void ImGuiManager::applyTheme()
{
    ImGuiStyle& style = ImGui::GetStyle();

    if (m_theme == Theme::Dark) {
        ImGui::StyleColorsDark();

        ImVec4* c = style.Colors;
        c[ImGuiCol_WindowBg]           = ImVec4(0.10f, 0.10f, 0.12f, m_bgAlpha);
        c[ImGuiCol_PopupBg]            = ImVec4(0.10f, 0.10f, 0.12f, 0.96f);
        c[ImGuiCol_Border]             = ImVec4(0.30f, 0.30f, 0.35f, 0.50f);
        c[ImGuiCol_FrameBg]            = ImVec4(0.16f, 0.16f, 0.19f, 1.00f);
        c[ImGuiCol_FrameBgHovered]     = ImVec4(0.22f, 0.22f, 0.26f, 1.00f);
        c[ImGuiCol_FrameBgActive]      = ImVec4(0.28f, 0.28f, 0.33f, 1.00f);
        c[ImGuiCol_TitleBg]            = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        c[ImGuiCol_TitleBgActive]      = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
        c[ImGuiCol_MenuBarBg]          = ImVec4(0.10f, 0.10f, 0.12f, m_bgAlpha);
        c[ImGuiCol_Tab]                = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
        c[ImGuiCol_TabSelected]        = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
        c[ImGuiCol_TabHovered]         = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        c[ImGuiCol_DockingPreview]     = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
        c[ImGuiCol_DockingEmptyBg]     = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        c[ImGuiCol_Header]             = ImVec4(0.20f, 0.20f, 0.24f, 1.00f);
        c[ImGuiCol_HeaderHovered]      = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        c[ImGuiCol_HeaderActive]       = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        c[ImGuiCol_Separator]          = ImVec4(0.30f, 0.30f, 0.35f, 0.50f);
        c[ImGuiCol_Button]             = ImVec4(0.20f, 0.20f, 0.24f, 1.00f);
        c[ImGuiCol_ButtonHovered]      = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        c[ImGuiCol_ButtonActive]       = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        c[ImGuiCol_SliderGrab]         = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        c[ImGuiCol_SliderGrabActive]   = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        c[ImGuiCol_CheckMark]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        c[ImGuiCol_PlotHistogram]      = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        c[ImGuiCol_TextSelectedBg]     = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);

    } else {
        ImGui::StyleColorsLight();

        ImVec4* c = style.Colors;
        c[ImGuiCol_WindowBg]       = ImVec4(0.94f, 0.94f, 0.94f, m_bgAlpha);
        c[ImGuiCol_PopupBg]        = ImVec4(0.98f, 0.98f, 0.98f, 0.96f);
        c[ImGuiCol_MenuBarBg]      = ImVec4(0.86f, 0.86f, 0.86f, m_bgAlpha);
        c[ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
        c[ImGuiCol_DockingEmptyBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    }

    // Common geometry/spacing settings for both themes.
    style.WindowRounding    = 6.0f;
    style.FrameRounding     = 4.0f;
    style.GrabRounding      = 4.0f;
    style.TabRounding       = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.WindowBorderSize  = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.PopupBorderSize   = 1.0f;
    style.WindowPadding     = ImVec2(10.0f, 10.0f);
    style.FramePadding      = ImVec2(8.0f, 4.0f);
    style.ItemSpacing       = ImVec2(8.0f, 6.0f);
    style.IndentSpacing     = 20.0f;
    style.ScrollbarSize     = 14.0f;
    style.GrabMinSize       = 12.0f;
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
        ImGui::Render();
        return;
    }

    // ── Dockspace over entire viewport ──────────────────────────────────────
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags dockspaceFlags =
        ImGuiWindowFlags_NoDocking             |
        ImGuiWindowFlags_NoTitleBar            |
        ImGuiWindowFlags_NoCollapse            |
        ImGuiWindowFlags_NoResize              |
        ImGuiWindowFlags_NoMove                |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus            |
        ImGuiWindowFlags_NoBackground          |
        ImGuiWindowFlags_MenuBar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("DockSpaceWindow", nullptr, dockspaceFlags);
    ImGui::PopStyleVar(3);

    ImGuiID dockspaceId = ImGui::GetID("MainDockSpace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);

    if (m_firstFrame) {
        m_firstFrame = false;
        buildDefaultLayout(dockspaceId);
    }

    // ── Menu bar ────────────────────────────────────────────────────────────
    if (ImGui::BeginMenuBar()) {

        // ── File ────────────────────────────────────────────────────────────
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Load glTF...", nullptr, false, false)) {
                // Placeholder — file dialog in a future phase
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_CAMERA " Screenshot", "F2")) {
                // Handled in main.cpp via F2; this is a mirror entry
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_RIGHT_FROM_BRACKET " Quit", "Esc")) {
                m_quitRequested = true;
            }
            ImGui::EndMenu();
        }

        // ── View ─────────────────────────────────────────────────────────────
        if (ImGui::BeginMenu("View")) {
            for (auto& panel : m_panels) {
                ImGui::MenuItem(panel.name.c_str(), nullptr, &panel.visible);
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_EXPAND " Fullscreen", "F11"))
                toggleVisible();
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_ARROWS_ROTATE " Reset Layout"))
                m_firstFrame = true;
            ImGui::EndMenu();
        }

        // ── Rendering ────────────────────────────────────────────────────────
        if (ImGui::BeginMenu("Rendering")) {
            if (m_renderToggles.wireframe)
                ImGui::MenuItem(ICON_FA_BORDER_ALL " Wireframe", nullptr,
                                m_renderToggles.wireframe);
            if (m_renderToggles.showNormals)
                ImGui::MenuItem(ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT " Show Normals", nullptr,
                                m_renderToggles.showNormals);
            ImGui::Separator();
            ImGui::MenuItem(ICON_FA_WAND_MAGIC_SPARKLES " SMAA (Phase 3)", nullptr, false, false);
            ImGui::MenuItem(ICON_FA_CLOUD_SUN " Shadows (Phase 4)", nullptr, false, false);
            ImGui::MenuItem(ICON_FA_GRIP " VRS (Phase 5)", nullptr, false, false);
            ImGui::EndMenu();
        }

        // ── Settings ─────────────────────────────────────────────────────────
        if (ImGui::BeginMenu("Settings")) {
            if (ImGui::MenuItem("Dark Theme",  nullptr, m_theme == Theme::Dark))
                setTheme(Theme::Dark);
            if (ImGui::MenuItem("Light Theme", nullptr, m_theme == Theme::Light))
                setTheme(Theme::Light);
            ImGui::Separator();
            float alpha = m_bgAlpha;
            if (ImGui::SliderFloat("Panel Opacity", &alpha, 0.3f, 1.0f, "%.2f"))
                setBackgroundAlpha(alpha);
            ImGui::EndMenu();
        }

        // ── Help ─────────────────────────────────────────────────────────────
        if (ImGui::BeginMenu("Help")) {
            ImGui::Text("ProjectOptimizedRenderer");
            ImGui::Separator();
            ImGui::Text("F1:   Toggle mouse capture");
            ImGui::Text("F2:   Screenshot");
            ImGui::Text("F11:  Toggle UI overlay");
            ImGui::Text("WASD + Mouse: FPS camera");
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
    if (!m_visible) return;

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
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    // Split: right column (25%) — capture both halves; dockCenter is the left remainder.
    // DockBuilderSplitNode returns out_id_at_dir; the opposite side comes via out_id_at_opposite_dir.
    ImGuiID dockRight;
    ImGuiID dockCenter;
    ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Right, 0.25f, &dockRight, &dockCenter);

    // Split left remainder into bottom strip (25% height) and main viewport.
    ImGuiID dockBottom;
    ImGuiID dockMain;
    ImGui::DockBuilderSplitNode(dockCenter, ImGuiDir_Down, 0.25f, &dockBottom, &dockMain);

    // Split right column into top and bottom halves (panels tab together within each).
    ImGuiID dockRightTop;
    ImGuiID dockRightBottom;
    ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.5f, &dockRightBottom, &dockRightTop);

    bool rightTopUsed = false;
    for (const auto& panel : m_panels) {
        ImGuiID targetNode;
        switch (panel.dockLocation) {
            case DockLocation::Right:
                targetNode    = rightTopUsed ? dockRightBottom : dockRightTop;
                rightTopUsed  = true;
                break;
            case DockLocation::Bottom:
                targetNode = dockBottom;
                break;
            case DockLocation::Floating:
            default:
                continue;
        }
        ImGui::DockBuilderDockWindow(panel.name.c_str(), targetNode);
    }

    ImGui::DockBuilderFinish(dockspaceId);
    spdlog::info("Default dock layout applied");
}
