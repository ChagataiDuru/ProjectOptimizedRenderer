#pragma once

#include "core/VulkanContext.h"
#include <string>
#include <unordered_map>
#include <vector>

// GPU timestamp profiler using Vulkan timestamp queries (core in Vulkan 1.3/1.4).
//
// Usage per frame:
//   timer.beginFrame(cmd);                    // reset query pool slots on GPU
//   timer.writeTimestamp(cmd, "Scene_Begin");
//   // ... record scene draw commands ...
//   timer.writeTimestamp(cmd, "Scene_End");
//   // ... submit + wait on fence ...
//   timer.collectResults();                   // read back previous frame's values
//   float ms = timer.getElapsedMs("Scene_Begin", "Scene_End");

class GPUTimer {
public:
    explicit GPUTimer(VulkanContext& ctx);
    ~GPUTimer();

    GPUTimer(const GPUTimer&)            = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    // Create the query pool. maxTimestamps = max writeTimestamp calls per frame.
    void init(uint32_t maxTimestamps = 32);
    void shutdown();

    // Reset query slots for the upcoming frame. Call at the top of command buffer recording.
    void beginFrame(VkCommandBuffer cmd);

    // Inject a timestamp at the current pipeline point.
    // Defaults to ALL_COMMANDS so the timestamp comes after all prior work.
    void writeTimestamp(VkCommandBuffer cmd, const std::string& name,
                        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    // Read the previous frame's timestamps from the query pool.
    // Call after the previous frame's in-flight fence has signaled.
    void collectResults();

    // Return elapsed milliseconds between two named timestamps.
    // Returns 0.0f if either name is missing or no results are available yet.
    float getElapsedMs(const std::string& beginName, const std::string& endName) const;

    bool isValid() const { return m_queryPool != VK_NULL_HANDLE; }

private:
    VulkanContext& m_ctx;
    VkQueryPool    m_queryPool      = VK_NULL_HANDLE;
    uint32_t       m_maxTimestamps  = 32;
    uint32_t       m_currentIndex   = 0;     // next free slot this frame
    float          m_timestampPeriod = 1.0f; // nanoseconds per GPU tick

    // name → query index within the pool (populated fresh each beginFrame)
    std::unordered_map<std::string, uint32_t> m_nameToIndex;
    // Raw 64-bit results from the previous completed frame
    std::vector<uint64_t> m_results;
    // How many slots were recorded in the frame that produced m_results
    uint32_t m_resultCount = 0;
    bool     m_hasResults  = false;
};
