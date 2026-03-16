#include "debug/GPUTimer.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

GPUTimer::GPUTimer(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

GPUTimer::~GPUTimer()
{
    shutdown();
}

void GPUTimer::init(uint32_t maxTimestamps)
{
    m_maxTimestamps = maxTimestamps;

    // Fetch the timestamp period (nanoseconds per GPU clock tick).
    // A value of 0 means the device does not support timestamp queries.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_ctx.getPhysicalDevice(), &props);

    if (props.limits.timestampPeriod == 0.0f) {
        spdlog::warn("GPUTimer: device does not support timestamp queries — profiling disabled");
        return;
    }
    m_timestampPeriod = props.limits.timestampPeriod;

    const VkQueryPoolCreateInfo poolCI{
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = m_maxTimestamps,
    };
    VK_CHECK(vkCreateQueryPool(m_ctx.getDevice(), &poolCI, nullptr, &m_queryPool));

    // Pre-allocate result buffer to avoid per-frame allocation.
    m_results.resize(m_maxTimestamps, 0);

    spdlog::info("GPUTimer initialized: {} slots, {:.2f} ns/tick",
                 m_maxTimestamps, m_timestampPeriod);
}

void GPUTimer::shutdown()
{
    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_ctx.getDevice(), m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }
    m_hasResults = false;
}

void GPUTimer::beginFrame(VkCommandBuffer cmd)
{
    if (m_queryPool == VK_NULL_HANDLE) return;

    // Save how many timestamps were written last frame so collectResults knows
    // the range to read (avoids reading unwritten slots).
    m_resultCount = m_currentIndex;

    // Reset all query slots at the start of the new frame.
    // This executes on the GPU timeline — safe to call even before the previous
    // frame's results have been read on the CPU (collectResults uses WAIT_BIT).
    vkCmdResetQueryPool(cmd, m_queryPool, 0, m_maxTimestamps);

    m_nameToIndex.clear();
    m_currentIndex = 0;
}

void GPUTimer::writeTimestamp(VkCommandBuffer cmd, const std::string& name,
                               VkPipelineStageFlags2 stage)
{
    if (m_queryPool == VK_NULL_HANDLE) return;
    if (m_currentIndex >= m_maxTimestamps) {
        spdlog::warn("GPUTimer: exceeded maxTimestamps ({}), ignoring '{}'",
                     m_maxTimestamps, name);
        return;
    }

    m_nameToIndex[name] = m_currentIndex;
    vkCmdWriteTimestamp2(cmd, stage, m_queryPool, m_currentIndex);
    ++m_currentIndex;
}

void GPUTimer::collectResults()
{
    if (m_queryPool == VK_NULL_HANDLE) return;
    if (m_resultCount == 0) return; // Nothing was written last frame

    // WAIT_BIT: block until all queries in [0, resultCount) are available.
    // Since this is called after the in-flight fence has signaled, the GPU has
    // already completed the frame that wrote these timestamps — the wait is instant.
    const VkResult res = vkGetQueryPoolResults(
        m_ctx.getDevice(),
        m_queryPool,
        0, m_resultCount,
        m_resultCount * sizeof(uint64_t),
        m_results.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (res == VK_SUCCESS) {
        m_hasResults = true;
    } else {
        spdlog::warn("GPUTimer: vkGetQueryPoolResults returned {}", static_cast<int>(res));
        m_hasResults = false;
    }
}

float GPUTimer::getElapsedMs(const std::string& beginName, const std::string& endName) const
{
    if (!m_hasResults) return 0.0f;

    const auto itBegin = m_nameToIndex.find(beginName);
    const auto itEnd   = m_nameToIndex.find(endName);

    if (itBegin == m_nameToIndex.end() || itEnd == m_nameToIndex.end()) return 0.0f;

    const uint32_t idxBegin = itBegin->second;
    const uint32_t idxEnd   = itEnd->second;

    if (idxBegin >= m_resultCount || idxEnd >= m_resultCount) return 0.0f;
    if (m_results[idxEnd] < m_results[idxBegin]) return 0.0f; // overflow guard

    const uint64_t ticks = m_results[idxEnd] - m_results[idxBegin];
    // timestampPeriod is in nanoseconds/tick; divide by 1e6 to get milliseconds
    return static_cast<float>(ticks) * m_timestampPeriod * 1e-6f;
}
