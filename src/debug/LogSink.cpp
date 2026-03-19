#include "debug/LogSink.h"

ImGuiLogSink::ImGuiLogSink(size_t maxEntries)
    : m_maxEntries(maxEntries)
{
    m_entries.reserve(maxEntries);
}

void ImGuiLogSink::sink_it_(const spdlog::details::log_msg& msg)
{
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);

    LogEntry entry;
    entry.level   = msg.level;
    entry.message = fmt::to_string(formatted);

    // Ring buffer: wrap around once capacity is reached.
    if (m_entries.size() < m_maxEntries) {
        m_entries.push_back(std::move(entry));
    } else {
        m_entries[m_entryCount % m_maxEntries] = std::move(entry);
    }
    ++m_entryCount;
}

void ImGuiLogSink::clear()
{
    m_entries.clear();
    m_entryCount = 0;
}
