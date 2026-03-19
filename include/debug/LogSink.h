#pragma once

#include <spdlog/sinks/base_sink.h>
#include <mutex>
#include <string>
#include <vector>

struct LogEntry {
    spdlog::level::level_enum level;
    std::string               message;  // Pre-formatted by spdlog (includes timestamp + level)
};

// Custom spdlog sink that stores the last N log entries in a ring buffer.
// The ImGui console panel reads from getEntries() on the main thread.
// All log calls in this project are on the main thread, so getEntries() needs
// no additional locking beyond what base_sink<std::mutex> provides for sink_it_.
class ImGuiLogSink : public spdlog::sinks::base_sink<std::mutex> {
public:
    explicit ImGuiLogSink(size_t maxEntries = 500);

    // Read-only access for ImGui rendering (call from main thread only).
    const std::vector<LogEntry>& getEntries() const { return m_entries; }

    // Total entries ever written — used by the console panel to detect new output
    // and auto-scroll to bottom.
    size_t getEntryCount() const { return m_entryCount; }

    void clear();

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;
    void flush_() override {}

private:
    std::vector<LogEntry> m_entries;
    size_t                m_maxEntries;
    size_t                m_entryCount = 0;
};
