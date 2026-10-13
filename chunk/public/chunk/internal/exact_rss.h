/*
 * Copyright 2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file exact_rss.h
 * @brief The process's resident size, EXACT: /proc/self/smaps_rollup.
 *
 * VmRSS (/proc/self/status), /proc/self/statm and getrusage()'s ru_maxrss all
 * read the kernel's RSS counters, which are kept per thread and folded into
 * the process total only every so many events. With many threads faulting
 * pages the total can trail reality by gigabytes. smaps_rollup walks the page
 * tables when it is read: exact, but the cost grows with the resident size
 * (tens of ms on a large process). Fine for traces and checkpoints; never on a
 * periodic hot path.
 */

#include <array>    // std::array (the smaps_rollup line buffer)
#include <atomic>   // std::atomic, std::memory_order
#include <chrono>   // std::chrono::milliseconds
#include <cstdio>   // std::FILE, std::fopen, std::fclose
#include <thread>   // std::thread, std::this_thread::sleep_for
#include <unistd.h> // sysconf(_SC_PAGESIZE)

namespace shrink_array
{

/// @brief Exact resident size in KiB (smaps_rollup "Rss:"); statm as a
///        fallback when smaps_rollup is missing; -1 if neither is readable.
inline long exact_rss_kib() noexcept
{
    if (std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r"))
    {
        std::array<char, 256> line{};
        long                  kib = -1;
        while (std::fgets(line.data(), static_cast<int>(line.size()), f) != nullptr)
        {
            if (std::sscanf(line.data(), "Rss: %ld kB", &kib) == 1)
            {
                break;
            }
        }
        (void)std::fclose(f); // read-only: nothing to flush, nothing to report
        if (kib >= 0)
        {
            return kib;
        }
    }
    if (std::FILE* f = std::fopen("/proc/self/statm", "r"))
    {
        long      size = 0;
        long      res  = 0;
        const int n    = std::fscanf(f, "%ld %ld", &size, &res);
        (void)std::fclose(f);
        const long ps = ::sysconf(_SC_PAGESIZE);
        if (n == 2 && ps > 0)
        {
            return res * (ps / 1024);
        }
    }
    return -1;
}

/**
 * @brief Exact peak resident size: a background thread samples
 *        exact_rss_kib() every @p period_ms and keeps the maximum.
 * @details The kernel's high-water mark (ru_maxrss, VmHWM) is taken from the
 *          same lagging counters as VmRSS, so an exact peak needs samples.
 *          A peak shorter than the period can be missed; note() folds in the
 *          samples other code takes at the moments that matter.
 */
class RssPeakSampler
{
  public:
    /// @brief Start the sampling thread when @p period_ms is positive.
    /// @param period_ms Sampling period, in milliseconds.
    explicit RssPeakSampler(int period_ms) : period_ms_(period_ms)
    {
        if (period_ms_ > 0)
        {
            thread_ = std::thread([this] { run(); });
        }
    }
    /// @brief Signal the sampling thread to stop and join it.
    ~RssPeakSampler()
    {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable())
        {
            thread_.join();
        }
    }
    // Owns the sampling thread, which captured `this`: neither copied nor moved.
    RssPeakSampler(const RssPeakSampler&)            = delete;
    RssPeakSampler& operator=(const RssPeakSampler&) = delete;
    RssPeakSampler(RssPeakSampler&&)                 = delete;
    RssPeakSampler& operator=(RssPeakSampler&&)      = delete;

    /// @brief Fold in a sample taken elsewhere.
    void note(long kib) noexcept
    {
        long cur = peak_.load(std::memory_order_relaxed);
        while (kib > cur && !peak_.compare_exchange_weak(cur, kib, std::memory_order_relaxed))
        {
        }
    }
    /// @brief Largest sample so far, in KiB (0 before the first one).
    [[nodiscard]] long peak_kib() const noexcept
    {
        return peak_.load(std::memory_order_relaxed);
    }

  private:
    void run() noexcept
    {
        while (!stop_.load(std::memory_order_relaxed))
        {
            note(exact_rss_kib());
            std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
        }
    }

    int               period_ms_;
    std::atomic<bool> stop_{false};
    std::atomic<long> peak_{0};
    std::thread       thread_;
};

} // namespace shrink_array
