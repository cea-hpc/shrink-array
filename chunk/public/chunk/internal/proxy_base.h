/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
 * @file proxy_base.h
 * @brief Common base class for Proxy2 and Proxy2A using CRTP pattern
 *
 * This refactoring extracts all common functionality into ProxyBase,
 * leaving only chunk-specific logic in derived classes.
 */

#include <type_traits>
#include <atomic>
#include <complex>
#include <cstdlib>
#include <functional>
#include <vector>
#include <memory>

#include "chunk/internal/element_access.h" // elem_get / elem_set, is_packed_bit_v: Bit buffers are words of bits
#include "chunk/config.h"
#include "chunk/internal/chunk_stats.h" // increm_chunk_* in the bodies (dependent calls the include tools cannot attribute)
#include "chunk/internal/in_t.h"
#include "chunk/internal/invariant_check.h" // INVARIANT_FAILED
#include "chunk/internal/proxy_operators.h"
#include "chunk/internal/compression_manager.h"
#include "chunk/internal/tab_cst_val2_holder.h" // TabCstVal2Holder (CHUNK_CST_VAL2_NO_ARRAYS)
#include "chunk/internal/relaxed_element.h"     // store_value: Chunk2A's relaxed element stores
#include "chunk/internal/concept_is_clean_assignement.h"
#include "chunk/internal/crp_helper.h"
#include "chunk/internal/arithmetic_context.h"

namespace shrink_array
{

template <typename T>
struct ArithmeticProxyContext;

template <typename T, typename Op>
struct ArithmeticOperationProxyContext;

template <typename T>
class Proxy;

/// @brief Whether T is a Proxy<U> (Proxy2 and Proxy2A are not): false by default.
template <typename T>
struct is_simple_proxy : std::false_type
{
};

/// @brief Proxy<T> is a simple proxy.
template <typename T>
struct is_simple_proxy<Proxy<T>> : std::true_type
{
};
/**
 * @brief The compression-bookkeeping half of ::ProxyBase.
 *
 * @tparam Derived The concrete proxy (CRTP).
 * @tparam T Element type.
 * @tparam ChunkType The chunk this proxy points into.
 * @tparam Enabled chunk_config::use_fast_block_compression.
 *
 * @details Split out so a build WITHOUT fast block compression pays nothing for
 *          it: the false specialisation below keeps the member names, so
 *          ProxyBase still compiles, but every entry point aborts. Nothing is
 *          supposed to reach them.
 *
 * @note The `*_async` members take a context rather than reading the proxy's
 *       own fields, because on the deferred-replay path there is no live proxy
 *       -- the write is being applied long after the one that enqueued it went
 *       out of scope. @see make_proxy_with_info
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
struct ProxyBaseFastCompPolicy : public CRTPSupport<Derived>
{
    using CRTPSupport<Derived>::derived;

    /// @brief The one compression manager for this (T, ChunkType) pair.
    static inline CompressionManager<T, ChunkType> compression_mgr;

    /// @brief This proxy's entry in #tab_is_inside_chunk_list_to_compress, or
    ///        null before set_ptr_is_inside_chunk_list_to_compress() runs.
    std::atomic<char>* ptr_is_inside_chunk_list_to_compress = nullptr;

    /**
     * @brief Per-tab array of atomic flags indicating whether each chunk is marked for compression.
     * Each entry holds nb_chunk atomic<char> flags, built whole (atomics are not
     * movable). char, not bool: vector<bool> is the packed specialisation.
     */

    static inline std::vector<std::vector<std::atomic<char>>> tab_is_inside_chunk_list_to_compress;

    /**
     * @brief Grow the per-array table of candidate-flag arrays to @p nb_tab entries.
     * @param nb_tab Number of arrays to make room for.
     */
    static void resize_tab_is_int(int nb_tab)
    {
        tab_is_inside_chunk_list_to_compress.resize(nb_tab);
    }

    /**
     * @brief Allocate and zero one array's per-chunk candidate flags.
     * @param num_tab Which array's flags to (re)allocate.
     * @param nb_chunk How many chunks it holds.
     * @note Replacing the vector frees the previous flags, so this doubles as
     *       the release path on a re-chunk.
     */
    static void resize_tab_is_int(int num_tab, int nb_chunk)
    {
        // Value-initialised: every flag 0 (C++20 atomics).
        tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)] =
            std::vector<std::atomic<char>>(static_cast<std::size_t>(nb_chunk));
    }

    /**
     * @brief Release one array's candidate flags (its teardown).
     * @details Called once the array's candidates are purged and its chunks
     *          freed. Until Sept '26 nothing did: an entry was only replaced
     *          when its num_tab was reused, and a re-chunk builds the new
     *          storage under a NEW num_tab -- one byte per old chunk leaked
     *          at every re-chunk. Swapped with an empty row, which frees a
     *          vector's capacity as well as a unique_ptr's array. A later
     *          resize_tab_is_int() reallocates.
     * @param num_tab The dying array's index.
     */
    static void release_tab_is_int(int num_tab) noexcept
    {
        using row_t = typename decltype(tab_is_inside_chunk_list_to_compress)::value_type;
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_is_inside_chunk_list_to_compress.size())
            row_t{}.swap(tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)]);
    }

    /// @brief Write @p v at the offset @p ctx recorded, on the replay path.
    /// @warning Reloads the chunk's value pointer from @p ctx: the one captured
    ///          when the write was enqueued may since have been reallocated.
    static void store_value_async(auto* ctx, in_t<T> v) noexcept
    {
        T* const values_loc = ctx->chunk.values_load();
        elem_set(values_loc, static_cast<std::size_t>(ctx->num), static_cast<T>(v));
    }

    /**
     * @brief Rebuild a usable proxy from a deferred operation's context.
     * @tparam ProxyType The concrete proxy to produce.
     * @param ctx The context captured when the write was enqueued.
     * @details Replay happens after the original proxy is gone, so the parts of
     *          it the write body needs are reconstructed here -- including the
     *          frozen chunk geometry, so the replayed write uses the geometry in
     *          force when it was enqueued rather than the current one.
     */
    template <typename ProxyType, typename Ctx>
    static ProxyType make_proxy_with_info(Ctx* ctx);

    /// @brief Read at the offset @p ctx recorded, on the replay path.
    /// @warning Same reload caveat as store_value_async().
    [[nodiscard]] static T read_values_async(auto* ctx) noexcept
    {
        const T* const values_loc = ctx->chunk.values_load();
        return elem_get(values_loc, static_cast<std::size_t>(ctx->num));
    }

    /// @brief set_ptr_is_inside_chunk_list_to_compress() for the replay path,
    ///        taking the tab/chunk indices from @p ctx.
    void set_ptr_is_inside_chunk_list_to_compress_async(auto* ctx);

    /// @brief Queue the chunk @p ctx names as a compression candidate, on the
    ///        replay path.
    // Explicit object parameter (C++23) in place of CRTPSupport::derived().
    void add_to_compression_list_async(this const auto& self, auto* ctx);
};

/**
 * @brief ::ProxyBaseFastCompPolicy with fast block compression compiled out.
 *
 * @details Keeps the member NAMES so ProxyBase's `using` declarations still
 *          resolve, but the entry points abort. Reaching one means a caller
 *          took a compression path in a build that has none, which is a bug in
 *          the caller, not a case to handle.
 */
template <typename Derived, typename T, typename ChunkType>
struct ProxyBaseFastCompPolicy<Derived, T, ChunkType, false> : public CRTPSupport<Derived>
{
    // Management members
    static inline CompressionManager<T, ChunkType>            compression_mgr;
    std::atomic<char>*                                        ptr_is_inside_chunk_list_to_compress;
    static inline std::vector<std::vector<std::atomic<char>>> tab_is_inside_chunk_list_to_compress;


    /// @brief No candidate flags in this configuration; nothing to size.
    /// @note Still declared: ArrayChunkBase::create_common() calls it
    ///       unconditionally, because the call site is shared with variants
    ///       that do need it.
    static void resize_tab_is_int([[maybe_unused]] int nb_tab) noexcept {}
    /// @copydoc resize_tab_is_int(int)
    static void resize_tab_is_int([[maybe_unused]] int num_tab, [[maybe_unused]] int nb_chunk) noexcept {}
    /// @copydoc resize_tab_is_int(int)
    static void release_tab_is_int([[maybe_unused]] int num_tab) noexcept {}

    /// @brief Never callable in this configuration. Aborts.
    template <typename ProxyType, typename Ctx>
    static ProxyType make_proxy_with_info(Ctx* /*ctx*/)
    {
        std::abort();
    }
};

// ============================================================================
// ProxyBase: Base CRTP class containing all shared functionality
// ============================================================================

/**
 * @brief Base class for proxy types using CRTP pattern
 * @tparam Derived The derived proxy class (Proxy, Proxy2 or Proxy2A)
 * @tparam T The underlying value type
 * @tparam ChunkType The chunk type (Chunk<T>, Chunk2<T> or Chunk2A<T>)
 *
 * Contains all common operators, compression logic, and thread-safety mechanisms.
 *
 * A proxy stands for ONE element of an array and behaves like a value: it
 * converts to T by reading the element (operator T() -> read_pure(), which
 * first synchronises any deferred write to it), and every write goes through
 * it. It is a handle, not a copy:
 *   - it holds references into the array's chunk storage, so it must not
 *     outlive an operation that rebuilds that storage -- in particular a
 *     re-chunk, explicit or decided by a RechunkDecider (full list on the
 *     array's non-const operator[]);
 *   - `auto v = a[i];` keeps the proxy, which then follows the element;
 *     `T v = a[i];` takes the value.
 * The compound operators (+= -= *= /= %= &= |= ^= <<= >>=) and the prefix
 * ++/-- return the proxy by value: discarding it, as `a[i] += x;` does, keeps
 * the operation fully deferrable, while `T y = (a[i] += x);` reads the result.
 * The postfix ++/-- return the old value as a T.
 *
 * Those operators take the proxy as an explicit object parameter deduced as
 * `this auto&& self` (C++23), not the `this Derived& self` ArrayChunkBase
 * uses: `a[i]` is a prvalue, which a typed lvalue reference would not bind.
 * The reason ArrayChunkBase types its self -- a deduced one would outrank the
 * const read overloads -- does not arise, none of them having a const form.
 */
template <typename Derived, typename T, typename ChunkType>
class ProxyBase : public ProxyOperators<Derived, T>,
                  public ProxyBaseFastCompPolicy<Derived, T, ChunkType, chunk_config::use_fast_block_compression>

{
  public:
    // Required for template concepts to recognize this as a Proxy
    using is_shrink_proxy_tag = void;
    using value_type          = T;

    using FastCompPolicy = ProxyBaseFastCompPolicy<Derived, T, ChunkType, chunk_config::use_fast_block_compression>;

    using FastCompPolicy::compression_mgr;
    using FastCompPolicy::derived;
    using FastCompPolicy::ptr_is_inside_chunk_list_to_compress;

    template <typename U, typename D, typename C, bool E>
    friend struct ProxyBaseFastCompPolicy;

  protected:
    // Alias for internal usage

    // ========================================================================
    // Common type aliases
    // ========================================================================
    using atomic_ptr_ref_t = std::conditional_t<build_config::use_openmp, std::atomic<T*>&, T*&>;

    // ========================================================================
    // Common member variables
    // ========================================================================
    /// @brief Reference to the chunk's value pointer, atomic under OpenMP.
    /// @warning A REFERENCE, captured at construction. The chunk may reallocate
    ///          under it, so a write path must reload through it, never cache it.
    atomic_ptr_ref_t values;

    const int  num;   ///< Offset of this element WITHIN its chunk.
    ChunkType& chunk; ///< The chunk that owns the element.

    mutable const void* ptr_on_array; ///< The array the chunk belongs to.
    mutable int         num_chunk;    ///< Index of the chunk within that array.
    mutable int         num_tab = -1; ///< Index of the array in the registry; -1 until armed.

    // fw14 (dyn UAF fix): frozen chunk geometry for the deferred-replay path.
    // 0 means "not armed" -> chunk_size() falls back to the live ptr_on_array
    // read (the direct write path, array alive). make_proxy_with_info arms this
    // from ctx->chunk_size_frozen on replay, so chunk_size() there returns the
    // enqueue-time value WITHOUT dereferencing a possibly-dead ptr_on_array.
    mutable unsigned int chunk_size_frozen_ = 0;
    /// @brief Affinity key captured with chunk_size_frozen_ for the same reason:
    ///        on a deferred replay ptr_on_array may be dead. 0 until frozen.
    mutable unsigned int alloc_key_frozen_ = 0;

    using CompressionManagerType = CompressionManager<T, ChunkType>;

  public:
    /// @brief Bind a proxy to one element.
    /// @note Leaves #ptr_on_array, #num_chunk and #num_tab unset --
    ///       set_this_and_num_chunk() and set_num_tab() finish the job, and the
    ///       compression paths need them.
    explicit ProxyBase(atomic_ptr_ref_t values_, int num_, ChunkType& chunk_) noexcept
        : values(values_), num(num_), chunk(chunk_)
    {
    }
    // ========================================================================
    // Common getters
    // ========================================================================

    /// @brief Index of this element's chunk within its array.
    [[nodiscard]] int get_num_chunk() const noexcept
    {
        return num_chunk;
    }

    /**
     * @brief Elements per chunk, as this proxy sees them.
     *
     * @details A compile-time constant with a static chunk size; with a dynamic
     *          one it reads the array's runtime geometry through #ptr_on_array
     *          -- unless set_chunk_size_frozen() armed a value, which the replay
     *          path does precisely so it need not touch a possibly-dead
     *          #ptr_on_array.
     *
     * @note Use this rather than the process-wide geometry on any write path:
     *       during a re-chunk the two disagree.
     */
    [[nodiscard]] unsigned int chunk_size() const noexcept;
    /// @brief The owning array's slab-affinity key (0 = shared pool); frozen
    ///        twin on the replay path like chunk_size().
    [[nodiscard]] unsigned int alloc_key() const noexcept;

    /// @brief Offset of this element within its chunk.
    [[nodiscard]] int get_num() const noexcept
    {
        return num;
    }

    /// @brief Index of this element's array in the proxy registry.
    [[nodiscard]] int get_num_tab() const noexcept
    {
        return num_tab;
    }

    /// @brief Record which array this proxy points into. @see get_num_tab
    void set_num_tab(int nt) noexcept
    {
        num_tab = nt;
    }

    /// @brief The chunk that owns this element.
    [[nodiscard]] ChunkType& get_chunk() const noexcept
    {
        return chunk;
    }

    // ========================================================================
    // Common chunk state queries
    // ========================================================================

    /// @brief Is the chunk compressed to its single first value?
    [[nodiscard]] bool is_cst_val1() const noexcept
    {
        return chunk.is_cst_val1();
    }

    /// @brief is_cst_val1() judged from an already-loaded value pointer.
    /// @param values_loc The pointer the caller loaded; saves a reload on a path
    ///        that has one in hand, and keeps the whole test on ONE snapshot.
    [[nodiscard]] bool is_cst_val1(const T* values_loc) const noexcept
    {
        return chunk.is_cst_val1(values_loc);
    }

    /// @brief Complement of is_cst_val1().
    [[nodiscard]] bool is_not_cst_val1() const noexcept
    {
        return !chunk.is_cst_val1();
    }

    /// @brief Would writing @p v leave the chunk compressed exactly as it is?
    /// @details The write fast path's early-out: the chunk holds only its first
    ///          value and @p v IS that value, so there is nothing to do and no
    ///          reason to materialise the chunk.
    ///
    ///          Never while the chunk has a deferred op pending
    ///          (pending_any()): that op is OLDER than this write in program
    ///          order and would replay after it, so the write must reach the
    ///          engine, whose WAW guard retires the op before the store. The
    ///          early-outs sit before the engine; this is their one hole,
    ///          found by deferred_path_validation (chunk2/bool lost 1 500
    ///          writes in a million). pending_any() is one acquire load of
    ///          the chunk header, constant-false in a serial build.
    [[nodiscard]] bool is_cst_val1_and_v_equal_first_value(in_t<T> v) const noexcept
    {
        return v == static_cast<T>(ChunkType::first_value) && is_cst_val1() && !chunk.pending_any();
    }

    /// @brief is_cst_val1_and_v_equal_first_value() from a loaded value pointer.
    [[nodiscard]] bool is_cst_val1_and_v_equal_first_value(const T* values_loc, in_t<T> v) const noexcept
    {
        return v == static_cast<T>(ChunkType::first_value) && is_cst_val1(values_loc) && !chunk.pending_any();
    }
    /**
     * @brief Initialize per-chunk pointers except second value.
     */

    static void set_tab_is_inside_chunk_list_to_compress(int num_tabl, int num_chunkl, char v) noexcept
    {
        FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tabl][num_chunkl] = v;
    }

    /// @brief Point this proxy at an already-known candidate flag.
    void set_ptr_is_inside_chunk_list_to_compress(std::atomic<char>* pt) noexcept
    {
        ptr_is_inside_chunk_list_to_compress = pt;
    }

    /// @brief The whole candidate-flag row for array @p num_tab.
    static std::vector<std::atomic<char>>& get_tab_is_inside_list_to_compress(int num_tab) noexcept
    {
        return FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tab];
    }

    /// @brief Clear one chunk's candidate flag, so it may be queued again.
    /// @note Release store: whatever made the chunk a candidate must be visible
    ///       to the thread that next reads the flag.
    static void reset_tab_inside_list_to_compress(int num_tab, int num_chunk) noexcept
    {
        FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tab][num_chunk].store(0, std::memory_order_release);
    }

    /**
     * @brief Set up pointer to this chunk's compression tracking flag
     * @pre num_tab and num_chunk must be valid
     * @post ptr_is_inside_chunk_list_to_compress points to correct atomic flag
     */

    void set_ptr_is_inside_chunk_list_to_compress() noexcept
    {
        ptr_is_inside_chunk_list_to_compress =
            &FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tab][num_chunk];
    }

    /// @brief Queue this chunk as a compression candidate in @p vec_tuple.
    void add_to_compression_list([[maybe_unused]] auto& vec_tuple) const;

    /// @brief add_to_compression_list() on the calling thread's own list.
    void add_to_compression_list(this const Derived& self);

    // ========================================================================
    // Common I/O operations
    //

    /// @brief This element, read raw: `values[num]`, or its bit for Bit
    ///        (element_access.h). No sync, no check -- the old spelling.
    [[nodiscard]] T raw_elem() const noexcept
    {
        return elem_get(static_cast<const T*>(values), static_cast<std::size_t>(num));
    }

    /// @brief This element, written raw: `values[num] = v`, or an atomic RMW
    ///        of its word for Bit. No sync, no check -- the old spelling.
    void raw_elem_set(in_t<T> v) const noexcept
    {
        elem_set(static_cast<T*>(values), static_cast<std::size_t>(num), static_cast<T>(v));
    }

    /// @brief Write @p v straight through #values, with no synchronisation.
    /// @pre The chunk is materialised and the caller holds the write path.
    void store_value(in_t<T> v) const noexcept
    {
        // The chunk must not be COMMITTED to a compressor at this point: a
        // writer on a marked chunk is sent to the slow path by
        // must_defer_write(), and a compressor commits its claims only once
        // its window is shut -- no writer flagged on a marked chunk. A chunk
        // merely marked (1) is legitimate here: the compressor may have marked
        // it after this writer's check #2 and is waiting on the flag; that
        // was the false positive of the first form of this guard, one run in
        // two at 8 threads. If this fires, the compressor passed its window
        // with this writer's flag up: the ordering of the gate's read, not the
        // propagation of the mark -- what TSan reports as this write racing
        // scan_constants()/read_value_scan().
        if constexpr (build_config::any_check && chunk_config::use_chunk_steal_mark)
        {
            if (chunk.steal_committed()) [[unlikely]]
                INVARIANT_FAILED("store_value on a chunk committed to a compressor (num={})", num);
        }
        // Chunk2A: the marking filter of another writer may be reading this
        // element -- relaxed atomic store, see relaxed_element.h.
        if constexpr (internal::relaxed_element_stores_v<ChunkType, T>)
        {
            T* const vals = values;
            internal::store_element_relaxed(vals + num, static_cast<T>(v));
        }
        else
        {
            // Bit: an atomic RMW of the word (element_access.h); values[num] otherwise.
            elem_set(static_cast<T*>(values), static_cast<std::size_t>(num), static_cast<T>(v));
        }
    }

    /// @brief Raw read for the WRITE path only. Deliberately does not sync.
    /// @warning Not an ordinary read -- use read_pure(). The reason it must not
    ///          sync is in the body comment.
    [[nodiscard]] T read_values() const noexcept
    {
        // WRITE-PATH helper ONLY (v_old capture + mark_for_compression new value
        // in execute_simple_operation / arithmetic_assign). It must NOT sync:
        // syncing here re-flushes from inside a write (begin_write held) and, when
        // pending_ops_>0, wait_for_availability's replay reallocates `values`
        // under the in-flight writer -> SIGSEGV. Same-offset visibility on the
        // write path is handled by the P2 WAW erase before the write body; pure
        // reads use read_pure() below. Kept raw in BOTH sync_on_read states.
        //
        // ... but NOT raw under CHUNK_CST_VAL2_NO_ARRAYS: there the shared
        // array of a second constant is never written, so this would read a
        // zero page and the compression decision downstream would be taken
        // against 0 instead of the chunk's constant. Resolving the registry
        // is arithmetic plus one table load -- it does not sync, so the
        // contract above still holds.
        //
        // The question is asked of the POINTER, not of the chunk. This
        // function runs with a write in flight -- that is its reason to
        // exist -- so the chunk's state and the pointer this proxy is about
        // to index can disagree, and asking chunk.is_cst_val2() then
        // converting a pointer that is NOT in the region gave an arbitrary
        // index into value_by_id[] and a SIGSEGV. One load, one exact range
        // test on it, one conversion of the same pointer.
        if constexpr (build_config::cst_val2_no_arrays && requires(const T* ptr) {
                          TabCstVal2Holder<T>::is_cst_val2_ptr(ptr);
                          TabCstVal2Holder<T>::value_of_ptr_checked(ptr);
                      })
        {
            const T* const vals = values;
            if (TabCstVal2Holder<T>::is_cst_val2_ptr(vals)) [[unlikely]]
                return TabCstVal2Holder<T>::value_of_ptr_checked(vals);
            // elem_get, not vals[num]: for Bit that indexed BYTES, up to
            // chunk_size of them in a chunk_size / 8 byte buffer.
            return elem_get(vals, static_cast<std::size_t>(num));
        }
        return elem_get(static_cast<const T*>(values), static_cast<std::size_t>(num));
    }

    // PURE-READ entry (operator T(), the *=1 / /=1 early-outs). Under
    // sync_on_read this pulls the offset to applied state then reads LIVE; the
    // ctor-time `values` snapshot may be stale after materialisation. This is the
    // ONLY place the read-side toll lives — never on the write path.
    /// @brief The ordinary read: synchronise this offset, then read it live.
    /// @details The only place the read-side synchronisation cost is paid, and
    ///          the entry point behind `operator T()`.
    [[nodiscard]] T read_pure() const noexcept
    {
        chunk.reader_sync(num);
        return chunk.read_value(num); // read_value accounts the read in stats
    }

    /// @brief Read through the chunk WITHOUT synchronising first.
    /// @warning Returns a stale value if a deferred write on this offset has
    ///          not been applied. Use read_pure() unless you know otherwise.
    [[nodiscard]] T load_value() const noexcept
    {
        return chunk.read_value(num);
    }

    // Public API used by writers (your Proxy2::operator= / manage_core_op_eq)
    // Use this to perform a write on (num_tab,num_chunk). The write_fn is the actual write body.
    // internal generic function: executes write_body() on fast- or slow-path

    /**
     * @brief Run @p write_body on the fast path, or defer it.
     * @tparam WriteCallable The write itself.
     * @tparam LBD_INFO Callable arming whatever metadata the write needs.
     * @return TRUE if the write was performed, FALSE if it was deferred and
     *         will be replayed later.
     * @note @p lbd_info runs BEFORE the body, and only when the metadata is
     *       actually needed -- it is a no-op without fast block compression.
     */
    template <typename WriteCallable, typename LBD_INFO>
    [[nodiscard]] bool write_with_optional_lock_impl(this Derived&             self,
                                                     WriteCallable&&           write_body,
                                                     [[maybe_unused]] LBD_INFO lbd_info) noexcept;

    // Removed: the [[deprecated]] write_with_optional_lock() overload that armed
    // NOTHING before the write (empty lbd_info). It had no caller left: Proxy
    // declares its own same-named member, which HIDES this one for the only call
    // site (proxy_inline.h), and Proxy2 / Proxy2A go through
    // write_with_optional_lock_info() below. Its own deprecation note asked for
    // exactly this removal.

    /**
     * @brief Executes a write operation without parameters.
     * @tparam WriteFn Callable to execute.
     * @param write_fn Function to execute.
     */
    template <typename WriteFn>
    [[nodiscard]] bool write_with_optional_lock_info(this Derived& self, WriteFn write_fn)
    {
        const auto lbd_info = [&self]
        {
            if constexpr (chunk_config::use_fast_block_compression)
            {
                // This code is only compiled and executed if the condition is true
                self.set_info_on_chunk();
            }
        };

        return self.write_with_optional_lock_impl([&] { write_fn(); }, lbd_info);
    }

    /**
     * @brief Has a thread's candidate list grown past the point where it must be drained?
     *
     * The single definition of the "list became dangerous" threshold, shared by
     * the trigger_compression_if_needed() of Proxy, Proxy2 and Proxy2A. Keep it
     * that way: the three used to spell the comparison out individually, so a
     * change to the rule reached only whichever variant was being edited.
     *
     * @param sz  Current candidate count for the thread.
     * @param szm Maximum tolerated before the list is flagged dangerous.
     * @return True when the list must be flagged.
     */
    static bool should_trigger_compression(int sz, int szm) noexcept
    {
        return sz > szm;
    }

    /**
     * @brief Deferred-path tail: a dangerous rank still drives its pass.
     *
     * A write that must be deferred (a compressor is in flight) leaves without
     * further ado -- unless this rank's candidate list is flagged dangerous, in
     * which case it runs perform_compression() as the fast path would have
     * after its store: the flush of the rank's own deferred ops, then a
     * start() that waits for the in-flight compressor rather than yielding
     * to a flush of its own making. The other ranks used to run the pass too,
     * and each one flushed -- and announced -- the single op it had just
     * enqueued.
     *
     * @param thr Calling thread's rank.
     */
    void drive_compression_if_dangerous(this Derived& self, int thr) noexcept
    {
        if (CompressionManagerType::is_dangerous_list(thr)) [[unlikely]]
        {
            compression_mgr.run_slow_path([&] { self.perform_compression(); });
        }
    }

    // ========================================================================
    // Common metadata management
    // ========================================================================

    /**
     * @brief Store pointer-to-array and chunk index for subsequent operations.
     * @param array_ptr pointer to the array
     * @param num_chunk_ chunk index
     */

    void set_this_and_num_chunk(const void* array_ptr, int num_chunk_) const noexcept
    {
        ptr_on_array = array_ptr;
        num_chunk    = num_chunk_;
        num_tab      = -1; // not armed: set_num_tab() looks the registry up
    }

    /// @brief Same, with the array's own table number, so the write path
    ///        needs no registry lookup.
    /// @param array_ptr  pointer to the array
    /// @param num_chunk_ chunk index
    /// @param num_tab_   the array's num_tab (ArrayChunkBase::num_tab), the
    ///                   value it registered under its address
    /// @note The registry lookup (share_info_array.get_num_tab) is a linear
    ///       scan of the live arrays of this type, or a hash probe past a
    ///       threshold, and it ran on EVERY write: 1.5% of the reference run.
    ///       The array knows the answer; it is checked against the registry
    ///       under any_check (Proxy2A::set_num_tab).
    void set_this_and_num_chunk(const void* array_ptr, int num_chunk_, int num_tab_) const noexcept
    {
        ptr_on_array = array_ptr;
        num_chunk    = num_chunk_;
        num_tab      = num_tab_;
    }

    // fw14: arm the frozen chunk size on the replay-reconstructed proxy.
    /// @brief Pin the chunk geometry a deferred write must be replayed with.
    /// @param sz Element count per chunk at enqueue time; 0 means "not armed",
    ///        and chunk_size() then reads the live geometry instead.
    /// @details Without this, replay would read the geometry through
    ///          #ptr_on_array -- which a re-chunk may have freed by then.
    void set_chunk_size_frozen(unsigned int sz, unsigned int key = 0) const noexcept
    {
        chunk_size_frozen_ = sz;
        alloc_key_frozen_  = key;
    }

    /// @brief Record the chunk index alone. @see set_this_and_num_chunk
    void set_num_chunk(int num_chunk_) const noexcept
    {
        num_chunk = num_chunk_;
    }

    // ========================================================================
    // DELETED OPERATORS
    // ========================================================================

    /// @brief Deleted: proxy-to-proxy assignment would be ambiguous -- rebind
    ///        the proxy, or copy the value? The derived classes define what
    ///        `a[i] = b[j]` means.
    const Derived& operator=(const Derived& from) = delete;

    /// @brief Deleted: there is no stable address to take. The chunk may be
    ///        compressed, or reallocate under the proxy.
    T* operator&() = delete;

    /// @copydoc operator&()
    const T* operator&() const = delete;

    /// @brief Copying a proxy is fine: it rebinds to the same element.
    ProxyBase(const ProxyBase& other) = default;

    /**
     * @brief Modulo struct for compound modulo operation
     * @details Wrapper for % operator to fit execute_arithmetic_compound_operation interface
     */

    struct modulo
    {
        /**
         * @brief Apply modulo operation
         * @param lhs Dividend
         * @param rhs Divisor
         * @return lhs % rhs
         */
        static constexpr T operator()(T lhs, T rhs) noexcept
        {
            return lhs % rhs;
        }
    };

    /// @brief Compare the element's value with @p v. Reads through read_pure().
    template <typename U>
    [[nodiscard]] bool operator==(const U& v) const noexcept
    {
        return read_pure() == v;
    }

    /// @brief Compare with a @p v already of the element type.
    [[nodiscard]] bool operator==(T v) const noexcept
    {
        return read_pure() == v;
    }

    /// @brief Complement of operator==().
    [[nodiscard]] bool operator!=(T v) const noexcept
    {
        return read_pure() != v;
    }

    /// @brief Is the element's value falsy?
    [[nodiscard]] bool operator!() const noexcept
    {
        return !read_pure();
    }

    /// @brief The element's value, negated. Does not modify it.
    [[nodiscard]] T operator-() const noexcept
    {
        return -read_pure();
    }

    /// @brief The element's value divided by @p v. Does not modify it.
    template <typename U>
    [[nodiscard]] T operator/(const U& v) const noexcept
        requires(!std::is_same_v<U, std::complex<T>>)
    {
        return read_pure() / static_cast<T>(v);
    }

    /// @brief Read the element. This is what makes a proxy behave like a value.
    /// @note Goes through read_pure(), so it pays the read-side synchronisation.
    [[nodiscard]] operator T() const noexcept
    {
        return read_pure();
    }

    /// @brief ArrayChunkBits: read the element straight as a bool. Through
    ///        operator T() it would be two user conversions (proxy -> Bit ->
    ///        bool), which C++ does not chain, so `bool b = a[i]` needs this one.
    [[nodiscard]] operator bool() const noexcept
        requires is_packed_bit_v<T>
    {
        return static_cast<bool>(read_pure());
    }

    /**
     * @brief Record a write to be applied later, when the fast path refuses it.
     * @param ctx Captures everything the replay needs -- offset, chunk indices,
     *        and the frozen geometry.
     * @param body The write, run at replay time against a proxy rebuilt by
     *        make_proxy_with_info().
     */
    template <typename CtxType, typename LambdaBody>
    void enqueue_deferred_write(CtxType* ctx, LambdaBody&& body) const;

    /// @brief enqueue_deferred_write() for a compound arithmetic operation,
    ///        carrying @p v and @p operation into the context.
    template <typename U, typename Op, typename AsyncBody>
    void enqueue_async_arithmetic(this Derived& self, const U& v, Op operation, AsyncBody&& async_body);

    /**
     * @brief Apply @p operation to the element: load, compute in
     *        common_type_t<T, U>, store -- inside the write protocol
     *        (write_with_optional_lock_info), deferred when the fast path
     *        refuses. Not a hardware atomic read-modify-write: there is no
     *        compare-exchange loop, and two threads updating the same element
     *        race as they would on a plain T.
     * @param v_old Unused by the body (kept for the callers' signature).
     */
    template <typename U, typename Op>
        requires std::is_invocable_r_v<T, Op, T, U>
    void apply_atomic_op(this Derived& self, Op operation, const U& v, in_t<T> v_old);

    /// @brief Run @p lambda as a simple write, through the derived proxy.
    template <typename Lambda>
    void execute_operation(this Derived& self, Lambda&& lambda)
    {
        self.execute_simple_operation(std::forward<Lambda>(lambda));
    }

    /// @brief execute_operation() on the replay path, against @p ctx.
    template <typename Lambda>
    void execute_operation_async(this Derived& self, auto* ctx, Lambda&& lambda)
    {
        self.execute_simple_operation_async(ctx, std::forward<Lambda>(lambda));
    }

    /**
     * @brief Generic helper for compound arithmetic operations
     *
     * Executes an arithmetic operation with synchronization and optional statistics.
     * Used to implement += and -=.
     *
     * @tparam U Type of the value to apply
     * @tparam Op Operation type (callable: T(T, U))
     * @param v Value to use in the operation
     * @param operation Binary function applying the operation (e.g., a + b)
     * @note Returns nothing: the operators return the proxy, whose conversion
     *       to T reads the resulting value (synchronised). On the deferred
     *       path the result does not exist yet on return, so nothing here
     *       could report it.
     * @note Thread-safe via write_with_optional_lock_info()
     * @note Increments CHUNK_STATS statistics if enabled
     */

    template <typename U, typename Op>
    void
    arithmetic_assign(this Derived& self, in_t<U> v, Op operation) noexcept; // in_t: see assign_impl in the proxies
    /**
     * @brief Compound addition assignment -- clean types (e.g. float += float,
     *        float += int).
     * @tparam U Type of operand (implicit conversion to T)
     * @param v Value to add
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires IsCleanCompoundOp<T, U> && ElementAddable<T>
    Derived operator+=(this auto&& self, const U& v) noexcept
    {
        self.template arithmetic_assign<T>(static_cast<T>(v), std::plus<T>{});
        return self;
    }

    /** @brief Compound addition — type mismatch (e.g. float += double). Precision warning. */
    template <typename U>
        requires(!IsCleanCompoundOp<T, U>) && std::is_convertible_v<U, T> && ElementAddable<T>
    [[deprecated("HPC WARNING: Type mismatch in +=. Possible precision loss. Use 'f' suffix for floats.")]]
    Derived operator+=(this auto&& self, const U& v) noexcept
    {
        self.template arithmetic_assign<T>(static_cast<T>(v), std::plus<T>{});
        return self;
    }

    /**
     * @brief Compound subtraction operator -- clean types.
     *
     * Subtracts a value from the element referenced by the proxy in a thread-safe manner.
     *
     * @tparam U Type of the value to subtract (must be compatible with T)
     * @param v Value to subtract
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note noexcept guaranteed
     * @note Increments CHUNK_STATS if enabled
     */
    template <typename U>
        requires IsCleanCompoundOp<T, U> && ElementSubtractable<T>
    Derived operator-=(this auto&& self, const U& v) noexcept
    {
        self.template arithmetic_assign<T>(static_cast<T>(v), std::minus<T>{});
        return self;
    }

    /** @brief Compound subtraction — type mismatch. */

    template <typename U>
        requires(!IsCleanCompoundOp<T, U>) && std::is_constructible_v<T, U> && ElementSubtractable<T>
    [[deprecated("HPC WARNING: Precision mismatch.")]]
    Derived operator-=(this auto&& self, const U& v) noexcept
    {
        self.template arithmetic_assign<T>(static_cast<T>(v), std::minus<T>{});
        return self;
    }
    /**
     * @brief Engine of every bitwise compound operator (&= |= ^= <<= >>=).
     *
     * Executes an in-place bitwise operation, synchronously or deferred.
     *
     * @tparam U Type of the value to apply (must be integral)
     * @tparam Op Operation type (callable: T(T, T))
     * @param v Value to use in the operation
     * @param operation Function applying the operation
     * @note Returns nothing: the operators return the proxy, whose conversion
     *       to T reads the resulting value (synchronised). On the deferred
     *       path the result does not exist yet on return, so nothing here
     *       could report it.
     * @note Thread-safe via write_with_optional_lock_info()
     * @note Increments CHUNK_STATS statistics if enabled -- formerly only the
     *       &= engine did, so |= ^= <<= >>= went uncounted
     * @note Requires T and U to be integral types
     */

    template <typename U, typename Op>
    void bitwise_assign(this Derived& self, in_t<U> v, Op operation) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<U>;

    /**
     * @brief Compound bitwise AND operator
     *
     * Applies a bitwise AND operation with a value to the referenced element.
     *
     * @tparam U Type of the value (must be integral)
     * @param v Value to use for the AND operation
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    template <typename U>
    Derived operator&=(this auto&& self, const U& v) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<base_t<U>>
    {
        self.template bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs & rhs; });
        return self;
    }

    /**
     * @brief Compound bitwise OR operator
     *
     * Applies a bitwise OR operation with a value to the referenced element.
     *
     * @tparam U Type of the value (must be integral)
     * @param v Value to use for the OR operation
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    template <typename U>
    Derived operator|=(this auto&& self, const U& v) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<base_t<U>>
    {
        self.template bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs | rhs; });
        return self;
    }

    /**
     * @brief Compound bitwise XOR operator
     *
     * Applies a bitwise XOR operation with a value to the referenced element.
     *
     * @tparam U Type of the value (must be integral)
     * @param v Value to use for the XOR operation
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    template <typename U>
    Derived operator^=(this auto&& self, const U& v) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<base_t<U>>
    {
        self.template bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs ^ rhs; });
        return self;
    }

    // ------------------------------------------------------------------------
    // ArrayChunkBits (T = Bit, bit.h): |= &= ^= with a bool operand.
    //
    // Bit is not integral, so the three above do not apply. A logical
    // operation needs no read-modify-write when the operand decides the result:
    // `|= true` is `= true`, `&= false` is `= false`, and the other operand
    // leaves the element as it is -- a plain write through the proxy's
    // operator=, with its whole protocol (marking, deferral), and no write at
    // all when nothing changes. Only `^= true` reads: the element, then its
    // negation written back -- as for a plain bool, two threads updating the
    // SAME element race; different elements of one word do not.
    // ------------------------------------------------------------------------

    /// @brief The operand kinds the Bit operators take: bool, Bit, or a proxy
    ///        of either.
    template <typename U>
    static constexpr bool bit_operand_v = std::is_same_v<base_t<U>, bool> || is_packed_bit_v<base_t<U>>;

    /// @brief `a[i] |= b` for ArrayChunkBits: sets the element when @p v is true.
    template <typename U>
    Derived operator|=(this auto&& self, const U& v) noexcept
        requires is_packed_bit_v<T> && bit_operand_v<U>
    {
        if (static_cast<bool>(v))
            self = true;
        return self;
    }

    /// @brief `a[i] &= b` for ArrayChunkBits: clears the element when @p v is false.
    template <typename U>
    Derived operator&=(this auto&& self, const U& v) noexcept
        requires is_packed_bit_v<T> && bit_operand_v<U>
    {
        if (!static_cast<bool>(v))
            self = false;
        return self;
    }

    /// @brief `a[i] ^= b` for ArrayChunkBits: flips the element when @p v is true.
    template <typename U>
    Derived operator^=(this auto&& self, const U& v) noexcept
        requires is_packed_bit_v<T> && bit_operand_v<U>
    {
        if (static_cast<bool>(v))
        {
            const bool cur = static_cast<bool>(self.read_pure()); // synchronised read, as operator bool()
            self           = !cur;
        }
        return self;
    }

    /**
     * @brief Compound left shift operator
     *
     * Shifts bits to the left by a specified number of positions.
     *
     * @tparam U Type of the shift amount (must be integral)
     * @param v Number of positions to shift
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    /** @brief Bitwise shift left — clean types (Integral <<= Integral). */
    template <typename U>
        requires IsCleanShiftOp<T, U>
    Derived operator<<=(this auto&& self, const U& v) noexcept
    {
        self.template bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs << rhs; });
        return self;
    }

    /** @brief Bitwise shift left — trap for non-integral shift amount. */
    template <typename U>
        requires(!IsCleanShiftOp<T, U>) && std::is_integral_v<T>
    [[deprecated("HPC WARNING: Shift amount must be an integer to avoid precision issues.")]]
    Derived operator<<=(this auto&& self, const U& v) noexcept
    {
        // Force conversion to int for the shift amount
        self.template bitwise_assign<int>(static_cast<int>(v), [](auto lhs, auto rhs) { return lhs << rhs; });
        return self;
    }

    /**
     * @brief Compound right shift operator
     *
     * Shifts bits to the right by a specified number of positions.
     *
     * @tparam U Type of the shift amount (must be integral)
     * @param v Number of positions to shift
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     * @note Divides by 2^v for unsigned types, implementation-defined for signed
     */
    template <typename U>
        requires IsCleanShiftOp<T, U>
    Derived operator>>=(this auto&& self, const U& v) noexcept
    {
        self.template bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs >> rhs; });
        return self;
    }

    /** @brief Bitwise shift right — trap. */
    template <typename U>
        requires(!IsCleanShiftOp<T, U>) && std::is_integral_v<T>
    [[deprecated("HPC WARNING: Shift amount must be an integer.")]]
    Derived operator>>=(this auto&& self, const U& v) noexcept
    {
        self.template bitwise_assign<int>(static_cast<int>(v), [](auto lhs, auto rhs) { return lhs >> rhs; });
        return self;
    }

    [[nodiscard]] T operator~() const noexcept
        requires std::is_integral_v<T>
    {
        return ~read_pure();
    }
    /**
     * @brief Helper to execute an increment/decrement operation with optional async handling
     *
     * @tparam SyncOp  Callable applying the operation in place, now.
     * @tparam AsyncOp Callable `(ctx, proxy)` applying it later, on replay.
     * @param sync_op  Body run on the fast path.
     * @param async_op Body enqueued when the fast path refuses the write.
     * @note Materialises any foreign deferred write on this offset first: an
     *       in-place operation reads the current value, so a stale read here
     *       would clobber that write.
     */

    template <typename SyncOp, typename AsyncOp>
    void apply_inplace_operation(this Derived& self, SyncOp sync_op, AsyncOp async_op) noexcept;

    /**
     * @brief Pre-increment operator
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @details Implements: ++proxy
     * Returns the value after increment.
     */
    Derived operator++(this auto&& self) noexcept
        requires ElementIncrementable<T>
    {
        // Synchronous version
        const auto inc_operation = [&self] { self.execute_inplace_operation([](T& elem) { ++elem; }); };

        // Asynchronous version
        const auto inc_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { ++elem; }); };

        self.apply_inplace_operation(inc_operation, inc_operation_async);

        // Return the proxy, not a value. Reading the element here would have to
        // go through read_pure(), never `values[num]`: `values` is the ctor-time
        // snapshot, and the call above may have materialised the chunk (snapshot
        // now stale, possibly the cst sentinel) or DEFERRED the op (cell not yet
        // updated, while another rank's replay may be writing it -- TSan, v1
        // static 256, 8 threads: such a read raced the async body above). The
        // returned proxy's operator T() IS read_pure(): it pulls the offset to
        // applied state, then reads live through the chunk. So `y = ++a[i]`
        // gets the right value, and the read -- with its sync -- is only paid
        // when the value is actually used.
        return self;
    }

    /**
     * @brief Pre-decrement operator
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @details Implements: --proxy
     * Returns the value after decrement.
     */
    Derived operator--(this auto&& self) noexcept
        requires ElementIncrementable<T>
    {
        const auto dec_operation = [&self] { self.execute_inplace_operation([](T& elem) { --elem; }); };

        const auto dec_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { --elem; }); };

        self.apply_inplace_operation(dec_operation, dec_operation_async);

        // Return the proxy, not a value. Reading the element here would have to
        // go through read_pure(), never `values[num]`: `values` is the ctor-time
        // snapshot, and the call above may have materialised the chunk (snapshot
        // now stale, possibly the cst sentinel) or DEFERRED the op (cell not yet
        // updated, while another rank's replay may be writing it -- TSan, v1
        // static 256, 8 threads: such a read raced the async body above). The
        // returned proxy's operator T() IS read_pure(): it pulls the offset to
        // applied state, then reads live through the chunk. So `y = ++a[i]`
        // gets the right value, and the read -- with its sync -- is only paid
        // when the value is actually used.
        return self;
    }

    /**
     * @brief Post-increment operator
     * @return Value before increment
     *
     * @details Implements: proxy++
     * Returns the original value before increment.
     */
    T operator++(this auto&& self, int) noexcept
        requires ElementIncrementable<T>
    {
        T temp(self);

        const auto inc_operation = [&self] { self.execute_inplace_operation([](T& elem) { elem++; }); };

        const auto inc_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { elem++; }); };

        self.apply_inplace_operation(inc_operation, inc_operation_async);
        return temp;
    }

    /**
     * @brief Post-decrement operator
     * @return Value before decrement
     *
     * @details Implements: proxy--
     * Returns the original value before decrement.
     */
    T operator--(this auto&& self, int) noexcept
        requires ElementIncrementable<T>
    {
        T          temp(self);
        const auto dec_operation = [&self] { self.execute_inplace_operation([](T& elem) { elem--; }); };

        const auto dec_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { elem--; }); };

        self.apply_inplace_operation(dec_operation, dec_operation_async);
        return temp;
    }
};

} // namespace shrink_array

#include "chunk/internal/proxy_base_inline.h"
