#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// Two-rank bf16 all-reduce over GPU peer access, NCCL's LL protocol in one CTA. Each rank
// pushes its vector into the PEER's VRAM inbox as 8-byte lines of {2 bf16, 32-bit sequence
// flag}; a line is valid exactly when its flag matches, so there is no __threadfence_system,
// no separate flag word, and every thread polls its own lines in local VRAM instead of
// crossing PCIe per poll. The sequence counter lives on the device, so a captured graph
// replays correctly. The inbox has two parity slots: a rank can only reach call seq+2 (reusing
// a slot) after its peer's seq+1 lines arrived, which the peer sends only after it finished
// reading slot seq.

namespace p2p_allreduce {

constexpr int kThreads = 512;

__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
    const auto fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a));
    const auto fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b));
    const auto r = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
    return *reinterpret_cast<const uint32_t*>(&r);
}

__global__ void __launch_bounds__(kThreads) kernel(
    uint32_t* __restrict__ x, int64_t n_words, uint64_t* seq_dev, const uint2* my_lines,
    uint2* peer_lines, int64_t slot_lines) {
    const uint64_t seq = *seq_dev + 1;
    const uint32_t flag = static_cast<uint32_t>(seq);
    const int64_t off = static_cast<int64_t>(seq & 1) * slot_lines;
    for (int64_t i = threadIdx.x; i < n_words; i += kThreads) {
        asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(peer_lines + off + i), "r"(x[i]),
                     "r"(flag)
                     : "memory");
    }
    for (int64_t i = threadIdx.x; i < n_words; i += kThreads) {
        uint32_t d, f;
        do {
            asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                         : "=r"(d), "=r"(f)
                         : "l"(my_lines + off + i)
                         : "memory");
        } while (f != flag);
        // a + b == b + a in IEEE arithmetic, so both ranks store bit-identical sums
        x[i] = add2(x[i], d);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *seq_dev = seq;
    }
}

// The same lines, slots and counter as `kernel`, so the two interleave freely in one stream:
// each rank's words are copied into its half of `out` instead of summed. `x` may be that half.
__global__ void __launch_bounds__(kThreads) gather_kernel(
    const uint32_t* x, uint32_t* out, int64_t n_words, int64_t rank, uint64_t* seq_dev,
    const uint2* my_lines, uint2* peer_lines, int64_t slot_lines) {
    const uint64_t seq = *seq_dev + 1;
    const uint32_t flag = static_cast<uint32_t>(seq);
    const int64_t off = static_cast<int64_t>(seq & 1) * slot_lines;
    uint32_t* mine = out + rank * n_words;
    uint32_t* theirs = out + (1 - rank) * n_words;
    for (int64_t i = threadIdx.x; i < n_words; i += kThreads) {
        const uint32_t v = x[i];
        asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(peer_lines + off + i), "r"(v),
                     "r"(flag)
                     : "memory");
        mine[i] = v;
    }
    for (int64_t i = threadIdx.x; i < n_words; i += kThreads) {
        uint32_t d, f;
        do {
            asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                         : "=r"(d), "=r"(f)
                         : "l"(my_lines + off + i)
                         : "memory");
        } while (f != flag);
        theirs[i] = d;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *seq_dev = seq;
    }
}

}  // namespace p2p_allreduce

inline void p2p_allreduce_bf16(
    tvm::ffi::TensorView x, tvm::ffi::TensorView seq, int64_t my_inbox, int64_t peer_inbox,
    int64_t slot_bytes) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto N = SymbolicSize{"numel"};
    auto x_dtype = SymbolicDType{};
    auto seq_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int16_t>(x_dtype).with_device<kDLCUDA>(device).verify(x);
    TensorMatcher({1}).with_dtype<int64_t>(seq_dtype).with_device<kDLCUDA>(device).verify(seq);
    const int64_t n = static_cast<int64_t>(N.unwrap());
    // a bf16 pair becomes one 8-byte line: a slot of slot_bytes holds slot_bytes / 4 elements
    RuntimeCheck(n % 2 == 0 && n * 4 <= slot_bytes, "p2p_allreduce: bad size");
    LaunchKernel(1, p2p_allreduce::kThreads, device.unwrap())(
        p2p_allreduce::kernel, static_cast<uint32_t*>(x.data_ptr()), n / 2,
        static_cast<uint64_t*>(seq.data_ptr()), reinterpret_cast<const uint2*>(my_inbox),
        reinterpret_cast<uint2*>(peer_inbox), slot_bytes / 8);
}

inline void p2p_allgather_bf16(
    tvm::ffi::TensorView x, tvm::ffi::TensorView out, tvm::ffi::TensorView seq, int64_t rank,
    int64_t my_inbox, int64_t peer_inbox, int64_t slot_bytes) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto N = SymbolicSize{"numel"};
    auto M = SymbolicSize{"out_numel"};
    auto x_dtype = SymbolicDType{};
    auto out_dtype = SymbolicDType{};
    auto seq_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int16_t>(x_dtype).with_device<kDLCUDA>(device).verify(x);
    TensorMatcher({M}).with_dtype<int16_t>(out_dtype).with_device<kDLCUDA>(device).verify(out);
    TensorMatcher({1}).with_dtype<int64_t>(seq_dtype).with_device<kDLCUDA>(device).verify(seq);
    const int64_t n = static_cast<int64_t>(N.unwrap());
    RuntimeCheck(static_cast<int64_t>(M.unwrap()) == 2 * n, "p2p_allgather: out must hold both halves");
    RuntimeCheck(n % 2 == 0 && n * 4 <= slot_bytes && (rank == 0 || rank == 1), "p2p_allgather: bad size");
    LaunchKernel(1, p2p_allreduce::kThreads, device.unwrap())(
        p2p_allreduce::gather_kernel, static_cast<const uint32_t*>(x.data_ptr()),
        static_cast<uint32_t*>(out.data_ptr()), n / 2, rank, static_cast<uint64_t*>(seq.data_ptr()),
        reinterpret_cast<const uint2*>(my_inbox), reinterpret_cast<uint2*>(peer_inbox), slot_bytes / 8);
}
