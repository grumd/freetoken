#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// Two-rank bf16 all-reduce through pinned host memory shared by both processes, for
// GPUs without peer access. One CTA: publish the local vector, release a sequence flag,
// acquire the peer's flag, read the peer's vector and sum. Each call bumps a device-side
// sequence counter, so a captured graph replays correctly; the data slots alternate on
// the sequence parity, which keeps a rank from overwriting a slot its peer still reads.

namespace host_allreduce {

constexpr int kThreads = 512;
constexpr int kFlagStride = 16;  // uint64 words: one 128-byte line per rank's flag

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
    const auto fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a));
    const auto fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b));
    const auto r = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
    return *reinterpret_cast<const uint32_t*>(&r);
}

__global__ void __launch_bounds__(kThreads) kernel(
    uint4* __restrict__ x, int64_t n_vec, uint64_t* seq_dev, uint64_t* flags, uint4* data,
    int64_t slot_vec, int rank) {
    __shared__ uint64_t s_seq;
    if (threadIdx.x == 0) {
        s_seq = *seq_dev + 1;
    }
    __syncthreads();
    const uint64_t seq = s_seq;
    const int64_t par = static_cast<int64_t>(seq & 1);
    uint4* mine = data + (static_cast<int64_t>(rank) * 2 + par) * slot_vec;
    const uint4* peer = data + (static_cast<int64_t>(1 - rank) * 2 + par) * slot_vec;

    for (int64_t i = threadIdx.x; i < n_vec; i += kThreads) {
        mine[i] = x[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        st_release_sys(flags + rank * kFlagStride, seq);
        const uint64_t* peer_flag = flags + (1 - rank) * kFlagStride;
        while (ld_acquire_sys(peer_flag) < seq) {
        }
        *seq_dev = seq;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < n_vec; i += kThreads) {
        const uint4 a = x[i];
        const uint4 b = ld_volatile(peer + i);
        // a + b == b + a in IEEE arithmetic, so both ranks store bit-identical sums
        x[i] = uint4{add2(a.x, b.x), add2(a.y, b.y), add2(a.z, b.z), add2(a.w, b.w)};
    }
}

}  // namespace host_allreduce

inline void host_allreduce_bf16(
    tvm::ffi::TensorView x, tvm::ffi::TensorView seq, int64_t flags_ptr, int64_t data_ptr,
    int64_t slot_bytes, int64_t rank) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto N = SymbolicSize{"numel"};
    auto x_dtype = SymbolicDType{};
    auto seq_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int16_t>(x_dtype).with_device<kDLCUDA>(device).verify(x);
    TensorMatcher({1}).with_dtype<int64_t>(seq_dtype).with_device<kDLCUDA>(device).verify(seq);
    const int64_t n = static_cast<int64_t>(N.unwrap());
    RuntimeCheck(n % 8 == 0 && n * 2 <= slot_bytes, "host_allreduce: bad size");
    LaunchKernel(1, host_allreduce::kThreads, device.unwrap())(
        host_allreduce::kernel, static_cast<uint4*>(x.data_ptr()), n / 8,
        static_cast<uint64_t*>(seq.data_ptr()), reinterpret_cast<uint64_t*>(flags_ptr),
        reinterpret_cast<uint4*>(data_ptr), slot_bytes / 16, static_cast<int>(rank));
}
