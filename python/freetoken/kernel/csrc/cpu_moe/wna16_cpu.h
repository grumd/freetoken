// CPU W4A16 (AutoGPTQ int4) routed-expert compute for prefill.
//
// The GPU offload path moves every missed expert over PCIe before it can compute it; on a
// host whose RAM bandwidth exceeds its PCIe link, computing part of the misses here, straight
// from the pinned host banks, finishes the layer sooner. Unlike the decode executor
// (cpu_moe_ext.cpp, one GEMV per (token, expert) route), work is grouped by expert so each
// expert's weights are read from RAM once however many tokens it serves.
//
// Bank layout (one expert, this rank's intermediate band I; K = input, N = output, last axis
// contiguous):  gate_up qweight [H/8, 2I] int32, code j of a word -> k = 8w + j;
//               gate_up qzeros [H/128, 2I/8], scales [H/128, 2I] fp16;
//               down qweight [I/8, H], qzeros [dG, H/8], scales [dG, H] (k shifted by k_off).
// W[n, k] = (code - (zero + 1)) * scale. The intermediates are rounded to bf16 where the GPU
// path stores bf16 (gate/up output, activation output, per-route down output).
#pragma once

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace wna16cpu {

inline float bf16_to_f32(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// round-to-nearest-even to bf16, returned as the bf16 bit pattern in the high half of each lane
inline __m512i bf16_bits_16(__m512 v) {
  __m512i u = _mm512_castps_si512(v);
  const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
  u = _mm512_add_epi32(u, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)));
  return _mm512_and_si512(u, _mm512_set1_epi32(int(0xffff0000u)));
}

inline __m512 round_bf16_16(__m512 v) { return _mm512_castsi512_ps(bf16_bits_16(v)); }

struct Layer {
  const int32_t* gu_qw;
  const int32_t* gu_qz;
  const uint16_t* gu_sc;
  const int32_t* dn_qw;
  const int32_t* dn_qz;
  const uint16_t* dn_sc;
};

struct Dims {
  int H, I, dG, k_off;
  int64_t gu_qw_s, gu_qz_s, gu_sc_s, dn_qw_s, dn_qz_s, dn_sc_s;  // per-expert strides (elements)
};

constexpr int MT = 4;  // tokens per register tile

// zero points (+1, GPTQ) for 16 columns starting at n (n % 16 == 0) as floats.
inline __m512 load_zero16(const int32_t* qz_row, int n) {
  const __m512i idx = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m512i sh = _mm512_set_epi32(28, 24, 20, 16, 12, 8, 4, 0, 28, 24, 20, 16, 12, 8, 4, 0);
  __m512i w = _mm512_castsi128_si512(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qz_row + n / 8)));
  w = _mm512_permutexvar_epi32(idx, w);
  w = _mm512_and_si512(_mm512_srlv_epi32(w, sh), _mm512_set1_epi32(15));
  return _mm512_cvtepi32_ps(_mm512_add_epi32(w, _mm512_set1_epi32(1)));
}

inline __m512 load_scale16(const uint16_t* sc_row, int n) {
  return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sc_row + n)));
}

// acc[m][v] += sum over `nwords` packed rows (stride `rs` int32) of code(k, n0 + 16v + lane) * x[m][k]
// for k = 8w + j. Each 16-bit half of a packed word holds four codes (k = 8w..8w+3 low,
// 8w+4..8w+7 high); a 16-bit shift by 4p puts codes p and p+4 in the low nibble of each half,
// vpermw maps them to bf16 through a 32-entry table (indexes use 5 bits, so the table repeats),
// and one vdpbf16ps multiplies the pair by the token's (x[8w+p], x[8w+p+4]). `xp[m]` holds
// those activation pairs, four per packed row, starting at the row block.
template <int M>
inline void strip_dot(const int32_t* rows, int64_t rs, int nwords, const uint32_t* const* xp, __m512 (&acc)[MT][4]) {
  const __m512i lut = _mm512_set_epi16(
      0x4170, 0x4160, 0x4150, 0x4140, 0x4130, 0x4120, 0x4110, 0x4100, 0x40e0, 0x40c0, 0x40a0, 0x4080, 0x4040, 0x4000, 0x3f80, 0x0000,
      0x4170, 0x4160, 0x4150, 0x4140, 0x4130, 0x4120, 0x4110, 0x4100, 0x40e0, 0x40c0, 0x40a0, 0x4080, 0x4040, 0x4000, 0x3f80, 0x0000);
  for (int w = 0; w < nwords; ++w) {
    const int32_t* r = rows + w * rs;
    const __m512i q0 = _mm512_loadu_si512(r);
    const __m512i q1 = _mm512_loadu_si512(r + 16);
    const __m512i q2 = _mm512_loadu_si512(r + 32);
    const __m512i q3 = _mm512_loadu_si512(r + 48);
#pragma GCC unroll 4
    for (int p = 0; p < 4; ++p) {
      const __m512bh c0 = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(q0, 4 * p), lut);
      const __m512bh c1 = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(q1, 4 * p), lut);
      const __m512bh c2 = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(q2, 4 * p), lut);
      const __m512bh c3 = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(q3, 4 * p), lut);
#pragma GCC unroll 4
      for (int m = 0; m < M; ++m) {
        const __m512bh xb = (__m512bh)_mm512_set1_epi32(int(xp[m][4 * w + p]));
        acc[m][0] = _mm512_dpbf16_ps(acc[m][0], c0, xb);
        acc[m][1] = _mm512_dpbf16_ps(acc[m][1], c1, xb);
        acc[m][2] = _mm512_dpbf16_ps(acc[m][2], c2, xb);
        acc[m][3] = _mm512_dpbf16_ps(acc[m][3], c3, xb);
      }
    }
  }
}

// bf16 activations [n] (n % 8 == 0) -> the pair layout strip_dot reads
inline void pack_pairs(const uint16_t* x, int n, uint32_t* out) {
  for (int w = 0; w < n / 8; ++w)
    for (int p = 0; p < 4; ++p) out[4 * w + p] = uint32_t(x[8 * w + p]) | (uint32_t(x[8 * w + p + 4]) << 16);
}

// out[m][n] += s[n] * (sum_k code * x - zero[n] * xsum[m]) over one quantization group's rows,
// for every 64-column strip of an N-wide output.
template <int M>
inline void group_gemm(const int32_t* qw, int N, int w0, int nwords, const int32_t* qz_row,
                       const uint16_t* sc_row, const uint32_t* const* x, const float* xsum, int xsum_stride,
                       float* out, int out_stride) {
  for (int n0 = 0; n0 < N; n0 += 64) {
    __m512 acc[MT][4];
    for (int m = 0; m < M; ++m)
      for (int v = 0; v < 4; ++v) acc[m][v] = _mm512_setzero_ps();
    strip_dot<M>(qw + int64_t(w0) * N + n0, N, nwords, x, acc);
    for (int v = 0; v < 4; ++v) {
      const int n = n0 + 16 * v;
      const __m512 z = load_zero16(qz_row, n);
      const __m512 s = load_scale16(sc_row, n);
      for (int m = 0; m < M; ++m) {
        float* o = out + m * out_stride + n;
        const __m512 corr = _mm512_fnmadd_ps(z, _mm512_set1_ps(xsum[m * xsum_stride]), acc[m][v]);
        _mm512_storeu_ps(o, _mm512_fmadd_ps(s, corr, _mm512_loadu_ps(o)));
      }
    }
  }
}

struct Scratch {
  std::vector<float> xs, gu, hs, dn;
  std::vector<uint32_t> x, h;
  std::vector<uint16_t> hb;
  void ensure(const Dims& d) {
    x.resize(size_t(MT) * d.H / 2);
    hb.resize(size_t(MT) * d.I);
    xs.resize(size_t(MT) * (d.H / 128));
    gu.resize(size_t(MT) * 2 * d.I);
    h.resize(size_t(MT) * d.I / 2);
    hs.resize(size_t(MT) * d.dG);
    dn.resize(size_t(MT) * d.H);
  }
};

struct Route {
  int token;
  float weight;
};

// One expert over `n` routes (n <= MT): y[token] += round_bf16(w * expert(x[token])).
template <int M>
void expert_tile(const Layer& L, const Dims& d, int e, const Route* rt, const uint16_t* x_bf16,
                 float* y, Scratch& s) {
  const int H = d.H, I = d.I, N1 = 2 * I, G1 = H / 128;
  const int32_t* gqw = L.gu_qw + e * d.gu_qw_s;
  const int32_t* gqz = L.gu_qz + e * d.gu_qz_s;
  const uint16_t* gsc = L.gu_sc + e * d.gu_sc_s;
  const int32_t* dqw = L.dn_qw + e * d.dn_qw_s;
  const int32_t* dqz = L.dn_qz + e * d.dn_qz_s;
  const uint16_t* dsc = L.dn_sc + e * d.dn_sc_s;

  uint32_t* xq = s.x.data();
  for (int m = 0; m < M; ++m) {
    const uint16_t* src = x_bf16 + int64_t(rt[m].token) * H;
    pack_pairs(src, H, xq + m * (H / 2));
    for (int g = 0; g < G1; ++g) {
      __m512 a = _mm512_setzero_ps();
      for (int k = 0; k < 128; k += 16) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + g * 128 + k));
        a = _mm512_add_ps(a, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16)));
      }
      s.xs[m * G1 + g] = _mm512_reduce_add_ps(a);
    }
  }

  // gate_up: one quantization group (16 packed rows) at a time, every column strip
  float* gu = s.gu.data();
  std::memset(gu, 0, sizeof(float) * M * N1);
  for (int g = 0; g < G1; ++g) {
    const uint32_t* xp[MT];
    for (int m = 0; m < M; ++m) xp[m] = xq + m * (H / 2) + g * 64;
    group_gemm<M>(gqw, N1, g * 16, 16, gqz + int64_t(g) * (N1 / 8), gsc + int64_t(g) * N1, xp,
                  s.xs.data() + g, G1, gu, N1);
  }

  // silu(gate) * up, bf16 at the GPU's store points; group sums for the down epilogue
  uint16_t* hb = s.hb.data();
  uint32_t* h = s.h.data();
  for (int m = 0; m < M; ++m) {
    for (int i = 0; i < I; ++i) {
      const float gt = round_bf16(gu[m * N1 + i]);
      const float up = round_bf16(gu[m * N1 + I + i]);
      hb[m * I + i] = f32_to_bf16(gt / (1.0f + std::exp(-gt)) * up);
    }
    pack_pairs(hb + m * I, I, h + m * (I / 2));
    for (int gi = 0; gi < d.dG; ++gi) {
      const int k_lo = std::max(0, gi * 128 - d.k_off), k_hi = std::min(I, (gi + 1) * 128 - d.k_off);
      float a = 0.f;
      for (int k = k_lo; k < k_hi; ++k) a += bf16_to_f32(hb[m * I + k]);
      s.hs[m * d.dG + gi] = a;
    }
  }

  float* dn = s.dn.data();
  std::memset(dn, 0, sizeof(float) * M * H);
  for (int gi = 0; gi < d.dG; ++gi) {
    const int k_lo = std::max(0, gi * 128 - d.k_off), k_hi = std::min(I, (gi + 1) * 128 - d.k_off);
    if (k_hi <= k_lo) continue;
    const uint32_t* hp[MT];
    for (int m = 0; m < M; ++m) hp[m] = h + m * (I / 2) + k_lo / 2;
    group_gemm<M>(dqw, H, k_lo / 8, (k_hi - k_lo) / 8, dqz + int64_t(gi) * (H / 8),
                  dsc + int64_t(gi) * H, hp, s.hs.data() + gi, d.dG, dn, H);
  }
  for (int m = 0; m < M; ++m) {
    float* yr = y + int64_t(rt[m].token) * H;
    const __m512 w = _mm512_set1_ps(rt[m].weight);
    const float* dr = dn + m * H;
    for (int n = 0; n < H; n += 16)
      _mm512_storeu_ps(yr + n, _mm512_add_ps(_mm512_loadu_ps(yr + n),
                                             round_bf16_16(_mm512_mul_ps(w, _mm512_loadu_ps(dr + n)))));
  }
}

inline void expert_routes(const Layer& L, const Dims& d, int e, const Route* rt, int n,
                          const uint16_t* x_bf16, float* y, Scratch& s) {
  for (int i = 0; i < n; i += MT) {
    switch (std::min(MT, n - i)) {
      case 1: expert_tile<1>(L, d, e, rt + i, x_bf16, y, s); break;
      case 2: expert_tile<2>(L, d, e, rt + i, x_bf16, y, s); break;
      case 3: expert_tile<3>(L, d, e, rt + i, x_bf16, y, s); break;
      default: expert_tile<4>(L, d, e, rt + i, x_bf16, y, s); break;
    }
  }
}

// Persistent pinned worker pool: run(fn) executes fn(tid) on every worker and returns when all
// are done. Workers spin briefly between jobs (layers arrive milliseconds apart), then sleep.
class Pool {
 public:
  Pool(int n, const std::vector<int>& cpus) : n_(n) {
    for (int t = 0; t < n; ++t) {
      int cpu = cpus.empty() ? -1 : cpus[t % cpus.size()];
      th_.emplace_back([this, t, cpu] { loop(t, cpu); });
    }
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      gen_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  int size() const { return n_; }
  void run(const std::function<void(int)>& fn) {
    job_ = &fn;
    done_.store(0);
    {
      std::lock_guard<std::mutex> lk(mu_);
      gen_.fetch_add(1);
    }
    cv_.notify_all();
    while (done_.load(std::memory_order_acquire) < n_) _mm_pause();
    job_ = nullptr;
  }

 private:
  void loop(int t, int cpu) {
    if (cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    uint64_t seen = gen_.load();
    while (true) {
      int spins = 0;
      while (gen_.load(std::memory_order_acquire) == seen && spins < 200000) {
        _mm_pause();
        ++spins;
      }
      if (gen_.load(std::memory_order_acquire) == seen) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return gen_.load() != seen; });
      }
      seen = gen_.load();
      if (stop_) return;
      (*job_)(t);
      done_.fetch_add(1, std::memory_order_release);
    }
  }
  int n_;
  std::vector<std::thread> th_;
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> done_{0};
  const std::function<void(int)>* job_ = nullptr;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

// Routed-expert sum for one layer: y[T, H] = sum over routes with id >= 0 of w * expert_id(x),
// written as bf16 (y) or unrounded fp32 (y32, for a caller that adds it to another partial
// before rounding once). Experts are bucketed and handed out largest-first; each worker
// accumulates into its own fp32 rows, then the rows are reduced in parallel.
class Executor {
 public:
  Executor(int threads, const std::vector<int>& cpus, const Dims& d, int num_experts)
      : d_(d), E_(num_experts), pool_(threads, cpus), scratch_(threads), acc_(threads) {
    for (auto& s : scratch_) s.ensure(d_);
  }
  void set_layer(int layer, const Layer& L) {
    if (int(layers_.size()) <= layer) layers_.resize(layer + 1);
    layers_[layer] = L;
  }
  int threads() const { return pool_.size(); }

  // returns the number of distinct experts computed
  int forward(int layer, const uint16_t* x, int T, const int32_t* ids, const float* w, int topk,
              uint16_t* y, float* y32) {
    const Layer& L = layers_.at(layer);
    const int H = d_.H, nt = pool_.size();
    count_.assign(E_ + 1, 0);
    for (int i = 0; i < T * topk; ++i)
      if (ids[i] >= 0) ++count_[ids[i] + 1];
    for (int e = 0; e < E_; ++e) count_[e + 1] += count_[e];
    routes_.resize(count_[E_]);
    fill_.assign(count_.begin(), count_.end() - 1);
    for (int i = 0; i < T * topk; ++i)
      if (ids[i] >= 0) routes_[fill_[ids[i]]++] = Route{i / topk, w[i]};
    order_.clear();
    for (int e = 0; e < E_; ++e)
      if (count_[e + 1] > count_[e]) order_.push_back(e);
    std::sort(order_.begin(), order_.end(), [&](int a, int b) {
      return count_[a + 1] - count_[a] > count_[b + 1] - count_[b];
    });
    for (auto& a : acc_) {
      a.rows.resize(size_t(T) * H);
      a.touched.assign(T, 0);
    }
    next_.store(0);
    const int nexp = int(order_.size());
    std::function<void(int)> work = [&](int tid) {
      Acc& a = acc_[tid];
      Scratch& s = scratch_[tid];
      for (int i; (i = next_.fetch_add(1)) < nexp;) {
        const int e = order_[i];
        const Route* rt = routes_.data() + count_[e];
        const int n = count_[e + 1] - count_[e];
        for (int r = 0; r < n; ++r)
          if (!a.touched[rt[r].token]) {
            a.touched[rt[r].token] = 1;
            std::memset(a.rows.data() + int64_t(rt[r].token) * H, 0, sizeof(float) * H);
          }
        expert_routes(L, d_, e, rt, n, x, a.rows.data(), s);
      }
    };
    pool_.run(work);
    std::function<void(int)> reduce = [&](int tid) {
      for (int t = tid; t < T; t += nt) {
        uint16_t* yr = y32 ? nullptr : y + int64_t(t) * H;
        float* yr32 = y32 ? y32 + int64_t(t) * H : nullptr;
        bool any = false;
        for (int k = 0; k < nt; ++k) any |= acc_[k].touched[t] != 0;
        if (!any) {
          if (yr32)
            std::memset(yr32, 0, sizeof(float) * H);
          else
            std::memset(yr, 0, sizeof(uint16_t) * H);
          continue;
        }
        for (int n = 0; n < H; n += 16) {
          __m512 sum = _mm512_setzero_ps();
          for (int k = 0; k < nt; ++k)
            if (acc_[k].touched[t]) sum = _mm512_add_ps(sum, _mm512_loadu_ps(acc_[k].rows.data() + int64_t(t) * H + n));
          if (yr32) {
            _mm512_storeu_ps(yr32 + n, sum);
          } else {
            const __m512i u = _mm512_srli_epi32(bf16_bits_16(sum), 16);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(yr + n), _mm512_cvtepi32_epi16(u));
          }
        }
      }
    };
    pool_.run(reduce);
    return nexp;
  }

 private:
  struct Acc {
    std::vector<float> rows;
    std::vector<uint8_t> touched;
  };
  Dims d_;
  int E_;
  Pool pool_;
  std::vector<Scratch> scratch_;
  std::vector<Acc> acc_;
  std::vector<Layer> layers_;
  std::vector<int> count_, fill_, order_;
  std::vector<Route> routes_;
  std::atomic<int> next_{0};
};

}  // namespace wna16cpu
