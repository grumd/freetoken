// CPU NVFP4 (ModelOpt e2m1 + e4m3 per-16 scales + fp32 row global) routed-expert compute for
// prefill: the NVFP4 policy for wna16cpu::ExecutorT (same routing, threading and reduction as
// the W4A16 executor, see wna16_cpu.h).
//
// Bank layout (one expert, this rank's intermediate band I; rows = outputs, K contiguous):
//   gate_up packed [2I, H/2] u8 (low nibble = even k), scale [2I, H/16] e4m3, global [2I] f32;
//   down packed [H, I/2], scale [H, I/16], global [H].
// W[n, k] = e2m1(code) * e4m3(scale[n, k/16]) * global[n].
//
// A 64-k chunk of a row is 32 bytes. Widened to 16-bit lanes, the low nibbles index an e2m1 ->
// bf16 table (exact) for k = 64c + 2i and the high nibbles for k = 64c + 2i + 1. vdpbf16ps pairs
// words (2j, 2j+1), so the activations are packed to match: lane j takes (x[64c+4j], x[64c+4j+2])
// against the low vector and (x[64c+4j+1], x[64c+4j+3]) against the high one. Every lane then
// covers k in [64c+4j, 64c+4j+3], one 16-group (4c + j/4), so the block scale multiplies the
// fresh lane sums in fp32: the weights stay exact, as on the GPU. gate/up stay fp32 into
// SiLU * up; the activation is rounded to bf16 where the GPU stores it.
#pragma once

#include "wna16_cpu.h"

namespace nvfp4cpu {

using wna16cpu::f32_to_bf16;
using wna16cpu::Route;

struct Layer {
  const uint8_t* gu_p;
  const uint8_t* gu_s;
  const float* gu_g;
  const uint8_t* dn_p;
  const uint8_t* dn_s;
  const float* dn_g;
};

struct Dims {
  int H, I;
  int64_t gu_p_s, gu_s_s, gu_g_s, dn_p_s, dn_s_s, dn_g_s;  // per-expert strides (elements)
};

constexpr int MT = 8;  // tokens per tile: a chunk is unpacked once for all of them

inline __m512i e2m1_bf16_lut() {
  // index = 4-bit code (the table repeats: vpermw reads 5 index bits)
  return _mm512_set_epi16(
      int16_t(0xC0C0), int16_t(0xC080), int16_t(0xC040), int16_t(0xC000), int16_t(0xBFC0), int16_t(0xBF80),
      int16_t(0xBF00), int16_t(0x8000), 0x40C0, 0x4080, 0x4040, 0x4000, 0x3FC0, 0x3F80, 0x3F00, 0x0000,
      int16_t(0xC0C0), int16_t(0xC080), int16_t(0xC040), int16_t(0xC000), int16_t(0xBFC0), int16_t(0xBF80),
      int16_t(0xBF00), int16_t(0x8000), 0x40C0, 0x4080, 0x4040, 0x4000, 0x3FC0, 0x3F80, 0x3F00, 0x0000);
}

// bf16 activations [K] (K % 64 == 0) -> the lane pairs described above: lo[K/4], hi[K/4]
inline void pack_pairs(const uint16_t* x, int K, uint32_t* lo, uint32_t* hi) {
  for (int c = 0; c < K / 64; ++c)
    for (int j = 0; j < 16; ++j) {
      const uint16_t* b = x + 64 * c + 4 * j;
      lo[16 * c + j] = uint32_t(b[0]) | (uint32_t(b[2]) << 16);
      hi[16 * c + j] = uint32_t(b[1]) | (uint32_t(b[3]) << 16);
    }
}

// four e4m3 scale bytes -> 16 lanes (lane j takes byte j / 4) as fp32 * 2^-8
inline __m512 scale_lanes(const uint8_t* s4) {
  const __m128i spread = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
  uint32_t v;
  std::memcpy(&v, s4, 4);
  const __m512i b = _mm512_cvtepu8_epi32(_mm_shuffle_epi8(_mm_cvtsi32_si128(int(v)), spread));
  const __m512i h = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 8),
                                    _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x7F)), 7));
  return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(h));
}

// out[m * ld + r] = global[r] * sum_k W[r, k] * x_m[k] for rows [0, rows), K = 64 * nchunks
template <int M>
inline void rows_dot(const uint8_t* P, const uint8_t* S, const float* G, int rows, int nchunks,
                     const uint32_t* const* xlo, const uint32_t* const* xhi, float* out, int ld) {
  const __m512i lut = e2m1_bf16_lut();
  const __m512i low4 = _mm512_set1_epi16(0xF);
  const int64_t prow = int64_t(nchunks) * 32, srow = int64_t(nchunks) * 4;
  for (int r = 0; r < rows; ++r) {
    const uint8_t* pr = P + r * prow;
    const uint8_t* sr = S + r * srow;
    __m512 acc[M];
    for (int m = 0; m < M; ++m) acc[m] = _mm512_setzero_ps();
    for (int c = 0; c < nchunks; ++c) {
      const __m512i w = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pr + 32 * c)));
      const __m512bh wlo = (__m512bh)_mm512_permutexvar_epi16(_mm512_and_si512(w, low4), lut);
      const __m512bh whi = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(w, 4), lut);
      const __m512 sv = scale_lanes(sr + 4 * c);
#pragma GCC unroll 8
      for (int m = 0; m < M; ++m) {
        __m512 t = _mm512_dpbf16_ps(_mm512_setzero_ps(), wlo,
                                    (__m512bh)_mm512_loadu_si512(xlo[m] + 16 * c));
        t = _mm512_dpbf16_ps(t, whi, (__m512bh)_mm512_loadu_si512(xhi[m] + 16 * c));
        acc[m] = _mm512_fmadd_ps(t, sv, acc[m]);
      }
    }
    const float g = G[r] * 256.0f;
    for (int m = 0; m < M; ++m) out[m * ld + r] = _mm512_reduce_add_ps(acc[m]) * g;
  }
}

struct Scratch {
  std::vector<uint32_t> xlo, xhi, hlo, hhi;
  std::vector<float> gu, dn;
  std::vector<uint16_t> hb;
  void ensure(const Dims& d) {
    xlo.resize(size_t(MT) * d.H / 4);
    xhi.resize(size_t(MT) * d.H / 4);
    hlo.resize(size_t(MT) * d.I / 4);
    hhi.resize(size_t(MT) * d.I / 4);
    gu.resize(size_t(MT) * 2 * d.I);
    dn.resize(size_t(MT) * d.H);
    hb.resize(d.I);
  }
};

// One expert over `n` routes (n <= MT): y[token] += w * expert(x[token]) in fp32.
template <int M>
void expert_tile(const Layer& L, const Dims& d, int e, const Route* rt, const uint16_t* x_bf16, float* y,
                 Scratch& s) {
  const int H = d.H, I = d.I;
  const uint32_t* xl[MT];
  const uint32_t* xh[MT];
  for (int m = 0; m < M; ++m) {
    uint32_t* lo = s.xlo.data() + m * (H / 4);
    uint32_t* hi = s.xhi.data() + m * (H / 4);
    pack_pairs(x_bf16 + int64_t(rt[m].token) * H, H, lo, hi);
    xl[m] = lo;
    xh[m] = hi;
  }
  rows_dot<M>(L.gu_p + e * d.gu_p_s, L.gu_s + e * d.gu_s_s, L.gu_g + e * d.gu_g_s, 2 * I, H / 64, xl, xh,
              s.gu.data(), 2 * I);
  const uint32_t* hl[MT];
  const uint32_t* hh[MT];
  for (int m = 0; m < M; ++m) {
    const float* g = s.gu.data() + m * 2 * I;
    for (int i = 0; i < I; ++i) s.hb[i] = f32_to_bf16(g[i] / (1.0f + std::exp(-g[i])) * g[I + i]);
    uint32_t* lo = s.hlo.data() + m * (I / 4);
    uint32_t* hi = s.hhi.data() + m * (I / 4);
    pack_pairs(s.hb.data(), I, lo, hi);
    hl[m] = lo;
    hh[m] = hi;
  }
  rows_dot<M>(L.dn_p + e * d.dn_p_s, L.dn_s + e * d.dn_s_s, L.dn_g + e * d.dn_g_s, H, I / 64, hl, hh,
              s.dn.data(), H);
  for (int m = 0; m < M; ++m) {
    float* yr = y + int64_t(rt[m].token) * H;
    const __m512 w = _mm512_set1_ps(rt[m].weight);
    const float* dr = s.dn.data() + m * H;
    for (int n = 0; n < H; n += 16)
      _mm512_storeu_ps(yr + n, _mm512_fmadd_ps(w, _mm512_loadu_ps(dr + n), _mm512_loadu_ps(yr + n)));
  }
}

struct Nvfp4Kernel {
  using Layer = nvfp4cpu::Layer;
  using Dims = nvfp4cpu::Dims;
  using Scratch = nvfp4cpu::Scratch;
  // `precise` is moot: the weights are exact and gate/up always stay fp32
  static void routes(const Layer& L, const Dims& d, int e, const Route* rt, int n, const uint16_t* x, float* y,
                     Scratch& s, bool /*precise*/) {
    for (int i = 0; i < n; i += MT) {
      switch (std::min(MT, n - i)) {
        case 1: expert_tile<1>(L, d, e, rt + i, x, y, s); break;
        case 2: expert_tile<2>(L, d, e, rt + i, x, y, s); break;
        case 3: expert_tile<3>(L, d, e, rt + i, x, y, s); break;
        case 4: expert_tile<4>(L, d, e, rt + i, x, y, s); break;
        case 5: expert_tile<5>(L, d, e, rt + i, x, y, s); break;
        case 6: expert_tile<6>(L, d, e, rt + i, x, y, s); break;
        case 7: expert_tile<7>(L, d, e, rt + i, x, y, s); break;
        default: expert_tile<8>(L, d, e, rt + i, x, y, s); break;
      }
    }
  }
};

using Executor = wna16cpu::ExecutorT<Nvfp4Kernel>;

}  // namespace nvfp4cpu
