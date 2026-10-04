// Python binding for the CPU W4A16 prefill expert executor (wna16_cpu.h).
#include <torch/extension.h>

#include <memory>

#include "wna16_cpu.h"

namespace {

struct Wna16CpuPrefill {
  std::unique_ptr<wna16cpu::Executor> ex;
  wna16cpu::Dims d{};
  int num_experts;

  Wna16CpuPrefill(int threads, std::vector<int> cpus, int hidden, int inter, int down_groups, int k_off,
                  int num_experts_)
      : num_experts(num_experts_) {
    TORCH_CHECK(hidden % 128 == 0 && inter % 64 == 0 && (2 * inter) % 64 == 0, "unsupported wna16 shape");
    TORCH_CHECK(k_off % 8 == 0, "down k_off must be a multiple of 8");
    d.H = hidden;
    d.I = inter;
    d.dG = down_groups;
    d.k_off = k_off;
    d.gu_qw_s = int64_t(hidden / 8) * 2 * inter;
    d.gu_qz_s = int64_t(hidden / 128) * (2 * inter / 8);
    d.gu_sc_s = int64_t(hidden / 128) * 2 * inter;
    d.dn_qw_s = int64_t(inter / 8) * hidden;
    d.dn_qz_s = int64_t(down_groups) * (hidden / 8);
    d.dn_sc_s = int64_t(down_groups) * hidden;
    ex = std::make_unique<wna16cpu::Executor>(threads, cpus, d, num_experts);
  }

  static void check_bank(const torch::Tensor& t, at::ScalarType dt, int64_t per_expert, int64_t E, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a host tensor");
    TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
    TORCH_CHECK(t.is_contiguous() && t.size(0) == E && t.numel() == per_expert * E, name, " has the wrong shape");
  }

  void set_layer(int layer, torch::Tensor gu_qw, torch::Tensor gu_qz, torch::Tensor gu_sc, torch::Tensor dn_qw,
                 torch::Tensor dn_qz, torch::Tensor dn_sc) {
    const int64_t E = num_experts;
    check_bank(gu_qw, at::kInt, d.gu_qw_s, E, "gate_up");
    check_bank(gu_qz, at::kInt, d.gu_qz_s, E, "gate_up_zero");
    check_bank(gu_sc, at::kHalf, d.gu_sc_s, E, "gate_up_scale");
    check_bank(dn_qw, at::kInt, d.dn_qw_s, E, "down");
    check_bank(dn_qz, at::kInt, d.dn_qz_s, E, "down_zero");
    check_bank(dn_sc, at::kHalf, d.dn_sc_s, E, "down_scale");
    ex->set_layer(layer, wna16cpu::Layer{gu_qw.data_ptr<int32_t>(), gu_qz.data_ptr<int32_t>(),
                                         reinterpret_cast<const uint16_t*>(gu_sc.data_ptr<at::Half>()),
                                         dn_qw.data_ptr<int32_t>(), dn_qz.data_ptr<int32_t>(),
                                         reinterpret_cast<const uint16_t*>(dn_sc.data_ptr<at::Half>())});
  }

  // y[T, H] (bf16, or fp32 unrounded) = sum over routes with ids >= 0 of weight * expert(x); returns
  // #experts computed.
  int64_t forward(int layer, torch::Tensor x, torch::Tensor ids, torch::Tensor w, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == d.H,
                "x must be a contiguous host bf16 [T, hidden]");
    const int64_t T = x.size(0);
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.dim() == 2 &&
                    ids.size(0) >= T,
                "ids must be a contiguous host int32 [T, top_k]");
    TORCH_CHECK(w.device().is_cpu() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.sizes() == ids.sizes(),
                "weights must be a contiguous host float32 shaped like ids");
    const bool y32 = y.scalar_type() == at::kFloat;
    TORCH_CHECK(y.device().is_cpu() && (y32 || y.scalar_type() == at::kBFloat16) && y.is_contiguous() &&
                    y.size(0) >= T && y.size(1) == d.H,
                "y must be a contiguous host bf16 or float32 [T, hidden]");
    const int topk = int(ids.size(1));
    const int32_t* idp = ids.data_ptr<int32_t>();
    for (int64_t i = 0; i < T * topk; ++i)
      TORCH_CHECK(idp[i] < num_experts, "expert id out of range");
    py::gil_scoped_release release;
    uint16_t* yb = y32 ? nullptr : reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
    float* yf = y32 ? y.data_ptr<float>() : nullptr;
    return ex->forward(layer, reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()), int(T), idp,
                       w.data_ptr<float>(), topk, yb, yf);
  }

  int threads() const { return ex->threads(); }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Wna16CpuPrefill>(m, "Wna16CpuPrefill")
      .def(py::init<int, std::vector<int>, int, int, int, int, int>(), py::arg("threads"), py::arg("cpus"),
           py::arg("hidden"), py::arg("inter"), py::arg("down_groups"), py::arg("k_off"), py::arg("num_experts"))
      .def("set_layer", &Wna16CpuPrefill::set_layer)
      .def("forward", &Wna16CpuPrefill::forward)
      .def_property_readonly("threads", &Wna16CpuPrefill::threads);
}
