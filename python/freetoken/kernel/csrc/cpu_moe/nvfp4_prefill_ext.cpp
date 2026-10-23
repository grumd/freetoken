// Python binding for the CPU NVFP4 prefill expert executor (nvfp4_cpu.h).
#include <torch/extension.h>

#include <memory>

#include "nvfp4_cpu.h"

namespace {

struct Nvfp4CpuPrefill {
  std::unique_ptr<nvfp4cpu::Executor> ex;
  nvfp4cpu::Dims d{};
  int num_experts;

  Nvfp4CpuPrefill(int threads, std::vector<int> cpus, int hidden, int inter, int num_experts_)
      : num_experts(num_experts_) {
    TORCH_CHECK(hidden % 64 == 0 && inter % 64 == 0, "unsupported nvfp4 shape (hidden and inter must be multiples of 64)");
    d.H = hidden;
    d.I = inter;
    d.gu_p_s = int64_t(2 * inter) * (hidden / 2);
    d.gu_s_s = int64_t(2 * inter) * (hidden / 16);
    d.gu_g_s = 2 * inter;
    d.dn_p_s = int64_t(hidden) * (inter / 2);
    d.dn_s_s = int64_t(hidden) * (inter / 16);
    d.dn_g_s = hidden;
    ex = std::make_unique<nvfp4cpu::Executor>(threads, cpus, d, num_experts);
  }

  static void check_bank(const torch::Tensor& t, int64_t elem, int64_t per_expert, int64_t E, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a host tensor");
    TORCH_CHECK(t.element_size() == elem, name, " has the wrong element size");
    TORCH_CHECK(t.is_contiguous() && t.size(0) == E && t.numel() == per_expert * E, name, " has the wrong shape");
  }

  void set_layer(int layer, torch::Tensor gu_p, torch::Tensor gu_s, torch::Tensor gu_g, torch::Tensor dn_p,
                 torch::Tensor dn_s, torch::Tensor dn_g) {
    const int64_t E = num_experts;
    check_bank(gu_p, 1, d.gu_p_s, E, "gate_up");
    check_bank(gu_s, 1, d.gu_s_s, E, "gate_up_scale");
    check_bank(gu_g, 4, d.gu_g_s, E, "gate_up_global");
    check_bank(dn_p, 1, d.dn_p_s, E, "down");
    check_bank(dn_s, 1, d.dn_s_s, E, "down_scale");
    check_bank(dn_g, 4, d.dn_g_s, E, "down_global");
    TORCH_CHECK(gu_g.scalar_type() == at::kFloat && dn_g.scalar_type() == at::kFloat, "globals must be float32");
    auto u8 = [](const torch::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); };
    ex->set_layer(layer, nvfp4cpu::Layer{u8(gu_p), u8(gu_s), gu_g.data_ptr<float>(), u8(dn_p), u8(dn_s),
                                         dn_g.data_ptr<float>()});
  }

  int64_t forward(int layer, torch::Tensor x, torch::Tensor ids, torch::Tensor w, torch::Tensor y, bool precise) {
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
    for (int64_t i = 0; i < T * topk; ++i) TORCH_CHECK(idp[i] < num_experts, "expert id out of range");
    py::gil_scoped_release release;
    uint16_t* yb = y32 ? nullptr : reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
    float* yf = y32 ? y.data_ptr<float>() : nullptr;
    return ex->forward(layer, reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()), int(T), idp,
                       w.data_ptr<float>(), topk, yb, yf, precise);
  }

  int threads() const { return ex->threads(); }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Nvfp4CpuPrefill>(m, "Nvfp4CpuPrefill")
      .def(py::init<int, std::vector<int>, int, int, int>(), py::arg("threads"), py::arg("cpus"), py::arg("hidden"),
           py::arg("inter"), py::arg("num_experts"))
      .def("set_layer", &Nvfp4CpuPrefill::set_layer)
      .def("forward", &Nvfp4CpuPrefill::forward, py::arg("layer"), py::arg("x"), py::arg("ids"), py::arg("w"),
           py::arg("y"), py::arg("precise") = false)
      .def_property_readonly("threads", &Nvfp4CpuPrefill::threads);
}
