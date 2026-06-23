// sherpa-onnx/python/csrc/offline-tts-moss-model-config.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/python/csrc/offline-tts-moss-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-moss-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsMossModelConfig(py::module *m) {
  using PyClass = OfflineTtsMossModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsMossModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("prefill"), py::arg("decode_step"),
           py::arg("local_fixed_sampled_frame"), py::arg("codec_encoder"),
           py::arg("codec_decoder"), py::arg("tokenizer_vocab"),
           py::arg("tokenizer_scores"))
      .def_readwrite("prefill", &PyClass::prefill)
      .def_readwrite("decode_step", &PyClass::decode_step)
      .def_readwrite("local_fixed_sampled_frame",
                     &PyClass::local_fixed_sampled_frame)
      .def_readwrite("codec_encoder", &PyClass::codec_encoder)
      .def_readwrite("codec_decoder", &PyClass::codec_decoder)
      .def_readwrite("tokenizer_vocab", &PyClass::tokenizer_vocab)
      .def_readwrite("tokenizer_scores", &PyClass::tokenizer_scores)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
