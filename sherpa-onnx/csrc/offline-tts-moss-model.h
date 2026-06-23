// sherpa-onnx/csrc/offline-tts-moss-model.h
//
// Copyright (c)  2026  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

struct MossGlobalOutput {
  Ort::Value global_hidden_state{nullptr};
  std::vector<Ort::Value> present_kv;
};

struct MossSampledFrame {
  bool should_continue = false;
  std::vector<int32_t> frame_token_ids;
};

struct MossCodecDecodeOutput {
  Ort::Value audio{nullptr};
  Ort::Value audio_lengths{nullptr};
};

struct MossCodecEncodeOutput {
  Ort::Value audio_codes{nullptr};
  Ort::Value audio_code_lengths{nullptr};
};

class OfflineTtsMossModel {
 public:
  explicit OfflineTtsMossModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsMossModel(Manager *mgr, const OfflineTtsModelConfig &config);

  ~OfflineTtsMossModel();

  int32_t NumLayers() const;
  int32_t NumHeads() const;
  int32_t HeadDim() const;
  int32_t HiddenSize() const;

  const std::string &ModelConfigJson() const;
  const std::string &PromptTemplatesJson() const;
  const std::string &GenerationDefaultsJson() const;
  const std::string &DefaultPromptAudioCodesJson() const;

  MossGlobalOutput RunPrefill(Ort::Value input_ids,
                              Ort::Value attention_mask) const;

  MossGlobalOutput RunDecodeStep(Ort::Value input_ids,
                                 Ort::Value past_valid_lengths,
                                 std::vector<Ort::Value> *past_kv) const;

  MossSampledFrame RunLocalFixedSampledFrame(Ort::Value global_hidden,
                                             Ort::Value repetition_seen_mask,
                                             Ort::Value assistant_random_u,
                                             Ort::Value audio_random_u) const;

  MossCodecEncodeOutput RunCodecEncode(Ort::Value input_values,
                                       Ort::Value input_lengths) const;

  MossCodecDecodeOutput RunCodecDecode(Ort::Value audio_codes,
                                       Ort::Value audio_codes_lengths) const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_MODEL_H_
