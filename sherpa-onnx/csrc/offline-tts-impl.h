// sherpa-onnx/csrc/offline-tts-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/fst-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsImpl {
 public:
  virtual ~OfflineTtsImpl() = default;

  static std::unique_ptr<OfflineTtsImpl> Create(const OfflineTtsConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineTtsImpl> Create(Manager *mgr,
                                                const OfflineTtsConfig &config);

  [[deprecated("Use Generate(text, GenerationConfig, callback) instead")]]
  virtual GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const {
    SHERPA_ONNX_LOGE("Not implemented yet. Only some models support this");
    SHERPA_ONNX_LOGE("Please use sherpa-onnx > v1.12.30");
    return {};
  }

  virtual GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const {
    SHERPA_ONNX_LOGE("Not implemented yet. Only some models support this");
    return {};
  }

  virtual GeneratedAudio Generate(
      const std::string &text, const std::string &prompt_text,
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float speed = 1.0, int32_t num_step = 4,
      GeneratedAudioCallback callback = nullptr) const {
    SHERPA_ONNX_LOGE("Not implemented yet. Only some models support this");
    return {};
  }

  // Return the sample rate of the generated audio
  virtual int32_t SampleRate() const = 0;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  virtual int32_t NumSpeakers() const { return 1; }

  std::vector<int64_t> AddBlank(const std::vector<int64_t> &x,
                                int32_t blank_id = 0) const;

 protected:
  using TextNormalizerList =
      std::vector<std::unique_ptr<kaldifst::TextNormalizer>>;

  void InitTextNormalizer(const OfflineTtsConfig &config,
                          TextNormalizerList *tn_list) const;

  template <typename Manager>
  void InitTextNormalizer(Manager *mgr, const OfflineTtsConfig &config,
                          TextNormalizerList *tn_list) const;

  std::string ApplyTextNormalizer(std::string text,
                                  const TextNormalizerList &tn_list,
                                  bool debug) const;
};

template <typename Manager>
void OfflineTtsImpl::InitTextNormalizer(Manager *mgr,
                                        const OfflineTtsConfig &config,
                                        TextNormalizerList *tn_list) const {
  if (!config.rule_fsts.empty()) {
    std::vector<std::string> files = SplitStringAndTrim(config.rule_fsts, ',');
    tn_list->reserve(files.size());
    for (const auto &f : files) {
      if (config.model.debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
      }
      auto buf = ReadFile(mgr, f);
      std::istringstream is(std::string(buf.data(), buf.size()));
      tn_list->push_back(std::make_unique<kaldifst::TextNormalizer>(is));
    }
  }

  if (!config.rule_fars.empty()) {
    std::vector<std::string> files = SplitStringAndTrim(config.rule_fars, ',');
    tn_list->reserve(files.size() + tn_list->size());

    for (const auto &f : files) {
      if (config.model.debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
      }

      auto buf = ReadFile(mgr, f);

      auto fsts = ReadFstsFromFar(buf);
      for (auto &r : fsts) {
        tn_list->push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }
    }
  }
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
