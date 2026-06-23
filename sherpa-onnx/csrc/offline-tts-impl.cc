// sherpa-onnx/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-impl.h"

#include <memory>
#include <string>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-kitten-impl.h"
#include "sherpa-onnx/csrc/offline-tts-kokoro-impl.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-impl.h"
#include "sherpa-onnx/csrc/offline-tts-moss-impl.h"
#include "sherpa-onnx/csrc/offline-tts-pocket-impl.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-impl.h"

namespace sherpa_onnx {

std::vector<int64_t> OfflineTtsImpl::AddBlank(const std::vector<int64_t> &x,
                                              int32_t blank_id /*= 0*/) const {
  // we assume the blank ID is 0
  std::vector<int64_t> buffer(x.size() * 2 + 1, blank_id);
  int32_t i = 1;
  for (auto k : x) {
    buffer[i] = k;
    i += 2;
  }
  return buffer;
}

void OfflineTtsImpl::InitTextNormalizer(const OfflineTtsConfig &config,
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
      tn_list->push_back(std::make_unique<kaldifst::TextNormalizer>(f));
    }
  }

  if (!config.rule_fars.empty()) {
    if (config.model.debug) {
      SHERPA_ONNX_LOGE("Loading FST archives");
    }
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
      std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
          fst::FarReader<fst::StdArc>::Open(f));
      for (; !reader->Done(); reader->Next()) {
        std::unique_ptr<fst::StdConstFst> r(
            fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

        tn_list->push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }
    }

    if (config.model.debug) {
      SHERPA_ONNX_LOGE("FST archives loaded!");
    }
  }
}

std::string OfflineTtsImpl::ApplyTextNormalizer(
    std::string text, const TextNormalizerList &tn_list, bool debug) const {
  for (const auto &tn : tn_list) {
    text = tn->Normalize(text);
    if (debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("After normalizing: %{public}s", text.c_str());
#else
      SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
#endif
    }
  }

  return text;
}

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(config);
  } else if (!config.model.zipvoice.encoder.empty() &&
             !config.model.zipvoice.decoder.empty()) {
    return std::make_unique<OfflineTtsZipvoiceImpl>(config);
  } else if (!config.model.kokoro.model.empty()) {
    return std::make_unique<OfflineTtsKokoroImpl>(config);
  } else if (!config.model.kitten.model.empty()) {
    return std::make_unique<OfflineTtsKittenImpl>(config);
  } else if (!config.model.pocket.lm_flow.empty()) {
    return std::make_unique<OfflineTtsPocketImpl>(config);
  } else if (!config.model.supertonic.tts_json.empty()) {
    return std::make_unique<OfflineTtsSupertonicImpl>(config);
  } else if (!config.model.moss.prefill.empty()) {
    return std::make_unique<OfflineTtsMossImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please provide a tts model.");

  return {};
}

template <typename Manager>
std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    Manager *mgr, const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(mgr, config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(mgr, config);
  } else if (!config.model.zipvoice.encoder.empty() &&
             !config.model.zipvoice.decoder.empty()) {
    return std::make_unique<OfflineTtsZipvoiceImpl>(mgr, config);
  } else if (!config.model.kokoro.model.empty()) {
    return std::make_unique<OfflineTtsKokoroImpl>(mgr, config);
  } else if (!config.model.kitten.model.empty()) {
    return std::make_unique<OfflineTtsKittenImpl>(mgr, config);
  } else if (!config.model.pocket.lm_flow.empty()) {
    return std::make_unique<OfflineTtsPocketImpl>(mgr, config);
  } else if (!config.model.supertonic.tts_json.empty()) {
    return std::make_unique<OfflineTtsSupertonicImpl>(mgr, config);
  } else if (!config.model.moss.prefill.empty()) {
    return std::make_unique<OfflineTtsMossImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please provide a tts model.");
  return {};
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    NativeResourceManager *mgr, const OfflineTtsConfig &config);
#endif

}  // namespace sherpa_onnx
