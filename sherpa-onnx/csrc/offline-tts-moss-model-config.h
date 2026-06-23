// sherpa-onnx/csrc/offline-tts-moss-model-config.h
//
// Copyright (c)  2026  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsMossModelConfig {
  std::string prefill;
  std::string decode_step;
  std::string local_fixed_sampled_frame;
  std::string codec_encoder;
  std::string codec_decoder;
  std::string tokenizer_vocab;
  std::string tokenizer_scores;

  OfflineTtsMossModelConfig() = default;

  OfflineTtsMossModelConfig(const std::string &prefill,
                            const std::string &decode_step,
                            const std::string &local_fixed_sampled_frame,
                            const std::string &codec_encoder,
                            const std::string &codec_decoder,
                            const std::string &tokenizer_vocab,
                            const std::string &tokenizer_scores)
      : prefill(prefill),
        decode_step(decode_step),
        local_fixed_sampled_frame(local_fixed_sampled_frame),
        codec_encoder(codec_encoder),
        codec_decoder(codec_decoder),
        tokenizer_vocab(tokenizer_vocab),
        tokenizer_scores(tokenizer_scores) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;

  std::string GetTokenizerVocabPath() const;
  std::string GetTokenizerScoresPath() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_MODEL_CONFIG_H_
