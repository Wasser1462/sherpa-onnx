// sherpa-onnx/csrc/offline-tts-moss-bpe-tokenizer.h
//
// Copyright (c)  2026  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_BPE_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_BPE_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

class OfflineTtsMossBpeTokenizer {
 public:
  OfflineTtsMossBpeTokenizer(const std::string &vocab_json,
                             const std::string &token_scores_json);

  ~OfflineTtsMossBpeTokenizer();

  std::vector<int32_t> Encode(const std::string &text) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_BPE_TOKENIZER_H_
