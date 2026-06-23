// sherpa-onnx/csrc/offline-tts-moss-model-config.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/offline-tts-moss-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsMossModelConfig::Register(ParseOptions *po) {
  po->Register("moss-prefill", &prefill,
               "Path to official MOSS TTS moss_tts_prefill.onnx");
  po->Register("moss-decode-step", &decode_step,
               "Path to official MOSS TTS moss_tts_decode_step.onnx");
  po->Register(
      "moss-local-fixed-sampled-frame", &local_fixed_sampled_frame,
      "Path to official MOSS TTS moss_tts_local_fixed_sampled_frame.onnx");
  po->Register("moss-codec-encoder", &codec_encoder,
               "Path to official MOSS Audio Tokenizer encode ONNX model");
  po->Register("moss-codec-decoder", &codec_decoder,
               "Path to official MOSS Audio Tokenizer decode_full ONNX model");
  po->Register("moss-tokenizer-vocab", &tokenizer_vocab,
               "Path to MOSS tokenizer_vocab.json");
  po->Register("moss-tokenizer-scores", &tokenizer_scores,
               "Path to MOSS tokenizer_scores.json");
}

bool OfflineTtsMossModelConfig::Validate() const {
  if (prefill.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-prefill");
    return false;
  }

  if (!FileExists(prefill)) {
    SHERPA_ONNX_LOGE("--moss-prefill '%s' does not exist", prefill.c_str());
    return false;
  }

  if (decode_step.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-decode-step");
    return false;
  }

  if (!FileExists(decode_step)) {
    SHERPA_ONNX_LOGE("--moss-decode-step '%s' does not exist",
                     decode_step.c_str());
    return false;
  }

  if (local_fixed_sampled_frame.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-local-fixed-sampled-frame");
    return false;
  }

  if (!FileExists(local_fixed_sampled_frame)) {
    SHERPA_ONNX_LOGE("--moss-local-fixed-sampled-frame '%s' does not exist",
                     local_fixed_sampled_frame.c_str());
    return false;
  }

  if (codec_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-codec-encoder");
    return false;
  }

  if (!FileExists(codec_encoder)) {
    SHERPA_ONNX_LOGE("--moss-codec-encoder '%s' does not exist",
                     codec_encoder.c_str());
    return false;
  }

  if (codec_decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-codec-decoder");
    return false;
  }

  if (!FileExists(codec_decoder)) {
    SHERPA_ONNX_LOGE("--moss-codec-decoder '%s' does not exist",
                     codec_decoder.c_str());
    return false;
  }

  if (tokenizer_vocab.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-tokenizer-vocab");
    return false;
  }

  if (!FileExists(tokenizer_vocab)) {
    SHERPA_ONNX_LOGE("--moss-tokenizer-vocab '%s' does not exist",
                     tokenizer_vocab.c_str());
    return false;
  }

  if (tokenizer_scores.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moss-tokenizer-scores");
    return false;
  }

  if (!FileExists(tokenizer_scores)) {
    SHERPA_ONNX_LOGE("--moss-tokenizer-scores '%s' does not exist",
                     tokenizer_scores.c_str());
    return false;
  }

  return true;
}

std::string OfflineTtsMossModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsMossModelConfig(";
  os << "prefill=\"" << prefill << "\", ";
  os << "decode_step=\"" << decode_step << "\", ";
  os << "local_fixed_sampled_frame=\"" << local_fixed_sampled_frame << "\", ";
  os << "codec_encoder=\"" << codec_encoder << "\", ";
  os << "codec_decoder=\"" << codec_decoder << "\", ";
  os << "tokenizer_vocab=\"" << tokenizer_vocab << "\", ";
  os << "tokenizer_scores=\"" << tokenizer_scores << "\")";

  return os.str();
}

std::string OfflineTtsMossModelConfig::GetTokenizerVocabPath() const {
  return tokenizer_vocab;
}

std::string OfflineTtsMossModelConfig::GetTokenizerScoresPath() const {
  return tokenizer_scores;
}

}  // namespace sherpa_onnx
