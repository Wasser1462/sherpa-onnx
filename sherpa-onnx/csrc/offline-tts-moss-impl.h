// sherpa-onnx/csrc/offline-tts-moss-impl.h
//
// Copyright (c)  2026  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_IMPL_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-moss-bpe-tokenizer.h"
#include "sherpa-onnx/csrc/offline-tts-moss-model.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {
using json = nlohmann::json;

static std::string ToString(const std::vector<char> &buf) {
  return std::string(buf.data(), buf.size());
}

static json ParseJsonText(const std::string &s, const char *name) {
  try {
    return json::parse(s);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to parse %s: %s", name, e.what());
    SHERPA_ONNX_EXIT(-1);
  }
}

static std::vector<int32_t> JsonToIntVector(const json &j) {
  std::vector<int32_t> ans;
  ans.reserve(j.size());
  for (const auto &v : j) {
    ans.push_back(v.get<int32_t>());
  }
  return ans;
}

static std::vector<std::vector<int32_t>> JsonToIntMatrix(const json &j) {
  std::vector<std::vector<int32_t>> ans;
  ans.reserve(j.size());
  for (const auto &row : j) {
    ans.push_back(JsonToIntVector(row));
  }
  return ans;
}

}  // namespace

class OfflineTtsMossImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsMossImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMossModel>(config.model)) {
    InitTokenizer();
    InitPromptData();
    InitTextNormalizer(config, &tn_list_);
  }

  template <typename Manager>
  OfflineTtsMossImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMossModel>(mgr, config.model)) {
    InitTokenizer(mgr);
    InitPromptData(mgr);
    InitTextNormalizer(mgr, config, &tn_list_);
  }

  int32_t SampleRate() const override { return prompt_data_.sample_rate; }

  int32_t NumSpeakers() const override { return 1; }

  GeneratedAudio Generate(
      const std::string &_text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", gen_config.ToString().c_str());
    }

    std::string text =
        ApplyTextNormalizer(_text, tn_list_, config_.model.debug);
    auto sentences = SplitByPunctuation(text);
    if (sentences.empty()) {
      return {};
    }

    int32_t max_char_in_sentence =
        gen_config.GetExtraInt("max_char_in_sentence", 200);
    int32_t min_char_in_sentence =
        gen_config.GetExtraInt("min_char_in_sentence", 30);
    if (max_char_in_sentence <= 0) {
      SHERPA_ONNX_LOGE(
          "max_char_in_sentence must be positive. Given: %d. Use 200.",
          max_char_in_sentence);
      max_char_in_sentence = 200;
    }
    if (min_char_in_sentence < 0) {
      SHERPA_ONNX_LOGE(
          "min_char_in_sentence must be non-negative. Given: %d. Use 30.",
          min_char_in_sentence);
      min_char_in_sentence = 30;
    }

    sentences = MergeShortSentences(sentences, min_char_in_sentence);

    std::vector<std::string> chunks;
    for (const auto &s : sentences) {
      auto pieces = SplitLongSentence(s, max_char_in_sentence);
      chunks.insert(chunks.end(), pieces.begin(), pieces.end());
    }
    sentences = std::move(chunks);

    std::vector<std::vector<int32_t>> prompt_audio_codes =
        GetPromptAudioCodes(gen_config);
    if (prompt_audio_codes.empty()) {
      SHERPA_ONNX_LOGE("No MOSS prompt audio codes available");
      return {};
    }

    GeneratedAudio result;
    result.sample_rate = SampleRate();

    const int32_t total = static_cast<int32_t>(sentences.size());
    bool should_continue = true;

    for (int32_t i = 0; i < total && should_continue; ++i) {
      GeneratedAudioCallback wrapped_cb = nullptr;
      if (callback) {
        wrapped_cb = [&, i](const float *samples, int32_t n,
                            float sentence_progress) -> bool {
          float global_progress = (i + sentence_progress) / total;
          return callback(samples, n, global_progress);
        };
      }

      GeneratedAudio cur =
          GenerateSingleSentence(sentences[i], gen_config, prompt_audio_codes,
                                 i, &should_continue, wrapped_cb);

      if (cur.samples.empty()) {
        continue;
      }

      result.samples.insert(result.samples.end(), cur.samples.begin(),
                            cur.samples.end());
    }

    return result;
  }

 private:
  struct PromptData {
    int32_t n_vq = 16;
    int32_t row_width = 17;
    int32_t sample_rate = 48000;
    int32_t downsample_rate = 3840;
    int32_t audio_codebook_size = 1024;

    int32_t audio_pad_token_id = 1024;
    int32_t im_start_token_id = 4;
    int32_t im_end_token_id = 5;
    int32_t audio_start_token_id = 6;
    int32_t audio_end_token_id = 7;
    int32_t audio_user_slot_token_id = 8;
    int32_t audio_assistant_slot_token_id = 9;

    int32_t max_new_frames = 375;

    std::vector<int32_t> user_prompt_prefix_token_ids;
    std::vector<int32_t> user_prompt_after_reference_token_ids;
    std::vector<int32_t> assistant_prompt_prefix_token_ids;
    std::vector<std::vector<int32_t>> default_prompt_audio_codes;
  };

  void InitTokenizer() {
    auto vocab_path = config_.model.moss.GetTokenizerVocabPath();
    auto scores_path = config_.model.moss.GetTokenizerScoresPath();
    auto vocab_json = ToString(ReadFile(vocab_path));
    auto scores_json = ToString(ReadFile(scores_path));
    if (vocab_json.empty() || scores_json.empty()) {
      SHERPA_ONNX_LOGE(
          "MOSS tokenizer json is missing. Please check %s and %s.",
          vocab_path.c_str(), scores_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    tokenizer_ =
        std::make_unique<OfflineTtsMossBpeTokenizer>(vocab_json, scores_json);
  }

  template <typename Manager>
  void InitTokenizer(Manager *mgr) {
    auto vocab_path = config_.model.moss.GetTokenizerVocabPath();
    auto scores_path = config_.model.moss.GetTokenizerScoresPath();
    auto vocab_json = ToString(ReadFile(mgr, vocab_path));
    auto scores_json = ToString(ReadFile(mgr, scores_path));
    if (vocab_json.empty() || scores_json.empty()) {
      SHERPA_ONNX_LOGE(
          "MOSS tokenizer json is missing. Please check %s and %s.",
          vocab_path.c_str(), scores_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    tokenizer_ =
        std::make_unique<OfflineTtsMossBpeTokenizer>(vocab_json, scores_json);
  }

  void InitPromptData() { InitPromptDataFromMetadata(); }

  template <typename Manager>
  void InitPromptData(Manager * /*mgr*/) {
    InitPromptData();
  }

  void InitPromptDataFromMetadata() {
    const auto &model_config_json = model_->ModelConfigJson();
    const auto &prompt_templates_json = model_->PromptTemplatesJson();
    const auto &generation_defaults_json = model_->GenerationDefaultsJson();
    const auto &default_prompt_audio_codes_json =
        model_->DefaultPromptAudioCodesJson();

    if (model_config_json.empty() || prompt_templates_json.empty() ||
        default_prompt_audio_codes_json.empty()) {
      SHERPA_ONNX_LOGE(
          "MOSS prompt metadata is missing. Please export the official models "
          "with scripts/moss-tts-nano/run.sh or export.py.");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto model_config =
        ParseJsonText(model_config_json, "moss.model_config_json");
    prompt_data_.n_vq = model_config.value("n_vq", prompt_data_.n_vq);
    prompt_data_.row_width =
        model_config.value("row_width", prompt_data_.row_width);
    prompt_data_.audio_pad_token_id = model_config.value(
        "audio_pad_token_id", prompt_data_.audio_pad_token_id);
    prompt_data_.im_start_token_id =
        model_config.value("im_start_token_id", prompt_data_.im_start_token_id);
    prompt_data_.im_end_token_id =
        model_config.value("im_end_token_id", prompt_data_.im_end_token_id);
    prompt_data_.audio_start_token_id = model_config.value(
        "audio_start_token_id", prompt_data_.audio_start_token_id);
    prompt_data_.audio_end_token_id = model_config.value(
        "audio_end_token_id", prompt_data_.audio_end_token_id);
    prompt_data_.audio_user_slot_token_id = model_config.value(
        "audio_user_slot_token_id", prompt_data_.audio_user_slot_token_id);
    prompt_data_.audio_assistant_slot_token_id =
        model_config.value("audio_assistant_slot_token_id",
                           prompt_data_.audio_assistant_slot_token_id);

    const auto templates =
        ParseJsonText(prompt_templates_json, "moss.prompt_templates_json");
    prompt_data_.user_prompt_prefix_token_ids =
        JsonToIntVector(templates.at("user_prompt_prefix_token_ids"));
    prompt_data_.user_prompt_after_reference_token_ids =
        JsonToIntVector(templates.at("user_prompt_after_reference_token_ids"));
    prompt_data_.assistant_prompt_prefix_token_ids =
        JsonToIntVector(templates.at("assistant_prompt_prefix_token_ids"));

    if (!generation_defaults_json.empty()) {
      const auto defaults = ParseJsonText(generation_defaults_json,
                                          "moss.generation_defaults_json");
      prompt_data_.max_new_frames =
          defaults.value("max_new_frames", prompt_data_.max_new_frames);
    }

    prompt_data_.default_prompt_audio_codes =
        JsonToIntMatrix(ParseJsonText(default_prompt_audio_codes_json,
                                      "moss.default_prompt_audio_codes_json"));

    ValidatePromptData();
  }

  void ValidatePromptData() const {
    if (prompt_data_.n_vq <= 0 || prompt_data_.row_width <= 1 ||
        prompt_data_.row_width < prompt_data_.n_vq + 1 ||
        prompt_data_.audio_codebook_size <= 0 ||
        prompt_data_.sample_rate <= 0 || prompt_data_.downsample_rate <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid MOSS metadata: n_vq=%d, row_width=%d, "
          "audio_codebook_size=%d, sample_rate=%d, downsample_rate=%d",
          prompt_data_.n_vq, prompt_data_.row_width,
          prompt_data_.audio_codebook_size, prompt_data_.sample_rate,
          prompt_data_.downsample_rate);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::vector<std::vector<int32_t>> GetPromptAudioCodes(
      const GenerationConfig &gen_config) const {
    if (!gen_config.reference_audio.empty()) {
      return EncodeReferenceAudio(gen_config);
    }

    return prompt_data_.default_prompt_audio_codes;
  }

  void AppendTextRows(const std::vector<int32_t> &tokens,
                      std::vector<int32_t> *input_data) const {
    for (int32_t token : tokens) {
      size_t old_size = input_data->size();
      input_data->resize(old_size + prompt_data_.row_width,
                         prompt_data_.audio_pad_token_id);
      (*input_data)[old_size] = token;
    }
  }

  std::pair<Ort::Value, Ort::Value> BuildPromptInputIds(
      const std::string &text,
      const std::vector<std::vector<int32_t>> &prompt_audio_codes) const {
    std::vector<int32_t> text_token_ids = tokenizer_->Encode(text);

    std::vector<int32_t> input_data;
    input_data.reserve(
        (text_token_ids.size() + prompt_audio_codes.size() + 128) *
        prompt_data_.row_width);

    std::vector<int32_t> head;
    head.reserve(2 + prompt_data_.user_prompt_prefix_token_ids.size());
    head.push_back(prompt_data_.im_start_token_id);
    head.insert(head.end(), prompt_data_.user_prompt_prefix_token_ids.begin(),
                prompt_data_.user_prompt_prefix_token_ids.end());
    head.push_back(prompt_data_.audio_start_token_id);
    AppendTextRows(head, &input_data);

    for (const auto &frame : prompt_audio_codes) {
      if (static_cast<int32_t>(frame.size()) != prompt_data_.n_vq) {
        SHERPA_ONNX_LOGE("Expected %d audio codes per frame. Got %d",
                         prompt_data_.n_vq, static_cast<int32_t>(frame.size()));
        SHERPA_ONNX_EXIT(-1);
      }

      size_t old_size = input_data.size();
      input_data.resize(old_size + prompt_data_.row_width,
                        prompt_data_.audio_pad_token_id);
      input_data[old_size] = prompt_data_.audio_user_slot_token_id;
      std::copy(frame.begin(), frame.end(), input_data.begin() + old_size + 1);
    }

    std::vector<int32_t> tail;
    tail.reserve(2 + prompt_data_.user_prompt_after_reference_token_ids.size() +
                 text_token_ids.size() +
                 prompt_data_.assistant_prompt_prefix_token_ids.size());
    tail.push_back(prompt_data_.audio_end_token_id);
    tail.insert(tail.end(),
                prompt_data_.user_prompt_after_reference_token_ids.begin(),
                prompt_data_.user_prompt_after_reference_token_ids.end());
    tail.insert(tail.end(), text_token_ids.begin(), text_token_ids.end());
    tail.insert(tail.end(),
                prompt_data_.assistant_prompt_prefix_token_ids.begin(),
                prompt_data_.assistant_prompt_prefix_token_ids.end());
    tail.push_back(prompt_data_.audio_start_token_id);
    AppendTextRows(tail, &input_data);

    int64_t seq_len =
        static_cast<int64_t>(input_data.size() / prompt_data_.row_width);

    std::array<int64_t, 3> ids_shape = {
        1, seq_len, static_cast<int64_t>(prompt_data_.row_width)};
    Ort::Value input_ids = Ort::Value::CreateTensor<int32_t>(
        model_->Allocator(), ids_shape.data(), ids_shape.size());
    std::copy(input_data.begin(), input_data.end(),
              input_ids.GetTensorMutableData<int32_t>());

    std::array<int64_t, 2> mask_shape = {1, seq_len};
    Ort::Value attention_mask = Ort::Value::CreateTensor<int32_t>(
        model_->Allocator(), mask_shape.data(), mask_shape.size());
    int32_t *mask = attention_mask.GetTensorMutableData<int32_t>();
    std::fill(mask, mask + seq_len, 1);

    if (config_.model.debug) {
      SHERPA_ONNX_LOGE(
          "MOSS prompt seq_len=%d, text_tokens=%d, prompt_frames=%d",
          static_cast<int32_t>(seq_len),
          static_cast<int32_t>(text_token_ids.size()),
          static_cast<int32_t>(prompt_audio_codes.size()));
    }

    return {std::move(input_ids), std::move(attention_mask)};
  }

  std::vector<std::vector<int32_t>> EncodeReferenceAudio(
      const GenerationConfig &gen_config) const {
    if (gen_config.reference_sample_rate <= 0) {
      SHERPA_ONNX_LOGE("reference_sample_rate %d is invalid.",
                       gen_config.reference_sample_rate);
      return {};
    }

    const float *p_audio = gen_config.reference_audio.data();
    int32_t num_samples =
        static_cast<int32_t>(gen_config.reference_audio.size());
    std::vector<float> resampled;

    if (gen_config.reference_sample_rate != prompt_data_.sample_rate) {
      float min_freq = std::min<int32_t>(gen_config.reference_sample_rate,
                                         prompt_data_.sample_rate);
      float lowpass_cutoff = 0.99f * 0.5f * min_freq;
      LinearResample resampler(gen_config.reference_sample_rate,
                               prompt_data_.sample_rate, lowpass_cutoff, 6);
      resampler.Resample(gen_config.reference_audio.data(),
                         gen_config.reference_audio.size(), true, &resampled);
      p_audio = resampled.data();
      num_samples = static_cast<int32_t>(resampled.size());
    }

    float max_ref_len =
        gen_config.GetExtraFloat("max_reference_audio_len", -1.0f);
    int32_t max_len =
        static_cast<int32_t>(max_ref_len * prompt_data_.sample_rate);
    if (max_ref_len > 0 && num_samples > max_len) {
      num_samples = max_len;
    }

    int32_t pad = (prompt_data_.downsample_rate -
                   (num_samples % prompt_data_.downsample_rate)) %
                  prompt_data_.downsample_rate;
    int32_t padded_len = num_samples + pad;

    std::vector<float> stereo(2 * padded_len, 0.0f);
    std::copy(p_audio, p_audio + num_samples, stereo.data());
    std::copy(p_audio, p_audio + num_samples, stereo.data() + padded_len);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> input_shape = {1, 2, padded_len};
    Ort::Value input_values =
        Ort::Value::CreateTensor(memory_info, stereo.data(), stereo.size(),
                                 input_shape.data(), input_shape.size());

    std::array<int64_t, 1> len_shape = {1};
    int32_t original_length = num_samples;
    Ort::Value input_lengths = Ort::Value::CreateTensor(
        memory_info, &original_length, 1, len_shape.data(), len_shape.size());

    MossCodecEncodeOutput encoded = model_->RunCodecEncode(
        std::move(input_values), std::move(input_lengths));

    auto shape = encoded.audio_codes.GetTensorTypeAndShapeInfo().GetShape();
    const int32_t *p = encoded.audio_codes.GetTensorData<int32_t>();
    const int32_t *audio_code_lengths =
        encoded.audio_code_lengths.GetTensorData<int32_t>();

    int64_t code_len = std::min<int64_t>(audio_code_lengths[0], shape[1]);
    int64_t n_vq = shape[2];
    std::vector<std::vector<int32_t>> ans(code_len, std::vector<int32_t>(n_vq));
    for (int64_t t = 0; t < code_len; ++t) {
      for (int64_t q = 0; q < n_vq; ++q) {
        ans[t][q] = p[t * n_vq + q];
      }
    }

    return ans;
  }

  std::vector<float> DecodeAudio(
      const std::vector<std::vector<int32_t>> &audio_frames) const {
    if (audio_frames.empty()) {
      return {};
    }

    int32_t num_frames = static_cast<int32_t>(audio_frames.size());
    std::vector<int32_t> codes(num_frames * prompt_data_.n_vq);
    for (int32_t t = 0; t < num_frames; ++t) {
      if (static_cast<int32_t>(audio_frames[t].size()) != prompt_data_.n_vq) {
        SHERPA_ONNX_LOGE("Expected %d audio codes per frame. Got %d",
                         prompt_data_.n_vq,
                         static_cast<int32_t>(audio_frames[t].size()));
        return {};
      }
      std::copy(audio_frames[t].begin(), audio_frames[t].end(),
                codes.begin() + t * prompt_data_.n_vq);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> codes_shape = {
        1, static_cast<int64_t>(num_frames),
        static_cast<int64_t>(prompt_data_.n_vq)};
    Ort::Value audio_codes =
        Ort::Value::CreateTensor(memory_info, codes.data(), codes.size(),
                                 codes_shape.data(), codes_shape.size());

    std::array<int64_t, 1> len_shape = {1};
    int32_t code_length = num_frames;
    Ort::Value audio_codes_lengths = Ort::Value::CreateTensor(
        memory_info, &code_length, 1, len_shape.data(), len_shape.size());

    auto out = model_->RunCodecDecode(std::move(audio_codes),
                                      std::move(audio_codes_lengths));
    auto shape = out.audio.GetTensorTypeAndShapeInfo().GetShape();
    const float *audio = out.audio.GetTensorData<float>();
    const int32_t *audio_lengths = out.audio_lengths.GetTensorData<int32_t>();

    int32_t channels = static_cast<int32_t>(shape[1]);
    int32_t n =
        std::min<int32_t>(audio_lengths[0], static_cast<int32_t>(shape.back()));

    if (channels == 1) {
      return {audio, audio + n};
    }

    std::vector<float> mono(n);
    const float *ch0 = audio;
    const float *ch1 = audio + shape.back();
    for (int32_t i = 0; i < n; ++i) {
      mono[i] = 0.5f * (ch0[i] + ch1[i]);
    }

    return mono;
  }

  std::vector<float> ExtractLastHidden(Ort::Value *global_hidden) const {
    auto shape = global_hidden->GetTensorTypeAndShapeInfo().GetShape();
    int64_t seq_len = shape[1];
    int64_t hidden_size = shape[2];
    const float *p = global_hidden->GetTensorData<float>();
    p += (seq_len - 1) * hidden_size;
    return {p, p + hidden_size};
  }

  Ort::Value CreateHiddenTensor(std::vector<float> *hidden) const {
    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(hidden->size())};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(memory_info, hidden->data(), hidden->size(),
                                    shape.data(), shape.size());
  }

  Ort::Value CreateSeenMaskTensor(std::vector<int32_t> *seen_mask) const {
    std::array<int64_t, 3> shape = {
        1, static_cast<int64_t>(prompt_data_.n_vq),
        static_cast<int64_t>(prompt_data_.audio_codebook_size)};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(memory_info, seen_mask->data(),
                                    seen_mask->size(), shape.data(),
                                    shape.size());
  }

  GeneratedAudio GenerateSingleSentence(
      const std::string &text, const GenerationConfig &gen_config,
      const std::vector<std::vector<int32_t>> &prompt_audio_codes,
      int32_t sentence_index, bool *should_continue,
      GeneratedAudioCallback callback) const {
    auto prompt = BuildPromptInputIds(text, prompt_audio_codes);
    int32_t prompt_len = static_cast<int32_t>(
        prompt.first.GetTensorTypeAndShapeInfo().GetShape()[1]);

    MossGlobalOutput prefill =
        model_->RunPrefill(std::move(prompt.first), std::move(prompt.second));

    std::vector<float> hidden = ExtractLastHidden(&prefill.global_hidden_state);
    std::vector<Ort::Value> caches = std::move(prefill.present_kv);
    int32_t past_len = prompt_len;

    int32_t max_new_frames =
        gen_config.GetExtraInt("max_new_frames", prompt_data_.max_new_frames);
    if (max_new_frames <= 0) {
      SHERPA_ONNX_LOGE("max_new_frames must be positive. Given: %d",
                       max_new_frames);
      return {};
    }

    int32_t seed = gen_config.GetExtraInt("seed", -1);
    std::mt19937 gen;
    if (seed >= 0) {
      gen.seed(static_cast<uint32_t>(seed + sentence_index));
    } else {
      std::random_device rd;
      gen.seed(rd());
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<int32_t> seen_mask(
        prompt_data_.n_vq * prompt_data_.audio_codebook_size, 0);
    std::vector<std::vector<int32_t>> generated_frames;
    generated_frames.reserve(max_new_frames);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    for (int32_t step = 0; step < max_new_frames && *should_continue; ++step) {
      std::array<float, 1> assistant_random = {dist(gen)};
      std::vector<float> audio_random(prompt_data_.n_vq);
      for (auto &v : audio_random) {
        v = dist(gen);
      }

      std::array<int64_t, 1> assistant_shape = {1};
      Ort::Value assistant_random_u = Ort::Value::CreateTensor(
          memory_info, assistant_random.data(), assistant_random.size(),
          assistant_shape.data(), assistant_shape.size());

      std::array<int64_t, 2> audio_random_shape = {
          1, static_cast<int64_t>(prompt_data_.n_vq)};
      Ort::Value audio_random_u = Ort::Value::CreateTensor(
          memory_info, audio_random.data(), audio_random.size(),
          audio_random_shape.data(), audio_random_shape.size());

      MossSampledFrame sampled = model_->RunLocalFixedSampledFrame(
          CreateHiddenTensor(&hidden), CreateSeenMaskTensor(&seen_mask),
          std::move(assistant_random_u), std::move(audio_random_u));

      if (!sampled.should_continue) {
        break;
      }
      if (static_cast<int32_t>(sampled.frame_token_ids.size()) !=
          prompt_data_.n_vq) {
        SHERPA_ONNX_LOGE("Expected %d sampled audio codes. Got %d",
                         prompt_data_.n_vq,
                         static_cast<int32_t>(sampled.frame_token_ids.size()));
        break;
      }

      generated_frames.push_back(sampled.frame_token_ids);
      for (int32_t q = 0; q < prompt_data_.n_vq; ++q) {
        int32_t token = sampled.frame_token_ids[q];
        if (token >= 0 && token < prompt_data_.audio_codebook_size) {
          seen_mask[q * prompt_data_.audio_codebook_size + token] = 1;
        }
      }

      if (step + 1 == max_new_frames) {
        break;
      }

      std::vector<int32_t> row(prompt_data_.row_width,
                               prompt_data_.audio_pad_token_id);
      row[0] = prompt_data_.audio_assistant_slot_token_id;
      std::copy(sampled.frame_token_ids.begin(), sampled.frame_token_ids.end(),
                row.begin() + 1);

      std::array<int64_t, 3> row_shape = {
          1, 1, static_cast<int64_t>(prompt_data_.row_width)};
      Ort::Value input_ids =
          Ort::Value::CreateTensor(memory_info, row.data(), row.size(),
                                   row_shape.data(), row_shape.size());

      std::array<int64_t, 1> len_shape = {1};
      Ort::Value past_valid_lengths = Ort::Value::CreateTensor(
          memory_info, &past_len, 1, len_shape.data(), len_shape.size());

      MossGlobalOutput decode = model_->RunDecodeStep(
          std::move(input_ids), std::move(past_valid_lengths), &caches);
      hidden = ExtractLastHidden(&decode.global_hidden_state);
      caches = std::move(decode.present_kv);
      ++past_len;
    }

    std::vector<float> audio = DecodeAudio(generated_frames);
    if (callback && !audio.empty()) {
      *should_continue = callback(audio.data(), audio.size(), 1.0f);
    }

    GeneratedAudio result;
    result.sample_rate = SampleRate();
    result.samples = std::move(audio);
    return result;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsMossModel> model_;
  std::unique_ptr<OfflineTtsMossBpeTokenizer> tokenizer_;
  PromptData prompt_data_;
  TextNormalizerList tn_list_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MOSS_IMPL_H_
