// sherpa-onnx/csrc/offline-tts-moss-model.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/offline-tts-moss-model.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsMossModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)) {
    prefill_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moss.prefill), sess_opts_);
    InitPrefill();

    decode_step_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moss.decode_step), sess_opts_);
    InitDecodeStep();

    local_fixed_sampled_frame_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moss.local_fixed_sampled_frame),
        sess_opts_);
    InitLocalFixedSampledFrame();

    codec_encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moss.codec_encoder), sess_opts_);
    InitCodecEncoder();

    codec_decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moss.codec_decoder), sess_opts_);
    InitCodecDecoder();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)) {
    {
      auto buf = ReadFile(mgr, config.moss.prefill);
      prefill_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                     buf.size(), sess_opts_);
      InitPrefill();
    }
    {
      auto buf = ReadFile(mgr, config.moss.decode_step);
      decode_step_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_);
      InitDecodeStep();
    }
    {
      auto buf = ReadFile(mgr, config.moss.local_fixed_sampled_frame);
      local_fixed_sampled_frame_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_);
      InitLocalFixedSampledFrame();
    }
    {
      auto buf = ReadFile(mgr, config.moss.codec_encoder);
      codec_encoder_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_);
      InitCodecEncoder();
    }
    {
      auto buf = ReadFile(mgr, config.moss.codec_decoder);
      codec_decoder_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_);
      InitCodecDecoder();
    }
  }

  int32_t NumLayers() const { return num_layers_; }
  int32_t NumHeads() const { return num_heads_; }
  int32_t HeadDim() const { return head_dim_; }
  int32_t HiddenSize() const { return hidden_size_; }

  const std::string &ModelConfigJson() const { return model_config_json_; }

  const std::string &PromptTemplatesJson() const {
    return prompt_templates_json_;
  }

  const std::string &GenerationDefaultsJson() const {
    return generation_defaults_json_;
  }

  const std::string &DefaultPromptAudioCodesJson() const {
    return default_prompt_audio_codes_json_;
  }

  MossGlobalOutput RunPrefill(Ort::Value input_ids,
                              Ort::Value attention_mask) const {
    std::array<Ort::Value, 2> inputs = {std::move(input_ids),
                                        std::move(attention_mask)};

    auto outputs = prefill_sess_->Run(
        {}, prefill_input_names_ptr_.data(), inputs.data(), inputs.size(),
        prefill_output_names_ptr_.data(), prefill_output_names_ptr_.size());

    MossGlobalOutput result;
    result.global_hidden_state = std::move(outputs[0]);
    for (size_t i = 1; i < outputs.size(); ++i) {
      result.present_kv.push_back(std::move(outputs[i]));
    }

    return result;
  }

  MossGlobalOutput RunDecodeStep(Ort::Value input_ids,
                                 Ort::Value past_valid_lengths,
                                 std::vector<Ort::Value> *past_kv) const {
    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + past_kv->size());
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(past_valid_lengths));
    for (auto &v : *past_kv) {
      inputs.push_back(View(&v));
    }

    auto outputs = decode_step_sess_->Run(
        {}, decode_step_input_names_ptr_.data(), inputs.data(), inputs.size(),
        decode_step_output_names_ptr_.data(),
        decode_step_output_names_ptr_.size());

    MossGlobalOutput result;
    result.global_hidden_state = std::move(outputs[0]);
    for (size_t i = 1; i < outputs.size(); ++i) {
      result.present_kv.push_back(std::move(outputs[i]));
    }

    return result;
  }

  MossSampledFrame RunLocalFixedSampledFrame(Ort::Value global_hidden,
                                             Ort::Value repetition_seen_mask,
                                             Ort::Value assistant_random_u,
                                             Ort::Value audio_random_u) const {
    std::array<Ort::Value, 4> inputs = {
        std::move(global_hidden), std::move(repetition_seen_mask),
        std::move(assistant_random_u), std::move(audio_random_u)};

    auto outputs = local_fixed_sampled_frame_sess_->Run(
        {}, local_fixed_sampled_frame_input_names_ptr_.data(), inputs.data(),
        inputs.size(), local_fixed_sampled_frame_output_names_ptr_.data(),
        local_fixed_sampled_frame_output_names_ptr_.size());

    const int32_t *should_continue = outputs[0].GetTensorData<int32_t>();
    const int32_t *frame = outputs[1].GetTensorData<int32_t>();

    auto shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    int32_t n = static_cast<int32_t>(shape.back());

    MossSampledFrame ans;
    ans.should_continue = should_continue[0] != 0;
    ans.frame_token_ids.assign(frame, frame + n);

    return ans;
  }

  MossCodecEncodeOutput RunCodecEncode(Ort::Value input_values,
                                       Ort::Value input_lengths) const {
    std::array<Ort::Value, 2> inputs = {std::move(input_values),
                                        std::move(input_lengths)};

    auto outputs = codec_encoder_sess_->Run(
        {}, codec_encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        codec_encoder_output_names_ptr_.data(),
        codec_encoder_output_names_ptr_.size());

    MossCodecEncodeOutput ans;
    ans.audio_codes = std::move(outputs[0]);
    ans.audio_code_lengths = std::move(outputs[1]);
    return ans;
  }

  MossCodecDecodeOutput RunCodecDecode(Ort::Value audio_codes,
                                       Ort::Value audio_codes_lengths) const {
    std::array<Ort::Value, 2> inputs = {std::move(audio_codes),
                                        std::move(audio_codes_lengths)};

    auto outputs = codec_decoder_sess_->Run(
        {}, codec_decoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        codec_decoder_output_names_ptr_.data(),
        codec_decoder_output_names_ptr_.size());

    MossCodecDecodeOutput ans;
    ans.audio = std::move(outputs[0]);
    ans.audio_lengths = std::move(outputs[1]);
    return ans;
  }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  std::string LookupMeta(const char *key) {
    Ort::ModelMetadata meta_data = prefill_sess_->GetModelMetadata();
    return LookupCustomModelMetaData(meta_data, key, allocator_);
  }

  void InitPrefill() {
    GetInputNames(prefill_sess_.get(), &prefill_input_names_,
                  &prefill_input_names_ptr_);
    GetOutputNames(prefill_sess_.get(), &prefill_output_names_,
                   &prefill_output_names_ptr_);

    auto shape = prefill_sess_->GetOutputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    hidden_size_ = static_cast<int32_t>(shape.back());

    shape = prefill_sess_->GetOutputTypeInfo(1)
                .GetTensorTypeAndShapeInfo()
                .GetShape();
    num_heads_ = static_cast<int32_t>(shape[2]);
    head_dim_ = shape[3] > 0 ? static_cast<int32_t>(shape[3]) : 64;

    num_layers_ = (static_cast<int32_t>(prefill_output_names_.size()) - 1) / 2;

    model_config_json_ = LookupMeta("moss.model_config_json");
    prompt_templates_json_ = LookupMeta("moss.prompt_templates_json");
    generation_defaults_json_ = LookupMeta("moss.generation_defaults_json");
    default_prompt_audio_codes_json_ =
        LookupMeta("moss.default_prompt_audio_codes_json");
  }

  void InitDecodeStep() {
    GetInputNames(decode_step_sess_.get(), &decode_step_input_names_,
                  &decode_step_input_names_ptr_);
    GetOutputNames(decode_step_sess_.get(), &decode_step_output_names_,
                   &decode_step_output_names_ptr_);
  }

  void InitLocalFixedSampledFrame() {
    GetInputNames(local_fixed_sampled_frame_sess_.get(),
                  &local_fixed_sampled_frame_input_names_,
                  &local_fixed_sampled_frame_input_names_ptr_);
    GetOutputNames(local_fixed_sampled_frame_sess_.get(),
                   &local_fixed_sampled_frame_output_names_,
                   &local_fixed_sampled_frame_output_names_ptr_);
  }

  void InitCodecEncoder() {
    GetInputNames(codec_encoder_sess_.get(), &codec_encoder_input_names_,
                  &codec_encoder_input_names_ptr_);
    GetOutputNames(codec_encoder_sess_.get(), &codec_encoder_output_names_,
                   &codec_encoder_output_names_ptr_);
  }

  void InitCodecDecoder() {
    GetInputNames(codec_decoder_sess_.get(), &codec_decoder_input_names_,
                  &codec_decoder_input_names_ptr_);
    GetOutputNames(codec_decoder_sess_.get(), &codec_decoder_output_names_,
                   &codec_decoder_output_names_ptr_);
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> prefill_sess_;
  std::unique_ptr<Ort::Session> decode_step_sess_;
  std::unique_ptr<Ort::Session> local_fixed_sampled_frame_sess_;
  std::unique_ptr<Ort::Session> codec_encoder_sess_;
  std::unique_ptr<Ort::Session> codec_decoder_sess_;

  int32_t num_layers_ = 0;
  int32_t num_heads_ = 0;
  int32_t head_dim_ = 0;
  int32_t hidden_size_ = 0;

  std::string model_config_json_;
  std::string prompt_templates_json_;
  std::string generation_defaults_json_;
  std::string default_prompt_audio_codes_json_;

  std::vector<std::string> prefill_input_names_;
  std::vector<const char *> prefill_input_names_ptr_;
  std::vector<std::string> prefill_output_names_;
  std::vector<const char *> prefill_output_names_ptr_;

  std::vector<std::string> decode_step_input_names_;
  std::vector<const char *> decode_step_input_names_ptr_;
  std::vector<std::string> decode_step_output_names_;
  std::vector<const char *> decode_step_output_names_ptr_;

  std::vector<std::string> local_fixed_sampled_frame_input_names_;
  std::vector<const char *> local_fixed_sampled_frame_input_names_ptr_;
  std::vector<std::string> local_fixed_sampled_frame_output_names_;
  std::vector<const char *> local_fixed_sampled_frame_output_names_ptr_;

  std::vector<std::string> codec_encoder_input_names_;
  std::vector<const char *> codec_encoder_input_names_ptr_;
  std::vector<std::string> codec_encoder_output_names_;
  std::vector<const char *> codec_encoder_output_names_ptr_;

  std::vector<std::string> codec_decoder_input_names_;
  std::vector<const char *> codec_decoder_input_names_ptr_;
  std::vector<std::string> codec_decoder_output_names_;
  std::vector<const char *> codec_decoder_output_names_ptr_;
};

OfflineTtsMossModel::OfflineTtsMossModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsMossModel::OfflineTtsMossModel(Manager *mgr,
                                         const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsMossModel::~OfflineTtsMossModel() = default;

int32_t OfflineTtsMossModel::NumLayers() const { return impl_->NumLayers(); }
int32_t OfflineTtsMossModel::NumHeads() const { return impl_->NumHeads(); }
int32_t OfflineTtsMossModel::HeadDim() const { return impl_->HeadDim(); }
int32_t OfflineTtsMossModel::HiddenSize() const { return impl_->HiddenSize(); }

const std::string &OfflineTtsMossModel::ModelConfigJson() const {
  return impl_->ModelConfigJson();
}

const std::string &OfflineTtsMossModel::PromptTemplatesJson() const {
  return impl_->PromptTemplatesJson();
}

const std::string &OfflineTtsMossModel::GenerationDefaultsJson() const {
  return impl_->GenerationDefaultsJson();
}

const std::string &OfflineTtsMossModel::DefaultPromptAudioCodesJson() const {
  return impl_->DefaultPromptAudioCodesJson();
}

MossGlobalOutput OfflineTtsMossModel::RunPrefill(
    Ort::Value input_ids, Ort::Value attention_mask) const {
  return impl_->RunPrefill(std::move(input_ids), std::move(attention_mask));
}

MossGlobalOutput OfflineTtsMossModel::RunDecodeStep(
    Ort::Value input_ids, Ort::Value past_valid_lengths,
    std::vector<Ort::Value> *past_kv) const {
  return impl_->RunDecodeStep(std::move(input_ids),
                              std::move(past_valid_lengths), past_kv);
}

MossSampledFrame OfflineTtsMossModel::RunLocalFixedSampledFrame(
    Ort::Value global_hidden, Ort::Value repetition_seen_mask,
    Ort::Value assistant_random_u, Ort::Value audio_random_u) const {
  return impl_->RunLocalFixedSampledFrame(
      std::move(global_hidden), std::move(repetition_seen_mask),
      std::move(assistant_random_u), std::move(audio_random_u));
}

MossCodecEncodeOutput OfflineTtsMossModel::RunCodecEncode(
    Ort::Value input_values, Ort::Value input_lengths) const {
  return impl_->RunCodecEncode(std::move(input_values),
                               std::move(input_lengths));
}

MossCodecDecodeOutput OfflineTtsMossModel::RunCodecDecode(
    Ort::Value audio_codes, Ort::Value audio_codes_lengths) const {
  return impl_->RunCodecDecode(std::move(audio_codes),
                               std::move(audio_codes_lengths));
}

OrtAllocator *OfflineTtsMossModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTtsMossModel::OfflineTtsMossModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsMossModel::OfflineTtsMossModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
