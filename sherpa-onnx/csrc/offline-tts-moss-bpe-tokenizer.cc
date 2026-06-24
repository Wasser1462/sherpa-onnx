// sherpa-onnx/csrc/offline-tts-moss-bpe-tokenizer.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/offline-tts-moss-bpe-tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

using json = nlohmann::json;

static json LoadJsonBuffer(const std::vector<char> &buf, const char *name) {
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Empty %s json metadata", name);
    SHERPA_ONNX_EXIT(-1);
  }

  return json::parse(buf.begin(), buf.end());
}

// Special tokens that must be matched as atomic units
static const char *kSpecialTokens[] = {
    "<|im_start|>",   "<|im_end|>",    "<|audio_start|>",  "<|audio_end|>",
    "<|vision_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
    "<|audio_pad|>",  "<user_inst>",   "</user_inst>",
};
static constexpr int32_t kNumSpecialTokens =
    sizeof(kSpecialTokens) / sizeof(kSpecialTokens[0]);

class OfflineTtsMossBpeTokenizer::Impl {
 public:
  Impl(const std::vector<char> &vocab_json,
       const std::vector<char> &token_scores_json) {
    Init(LoadJsonBuffer(vocab_json, "tokenizer_vocab.json"),
         LoadJsonBuffer(token_scores_json, "tokenizer_scores.json"));
  }

  std::vector<int32_t> Encode(const std::string &text) const {
    // Step 1: Normalize (NFKC-lite + whitespace) and escape whitespace
    std::string normalized = Normalize(text);
    if (normalized.empty()) {
      return {};
    }

    // Step 2: Split on special tokens
    std::vector<std::pair<bool, std::string>> segments;
    SplitSpecialTokens(normalized, &segments);

    // Step 3: BPE encode each non-special segment
    std::vector<int32_t> ids;
    ids.reserve(normalized.size());
    for (const auto &seg : segments) {
      if (seg.first) {
        // Special token
        auto it = token2id_.find(seg.second);
        if (it != token2id_.end()) {
          ids.push_back(it->second);
        }
      } else {
        // Regular text -> character split -> BPE merge
        BpeEncode(seg.second, &ids);
      }
    }
    return ids;
  }

 private:
  void Init(const json &vocab, const json &scores) {
    token2id_.reserve(vocab.size());
    for (const auto &item : vocab.items()) {
      token2id_[item.key()] = item.value();
    }

    token2score_.reserve(scores.size());
    for (const auto &item : scores.items()) {
      token2score_[item.key()] = item.value().get<float>();
    }

    // Build sorted special tokens (longest first)
    for (int32_t i = 0; i < kNumSpecialTokens; ++i) {
      special_tokens_.emplace_back(kSpecialTokens[i]);
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string &a, const std::string &b) {
                return a.size() > b.size();
              });
  }

  // Approximate SentencePiece nmt_nfkc normalization:
  //  - Fullwidth ASCII (U+FF01-U+FF5E) -> halfwidth (U+0021-U+007E)
  //  - Whitespace chars (\n, \t, \r) -> space
  //  - Collapse multiple spaces, strip leading/trailing
  //  - Add dummy prefix (prepend space)
  //  - Escape whitespaces (space -> ▁)
  std::string Normalize(const std::string &text) const {
    std::string buf;
    buf.reserve(text.size() + 4);

    const uint8_t *p = reinterpret_cast<const uint8_t *>(text.data());
    const uint8_t *end = p + text.size();

    while (p < end) {
      uint8_t c0 = *p;

      // Check for fullwidth ASCII: U+FF01-U+FF5E
      // UTF-8: EF BC 81 (U+FF01) to EF BD 9E (U+FF5E)
      if (c0 == 0xEF && p + 2 < end) {
        uint8_t c1 = p[1];
        uint8_t c2 = p[2];
        // U+FF01-U+FF3F: EF BC 81 to EF BC BF
        if (c1 == 0xBC && c2 >= 0x81 && c2 <= 0xBF) {
          // U+FF01 + (c2 - 0x81) maps to U+0021 + (c2 - 0x81)
          buf.push_back(static_cast<char>(0x21 + (c2 - 0x81)));
          p += 3;
          continue;
        }
        // U+FF40-U+FF5E: EF BD 80 to EF BD 9E
        if (c1 == 0xBD && c2 >= 0x80 && c2 <= 0x9E) {
          // U+FF40 + (c2 - 0x80) maps to U+0060 + (c2 - 0x80)
          buf.push_back(static_cast<char>(0x60 + (c2 - 0x80)));
          p += 3;
          continue;
        }
      }

      // Whitespace -> space
      if (c0 == '\n' || c0 == '\t' || c0 == '\r' || c0 == 0x0b || c0 == 0x0c) {
        buf.push_back(' ');
        ++p;
        continue;
      }

      // Regular byte
      buf.push_back(static_cast<char>(c0));
      ++p;
    }

    // Collapse multiple spaces
    std::string collapsed;
    collapsed.reserve(buf.size());
    bool prev_space = false;
    for (char c : buf) {
      if (c == ' ') {
        if (!prev_space) {
          collapsed.push_back(' ');
        }
        prev_space = true;
      } else {
        collapsed.push_back(c);
        prev_space = false;
      }
    }

    // Strip leading/trailing whitespace
    size_t start = 0;
    while (start < collapsed.size() && collapsed[start] == ' ') {
      ++start;
    }
    size_t stop = collapsed.size();
    while (stop > start && collapsed[stop - 1] == ' ') {
      --stop;
    }
    collapsed = collapsed.substr(start, stop - start);

    if (collapsed.empty()) {
      return {};
    }

    // Add dummy prefix (prepend space) and escape whitespaces (space -> ▁)
    // ▁ is UTF-8: E2 96 81
    std::string result;
    result.reserve(collapsed.size() * 3 + 3);
    result.append("\xE2\x96\x81");  // dummy prefix ▁
    for (char c : collapsed) {
      if (c == ' ') {
        result.append("\xE2\x96\x81");
      } else {
        result.push_back(c);
      }
    }
    return result;
  }

  void SplitSpecialTokens(
      const std::string &text,
      std::vector<std::pair<bool, std::string>> *segments) const {
    size_t i = 0;
    while (i < text.size()) {
      bool matched = false;
      for (const auto &st : special_tokens_) {
        if (i + st.size() <= text.size() &&
            text.compare(i, st.size(), st) == 0) {
          segments->emplace_back(true, st);
          i += st.size();
          matched = true;
          break;
        }
      }
      if (!matched) {
        if (!segments->empty() && !segments->back().first) {
          segments->back().second.push_back(text[i]);
        } else {
          segments->emplace_back(false, std::string(1, text[i]));
        }
        ++i;
      }
    }
  }

  void BpeEncode(const std::string &text, std::vector<int32_t> *ids) const {
    // Split into initial tokens (characters or byte fallback)
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    const uint8_t *p = reinterpret_cast<const uint8_t *>(text.data());
    const uint8_t *end = p + text.size();

    while (p < end) {
      // Determine UTF-8 character length
      int32_t char_len = 1;
      if (*p >= 0xF0 && p + 4 <= end) {
        char_len = 4;
      } else if (*p >= 0xE0 && p + 3 <= end) {
        char_len = 3;
      } else if (*p >= 0xC0 && p + 2 <= end) {
        char_len = 2;
      }

      std::string ch(reinterpret_cast<const char *>(p), char_len);
      if (token2id_.find(ch) != token2id_.end()) {
        tokens.push_back(std::move(ch));
      } else {
        // Byte fallback
        for (int32_t i = 0; i < char_len; ++i) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "<0x%02X>", p[i]);
          tokens.emplace_back(buf);
        }
      }
      p += char_len;
    }

    // Iteratively apply BPE merges
    while (tokens.size() > 1) {
      float best_score = -std::numeric_limits<float>::infinity();
      std::string best_merged;

      for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        std::string merged = tokens[i] + tokens[i + 1];
        auto it = token2score_.find(merged);
        if (it != token2score_.end() && it->second > best_score) {
          best_score = it->second;
          best_merged = merged;
        }
      }

      if (best_merged.empty()) {
        break;
      }

      // Apply merge: replace ALL occurrences of the best pair
      std::vector<std::string> new_tokens;
      new_tokens.reserve(tokens.size());
      size_t i = 0;
      while (i < tokens.size()) {
        if (i + 1 < tokens.size() && tokens[i] + tokens[i + 1] == best_merged) {
          new_tokens.push_back(best_merged);
          i += 2;
        } else {
          new_tokens.push_back(std::move(tokens[i]));
          ++i;
        }
      }
      tokens = std::move(new_tokens);
    }

    // Convert to IDs
    for (const auto &tok : tokens) {
      auto it = token2id_.find(tok);
      if (it != token2id_.end()) {
        ids->push_back(it->second);
      }
    }
  }

  std::unordered_map<std::string, int32_t> token2id_;
  std::unordered_map<std::string, float> token2score_;
  std::vector<std::string> special_tokens_;
};

OfflineTtsMossBpeTokenizer::OfflineTtsMossBpeTokenizer(
    const std::vector<char> &vocab_json,
    const std::vector<char> &token_scores_json)
    : impl_(std::make_unique<Impl>(vocab_json, token_scores_json)) {}

OfflineTtsMossBpeTokenizer::~OfflineTtsMossBpeTokenizer() = default;

std::vector<int32_t> OfflineTtsMossBpeTokenizer::Encode(
    const std::string &text) const {
  return impl_->Encode(text);
}

}  // namespace sherpa_onnx
