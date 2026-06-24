#!/bin/bash
# Copyright (c)  2026 zengyw
#
# For users in China, ModelScope is recommended:
#   https://modelscope.cn/models/openmoss/MOSS-TTS-Nano-100M-ONNX/files
#   https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer-Nano-ONNX/files
# Use: MOSS_DOWNLOAD_SOURCE=modelscope ./run.sh

set -e

STAGE=${1:-0}
STOP_STAGE=${2:-2}
DOWNLOAD_SOURCE="${MOSS_DOWNLOAD_SOURCE:-hf}"
SINGLE_FILE="${MOSS_SINGLE_FILE:-0}"

TTS_DIR="MOSS-TTS-Nano-100M-ONNX"
CODEC_DIR="MOSS-Audio-Tokenizer-Nano-ONNX"
OUT_DIR="sherpa-onnx-moss-tts-nano-2026-05-07"
INT8_OUT_DIR="sherpa-onnx-moss-tts-nano-2026-05-07-int8"

EXPORT_OPTS=()
if [ "${SINGLE_FILE}" != "1" ]; then
  EXPORT_OPTS+=(--external-data)
fi

if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
  if [ ! -f "${TTS_DIR}/moss_tts_prefill.onnx" ]; then
    if [ "${DOWNLOAD_SOURCE}" = "modelscope" ]; then
      modelscope download --model openmoss/MOSS-TTS-Nano-100M-ONNX --local_dir "${TTS_DIR}"
    else
      huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX \
        --local-dir "${TTS_DIR}" \
        --include "*.onnx" "*.data" "*.json" "tokenizer.model"
    fi
  fi

  if [ ! -f "${CODEC_DIR}/moss_audio_tokenizer_encode.onnx" ]; then
    if [ "${DOWNLOAD_SOURCE}" = "modelscope" ]; then
      modelscope download --model openmoss/MOSS-Audio-Tokenizer-Nano-ONNX --local_dir "${CODEC_DIR}"
    else
      huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX \
        --local-dir "${CODEC_DIR}" \
        --include "*.onnx" "*.data" "*.json"
    fi
  fi
fi

if [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
  python3 ./export.py \
    --tts-dir "./${TTS_DIR}" \
    --codec-dir "./${CODEC_DIR}" \
    --out-dir "./${OUT_DIR}" \
    "${EXPORT_OPTS[@]}"
fi

if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
  python3 ./export.py \
    --tts-dir "./${TTS_DIR}" \
    --codec-dir "./${CODEC_DIR}" \
    --out-dir "./${OUT_DIR}" \
    --int8-out-dir "./${INT8_OUT_DIR}" \
    "${EXPORT_OPTS[@]}"
fi
