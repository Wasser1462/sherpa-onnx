#!/usr/bin/env python3
# Copyright (c)  2026  zengyw

import argparse
import json
import shutil
from pathlib import Path

import onnx
import sentencepiece as spm


TTS_ONNX_FILES = (
    "moss_tts_prefill.onnx",
    "moss_tts_decode_step.onnx",
    "moss_tts_local_fixed_sampled_frame.onnx",
)

TTS_EXTERNAL_DATA_FILES = (
    "moss_tts_global_shared.data",
    "moss_tts_local_shared.data",
)

CODEC_ONNX_FILES = (
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_decode_full.onnx",
)

CODEC_EXTERNAL_DATA_FILES = (
    "moss_audio_tokenizer_encode.data",
    "moss_audio_tokenizer_decode_shared.data",
)

RUNTIME_FILES = TTS_ONNX_FILES + CODEC_ONNX_FILES
TOKENIZER_FILES = ("tokenizer_vocab.json", "tokenizer_scores.json")
OFFICIAL_FILES = (
    TTS_ONNX_FILES
    + TTS_EXTERNAL_DATA_FILES
    + CODEC_ONNX_FILES
    + CODEC_EXTERNAL_DATA_FILES
)


def int8_name(name: str) -> str:
    return name.removesuffix(".onnx") + ".int8.onnx"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tts-dir",
        type=Path,
        required=True,
        help="Official MOSS-TTS-Nano-100M-ONNX directory.",
    )
    parser.add_argument(
        "--codec-dir",
        type=Path,
        required=True,
        help="Official MOSS-Audio-Tokenizer-Nano-ONNX directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="FP32 export directory consumed by sherpa-onnx.",
    )
    parser.add_argument(
        "--int8-out-dir",
        type=Path,
        default=None,
        help="Optional int8 export directory. The ONNX files use *.int8.onnx names.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)


def load_sentencepiece(tokenizer: Path):
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    vocab = {}
    scores = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        vocab[piece] = i
        scores[piece] = sp.get_score(i)

    return vocab, scores


def set_metadata(model, metadata):
    existing = {p.key: p.value for p in model.metadata_props}
    existing.update({str(k): str(v) for k, v in metadata.items()})

    del model.metadata_props[:]
    for key in sorted(existing):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = existing[key]


def save_single_file_onnx(src: Path, dst: Path):
    model = onnx.load_model(str(src), load_external_data=True)
    onnx.save_model(model, str(dst), save_as_external_data=False)


def add_metadata(prefill: Path, tts_dir: Path, codec_dir: Path):
    tts_meta = read_json(tts_dir / "tts_browser_onnx_meta.json")
    codec_meta = read_json(codec_dir / "codec_browser_onnx_meta.json")
    manifest = read_json(tts_dir / "browser_poc_manifest.json")

    voices = manifest.get("builtin_voices", [])
    if not voices:
        raise ValueError("browser_poc_manifest.json contains no builtin voices")

    metadata = {
        "model_type": "moss-tts-nano",
        "moss.source": "openmoss/ModelScope official ONNX",
        "moss.export_format": "sherpa-onnx-moss-tts-nano",
        "moss.tts_onnx_meta_json": compact(tts_meta),
        "moss.codec_onnx_meta_json": compact(codec_meta),
        "moss.model_config_json": compact(tts_meta["model_config"]),
        "moss.prompt_templates_json": compact(manifest["prompt_templates"]),
        "moss.generation_defaults_json": compact(
            manifest.get("generation_defaults", {})
        ),
        "moss.default_voice": str(voices[0].get("voice", "")),
        "moss.default_prompt_audio_codes_json": compact(
            voices[0]["prompt_audio_codes"]
        ),
    }

    model = onnx.load_model(str(prefill), load_external_data=False)
    set_metadata(model, metadata)
    onnx.save_model(model, str(prefill), save_as_external_data=False)


def export_fp32(tts_dir: Path, codec_dir: Path, out_dir: Path):
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    for name in TTS_ONNX_FILES:
        save_single_file_onnx(tts_dir / name, tmp_dir / name)
    for name in CODEC_ONNX_FILES:
        save_single_file_onnx(codec_dir / name, tmp_dir / name)

    vocab, scores = load_sentencepiece(tts_dir / "tokenizer.model")
    (tmp_dir / "tokenizer_vocab.json").write_text(
        compact(vocab), encoding="utf-8")
    (tmp_dir / "tokenizer_scores.json").write_text(
        compact(scores), encoding="utf-8")

    add_metadata(tmp_dir / "moss_tts_prefill.onnx", tts_dir, codec_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def export_int8(fp32_dir: Path, int8_out_dir: Path):
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as e:
        raise RuntimeError(
            "onnxruntime.quantization is required for --int8-out-dir"
        ) from e

    tmp_dir = int8_out_dir.with_name(int8_out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    for name in RUNTIME_FILES:
        src = fp32_dir / name
        dst = tmp_dir / int8_name(name)
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
        )
    shutil.copy2(fp32_dir / "tokenizer_vocab.json",
                 tmp_dir / "tokenizer_vocab.json")
    shutil.copy2(fp32_dir / "tokenizer_scores.json",
                 tmp_dir / "tokenizer_scores.json")

    if int8_out_dir.exists():
        shutil.rmtree(int8_out_dir)
    tmp_dir.rename(int8_out_dir)


def main():
    args = get_args()
    tts_dir = args.tts_dir.resolve()
    codec_dir = args.codec_dir.resolve()
    out_dir = args.out_dir.resolve()

    for name in TTS_ONNX_FILES + TTS_EXTERNAL_DATA_FILES:
        require_file(tts_dir / name)
    for name in CODEC_ONNX_FILES + CODEC_EXTERNAL_DATA_FILES:
        require_file(codec_dir / name)
    for name in (
        "tokenizer.model",
        "tts_browser_onnx_meta.json",
        "browser_poc_manifest.json",
    ):
        require_file(tts_dir / name)
    require_file(codec_dir / "codec_browser_onnx_meta.json")

    export_fp32(tts_dir, codec_dir, out_dir)

    print(f"Exported FP32 MOSS TTS model to {out_dir}")
    print("Runtime files:")
    for name in RUNTIME_FILES:
        print(f"  {name}")
    print("Extra files:")
    print("  tokenizer_vocab.json")
    print("  tokenizer_scores.json")

    if args.int8_out_dir is not None:
        int8_out_dir = args.int8_out_dir.resolve()
        export_int8(out_dir, int8_out_dir)
        print(f"Exported int8 MOSS TTS model to {int8_out_dir}")
        print("Int8 runtime files:")
        for name in RUNTIME_FILES:
            print(f"  {int8_name(name)}")


if __name__ == "__main__":
    main()
