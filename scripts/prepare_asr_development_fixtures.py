"""Build traceable ASR format/concatenation diagnostics from downloaded Qwen demos."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fixtures/asr/publisher-qwen-20261020"
TARGET = ROOT / "fixtures/asr/development-20261020"


def pcm(raw):
    with wave.open(io.BytesIO(raw), "rb") as stream:
        params = stream.getparams()
        data = stream.readframes(params.nframes)
    if params.nchannels != 1 or params.comptype != "NONE":
        raise ValueError("Expected publisher mono uncompressed WAV")
    return params, data


def wav_bytes(data):
    out = io.BytesIO()
    with wave.open(out, "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(data)
    return out.getvalue()


def build():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    originals = {}
    for row in manifest["samples"]:
        path = ROOT / row["path"]
        if path.resolve().parent != SOURCE.resolve():
            raise ValueError("Source must stay in the downloaded publisher directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Publisher sample hash changed")
        originals[row["id"]] = raw
    zh_params, zh = pcm(originals["qwen-publisher-zh"])
    en_params, en = pcm(originals["qwen-publisher-en"])
    if (zh_params.framerate, zh_params.sampwidth) != (16000, 2):
        raise ValueError("Chinese source layout changed")
    if (en_params.framerate, en_params.sampwidth) != (48000, 3):
        raise ValueError("English source layout changed")
    # CPU data preparation only; never import Torch, vLLM or model packages.
    import numpy as np
    import scipy
    from scipy.signal import resample_poly
    packed = np.frombuffer(en, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    signed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
    signal = resample_poly(signed.astype(np.float64) / 8388608, 1, 3)
    en16 = np.clip(np.rint(signal * 32768), -32768, 32767).astype("<i2").tobytes()
    silence = b"\0\0" * 4000
    payloads = {"zh_16000_pcm16.wav": wav_bytes(zh),
                "en_16000_pcm16.wav": wav_bytes(en16),
                "zh_en_repeated_39s.wav": wav_bytes(silence.join((zh, en16, zh, en16)))}
    rows = []
    for name, raw in payloads.items():
        params, _ = pcm(raw)
        rows.append({"id": name.removesuffix(".wav"), "path": (TARGET / name).relative_to(ROOT).as_posix(),
                     "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                     "seconds": params.nframes / params.framerate,
                     "sample_rate": params.framerate, "channels": 1, "sample_width_bytes": 2,
                     "reference_text": None, "official_contest_data": False})
    report = {"development_only": True, "official_evaluation": "not_run", "sources": manifest["samples"],
              "transformations": {"zh": "decode and write same PCM16 samples without resampling",
                                  "en": "PCM24 signed samples / 2**23, scipy.signal.resample_poly(1,3), round/clip PCM16",
                                  "long": "zh,en,zh,en with 0.25s silence between segments; artificial concatenation"},
              "numpy": np.__version__, "scipy": scipy.__version__, "samples": rows,
              "natural_continuous_recording": False, "source_files_modified": False}
    TARGET.mkdir(exist_ok=False)
    for name, raw in payloads.items():
        (TARGET / name).write_bytes(raw)
    (TARGET / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"mode": "plan_only", "source": str(SOURCE), "target": str(TARGET),
                          "writes": False, "network": False, "model_loaded": False}))
