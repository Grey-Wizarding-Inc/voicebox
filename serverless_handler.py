"""RunPod Serverless handler for TADA-3B voice cloning.

Reuses voicebox's HumeTadaBackend so every solved dependency
(--no-deps chatterbox/hume-tada, git-only `tada`, the DAC shim, the ungated
Llama-tokenizer redirect) and the encoder `no_grad` fix come along unchanged.

Input  (job["input"]):
    text                 str   - text to speak (required)
    reference_audio_b64  str   - base64 WAV of the voice to clone (required)
    reference_text       str   - transcript of the reference clip (optional)
    language             str   - default "en"
    seed                 int   - optional

Output:
    audio_b64, sample_rate (24000), duration  -- or {"error": "..."}
"""

import os
import io
import time
import base64
import tempfile

# Cache HF downloads on the network volume when one is attached, so cold starts
# don't re-pull the ~8GB model every wake. Without a volume, falls back to the
# container default (re-downloads on each cold start) — fine for first load-time
# measurements; this is the optimization we deliberately defer.
# Fast parallel HF downloads (also set as ENV in the image; belt-and-suspenders)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# RunPod "Cached Models": when an endpoint has a cached model, RunPod pre-stages
# it on the host's LOCAL disk at /runpod-volume/huggingface-cache/hub/ in standard
# HF cache layout. Pointing HF_HOME there makes snapshot_download a cache HIT
# (no download, not billed) instead of pulling 8GB over a slow flex-worker link.
# This is NOT the slow object store — it's host-local and routed to warm hosts.
if os.path.isdir("/runpod-volume/huggingface-cache"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface-cache")
elif os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface")

import runpod
import soundfile as sf

from backend.backends.hume_backend import HumeTadaBackend

MODEL_SIZE = os.environ.get("TADA_MODEL_SIZE", "3B")

# One backend per worker. Model loads lazily on the first request (that first
# call's latency IS the cold-start load time); subsequent calls are warm.
_backend = HumeTadaBackend()


async def handler(job):
    inp = job.get("input", {}) or {}
    text = inp.get("text")
    ref_b64 = inp.get("reference_audio_b64")
    ref_text = inp.get("reference_text", "")
    language = inp.get("language", "en")
    seed = inp.get("seed")

    if not text:
        return {"error": "missing 'text'"}
    if not ref_b64:
        return {"error": "missing 'reference_audio_b64'"}

    ref_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(base64.b64decode(ref_b64))
            ref_path = f.name

        timings = {}
        _t = time.monotonic()
        await _backend.load_model(MODEL_SIZE)  # no-op once warm
        timings["load_total_s"] = round(time.monotonic() - _t, 1)
        # cold-load split (download_s vs load_s); 0/absent on a warm worker
        timings.update(getattr(_backend, "last_timings", {}))

        _t = time.monotonic()
        prompt, _ = await _backend.create_voice_prompt(ref_path, ref_text, use_cache=False)
        timings["encode_s"] = round(time.monotonic() - _t, 1)

        _t = time.monotonic()
        audio, sr = await _backend.generate(text, prompt, language=language, seed=seed)
        timings["gen_s"] = round(time.monotonic() - _t, 1)

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return {
            "audio_b64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": sr,
            "duration": float(len(audio) / sr),
            "timings": timings,
        }
    except Exception as exc:  # surface errors as job output, not worker crash
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


runpod.serverless.start({"handler": handler})
