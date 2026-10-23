import importlib.util
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import wave
from contextlib import redirect_stdout, redirect_stderr
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen3_asr_smoke.py"
SPEC = importlib.util.spec_from_file_location("qwen3_asr_smoke_under_test", SCRIPT)
asr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(asr)
TASK = SCRIPT.parents[1] / "tasks/qwen3-asr-bw-vllm/task.json"


def wav_bytes(channels=1, rate=16000, frames=1600):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels); audio.setsampwidth(2); audio.setframerate(rate)
        audio.writeframes(b"\0\0" * channels * frames)
    return output.getvalue()


class ASRSmokeTests(unittest.TestCase):
    def test_multipart_keeps_original_wav_and_has_no_reference_prompt(self):
        raw = wav_bytes()
        body, content_type = asr.multipart_request("qwen3-asr-1.7b", raw, "test-boundary")
        parsed = BytesParser(policy=policy.default).parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
        parts = {part.get_param("name", header="content-disposition"): part for part in parsed.iter_parts()}
        self.assertEqual(set(parts), {"model", "file", "response_format", "temperature"})
        self.assertEqual(parts["file"].get_payload(decode=True), raw)
        self.assertEqual(parts["file"].get_content_type(), "audio/wav")
        self.assertEqual(parts["temperature"].get_payload(decode=True), b"0")

    def test_wav_validation_does_not_silently_convert_or_trim(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.wav"
            path.write_bytes(wav_bytes())
            raw, info = asr.read_wav(path)
            self.assertEqual(raw, path.read_bytes())
            self.assertEqual(info["duration_seconds"], 0.1)
            self.assertFalse(info["resampled_or_trimmed"])
            for invalid in (wav_bytes(channels=2), wav_bytes(rate=8000), wav_bytes(frames=16000*31), wav_bytes()[:-2]):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    asr.read_wav(path)

    def test_expected_text_mismatch_is_not_official_metric(self):
        result = asr.transcript_diagnostic({"text": "a different transcript"}, "expected")
        self.assertFalse(result["exact_text_match_diagnostic"])
        self.assertTrue(result["transcription_nonempty"])
        self.assertFalse(result["semantic_accuracy_verified"])
        self.assertFalse(result["wer_computed"])
        self.assertFalse(result["truncation_verified_absent"])

    def test_long_audio_requires_explicit_flag_and_remains_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "long.wav"
            path.write_bytes(wav_bytes(frames=16000 * 31))
            with self.assertRaises(ValueError):
                asr.read_wav(path)
            raw, info = asr.read_wav(path, long_audio=True)
            self.assertEqual(raw, path.read_bytes())
            self.assertTrue(info["long_audio_explicitly_enabled"])
            self.assertEqual(info["duration_limit_seconds"], 120)
            path.write_bytes(wav_bytes(frames=16000 * 121))
            with self.assertRaises(ValueError):
                asr.read_wav(path, long_audio=True)

    def test_http_error_does_not_expose_uploaded_or_echoed_audio(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1/v1/audio/transcriptions", 500, "failure", {},
            io.BytesIO(b'uploaded-audio-content-that-must-not-be-logged'),
        )
        with self.assertRaises(ValueError) as raised:
            asr.request_json(opener, "http://127.0.0.1/v1/audio/transcriptions", 1, b"WAV")
        self.assertIn("HTTP 500", str(raised.exception))
        self.assertNotIn("uploaded-audio-content", str(raised.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_remote_destination_and_redirects_are_rejected(self):
        for url in ("https://example.com", "http://127.0.0.1/v1", "http://user:secret@localhost"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                asr.local_base_url(url)
        with self.assertRaises(ValueError):
            asr.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.com")

    def test_local_http_transcription_and_evidence_preservation(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"data":[{"id":"qwen3-asr-1.7b"}]}')
            def do_POST(self):
                captured.append((self.path, self.headers["Content-Type"], self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"text":"fixture transcript"}')
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path, log = Path(temp)/"sample.wav", Path(temp)/"new.jsonl"
                path.write_bytes(wav_bytes())
                args = ["--task", str(TASK), "--audio", str(path), "--expected-text", "expected-reference-never-uploaded", "--base-url", f"http://127.0.0.1:{server.server_port}", "--log", str(log)]
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(asr.main(args), 0)
                    self.assertEqual(asr.main(args), 1)
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0][0], "/v1/audio/transcriptions")
                self.assertNotIn(b"expected-reference-never-uploaded", captured[0][2])
                records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
                self.assertFalse(records[-2]["exact_text_match_diagnostic"])
                self.assertTrue(records[-1]["functional_transcription_received"])
                self.assertFalse(records[-1]["official_accuracy"])
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
