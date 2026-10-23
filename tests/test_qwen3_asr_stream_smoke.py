"""CPU-only SSE protocol checks; no ASR model, external service or GPU."""
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("asr_stream_diagnostic", SCRIPTS / "qwen3_asr_stream_smoke.py")
stream = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream)


def event(text="", finish=None, **extra):
    return "data: " + json.dumps({"model": "qwen3-asr-1.7b", "choices": [
        {"delta": {"content": text}, "finish_reason": finish}], **extra}, ensure_ascii=False) + "\n\n"


class AsrStreamSmokeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.audio = self.directory / "test.wav"
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 160)

    def consume(self, body):
        self.events = []
        return stream.consume_stream(io.BytesIO(body.encode("utf-8")), "qwen3-asr-1.7b", 1,
                                     lambda case, **values: self.events.append({"case": case, **values}))

    def test_real_sse_framing_preserves_split_prefix_text_usage_and_stop(self):
        body = ": comment\r\n\r\n" + event("language Chinese<asr_") + event("text>你好", "stop")
        body += 'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n'
        result = self.consume(body + "data: [DONE]\n\n")
        self.assertEqual(result["raw_text"]["text"], "language Chinese<asr_text>你好")
        self.assertEqual(result["usage"], {"completion_tokens": 7})
        self.assertTrue(result["done_marker_received"])
        self.assertFalse(result["asr_prefix_stripped"])
        self.assertTrue(result["transcript_body_nonempty"])

    def test_prefix_only_is_not_a_transcript(self):
        for text in ("language Chinese<asr_text>", "language English<asr_text> \t"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "after ASR prefix"):
                self.consume(event(text, "stop") + "data: [DONE]\n\n")
            self.assertEqual(self.events[0]["delta_text"]["text"], text)

    def test_terminal_finish_cannot_be_reused_for_later_choices(self):
        for later in (event("unfinished tail"), event("", "stop"), event("", "length")):
            with self.subTest(later=later), self.assertRaisesRegex(ValueError, "after terminal finish"):
                self.consume(event("complete first text", "stop") + later + "data: [DONE]\n\n")

    def test_late_done_read_and_record_cannot_bypass_absolute_deadline(self):
        body = (event("text", "stop") + "data: [DONE]\n\n").encode("utf-8")
        for delayed in ("read", "record"):
            clock = [0.0]

            class DelayedFinalLine(io.BytesIO):
                def readline(self, *args):
                    line = super().readline(*args)
                    if delayed == "read" and self.tell() == len(self.getvalue()):
                        clock[0] = 2.0
                    return line

            def record(case, **_values):
                if delayed == "record" and case == "stream_done":
                    clock[0] = 2.0

            with self.subTest(delayed=delayed), patch.object(stream.time, "monotonic", side_effect=lambda: clock[0]):
                with self.assertRaisesRegex(ValueError, "stream wall timeout"):
                    stream.consume_stream(DelayedFinalLine(body), "qwen3-asr-1.7b", 1, record)

    def test_error_after_partial_text_is_never_success(self):
        with self.assertRaisesRegex(ValueError, "SSE error"):
            self.consume(event("partial") + 'data: {"error":{"message":"broken"}}\n\ndata: [DONE]\n\n')
        self.assertEqual(self.events[0]["delta_text"]["text"], "partial")

    def test_missing_done_empty_text_and_length_finish_are_rejected(self):
        for body, error in ((event("text", "stop"), "without "),
                            (event("", "stop") + "data: [DONE]\n\n", "no transcript"),
                            (event("text", "length") + "data: [DONE]\n\n", "not stop"),
                            (event("text") + "data: [DONE]\n\n", "not stop")):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.consume(body)

    def test_malformed_model_and_byte_limits_are_rejected(self):
        cases = [(event("text", "stop").replace("qwen3-asr-1.7b", "other-model"), "another model"),
                 ("data: " + "x" * stream.MAX_LINE_BYTES, "byte cap"),
                 ('data: {"choices":"bad"}\n\n', "choices array"),
                 ("not SSE\n", "framing")]
        for body, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.consume(body)

    def test_multipart_has_stream_flags_without_audio_text_prompt(self):
        payload, content_type = stream.streaming_multipart("qwen3-asr-1.7b", self.audio.read_bytes())
        self.assertIn(b'name="stream"\r\n\r\ntrue', payload)
        self.assertIn(b'name="stream_include_usage"\r\n\r\ntrue', payload)
        self.assertIn(b'name="file"', payload)
        self.assertNotIn(b'name="prompt"', payload)
        self.assertNotIn(b'name="language"', payload)
        self.assertIn("multipart/form-data", content_type)

    def test_default_plan_performs_no_network_and_no_writes(self):
        output = io.StringIO()
        with patch.object(stream.urllib.request, "build_opener") as opener, contextlib.redirect_stdout(output):
            code = stream.main(["--audio", str(self.audio), "--log", str(self.directory / "unused.jsonl")])
        self.assertEqual(code, 0)
        opener.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertFalse(plan["run"])
        self.assertFalse(plan["realtime_audio_input_verified"])
        self.assertEqual(plan["network_requests"], 0)
        self.assertEqual(list(self.directory.iterdir()), [self.audio])

    def test_nonloopback_missing_log_and_existing_log_fail_before_network(self):
        existing = self.directory / "existing.jsonl"
        existing.write_bytes(b"previous")
        scenarios = [["--base-url", "https://example.com"], ["--run"],
                     ["--run", "--log", str(existing)]]
        for extra in scenarios:
            with self.subTest(extra=extra), patch.object(stream.urllib.request, "build_opener") as opener:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(stream.main(["--audio", str(self.audio), *extra]), 2)
                opener.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"previous")

    def test_explicit_run_uses_real_loopback_http_and_preserves_sse_evidence(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                received.append(self.path)
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(b'{"data":[{"id":"qwen3-asr-1.7b"}]}')

            def do_POST(self):
                received.append(self.path)
                received.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write((event("你好", "stop") + "data: [DONE]\n\n").encode("utf-8"))

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            log = self.directory / "stream.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                code = stream.main(["--audio", str(self.audio), "--run", "--log", str(log),
                                    "--base-url", f"http://127.0.0.1:{server.server_port}"])
            self.assertEqual(code, 0)
            self.assertEqual(received[:2], ["/v1/models", "/v1/audio/transcriptions"])
            self.assertIn(b'name="stream"\r\n\r\ntrue', received[2])
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["raw_text"]["text"], "你好")
            self.assertTrue(rows[-1]["done_marker_received"])
            self.assertFalse(rows[-1]["official_accuracy"])
            self.assertFalse(rows[-1]["realtime_audio_input_verified"])
            self.assertNotIn("RIFF", log.read_text(encoding="utf-8"))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_real_loopback_late_done_and_slow_trickle_stop_at_wall_deadline(self):
        for mode in ("late_done", "slow_trickle"):
            with self.subTest(mode=mode):
                stop = threading.Event()

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *_args):
                        pass

                    def do_GET(self):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"data":[{"id":"qwen3-asr-1.7b"}]}')

                    def do_POST(self):
                        self.rfile.read(int(self.headers["Content-Length"]))
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        try:
                            self.wfile.write(event("partial transcript", "stop").encode("utf-8"))
                            self.wfile.flush()
                            if mode == "late_done":
                                if not stop.wait(0.9):
                                    self.wfile.write(b"data: [DONE]\n\n")
                                    self.wfile.flush()
                            else:
                                for byte in b"data: [DONE]\n\n":
                                    if stop.wait(0.08):
                                        return
                                    self.wfile.write(bytes([byte]))
                                    self.wfile.flush()
                        except OSError:
                            pass

                server = HTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
                thread.start()
                log = self.directory / f"{mode}.jsonl"
                try:
                    started = time.monotonic()
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = stream.main(["--audio", str(self.audio), "--run", "--log", str(log),
                                            "--timeout", "0.2", "--base-url", f"http://127.0.0.1:{server.server_port}"])
                    elapsed = time.monotonic() - started
                    self.assertEqual(code, 2)
                    self.assertLess(elapsed, 0.65, f"{mode} continued beyond the absolute deadline")
                    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(rows[-1]["case"], "error")
                    self.assertFalse(rows[-1]["streaming_transcript_received"])
                    self.assertIn("timeout", rows[-1]["error"]["text"].lower())
                    self.assertTrue(any(row["case"] == "stream_event" for row in rows))
                    self.assertFalse(any(row["case"] == "summary" for row in rows))
                finally:
                    stop.set()
                    server.shutdown(); server.server_close(); thread.join(timeout=2)
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
