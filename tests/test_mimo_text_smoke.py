"""MiMo text transport/entry checks with CPU fixtures and a loopback mock only."""
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mimo_smoke_test", ROOT / "scripts/mimo_text_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
MODEL = "mimo-v2.6-distill-qwen-9b"


def answer(**overrides):
    value = {"model": MODEL, "choices": [{"index": 0, "message": {"content": "<think>raw reasoning</think> Hello"},
                                         "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    value.update(overrides)
    return value


def event(content="", finish=None, **extra):
    return "data: " + json.dumps({"model": MODEL, "choices": [{"index": 0, "delta": {"content": content},
                                                               "finish_reason": finish}], **extra}) + "\n\n"


def complete_stream():
    return event("Hello", "stop") + 'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n' + "data: [DONE]\n\n"


class TextSmokeTests(unittest.TestCase):
    def consume(self, body):
        self.events = []
        return smoke.consume_stream(io.BytesIO(body.encode()), MODEL, 1,
                                    lambda case, **values: self.events.append({"case": case, **values}))

    def test_raw_thinking_is_preserved_and_no_semantic_score(self):
        value = answer()
        self.assertEqual(smoke.response_check(value, MODEL), value["choices"][0]["message"]["content"])
        self.assertIn("<think>", value["choices"][0]["message"]["content"])

    def test_json_errors_truncation_reasoning_only_and_invalid_usage_rejected(self):
        for value in (answer(error={"message": "bad"}), answer(model="other"), answer(choices=[]),
                      answer(choices=[{"finish_reason": "length", "message": {"content": "partial"}}]),
                      answer(choices=[{"finish_reason": "stop", "message": {"content": None, "reasoning": "thought"}}]),
                      answer(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 14}),
                      answer(usage={"prompt_tokens": True, "completion_tokens": 5, "total_tokens": 6})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smoke.response_check(value, MODEL)

    def test_sse_raw_events_finish_usage_and_done_preserved(self):
        result = self.consume(complete_stream())
        self.assertEqual(result["raw_text"], "Hello")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["raw_usage"]["total_tokens"], 15)
        self.assertEqual(self.events[-1]["raw_data"], "[DONE]")

    def test_sse_truncation_missing_done_late_choices_and_error_rejected(self):
        cases = [event("text", "length") + "data: [DONE]\n\n", event("text", "stop"),
                 event("text", "stop") + event("late") + "data: [DONE]\n\n",
                 event("partial") + 'data: {"error":{"message":"bad"}}\n\ndata: [DONE]\n\n',
                 complete_stream().replace(MODEL, "wrong"),
                 complete_stream().replace('"total_tokens":15', '"total_tokens":14'),
                 complete_stream().replace("Hello", "")]
        for body in cases:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.consume(body)
        self.assertTrue(self.events)

    def test_sse_last_callback_cannot_bypass_deadline(self):
        now = [0.0]
        def record(case, **values):
            if values.get("raw_data") == "[DONE]":
                now[0] = 2.0
        with patch.object(smoke.time, "monotonic", side_effect=lambda: now[0]), self.assertRaisesRegex(ValueError, "wall timeout"):
            smoke.consume_stream(io.BytesIO(complete_stream().encode()), MODEL, 1, record)

    def test_complete_http_body_cannot_hide_elapsed_deadline(self):
        with patch.object(smoke.time, "monotonic", side_effect=[0.0, 2.0]), self.assertRaisesRegex(ValueError, "absolute wall timeout"):
            smoke.read_body(io.BytesIO(b"{}"), 1)

    def test_duplicate_and_premature_stream_usage_are_rejected(self):
        usage = 'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        for body in (usage + complete_stream(), complete_stream().replace("data: [DONE]", usage + "data: [DONE]")):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "usage"):
                self.consume(body)

    def test_default_plan_writes_nothing_and_never_opens_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(smoke.urllib.request, "build_opener", side_effect=AssertionError("No network")), redirect_stdout(io.StringIO()) as output:
            path = Path(temporary) / "not-created.log"
            self.assertEqual(smoke.main(["--dry-run", "--log", str(path)]), 0)
            self.assertFalse(path.exists())
            self.assertFalse(json.loads(output.getvalue())["temperature_top_p_top_k_or_thinking_override_sent"])

    def test_remote_and_credential_urls_rejected(self):
        for value in ("https://example.com", "http://127.0.0.1/path", "http://u:p@127.0.0.1", "http://127.0.0.1?x=y"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smoke.base_url(value)

    def server(self, broken_stream=False, missing_model=False):
        posted = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                raw = (b"ok" if self.path == "/health" else json.dumps({"data": [{"id": "other" if missing_model else MODEL}]}).encode())
                self.send_response(200); self.end_headers(); self.wfile.write(raw)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                posted.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if body["stream"] else "application/json")
                self.end_headers()
                raw = (event("partial") + 'data: {"error":{"message":"broken"}}\n\n') if broken_stream else complete_stream()
                self.wfile.write(raw.encode() if body["stream"] else json.dumps(answer()).encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(thread.join, 2); self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}", posted

    def test_real_loopback_four_requests_keep_sampling_and_thinking_defaults(self):
        url, posted = self.server()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            path = Path(temporary) / "fresh.log"
            self.assertEqual(smoke.main(["--base-url", url, "--log", str(path)]), 0)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(posted), 4)
        for body in posted:
            self.assertEqual(set(body), {"model", "messages", "max_tokens", "stream", *(["stream_options"] if body["stream"] else [])})
        self.assertEqual(posted[2]["messages"][1]["content"], answer()["choices"][0]["message"]["content"])
        self.assertTrue(rows[-1]["functional_checks_passed"])
        self.assertFalse(rows[-1]["answer_correctness_scored"])
        self.assertTrue(any(row["case"] == "stream_raw_event" for row in rows))

    def test_failure_raw_events_remain_in_new_log_and_existing_log_not_replaced(self):
        url, posted = self.server(broken_stream=True)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            path = Path(temporary) / "fresh.log"
            self.assertEqual(smoke.main(["--base-url", url, "--log", str(path)]), 2)
            original = path.read_bytes()
            self.assertIn(b'broken', original)
            self.assertEqual(smoke.main(["--base-url", url, "--log", str(path)]), 2)
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(posted), 4)

    def test_wrong_model_stops_before_any_generation_request(self):
        url, posted = self.server(missing_model=True)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(["--base-url", url, "--log", str(Path(temporary) / "new.log")]), 2)
        self.assertEqual(posted, [])


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "Bash entries run in WSL")
class MimoEntryTests(unittest.TestCase):
    def entry(self, name, *args, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MIMO_")}
        env.update(MODEL_ID="wrong", FRAMEWORK="sglang", PRECISION="int4", PORT="1", PYTHON_BIN="/wrong",
                   OFFICIAL_VERIFY_SCRIPT="/wrong", OFFICIAL_BENCH_OUTPUTS_JSON='["wrong.log"]',
                   MIMO_PYTHON_BIN=sys.executable)
        env.update(overrides)
        return subprocess.run(["bash", str(ROOT / "tasks/mimo-v2.6-distill-qwen9b-bw-vllm" / name), *args],
                              env=env, capture_output=True, text=True, timeout=15)

    def test_launch_plan_has_32k_native_identity_and_no_reasoning_or_sampling_override(self):
        result = self.entry("run.sh", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["model_id"], "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")
        self.assertEqual(plan["max_len"], 32768)
        self.assertEqual(plan["framework"], "vllm")
        self.assertEqual(plan["precision"], "bf16")
        self.assertNotIn("--reasoning-parser", plan["command"])

    def test_selfcheck_only_requires_real_script_and_bench_needs_t01_contract(self):
        result = self.entry("verify.sh", "--selfcheck", "--dry-run")
        self.assertEqual(result.returncode, 2)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["problems"], ["missing OFFICIAL_VERIFY_SCRIPT"])
        self.assertTrue(plan["selfcheck_only"])
        result = self.entry("bench.sh", "--dry-run")
        self.assertEqual(result.returncode, 2)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["track"], "T01")
        self.assertIn("missing OFFICIAL_BENCH_OUTPUTS_JSON", plan["problems"])
        self.assertNotIn("t02_template_load_plan", plan)

    def test_smoke_plan_and_core_extra_override_rejection(self):
        result = self.entry("smoke.sh", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["temperature_top_p_top_k_or_thinking_override_sent"])
        result = self.entry("run.sh", "--dry-run", MIMO_EXTRA_ARGS_JSON='["--dtype","float16"]')
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot override", result.stderr)


if __name__ == "__main__":
    unittest.main()
