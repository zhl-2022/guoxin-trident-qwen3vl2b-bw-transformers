import importlib.util
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/robobrain_protocol_smoke.py"
SPEC = importlib.util.spec_from_file_location("robobrain_protocol_under_test", SCRIPT)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)
TASK = SCRIPT.parents[1] / "tasks/robobrain-bw-vllm/task.json"


class ProtocolSmokeTests(unittest.TestCase):
    def test_request_uses_exact_template_and_one_image_greedy(self):
        config = protocol.load_configuration(TASK)
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}}
        for task in protocol.TASKS:
            with self.subTest(task=task):
                body = protocol.build_request(config, task, "the red square", image, "robo", 128)
                content = body["messages"][0]["content"]
                self.assertEqual(len([part for part in content if part["type"] == "image_url"]), 1)
                self.assertEqual(content[1]["text"], config["upstream_protocol"]["prompt_templates"][task].format(text="the red square"))
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["max_tokens"], 128)
                self.assertFalse(body["stream"])

    def test_reasoning_only_and_token_limit_are_not_success(self):
        for message, reason in (({"content": None, "reasoning_content": "thinking"}, "length"), ({"content": "[(1, 2)]"}, "length")):
            result = protocol.response_diagnostics("pointing", {"choices": [{"message": message, "finish_reason": reason}]})
            self.assertFalse(result["functional_protocol_check_passed"])
            self.assertTrue(result["token_limit_reached"])
            self.assertFalse(result["semantic_accuracy_verified"])

    def test_syntax_is_diagnostic_and_never_coordinate_correctness(self):
        result = protocol.response_diagnostics("pointing", {"choices": [{"message": {"content": "[(99999, 99999)]"}, "finish_reason": "stop"}]})
        self.assertTrue(result["syntax_pattern_observed"])
        self.assertFalse(result["coordinate_range_checked"])
        self.assertFalse(result["semantic_accuracy_verified"])

    def test_no_remote_image_destination_or_unsafe_template(self):
        for url in ("https://example.com", "http://user:password@127.0.0.1", "http://127.0.0.1/v1", "http://127.0.0.1?token=abc"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                protocol.local_base_url(url)
        with tempfile.TemporaryDirectory() as temp:
            config = json.loads(TASK.read_text(encoding="utf-8"))
            config["upstream_protocol"]["prompt_templates"]["pointing"] = "{text.__class__}"
            path = Path(temp) / "task.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                protocol.load_configuration(path)

    def test_local_http_error_is_saved_without_image_payload_or_retry(self):
        posts = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                raw = json.dumps({"data": [{"id": "robobrain2-5-8b-nv"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(raw)
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                posts.append(json.loads(raw))
                self.send_response(500); self.end_headers()
                self.wfile.write(b'{"error":"echo data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="}')
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                image = Path(temp) / "fixture.png"
                image.write_bytes(b"fake-image-bytes")
                output = Path(temp) / "new.jsonl"
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    code = protocol.main(["--task", str(TASK), "--image", str(image), "--target", "red square", "--trajectory-instruction", "reach for the red square", "--base-url", f"http://127.0.0.1:{server.server_port}", "--log", str(output)])
                self.assertEqual(code, 2)
                self.assertEqual(len(posts), 1)
                text = output.read_text(encoding="utf-8")
                self.assertIn("HTTP 500", text)
                self.assertNotIn("ZmFrZS1pbWFnZS1ieXRlcw==", text)
                records = [json.loads(line) for line in text.splitlines()]
                self.assertFalse(records[-1]["functional_protocol_checks_passed"])
                self.assertFalse(records[-1]["official_accuracy"])
                self.assertEqual(records[-1]["requests_sent"], 1)
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)

    def test_four_local_success_responses_are_only_functional(self):
        posts = []
        answers = ["A red square.", "[100, 100, 900, 900]", "[(500, 500)]", "[(500, 500, 0.25), (600, 600, 0.30)]"]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                raw = json.dumps({"data": [{"id": "robobrain2-5-8b-nv"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(raw)
            def do_POST(self):
                posts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": answers[len(posts)-1]}, "finish_reason": "stop"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(raw)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                image = Path(temp) / "fixture.png"
                image.write_bytes(b"fixture bytes")
                output = Path(temp) / "new.jsonl"
                args = ["--task", str(TASK), "--image", str(image), "--target", "red square", "--trajectory-instruction", "reach for the red square", "--base-url", f"http://127.0.0.1:{server.server_port}", "--log", str(output)]
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(protocol.main(args), 0)
                    self.assertEqual(protocol.main(args), 1)  # Existing evidence is preserved.
                self.assertEqual(len(posts), 4)
                records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(records[-1]["functional_protocol_checks_passed"])
                self.assertFalse(records[-1]["semantic_accuracy_verified"])
                self.assertFalse(records[-1]["official_accuracy"])
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
