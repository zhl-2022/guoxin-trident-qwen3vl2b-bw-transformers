import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_vllm_metrics_004.py"
SPEC = importlib.util.spec_from_file_location("collect_vllm_metrics_004", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
RUN = "a" * 32
MODEL = "/root/models/test-model"


class Clock:
    def __init__(self):
        self.value = 0
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


def raw_metrics(tokens=1024, requests=1):
    return (f'# TYPE vllm:generation_tokens_total counter\n'
            f'vllm:generation_tokens_total{{model_name="{MODEL}",engine="0"}} {tokens}\n'
            f'vllm:request_success_total{{model_name="{MODEL}",finished_reason="length"}} {requests}\n'
            f'vllm:request_generation_tokens_sum{{model_name="{MODEL}"}} {tokens}\n'
            f'vllm:request_generation_tokens_count{{model_name="{MODEL}"}} {requests}\n').encode()


class MetricsCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.modules_before = set(sys.modules)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.output = self.root / "collected"
        self.clock = Clock()
        self.command = ["/usr/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
                        "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", "30000"]
        self.config = {"run_id": RUN, "framework": "vllm", "host": "127.0.0.1", "port": 30000,
                       "url": "http://127.0.0.1:30000", "model_path": MODEL, "served_name": MODEL, "command": self.command}
        self.state = {"run_id": RUN, "status": "ready", "framework": "vllm", "url": self.config["url"], "server_pid": 12345,
                      "server_token": "98765", "config_path": str(self.runtime / ("config-" + RUN + ".json")), "command": self.command}
        self.write_runtime()

    def write_runtime(self):
        (self.runtime / "state.json").write_text(json.dumps(self.state))
        (self.runtime / ("config-" + RUN + ".json")).write_text(json.dumps(self.config))

    def run_collect(self, request=None, **kwargs):
        return collector.collect(self.runtime, RUN, self.output, max_seconds=kwargs.pop("max_seconds", 3), interval_seconds=1,
                                 token_reader=kwargs.pop("token_reader", lambda pid: "98765"),
                                 request=request or (lambda url, timeout: raw_metrics()),
                                 clock=self.clock.now, sleeper=self.clock.sleep, **kwargs)

    def test_bounded_one_http_per_sample_raw_sha_and_no_certification(self):
        requests = []
        def request(url, timeout):
            requests.append((url, timeout))
            return raw_metrics(len(requests) * 1024, len(requests))
        result = self.run_collect(request)
        self.assertEqual(result["reason"], "deadline_reached")
        self.assertEqual(result["samples"], 3)
        self.assertEqual(requests, [("http://127.0.0.1:30000/metrics", 2)] * 3)
        self.assertFalse(result["official_acceptance"])
        self.assertFalse(result["per_request_exactness_proven"])
        self.assertFalse(result["aggregate_token_comparison_performed"])
        rows = [json.loads(line) for line in (self.output / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["raw_sha256"], collector.digest((self.output / "metrics-first.txt").read_bytes()))
        self.assertEqual(rows[-1]["raw_sha256"], collector.digest((self.output / "metrics-last-good.txt").read_bytes()))
        self.assertEqual(len(rows[-1]["selected"]), 4)
        self.assertNotIn("torch", set(sys.modules) - self.modules_before)
        self.assertLess(sum(p.stat().st_size for p in self.output.iterdir()), 20_000_000)

    def test_identity_failure_sends_no_http_and_preserves_terminal(self):
        self.state["server_token"] = "wrong"
        self.write_runtime()
        request = mock.Mock()
        result = self.run_collect(request)
        self.assertEqual(result["samples"], 0)
        request.assert_not_called()
        self.assertIn("PID/start token", result["error"])
        self.assertTrue((self.output / "collector-final.json").is_file())

    def test_shutdown_preserves_last_good_without_stopping_service(self):
        calls = 0
        def request(url, timeout):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("server stopped independently")
            return raw_metrics()
        result = self.run_collect(request)
        self.assertEqual(result["samples"], 1)
        self.assertEqual((self.output / "metrics-last-good.txt").read_bytes(), raw_metrics())
        self.assertFalse(result["server_mutated"])
        self.assertEqual(json.loads((self.runtime / "state.json").read_text())["status"], "ready")

    def test_pid_reuse_and_model_change_stop_sampling(self):
        calls = 0
        def token(pid):
            nonlocal calls
            calls += 1
            return "98765" if calls <= 3 else "reused"
        result = self.run_collect(token_reader=token)
        self.assertEqual(result["samples"], 1)
        self.assertIn("PID/start token", result["error"])
        self.output = self.root / "second"
        self.clock = Clock()
        calls = 0
        def request(url, timeout):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.config["model_path"] = "/root/models/changed"
                self.write_runtime()
            return raw_metrics()
        result = self.run_collect(request)
        self.assertEqual(result["samples"], 1)
        self.assertIn("model/endpoint", result["error"])
        self.assertEqual((self.output / "metrics-last-good.txt").read_bytes(), raw_metrics())

    def test_loopback_model_runid_and_framework_required(self):
        for change, target in (({"host": "example.com"}, "config"), ({"run_id": "b" * 32}, "state"),
                               ({"framework": "sglang"}, "config"), ({"model_path": "relative-model"}, "config")):
            old_config, old_state = dict(self.config), dict(self.state)
            (self.config if target == "config" else self.state).update(change)
            self.write_runtime()
            with self.assertRaises(ValueError):
                collector.runtime_identity(self.runtime, RUN, lambda pid: "98765")
            self.config, self.state = old_config, old_state
        self.write_runtime()

    def test_selected_metric_model_mismatch_nonfinite_and_duplicates_rejected(self):
        identity = {"model_path": MODEL, "served_name": MODEL}
        for raw in (b'vllm:generation_tokens_total{model_name="other"} 1\n',
                    b'vllm:generation_tokens_total NaN\n', b'vllm:generation_tokens_total 1e999\n',
                    b'vllm:generation_tokens_total 1\nvllm:generation_tokens_total 2\n',
                    b'vllm:generation_tokens_total{model_name="x",model_name="x"} 1\n'):
            with self.assertRaises(ValueError):
                collector.selected_metrics(raw, identity)

    def test_storage_limit_preserves_final_and_lastgood(self):
        payload = raw_metrics() + (b"# ignored comment padding\n" * 200)
        result = self.run_collect(lambda url, timeout: payload, storage_limit=620_000, max_seconds=100)
        self.assertEqual(result["reason"], "collection_stopped")
        self.assertGreater(result["samples"], 0)
        self.assertIn("storage budget", result["error"])
        self.assertTrue((self.output / "metrics-last-good.txt").exists())
        self.assertLess(sum(p.stat().st_size for p in self.output.iterdir()), 620_000)
        self.assertTrue((self.output / "collector-final.json").exists())

    def test_existing_output_and_invalid_deadline_rejected(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_collect()
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        with self.assertRaises(ValueError):
            collector.collect(self.runtime, RUN, self.root / "invalid", max_seconds=5001)
        with self.assertRaises(ValueError):
            collector.collect(self.runtime, RUN, self.root / "invalid", interval_seconds=0.1)

    def test_real_http_ignores_proxy_refuses_redirect_and_body_limit(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                seen.append(self.path)
                if len(seen) == 2:
                    self.send_response(302)
                    self.send_header("Location", "/must-not-follow")
                    self.end_headers()
                elif len(seen) == 3:
                    self.send_response(200)
                    self.send_header("Content-Length", str(collector.MAX_BODY + 1))
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw_metrics())))
                    self.end_headers()
                    self.wfile.write(raw_metrics())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/metrics"
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}):
            self.assertEqual(collector.metrics_request(url), raw_metrics())
        with self.assertRaisesRegex(ValueError, "redirect"):
            collector.metrics_request(url)
        with self.assertRaisesRegex(ValueError, "size limit"):
            collector.metrics_request(url)
        self.assertEqual(seen, ["/metrics"] * 3)
        with self.assertRaises(ValueError):
            collector.metrics_request("http://example.com/metrics")

    @unittest.skipUnless(sys.platform == "linux", "Linux /proc start ticks")
    def test_real_linux_proc_token(self):
        self.assertTrue(collector.process_token(os.getpid()).isdigit())
        self.assertIsNone(collector.process_token(999999999))


if __name__ == "__main__":
    unittest.main()
