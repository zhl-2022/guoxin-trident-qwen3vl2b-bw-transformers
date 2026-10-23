"""Real CPU mock clients exercise the bounded upstream-client wrapper, never GPU."""
import contextlib
import ast
import asyncio
import __future__
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sglang_bench_under_test", ROOT / "scripts/t02_sglang_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class SGLangBenchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        self.output = self.root / "new-result"
        self.fixture = self.root / "mock.py"
        self.fixture.write_text("import json,sys\nfrom pathlib import Path\n"
                                "a=sys.argv\nn=int(a[a.index('--num-prompts')+1])\n"
                                "v={'completed':n,'output_lens':[1024]*n,'errors':['']*n,"
                                "'generated_texts':['CPU fixture']*n,'duration':1,'output_throughput':1}\n"
                                "Path(a[a.index('--output-file')+1]).write_text(json.dumps(v)+'\\n')\n"
                                "print('original CPU client log')\n")

    def plan(self, *extra, mock=True):
        argv = ["--model-path", str(self.model), "--served-model-name", "mock-model", "--output-dir", str(self.output)]
        if mock:
            argv += ["--client-command-json", json.dumps([sys.executable, "-B", str(self.fixture)])]
        return bench.build_plan(bench.parser().parse_args(argv + list(extra)))

    def execute(self, plan):
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.execute(plan)

    def test_plan_has_six_loads_two_rounds_and_explicit_upstream_semantics(self):
        plan = self.plan(mock=False)
        self.assertEqual(plan["client_type"], "upstream_sglang")
        self.assertEqual(len(plan["rows"]), 12)
        self.assertEqual(plan["measured_requests_planned"], 608)
        self.assertEqual(plan["extra_warmup_requests_planned"], 12)
        self.assertEqual(sum(r["scored_round"] for r in plan["rows"]), 6)
        for row in plan["rows"]:
            argv = row["command"]
            self.assertIn("sglang.bench_serving", argv)
            self.assertEqual(argv[argv.index("--backend")+1], "sglang-oai-chat")
            self.assertEqual(argv[argv.index("--dataset-name")+1], "image")
            self.assertEqual(argv[argv.index("--image-resolution")+1], "1024x1024")
            self.assertEqual(argv[argv.index("--random-range-ratio")+1], "1")
            extra = json.loads(argv[argv.index("--extra-request-body")+1])
            self.assertTrue(extra["stream_options"]["include_usage"])
            self.assertEqual(extra["temperature"], 0)
            self.assertTrue(extra["ignore_eos"])
            self.assertIn("--output-details", argv)
        self.assertFalse(plan["exact_output_tokens_independently_verified"])
        self.assertEqual(plan["server_framework"], "vllm")
        self.assertFalse(plan["server_framework_verified_from_runtime"])
        self.assertEqual(plan["upstream_sender"], "async_request_openai_chat_completions")
        self.assertTrue(plan["api_endpoint"].endswith("/v1/chat/completions"))
        self.assertFalse(self.output.exists())

    def test_primary_upstream_image_whitelist_and_sender_alias_mapping_are_compatible(self):
        source = ROOT / "evidence/session004/sglang-backend-004g-20261023"
        serving = ast.parse((source / "upstream-bench_serving.py").read_text(encoding="utf-8"))
        mapping = next(node.value for node in serving.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "ASYNC_REQUEST_FUNCS" for target in node.targets))
        aliases = {ast.literal_eval(key): value.id for key, value in zip(mapping.keys, mapping.values)}
        self.assertEqual(aliases[bench.CLIENT_BACKEND], "async_request_openai_chat_completions")
        self.assertEqual(aliases[bench.CLIENT_BACKEND], aliases["vllm-chat"])
        image = ast.parse((source / "upstream-image.py").read_text(encoding="utf-8"))
        function = next(node for node in image.body if isinstance(node, ast.FunctionDef) and node.name == "create_mm_data_row")
        scope = {"DatasetRow": lambda **values: types.SimpleNamespace(**values)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "primary-upstream-image", "exec"), scope)
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "a-templated-prompt"
            def __call__(self, *args, **kwargs):
                return {"input_ids": types.SimpleNamespace(numel=lambda: 100 if "images" in kwargs else 20)}
        call = lambda backend: scope["create_mm_data_row"]("raw text", [object()], ["data:image/jpeg;base64,CPU"], 1024, Processor(), backend)
        with self.assertRaisesRegex(ValueError, "supports backends"):
            call("vllm-chat")
        row = call(bench.CLIENT_BACKEND)
        self.assertEqual(row.prompt, "raw text")
        self.assertEqual(row.image_data, ["data:image/jpeg;base64,CPU"])
        self.assertEqual(row.output_len, 1024)

    def test_actual_unmodified_upstream_sender_builds_generic_openai_chat_image_payload(self):
        path = ROOT / "evidence/session004/sglang-backend-004g-20261023/upstream-bench_serving.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "async_request_openai_chat_completions")
        # Execute the real sender function only. Imports/model/client initialization
        # are omitted; its HTTP session points to an in-memory OpenAI contract mock.
        captured = {}
        class AsyncContext:
            def __init__(self, value):
                self.value = value
            async def __aenter__(self):
                return self.value
            async def __aexit__(self, *_args):
                return None
        class Response:
            status = 200
            async def json(self):
                return {"choices": [{"message": {"content": "CPU OpenAI response"}}], "usage": {"completion_tokens": 1024}}
        class Session:
            def post(self, **kwargs):
                captured.update(kwargs)
                return AsyncContext(Response())
        scope = {"args": types.SimpleNamespace(disable_stream=True, disable_ignore_eos=False, print_requests=False),
                 "_create_bench_client_session": lambda: AsyncContext(Session()),
                 "get_request_headers": lambda: {}, "time": time, "json": json,
                 "RequestFuncOutput": types.SimpleNamespace(init_new=lambda request: types.SimpleNamespace(success=False))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), scope)
        request = types.SimpleNamespace(prompt="raw user text", api_url="http://127.0.0.1:30000/v1/chat/completions",
                                        image_data=["data:image/jpeg;base64,CPU"], model="actual-vllm-served-model",
                                        output_len=1024, extra_request_body={"temperature": 0, "ignore_eos": True,
                                        "stream_options": {"include_usage": True}}, lora_name=None, routing_key=None)
        result = asyncio.run(scope[function.name](request))
        self.assertTrue(result.success)
        self.assertEqual(captured["url"], request.api_url)
        self.assertEqual(captured["json"]["model"], "actual-vllm-served-model")
        self.assertEqual(captured["json"]["messages"], [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CPU"}},
            {"type": "text", "text": "raw user text"}]}])
        self.assertEqual(captured["json"]["max_completion_tokens"], 1024)
        self.assertNotIn("sampling_params", captured["json"])
        route_branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                          and "sglang-oai-chat" in ast.unparse(node.test) and "vllm-chat" in ast.unparse(node.test)
                          and any(isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "api_url" for t in child.targets)
                                  for child in node.body)]
        self.assertTrue(any("/v1/chat/completions" in ast.unparse(node.body) for node in route_branches))

    def test_client_alias_cannot_relabel_a_different_runtime_server_framework(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        run_id = "b"*32
        (runtime / "state.json").write_text(json.dumps({"run_id": run_id, "status": "ready", "url": "http://127.0.0.1:30000"}))
        (runtime / ("config-"+run_id+".json")).write_text(json.dumps({"framework": "sglang", "model_path": str(self.model),
            "served_name": "mock-model", "host": "127.0.0.1", "port": 30000}))
        plan = self.plan("--runtime-dir", str(runtime), "--expected-run-id", run_id)
        with self.assertRaisesRegex(ValueError, "framework"):
            bench.runtime_snapshot(plan, self.output, "before")
        self.assertFalse(self.output.exists())

    def test_round_subset_remains_independent_and_warmups_are_counted_separately(self):
        plan = self.plan("--load", "16384:64", "--round", "2", "--warmup-requests", "0")
        self.assertEqual(len(plan["rows"]), 1)
        self.assertEqual(plan["rows"][0]["round"], 2)
        self.assertEqual(plan["measured_requests_planned"], 128)
        self.assertEqual(plan["extra_warmup_requests_planned"], 0)
        self.assertFalse(plan["full_matrix"])

    def test_default_still_hides_devices_and_records_original_visibility_without_probing(self):
        incoming = {"CUDA_VISIBLE_DEVICES": "2,3", "HIP_VISIBLE_DEVICES": "0", "PATH": "retained"}
        before = dict(incoming)
        env = bench.client_environment(False, incoming)
        metadata = bench.device_discovery_policy(False, incoming)
        self.assertEqual(incoming, before)
        for name in bench.VISIBILITY_VARIABLES:
            self.assertEqual(env[name], "")
            self.assertTrue(metadata["effective_visibility"][name]["is_set"])
            self.assertEqual(metadata["effective_visibility"][name]["value"], "")
        self.assertFalse(metadata["caller_visibility"]["ROCR_VISIBLE_DEVICES"]["is_set"])
        self.assertIsNone(metadata["caller_visibility"]["ROCR_VISIBLE_DEVICES"]["value"])
        self.assertFalse(metadata["allowed"])
        self.assertFalse(metadata["runtime_device_availability_verified"])

    def test_opt_in_preserves_set_unset_and_empty_variables_without_expanding_scope(self):
        for incoming in ({}, {"HIP_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "1,3", "ROCR_VISIBLE_DEVICES": "GPU-a"}):
            with self.subTest(incoming=incoming):
                original = dict(incoming)
                env = bench.client_environment(True, incoming)
                metadata = bench.device_discovery_policy(True, incoming)
                for name in bench.VISIBILITY_VARIABLES:
                    self.assertEqual(name in env, name in incoming)
                    self.assertEqual(env.get(name), incoming.get(name))
                self.assertEqual(incoming, original)
                self.assertTrue(metadata["allowed"])
                self.assertTrue(metadata["caller_visibility_preserved_exactly"])
                self.assertFalse(metadata["device_scope_expanded_by_driver"])
                self.assertFalse(metadata["driver_device_enumeration_performed"])

    def test_opt_in_plan_and_real_cpu_subprocess_receive_the_effective_visibility(self):
        self.fixture.write_text("import os,json\nprint(json.dumps({k:os.environ.get(k) for k in " + repr(bench.VISIBILITY_VARIABLES) + "}))\n" + self.fixture.read_text())
        incoming = {"CUDA_VISIBLE_DEVICES": "7", "HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""}
        for allow in (False, True):
            with self.subTest(allow=allow), patch.dict(os.environ, incoming):
                extra = ["--load", "4096:1", "--round", "2", "--output-dir", str(self.root / ("allowed" if allow else "hidden"))]
                if allow:
                    extra.append("--allow-client-device-discovery")
                plan = self.plan(*extra)
                self.assertEqual(plan["allow_client_device_discovery"], allow)
                self.assertEqual(plan["client_device_discovery"]["caller_visibility"]["HIP_VISIBLE_DEVICES"]["value"], "0")
                self.assertEqual(self.execute(plan), 0)
                observed = json.loads(Path(plan["rows"][0]["log"]).read_text().splitlines()[0])
                self.assertEqual(observed, incoming if allow else {k: "" for k in bench.VISIBILITY_VARIABLES})
                report = json.loads((Path(plan["output_dir"]) / "run.json").read_text())
                self.assertEqual(report["client_identity"]["client_device_discovery"], report["client_device_discovery"])

    @unittest.skipUnless(sys.platform == "linux", "Linux venv executable symlinks")
    def test_client_python_symlink_path_is_preserved_for_venv_discovery(self):
        import venv
        target = self.root / "isolated-client"
        venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(target)
        python = target / "bin/python"
        self.assertTrue(python.is_symlink())
        plan = self.plan("--client-python", str(python), mock=False)
        self.assertEqual(plan["client_python"], str(python))
        self.assertEqual(plan["rows"][0]["command"][0], str(python))
        result = subprocess.run([plan["client_python"], "-B", "-c", "import sys;print(sys.prefix)"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(target))

    def test_original_logs_sources_runtime_and_native_bytes_are_archived_without_official_claim(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        run_id = "a"*32
        (runtime / "state.json").write_text(json.dumps({"run_id": run_id, "status": "ready", "url": "http://127.0.0.1:30000"}))
        (runtime / "server.log").write_bytes(b"original runtime bytes\n")
        (runtime / ("config-"+run_id+".json")).write_text(json.dumps({"model_path": str(self.model),
            "served_name": "mock-model", "host": "127.0.0.1", "port": 30000}))
        plan = self.plan("--load", "4096:1", "--runtime-dir", str(runtime), "--expected-run-id", run_id)
        self.assertEqual(self.execute(plan), 0)
        report = json.loads((self.output / "run.json").read_text())
        self.assertEqual(report["status"], "complete_client_reports")
        self.assertFalse(report["official_acceptance"])
        self.assertFalse(report["exact_output_tokens_independently_verified"])
        self.assertFalse(report["client_identity"]["upstream_verified"])
        self.assertEqual(len((self.output / "round1.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((self.output / "round2.jsonl").read_text().splitlines()), 1)
        self.assertEqual((self.output / "sources/t02_sglang_bench.py").read_bytes(), Path(bench.__file__).read_bytes())
        for phase in ("before", "after"):
            self.assertEqual((self.output / "runtime" / phase / "server.log").read_bytes(), b"original runtime bytes\n")
        for entry in json.loads((self.output / "archive-manifest.json").read_text())["files"]:
            raw = (self.output / entry["path"]).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
        for row in report["rows"]:
            self.assertIn(b"original CPU client log", Path(row["log"]).read_bytes())
            self.assertFalse(row["result"]["exact_output_tokens_independently_verified"])

    def test_existing_output_and_invalid_model_are_rejected_without_overwrite(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_bytes(b"existing evidence")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.execute(self.plan())
        self.assertEqual(marker.read_bytes(), b"existing evidence")

    def test_unknown_served_alias_cannot_trigger_processor_download(self):
        with self.assertRaisesRegex(ValueError, "served_model_name"):
            self.execute(self.plan(mock=False))
        self.assertFalse(self.output.exists())

    def test_wrong_runtime_identity_fails_before_any_client_starts(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        (runtime / "state.json").write_text('{"run_id":"other"}')
        plan = self.plan("--load", "4096:1", "--runtime-dir", str(runtime), "--expected-run-id", "a"*32)
        with patch.object(bench.subprocess, "Popen") as popen:
            self.assertEqual(self.execute(plan), 1)
        popen.assert_not_called()
        self.assertFalse(plan["client_started"])

    def test_bad_native_details_do_not_become_complete_or_exact_output(self):
        result = self.root / "native.jsonl"
        for change in ({"errors": ["failure"]}, {"generated_texts": [""]}, {"output_lens": [1023]}, {"completed": 0}):
            data = {"completed": 1, "output_lens": [1024], "generated_texts": ["body"], "errors": [""]}
            data.update(change)
            result.write_text(json.dumps(data) + "\n")
            with self.subTest(change=change):
                checked = bench.inspect_result(result, 1)
                self.assertFalse(checked["client_report_complete"])
                self.assertFalse(checked["exact_output_tokens_independently_verified"])

    def test_total_deadline_is_bounded_and_later_round_not_attempted(self):
        self.fixture.write_text("import time\nprint('partial original bytes',flush=True)\ntime.sleep(60)\n")
        plan = self.plan("--load", "4096:1", "--timeout-seconds", "5", "--total-timeout-seconds", "0.8")
        self.assertEqual(self.execute(plan), 1)
        self.assertEqual(plan["rows"][0]["status"], "timeout")
        self.assertEqual(plan["rows"][1]["status"], "not_attempted_total_timeout")
        self.assertTrue(plan["rows"][0]["child_reaped"])
        self.assertLess(plan["total_wall_seconds"], 8)

    @unittest.skipUnless(sys.platform == "linux", "Linux child process groups")
    def test_timeout_removes_descendant_process_group(self):
        pidfile = self.root / "child.pid"
        script = self.root / "child-tree.py"
        script.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
                          "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                          "Path(" + repr(str(pidfile)) + ").write_text(str(p.pid))\ntime.sleep(60)\n")
        outcome = bench.bounded([sys.executable, str(script)], self.root / "tree.log", 0.4, os.environ.copy())
        self.assertEqual(outcome["process_status"], "timeout")
        self.assertTrue(outcome["child_reaped"])
        pid = int(pidfile.read_text())
        for _ in range(20):
            path = Path(f"/proc/{pid}/stat")
            if not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return
            time.sleep(0.05)
        self.fail("timed-out CPU client descendant remains running")


if __name__ == "__main__":
    unittest.main()
