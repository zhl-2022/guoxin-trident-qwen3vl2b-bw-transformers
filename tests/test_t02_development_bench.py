"""CPU mock coverage for the explicit random-mm development benchmark driver."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/t02_development_bench.py"
SPEC = importlib.util.spec_from_file_location("t02_development_bench", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class T02DevelopmentBenchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.output = self.root / "results"
        self.mock = self.root / "mock_vllm.py"
        self.mock.write_text("import sys,json,time\nfrom pathlib import Path\n"
                             "a=sys.argv[1:]\n"
                             "assert a[:2]==['bench','serve']\n"
                             "def get(k): return a[a.index(k)+1]\n"
                             "name=get('--result-filename')\n"
                             "n=int(get('--num-prompts'))\n"
                             "p=Path(get('--result-dir'))/name\n"
                             "p.write_text(json.dumps({'completed':n,'failed':0,'duration':0.01,'request_throughput':100,'output_throughput':1000,"
                             "'output_lens':[1024]*n,'generated_texts':['mock output']*n,'errors':['']*n,'total_output_tokens':1024*n}))\n"
                             "print('native mock',name)\n", encoding="utf-8")
        self.base = ["--model-path", str(self.model), "--served-model-name", "test-recon",
                     "--output-dir", str(self.output), "--vllm-command-json", json.dumps([sys.executable, str(self.mock)])]

    def run_driver(self, *args):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return bench.main([*self.base, *args])

    def test_default_plan_has_twelve_commands_and_exact_608_request_count_without_writes(self):
        with patch.object(bench.subprocess, "Popen") as popen:
            self.assertEqual(self.run_driver(), 0)
        popen.assert_not_called()
        plan = json.loads(self.stdout.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(len(plan["rows"]), 12)
        self.assertEqual(plan["requests_planned"], 608)
        self.assertTrue(plan["full_matrix"])
        self.assertFalse(plan["official_acceptance"])
        self.assertFalse(plan["client_started"])
        self.assertFalse(plan["filesystem_mutated"])
        self.assertEqual([row["round"] for row in plan["rows"][:4]], [1, 2, 1, 2])
        for row in plan["rows"]:
            args = row["command"]
            self.assertEqual(args[args.index("--backend") + 1], "openai-chat")
            self.assertEqual(args[args.index("--random-range-ratio") + 1], "0")
            self.assertNotIn("--enable-multimodal-chat", args)
            self.assertEqual(json.loads(args[args.index("--random-mm-limit-mm-per-prompt") + 1]), {"image": 1, "video": 0})
            self.assertEqual(args[args.index("--random-mm-bucket-config") + 1], "{(1024, 1024, 1): 1.0}")

    def test_run_preserves_twelve_native_results_logs_and_two_round_summaries(self):
        self.assertEqual(self.run_driver("--run"), 0, self.stderr.getvalue())
        report = json.loads((self.output / "run.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["official_acceptance"])
        self.assertEqual(len(list((self.output / "native").glob("*.json"))), 12)
        self.assertEqual(len(list(self.output.glob("*.log"))), 12)
        self.assertEqual(len((self.output / "round1.jsonl").read_text().splitlines()), 6)
        self.assertEqual(len((self.output / "round2.jsonl").read_text().splitlines()), 6)
        self.assertTrue((self.output / "summary.md").is_file())
        self.assertTrue(all(row["native_result_sha256"] and row["log_sha256"] and row["client_wall_seconds"] >= 0 for row in report["rows"]))

    def test_non_loopback_and_hostname_hosts_are_refused_without_writes(self):
        for host in ("0.0.0.0", "8.8.8.8", "example.com", "localhost", "127.0.0.1.evil", "http://127.0.0.1"):
            with self.subTest(host=host):
                self.assertEqual(self.run_driver("--host", host), 2)
                self.assertIn("loopback", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_existing_output_directory_is_never_overwritten(self):
        self.output.mkdir()
        original = self.output / "keep.txt"
        original.write_bytes(b"unchanged")
        with patch.object(bench.subprocess, "Popen") as popen:
            self.assertEqual(self.run_driver("--run"), 2)
        popen.assert_not_called()
        self.assertEqual(original.read_bytes(), b"unchanged")
        self.assertEqual(list(self.output.iterdir()), [original])

    def test_failed_commands_are_retained_and_do_not_drop_remaining_loads(self):
        self.mock.write_text("import sys\nprint('mock failure',flush=True)\nraise SystemExit(9)\n")
        self.assertEqual(self.run_driver("--run"), 1)
        report = json.loads((self.output / "run.json").read_text())
        self.assertEqual(len(report["rows"]), 12)
        self.assertTrue(all(row["status"] == "client_failed" and row["exit_code"] == 9 for row in report["rows"]))
        self.assertEqual(len(list(self.output.glob("*.log"))), 12)
        self.assertTrue(all("mock failure" in path.read_text() for path in self.output.glob("*.log")))

    def test_partial_request_success_is_not_complete(self):
        self.mock.write_text(self.mock.read_text().replace("'completed':n,'failed':0", "'completed':n-1,'failed':1"))
        self.assertEqual(self.run_driver("--run", "--load", "4096:1"), 1)
        report = json.loads((self.output / "run.json").read_text())
        self.assertTrue(all(row["status"] == "incomplete_or_failed" for row in report["rows"]))
        self.assertEqual(len(report["rows"]), 2)
        self.assertFalse(report["full_matrix"])

    def test_timeout_retains_attempt_and_deadline_keeps_unattempted_rows(self):
        self.mock.write_text("import time\nprint('slow mock',flush=True)\ntime.sleep(60)\n")
        self.assertEqual(self.run_driver("--run", "--timeout-seconds", "0.1", "--total-timeout-seconds", "0.25"), 1)
        report = json.loads((self.output / "run.json").read_text())
        self.assertEqual(len(report["rows"]), 12)
        self.assertTrue(any(row["status"] == "timeout" for row in report["rows"]))
        self.assertTrue(any(row["status"] == "not_attempted_total_timeout" for row in report["rows"]))

    def test_calibration_subset_is_explicit_and_image_size_is_developer_choice(self):
        self.assertEqual(self.run_driver("--load", "4096:1", "--round", "1", "--image-size", "512"), 0)
        plan = json.loads(self.stdout.getvalue())
        self.assertEqual(len(plan["rows"]), 1)
        self.assertEqual(plan["requests_planned"], 8)
        self.assertFalse(plan["full_matrix"])
        self.assertEqual(plan["image_size"], [512, 512])

    def altered_result(self, replacement):
        source = self.mock.read_text()
        self.mock.write_text(source.replace("p.write_text(json.dumps(", "data=")
                            .replace("1024*n}))", "1024*n}\n" + replacement + "\np.write_text(json.dumps(data))"))

    def test_empty_stream_is_rejected_even_if_native_success_counts_match(self):
        self.altered_result("data['generated_texts'][0]=''\ndata['output_lens'][0]=0\ndata['total_output_tokens']-=1024")
        self.assertEqual(self.run_driver("--run", "--load", "4096:1", "--round", "1"), 1)
        row = json.loads((self.output / "run.json").read_text())["rows"][0]
        self.assertEqual(row["status"], "incomplete_output_integrity")
        self.assertEqual(row["output_integrity"]["requests"][0]["output_tokens"], 0)
        self.assertEqual(row["output_integrity"]["requests"][0]["generated_text_characters"], 0)

    def test_truncated_stream_is_rejected_with_actual_lengths_preserved(self):
        self.altered_result("data['output_lens'][3]=31\ndata['total_output_tokens']-=993")
        self.assertEqual(self.run_driver("--run", "--load", "4096:1", "--round", "1"), 1)
        row = json.loads((self.output / "run.json").read_text())["rows"][0]
        self.assertEqual(row["status"], "incomplete_output_integrity")
        self.assertEqual(row["output_integrity"]["requests"][3]["output_tokens"], 31)

    def test_missing_details_cannot_be_verified(self):
        self.altered_result("del data['errors']")
        self.assertEqual(self.run_driver("--run", "--load", "4096:1", "--round", "1"), 1)
        row = json.loads((self.output / "run.json").read_text())["rows"][0]
        self.assertEqual(row["status"], "unverified_output_integrity")
        self.assertEqual(row["output_integrity"]["missing_fields"], ["errors"])

    def test_wrong_list_count_and_total_are_rejected(self):
        data = {"output_lens": [1024] * 7, "generated_texts": ["text"] * 8, "errors": [""] * 8, "total_output_tokens": 8192}
        result = bench.output_integrity(data, 8, 1024)
        self.assertFalse(result["verified"])
        self.assertTrue(any(issue["field"] == "output_lens" and "actual_entries" in issue for issue in result["issues"]))
        self.assertTrue(any(issue["field"] == "total_output_tokens" for issue in result["issues"]))

    def test_request_errors_are_preserved_and_not_counted_complete(self):
        self.altered_result("data['errors'][2]='SSE upstream error'")
        self.assertEqual(self.run_driver("--run", "--load", "4096:1", "--round", "1"), 1)
        row = json.loads((self.output / "run.json").read_text())["rows"][0]
        self.assertEqual(row["status"], "incomplete_output_integrity")
        self.assertEqual(row["output_integrity"]["requests"][2]["error"], "SSE upstream error")
        original = json.loads(Path(row["native_result"]).read_text())
        self.assertEqual(original["errors"][2], "SSE upstream error")


if __name__ == "__main__":
    unittest.main()
