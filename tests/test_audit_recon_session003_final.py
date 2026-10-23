"""CPU-only synthetic archive checks; mocks are never official evaluation evidence."""
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load("audit_recon_session003_final")
driver = load("t02_development_bench")


class ReconFinalAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()

    def write_snapshot(self, folder, run):
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        for round_ in (1, 2):
            (folder / f"round{round_}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in run["rows"] if row["round"] == round_), encoding="utf-8")
        (folder / "summary.md").write_text("CPU mock development run only\n", encoding="utf-8")

    def create_run(self, name="matrix", subset=False):
        folder = self.root / name
        argv = ["--run", "--model-path", str(self.model), "--served-model-name", audit.MODEL,
                "--output-dir", str(folder), "--total-timeout-seconds", "4000"]
        if subset:
            argv += ["--load", "16384:64"]
            if subset != "both":
                argv += ["--round", "2"]
        run = driver.build_plan(driver.parser().parse_args(argv))
        folder.mkdir()
        (folder / "native").mkdir()
        now = datetime(2026, 10, 19, 18, tzinfo=timezone.utc)
        for index, row in enumerate(run["rows"]):
            count = row["expected_requests"]
            tokens = count * 1024
            data = {"completed": count, "failed": 0, "duration": 1.0,
                    "request_throughput": float(count), "output_throughput": float(tokens),
                    "total_token_throughput": float(tokens + count * row["target_text_input_tokens"]),
                    "total_input_tokens": count * row["target_text_input_tokens"], "total_output_tokens": tokens,
                    "output_lens": [1024] * count, "generated_texts": ["CPU mock output"] * count, "errors": [""] * count}
            for field in ("mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2el_ms", "p99_e2el_ms"):
                data[field] = 1.0
            native = Path(row["native_result"])
            native.write_text(json.dumps(data), encoding="utf-8")
            log = Path(row["log"])
            log.write_text("CPU mock client log only\n", encoding="utf-8")
            row.update(status="complete", exit_code=0, client_wall_seconds=1.5,
                       started_utc=(now + timedelta(seconds=index * 2)).isoformat(),
                       ended_utc=(now + timedelta(seconds=index * 2 + 1.5)).isoformat(),
                       metrics={key: data[key] for key in driver.METRICS}, output_integrity={"verified": True},
                       native_result_sha256=driver.sha(native), log_sha256=driver.sha(log))
        run.update(status="complete", started_utc=now.isoformat(), ended_utc=(now + timedelta(minutes=1)).isoformat(),
                   driver_sha256=driver.sha(SCRIPTS / "t02_development_bench.py"), native_vllm_version="CPU_mock",
                   native_source_sha256={"serve": "mock", "datasets": "mock"})
        self.write_snapshot(folder, run)
        return folder, run

    def test_full_matrix_counts_output_and_read_only_default(self):
        folder, _ = self.create_run()
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = audit.audit(folder)
        self.assertTrue(result["matrix_complete_verified"])
        self.assertEqual((result["verified_cells"], result["verified_requests"], result["verified_output_tokens"]), (12, 608, 622592))
        self.assertEqual(sum(len(row["output_integrity"]["requests"]) for row in result["rows"]), 608)
        self.assertFalse(result["official_performance"])
        self.assertTrue(result["source_files_stable_during_audit"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_truncated_empty_or_error_native_output_is_never_complete(self):
        for field, value in (("output_lens", 1023), ("generated_texts", ""), ("errors", "incomplete")):
            with self.subTest(field=field):
                folder, run = self.create_run(name=field)
                row = run["rows"][0]
                native = Path(row["native_result"])
                data = json.loads(native.read_text())
                data[field][0] = value
                native.write_text(json.dumps(data), encoding="utf-8")
                row["native_result_sha256"] = driver.sha(native)
                self.write_snapshot(folder, run)
                result = audit.audit(folder)
                self.assertFalse(result["matrix_complete_verified"])
                self.assertEqual(result["rows"][0]["requests_with_full_output"], 7)
                self.assertEqual(result["verified_cells"], 11)

    def test_native_or_log_drift_and_missing_evidence_are_rejected(self):
        for artifact in ("native_result", "log", "missing_native"):
            with self.subTest(artifact=artifact):
                folder, run = self.create_run(name=artifact)
                path = Path(run["rows"][0]["native_result" if artifact == "missing_native" else artifact])
                if artifact == "missing_native":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes() + b"\n")
                result = audit.audit(folder)
                self.assertFalse(result["matrix_complete_verified"])
                if artifact == "missing_native":
                    self.assertIsNone(result["rows"][0]["completed_requests_reported"])
                    self.assertTrue(any(item["path"] == run["rows"][0]["log"] for item in result["source_files"]))

    def test_separate_retry_covers_only_missing_cell_without_rewriting_original(self):
        folder, run = self.create_run()
        run.update(status="incomplete_or_failed")
        run["rows"][-1].update(status="timeout", exit_code=124)
        self.write_snapshot(folder, run)
        original = (folder / "run.json").read_bytes()
        retry, _ = self.create_run(name="retry", subset=True)
        result = audit.audit(folder, retries=[retry])
        self.assertFalse(result["matrix_complete_verified"])
        self.assertTrue(result["combined_development_coverage_complete"])
        self.assertEqual(result["verified_cells"], 11)
        self.assertEqual(result["retries"][0]["expected_requests"], 128)
        self.assertFalse(result["retries"][0]["matrix_complete_verified"])
        self.assertTrue(result["combined_coverage"][-1]["source_is_retry"])
        self.assertEqual((folder / "run.json").read_bytes(), original)

    def test_retry_identity_mismatch_cannot_fill_coverage(self):
        folder, run = self.create_run()
        run["status"] = "incomplete_or_failed"
        run["rows"][-1].update(status="timeout", exit_code=124)
        self.write_snapshot(folder, run)
        retry, retry_run = self.create_run(name="retry", subset=True)
        retry_run["native_vllm_version"] = "different_native_build"
        self.write_snapshot(retry, retry_run)
        result = audit.audit(folder, retries=[retry])
        self.assertFalse(result["combined_development_coverage_complete"])

    def test_incomplete_two_cell_retry_keeps_its_completed_first_cell(self):
        folder, run = self.create_run()
        run["status"] = "incomplete_or_failed"
        run["rows"][-2].update(status="timeout", exit_code=124)
        run["rows"][-1].update(status="not_attempted_total_timeout", exit_code=124)
        self.write_snapshot(folder, run)
        retry, retry_run = self.create_run(name="retry", subset="both")
        retry_run["status"] = "incomplete_or_failed"
        retry_run["rows"][-1].update(status="timeout", exit_code=124)
        self.write_snapshot(retry, retry_run)
        result = audit.audit(folder, retries=[retry])
        self.assertFalse(result["matrix_complete_verified"])
        self.assertFalse(result["combined_development_coverage_complete"])
        self.assertEqual(sum(row["verified"] for row in result["combined_coverage"]), 11)
        self.assertTrue(result["combined_coverage"][-2]["source_is_retry"])
        self.assertFalse(result["combined_coverage"][-1]["verified"])
        self.assertEqual(result["retries"][0]["verified_requests"], 128)

    def test_duplicate_cell_and_round_summary_drift_are_not_coverage(self):
        folder, run = self.create_run()
        run["rows"][1] = run["rows"][0]
        self.write_snapshot(folder, run)
        result = audit.audit(folder)
        self.assertFalse(result["matrix_complete_verified"])
        self.assertFalse(result["combined_development_coverage_complete"])
        self.assertTrue(any("duplicate" in value for value in result["issues"]))

    def test_new_output_only_and_runtime_profile_is_observation_not_live_check(self):
        folder, _ = self.create_run()
        runtime = self.root / "runtime"
        runtime.mkdir()
        (runtime / "server.log").write_text("Available KV cache memory: 44.93 GiB\nGPU KV cache size: 327,168 tokens\n", encoding="utf-8")
        (runtime / "state.json").write_text('{"status":"stopped","run_id":"CPU_mock"}', encoding="utf-8")
        result = audit.audit(folder, runtime)
        self.assertTrue(result["runtime"]["recorded_stopped_state_present"])
        self.assertFalse(result["runtime"]["live_process_or_port_checked"])
        self.assertEqual(result["runtime"]["profiles"][1]["kv_tokens"], 327168)
        output = self.root / "report.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(audit.main(["--matrix-dir", str(folder), "--output", str(output)]), 0)
            before = output.read_bytes()
            self.assertEqual(audit.main(["--matrix-dir", str(folder), "--output", str(output)]), 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
