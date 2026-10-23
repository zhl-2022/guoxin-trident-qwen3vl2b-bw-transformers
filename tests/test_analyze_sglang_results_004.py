"""CPU archive fixtures, never real performance/acceptance evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


analyzer = module("posthoc_sglang004", "analyze_sglang_results_004.py")
producer = module("unchanged_sglang004", "t02_sglang_bench.py")


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / "archive"
        self.source.mkdir()
        (self.source / "native").mkdir()
        self.output = self.base / "new-analysis"
        model = self.base / "model"
        model.mkdir()
        self.run = producer.build_plan(producer.parser().parse_args([
            "--run",
            "--model-path", str(model), "--served-model-name", str(model),
            "--output-dir", str(self.source), "--load", "4096:1", "--round", "2"]))
        # The current004 archive is Linux. Preserve remote producer paths even
        # when this read-only postprocessor is run on Windows.
        self.run["output_dir"] = "/archived/remote/t02-sglang"
        row = self.run["rows"][0]
        original_native = row["native_result"]
        row["native_result"] = self.run["output_dir"] + "/native/" + row["name"] + ".jsonl"
        row["command"][row["command"].index("--output-file")+1] = row["native_result"]
        row.update(status="invalid_native_result", result_error="Infinity", process_status="exited", exit_code=0,
                   child_reaped=True, client_wall_seconds=15.0, allocated_timeout_seconds=100.0)
        self.native = {"backend": "sglang-oai-chat", "dataset_name": "image", "request_rate": float("inf"),
                       "max_concurrency": 1, "random_input_len":4096,"random_output_len":1024,"random_range_ratio":1,
                       "duration":10.0,"completed":8,"input_lens":[4608]*8,"output_lens":[1024]*8,
                       "generated_texts":["CPU fixture only"]*8,"errors":[""]*8,"ttfts":[0.1]*8,"itls":[[0.01,0.02] for _ in range(8)],
                       "total_input_tokens":4608*8,"total_input_text_tokens":4096*8,"total_input_vision_tokens":512*8,
                       "total_output_tokens":1024*8,"total_output_tokens_retokenized":16,
                       "request_throughput":0.8,"input_throughput":4608*8/10,"output_throughput":1024*8/10,
                       "total_throughput":(4608+1024)*8/10,"concurrency":0.8,"accept_length":None,
                       **{key:1.0 for key in analyzer.LATENCY_FIELDS}}
        entries = []
        for name in analyzer.SOURCE_FILES:
            raw = ((ROOT / "evidence/session004/sglang-backend-004g-20261023/upstream-bench_serving.py").read_bytes()
                   if name == "bench_serving.py" else b"# CPU archive manifest fixture; not real benchmark evidence\n")
            target = self.source / "sources/upstream-sglang" / name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
            entries.append({"path":name,"bytes":len(raw),"sha256":analyzer.sha(raw)})
        raw_driver = (ROOT / "scripts/t02_sglang_bench.py").read_bytes()
        (self.source / "sources/t02_sglang_bench.py").write_bytes(raw_driver)
        self.run.update(status="incomplete_or_failed",client_started=True,filesystem_mutated=True,driver_sha256=analyzer.sha(raw_driver),
                        client_identity={"version":"0.5.12","upstream_verified":True,"sources":entries,
                                         "identity_process":{"process_status":"exited","exit_code":0,"child_reaped":True}})
        self.save()

    def save(self, raw=None):
        row = self.run["rows"][0]
        raw = (json.dumps(self.native)+"\n").encode() if raw is None else raw
        (self.source / "native" / (row["name"]+".jsonl")).write_bytes(raw)
        row["native_result_sha256"] = analyzer.sha(raw)
        (self.source/"run.json").write_bytes(analyzer.encode(self.run))

    def test_only_top_level_positive_infinity_token_is_normalized(self):
        value, changes = analyzer.parse_json(b'{"request_rate":Infinity,"duration":1}',True)
        self.assertEqual(value,{"request_rate":"inf","duration":1})
        self.assertEqual(changes[0]["original_token"],"Infinity")
        for raw in (b'{"request_rate":-Infinity}',b'{"request_rate":NaN}',b'{"request_rate":1e999}',
                    b'{"request_rate":Infinity,"duration":NaN}',b'{"request_rate":Infinity,"nested":{"request_rate":Infinity}}',
                    b'{"request_rate":Infinity,"array":[Infinity]}',b'{"request_rate":Infinity,"request_rate":1}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                analyzer.parse_json(raw,True)
        with self.assertRaises(ValueError):
            analyzer.parse_json(b'{"request_rate":Infinity}')

    def test_real_contract_fixture_recovers_without_changing_original_artifacts(self):
        before = {path.relative_to(self.source):path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        report = analyzer.analyze(self.source/"run.json",self.output)
        self.assertEqual(report["summary"]["eligible_client_reports"],1)
        self.assertFalse(report["summary"]["full_two_round_matrix_complete"])
        self.assertFalse(report["summary"]["official_performance_accepted"])
        row=report["rows"][0]
        self.assertEqual(row["original_producer_status"],"invalid_native_result")
        self.assertEqual(row["original_producer_result_error"],"Infinity")
        self.assertEqual(row["parsed"]["normalizations"][0]["normalized_value"],"inf")
        self.assertFalse(row["parsed"]["exact_output_tokens_independently_verified"])
        self.assertFalse(row["parsed"]["raw_usage_received_independently_verified"])
        self.assertFalse(row["parsed"]["stream_terminal_framing_independently_verified"])
        self.assertEqual(before,{path.relative_to(self.source):path.read_bytes() for path in self.source.rglob("*") if path.is_file()})
        manifest=json.loads((self.output/"analysis-manifest.json").read_bytes())
        for item in manifest["files"]:
            raw=(self.output/item["path"]).read_bytes()
            self.assertEqual(analyzer.sha(raw),item["sha256"])
            self.assertEqual(len(raw),item["bytes"])

    def test_timeout_interruption_nonzero_unreaped_are_never_counted(self):
        for index,changes in enumerate(({"process_status":"timeout","exit_code":124},{"process_status":"interrupted","exit_code":130},
                                        {"exit_code":1},{"child_reaped":False},{"exit_code":False})):
            with self.subTest(changes=changes):
                self.run["rows"][0].update(process_status="exited",exit_code=0,child_reaped=True)
                self.run["rows"][0].update(changes);self.save()
                result=analyzer.analyze(self.source/"run.json",self.base/f"new-{index}")
                self.assertEqual(result["summary"]["eligible_client_reports"],0)
                self.assertTrue(result["rows"][0]["parsed"]["native_structure_complete"])

    def test_fake_override_client_and_missing_identity_cannot_count(self):
        for index,(field,value) in enumerate((("client_type","explicit_override_unverified"),("mode","plan_only"),("client_started",False),("client_identity",{}),("client_identity",[]))):
            self.run.update(client_type="upstream_sglang",mode="run",client_started=True)
            self.run[field]=value;self.save()
            result=analyzer.analyze(self.source/"run.json",self.base/f"bad-provenance-{index}")
            self.assertEqual(result["summary"]["eligible_client_reports"],0)
            self.assertFalse(result["source_provenance"]["verified_for_reanalysis"])

    def test_source_sha_tamper_and_native_sha_mismatch_fail_closed(self):
        (self.source/"sources/upstream-sglang/bench_serving.py").write_bytes(b"# tampered\n")
        self.run["rows"][0]["native_result_sha256"]="0"*64
        (self.source/"run.json").write_bytes(analyzer.encode(self.run))
        report=analyzer.analyze(self.source/"run.json",self.output)
        self.assertEqual(report["summary"]["eligible_client_reports"],0)
        self.assertTrue(any("native SHA" in issue for issue in report["rows"][0]["analysis_issues"]))
        self.assertFalse(report["source_provenance"]["verified_for_reanalysis"])

    def test_array_count_completion_error_and_metric_consistency_are_strict(self):
        row=self.run["rows"][0]
        cases=[("completed",True),("completed",7),("input_lens",[4608]*7),("output_lens",[True]*8),
               ("output_lens",[1023]*8),("generated_texts",[""]*8),("errors",[False]*8),
               ("ttfts",[-1]*8),("itls",[0.1]*8),("total_output_tokens",0),("output_throughput",1.0),
               ("request_rate","inf"),("max_concurrency",8)]
        for key,value in cases:
            with self.subTest(key=key,value=value):
                result=analyzer.native_result((json.dumps({**self.native,key:value})+"\n").encode(),row)
                self.assertFalse(result["native_structure_complete"])

    def test_other_nonfinite_native_keeps_failure_and_raw_sha(self):
        self.native["server_info"]={"x":float("inf")};self.save()
        report=analyzer.analyze(self.source/"run.json",self.output)
        self.assertEqual(report["summary"]["eligible_client_reports"],0)
        row=report["rows"][0]
        self.assertEqual(row["native_file_sha256"],self.run["rows"][0]["native_result_sha256"])
        self.assertTrue(any("Non-finite" in issue for issue in row["analysis_issues"]))

    def test_duplicate_rows_and_argv_mutation_refused(self):
        row=self.run["rows"][0]
        row["command"] += ["--disable-stream"]
        self.save()
        report=analyzer.analyze(self.source/"run.json",self.output)
        self.assertEqual(report["summary"]["eligible_client_reports"],0)
        self.run["rows"].append(dict(row));self.save()
        second=self.base/"duplicate-out"
        with self.assertRaisesRegex(ValueError,"Duplicate run row"):
            analyzer.analyze(self.source/"run.json",second)
        self.assertFalse(second.exists())

    def test_all_native_files_are_hashed_including_unreferenced(self):
        raw=b'{"orphan":true}\n'
        (self.source/"native/orphan.jsonl").write_bytes(raw)
        report=analyzer.analyze(self.source/"run.json",self.output)
        self.assertEqual(len(report["native_file_inventory"]),2)
        self.assertEqual(report["unreferenced_native_files"],["native/orphan.jsonl"])
        self.assertEqual(report["native_file_inventory"][0]["sha256"],analyzer.sha(raw))

    def test_new_output_and_source_symlinks_are_refused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError,"NEW"):
            analyzer.analyze(self.source/"run.json",self.output)
        with self.assertRaisesRegex(ValueError,"inside"):
            analyzer.analyze(self.source/"run.json",self.source/"analysis")
        native=self.source/"native"/(self.run["rows"][0]["name"]+".jsonl")
        outside=self.base/"outside.jsonl";native.rename(outside)
        try:native.symlink_to(outside)
        except OSError as error:self.skipTest(str(error))
        with self.assertRaisesRegex(ValueError,"Symlink"):
            analyzer.analyze(self.source/"run.json",self.base/"not-created")

    def test_standalone_copied_alias_cli_uses_no_sibling_import(self):
        copied=self.base/"uploaded-alias.py";shutil.copyfile(ROOT/"scripts/analyze_sglang_results_004.py",copied)
        result=subprocess.run([sys.executable,"-B",str(copied),"--run",str(self.source/"run.json"),"--analysis-dir",str(self.output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)["summary"]["eligible_client_reports"],1)

    def sealed_export(self):
        export = self.base / "export"
        source = export / "project/evidence/session004/sealed-run"
        source.parent.mkdir(parents=True)
        shutil.copytree(self.source,source)
        run=json.loads((source/"run.json").read_bytes())
        run["output_dir"]="/remote/project/evidence/session004/sealed-run"
        row=run["rows"][0]
        row["native_result"]=run["output_dir"]+"/native/"+row["name"]+".jsonl"
        row["command"][row["command"].index("--output-file")+1]=row["native_result"]
        (source/"run.json").write_bytes(analyzer.encode(run))
        payloads=[];scripts=[]
        for file in source.rglob("*"):
            if not file.is_file():continue
            name=file.relative_to(source).as_posix();raw=file.read_bytes()
            item={"archive_path":file.relative_to(export).as_posix(),"source":run["output_dir"]+"/"+name,
                  "size":len(raw),"sha256":analyzer.sha(raw)}
            if file.suffix==".py":scripts.append(item);file.unlink()
            else:payloads.append(item)
        manifest={"schema":"hg-session004-evidence-export-v1","project":"/remote/project","script_hashes_unchanged":True,
                  "script_hashes_before":scripts,"script_hashes_after":scripts,"files":payloads,
                  "capture_semantics":"CPU exporter manifest fixture only","finished_utc":"2026-10-23T00:00:00+00:00"}
        path=export/"export-manifest.json";path.write_bytes(analyzer.encode(manifest))
        return path,source

    def test_explicit_sealed_export_hash_proof_does_not_reconstruct_missing_source_payload(self):
        manifest,source=self.sealed_export()
        report=analyzer.analyze(source/"run.json",self.output,manifest,analyzer.sha(manifest.read_bytes()))
        self.assertEqual(report["summary"]["eligible_client_reports"],1)
        proofs=report["source_provenance"]["artifacts"]
        self.assertEqual(len(proofs),6)
        self.assertTrue(all(item["source_payload_archived"] is False for item in proofs))
        self.assertFalse(any((source/"sources").rglob("*.py")))

    def test_seal_wrong_sha_changed_before_after_or_payload_cannot_pass(self):
        manifest,source=self.sealed_export()
        with self.assertRaisesRegex(ValueError,"SHA"):
            analyzer.analyze(source/"run.json",self.output,manifest,"0"*64)
        parsed=json.loads(manifest.read_bytes())
        parsed["script_hashes_after"][0]["sha256"]="0"*64
        manifest.write_bytes(analyzer.encode(parsed))
        with self.assertRaisesRegex(ValueError,"before/after"):
            analyzer.analyze(source/"run.json",self.output,manifest,analyzer.sha(manifest.read_bytes()))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
