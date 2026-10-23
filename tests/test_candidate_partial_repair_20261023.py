import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repair_candidate_partial_20261023 as repair

POOL = ROOT / "scripts/candidate_cpu_wheels_20261023.json"


class RepairTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory) / "candidates"
        target = base / "envs/forcedaligner-001"
        target.mkdir(parents=True)
        (target / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        pool = repair.read(POOL)["wheels"]
        wheel = copy.deepcopy(next(x for x in pool if x["name"] == "packaging"))
        cpu = {**wheel, "mode": "install_cpu_wheel"}
        native = {"name":"torch","version":"2.10.0","root":"/native","requires_dist":[],"mode":"reuse_native"}
        record = {"run_id":"original","status":"preparing","target":str(target),
                  "native_before":{"torch":{"version":"2.10.0","root":"/native"}},
                  "plan":{"task":"forcedaligner","selected":[cpu,native],"protected_native_expectations":{"torch":"2.10.0"},
                          "requirements":["packaging==25.0","torch>=2.6"]}}
        path = target / "candidate-environment.json"
        path.write_text(json.dumps(record),encoding="utf-8")
        cache = base / "cpu-wheels"
        cache.mkdir()
        shutil.copyfile(ROOT / "tasks/qwen3-forcedaligner-bw-transformers/validation-deps/packaging-25.0-py3-none-any.whl", cache / wheel["filename"])
        actual = {"packaging":{"name":"packaging","version":"24.0","root":"/native","requires_dist":[],"metadata_sha256":"b"*64},
                  "torch":{**native,"metadata_sha256":"a"*64}}
        actual["torch"].pop("mode")
        protected = {name:None for name in repair.PROTECTED}
        protected["torch"] = actual["torch"]
        return base,target,path,record,wheel,actual,protected

    def test_only_unsatisfied_cpu_selected_and_satisfied_candidate_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            base,target,path,record,wheel,actual,protected=self.fixture(d)
            pending,reused=repair.choose_repairs(record["plan"],actual,target,{("packaging","25.0"):wheel})
            self.assertEqual([x["name"] for x in pending],["packaging"])
            actual["packaging"].update(version="25.0",root=str(target/"lib/python3.10/site-packages"))
            pending,reused=repair.choose_repairs(record["plan"],actual,target,{("packaging","25.0"):wheel})
            self.assertEqual(pending,[])
            self.assertEqual(len(reused),2)

    def test_original_record_sha_and_approved_pool_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            base,target,path,record,wheel,actual,protected=self.fixture(d)
            with mock.patch.object(repair,"BASE",base):
                result=repair.load_identity(path,repair.sha(path),POOL,repair.APPROVED_POOL_SHA256)
                self.assertEqual(result[1],target)
                with self.assertRaises(ValueError):repair.load_identity(path,"0"*64,POOL,repair.APPROVED_POOL_SHA256)
                with self.assertRaises(ValueError):repair.load_identity(path,repair.sha(path),POOL,"0"*64)

    def test_native_protection_refuses_version_or_path_change(self):
        with tempfile.TemporaryDirectory() as d:
            base,target,path,record,wheel,actual,protected=self.fixture(d)
            repair.require_native(record,protected)
            for field,value in (("root","/elsewhere"),("version","2.9.0")):
                other=copy.deepcopy(protected);other["torch"][field]=value
                with self.assertRaises(ValueError):repair.require_native(record,other)

    def test_uncertain_venv_old_version_is_not_uninstalled(self):
        with tempfile.TemporaryDirectory() as d:
            base,target,path,record,wheel,actual,protected=self.fixture(d)
            actual["packaging"]["root"]=str(target/"lib/python3.10/site-packages")
            with self.assertRaisesRegex(ValueError,"uninstall ownership"):
                repair.choose_repairs(record["plan"],actual,target,{("packaging","25.0"):wheel})

    def execute_fixture(self, directory, *, timeout=False, broken_wheel=False):
        base,target,path,record,wheel,actual,protected=self.fixture(directory)
        if broken_wheel:(base/"cpu-wheels"/wheel["filename"]).write_bytes(b"corrupt")
        old=path.read_bytes();commands=[]
        def capture(command,stdout,stderr,seconds):
            commands.append(command)
            Path(stderr).write_text("")
            if "metadata" in command:
                Path(stdout).write_text(json.dumps({"selected":actual,"protected":protected}))
            elif "pip" in command:
                Path(stdout).write_text("simulated approved cached wheel install")
                if timeout:raise subprocess.TimeoutExpired(command,seconds)
            elif "imports" in command:
                Path(stdout).write_text(json.dumps({"dependency_imports_passed":True,"gpu_initialized_after_imports":True,
                                                  "no_gpu_initialization_observed":False,"device_discovery_allowed":True}))
            return 0
        args=["--record",str(path),"--record-sha256",repair.sha(path),"--pool",str(POOL),"--apply","--allow-device-discovery"]
        with mock.patch.object(repair,"BASE",base),mock.patch.object(sys,"platform","linux"), \
             mock.patch.object(sys,"version_info",(3,10,0)),mock.patch.object(repair.platform,"machine",return_value="x86_64"), \
             mock.patch.object(repair,"protected_snapshot",return_value=protected),mock.patch.object(repair,"run_capture",side_effect=capture), \
             mock.patch("builtins.print"):
            code=repair.main(args)
        result=repair.read(next((target/"partial-repair-evidence").glob("*/result.json")))
        self.assertEqual(path.read_bytes(),old)
        self.assertTrue(result["original_record_unchanged"])
        self.assertTrue(result["protected_native_metadata_unchanged"])
        return code,result,commands,target

    def test_apply_only_new_venv_pip_no_download_and_discovery_is_honest(self):
        with tempfile.TemporaryDirectory() as d:
            code,result,commands,target=self.execute_fixture(d)
            self.assertEqual(code,0)
            pip=next(x for x in commands if "pip" in x)
            self.assertEqual(pip[0],str(target/"bin/python"))
            for flag in ("--no-deps","--no-index","--ignore-installed","--require-hashes","--only-binary=:all:"):
                self.assertIn(flag,pip)
            self.assertEqual(result["install_names"],["packaging"])
            self.assertEqual(result["status"],"dependencies_imports_verified_device_discovery_observed")
            self.assertFalse(result["validation"]["no_gpu_initialization_observed"])
            self.assertFalse(result["downloads_performed"])

    def test_timeout_retains_failure_attempt_and_original_record(self):
        with tempfile.TemporaryDirectory() as d:
            code,result,commands,target=self.execute_fixture(d,timeout=True)
            self.assertEqual(code,1)
            self.assertEqual(result["error"]["type"],"TimeoutExpired")
            self.assertEqual(result["status"],"failed")
            self.assertFalse(any("imports" in x for x in commands))

    def test_bad_cached_wheel_refused_before_pip(self):
        with tempfile.TemporaryDirectory() as d:
            code,result,commands,target=self.execute_fixture(d,broken_wheel=True)
            self.assertEqual(code,1)
            self.assertFalse(any("pip" in x for x in commands))

    def test_import_discovery_policy_records_true_gpu_state(self):
        torch=mock.Mock();torch.cuda.is_initialized.side_effect=[False,True]
        fake_transformers=mock.Mock(__version__="5.5.0")
        modules={"torch":torch,"transformers":fake_transformers}
        with mock.patch.dict(sys.modules,modules):
            result=repair.check_imports("mimo9b",True)
        self.assertTrue(result["gpu_initialized_after_imports"])
        self.assertFalse(result["no_gpu_initialization_observed"])
        self.assertFalse(result["cpu_only_hardware_access_proven"])
        torch.cuda.is_initialized.side_effect=[False,True]
        with mock.patch.dict(sys.modules,modules),self.assertRaisesRegex(ValueError,"initialized GPU"):
            repair.check_imports("mimo9b",False)

    def test_arbitrary_uploaded_alias_has_no_sibling_import_dependency(self):
        with tempfile.TemporaryDirectory() as d:
            alias=Path(d)/"upload_alias.py"
            shutil.copyfile(ROOT/"scripts/repair_candidate_partial_20261023.py",alias)
            result=subprocess.run([sys.executable,"-B",str(alias),"--help"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_evidence_parent_symlink_refused_before_outside_write(self):
        with tempfile.TemporaryDirectory() as d:
            base,target,path,record,wheel,actual,protected=self.fixture(d)
            outside=Path(d)/"outside";outside.mkdir()
            try:(target/"partial-repair-evidence").symlink_to(outside,target_is_directory=True)
            except OSError:self.skipTest("symlink creation unavailable")
            with mock.patch.object(repair,"BASE",base),self.assertRaisesRegex(ValueError,"symlink"):
                repair.load_identity(path,repair.sha(path),POOL,repair.APPROVED_POOL_SHA256)
            self.assertEqual(list(outside.iterdir()),[])

    def test_original_native_metadata_sha_checked_when_record_contains_it(self):
        with tempfile.TemporaryDirectory() as d:
            base,target,path,record,wheel,actual,protected=self.fixture(d)
            record["native_before"]["torch"]["metadata_sha256"]="different"
            with self.assertRaisesRegex(ValueError,"METADATA SHA"):
                repair.require_native(record,protected)


if __name__=="__main__":unittest.main()
