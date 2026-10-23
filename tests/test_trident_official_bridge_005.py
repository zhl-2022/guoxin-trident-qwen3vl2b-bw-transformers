"""CPU external-execution mocks, never official model/evaluator acceptance evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import trident_official_bridge_005 as bridge

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.task=self.base/'trident-qwen3vl2b-bw-transformers';self.task.mkdir()
        shutil.copyfile(ROOT/'tasks/trident-qwen3vl2b-bw-transformers/task.json',self.task/'task.json')
        self.data=self.base/'data';self.data.mkdir();(self.data/'sample.txt').write_text('CPU fixture data only')
        self.qrels=self.base/'qrels.txt';self.qrels.write_text('CPU fixture qrels only')
        self.baseline=self.base/'baseline.txt';self.baseline.write_text('CPU fixture opaque baseline only')
        self.script=self.base/'external.py';self.script.write_text("from pathlib import Path\nprint('opaque external execution')\nPath('result.bin').write_bytes(b'opaque result bytes')\n")
        self.output=self.base/'fresh-output';self.env=dict(os.environ)
        self.env.update(OFFICIAL_TRIDENT_DATA_DIR=str(self.data),OFFICIAL_TRIDENT_DATA_FILES_JSON='["sample.txt"]',
            OFFICIAL_TRIDENT_QRELS=str(self.qrels),OFFICIAL_TRIDENT_BASELINE=str(self.baseline),
            OFFICIAL_TRIDENT_OUTPUT_DIR=str(self.output),OFFICIAL_TRIDENT_TIMEOUT_SECONDS='2')
        for phase in ['VERIFY','SELFCHECK','BENCH']:
            self.env['OFFICIAL_TRIDENT_'+phase+'_SCRIPT']=str(self.script)
            self.env['OFFICIAL_TRIDENT_'+phase+'_ARGS_JSON']='[]'
            self.env['OFFICIAL_TRIDENT_'+phase+'_OUTPUTS_JSON']='["result.bin"]'
    def run_bridge(self,mode='verify',extra=()):
        with contextlib.redirect_stdout(io.StringIO()) as out:code=bridge.main([mode,'--task-dir',str(self.task),*extra],self.env)
        return code,json.loads(out.getvalue())
    def test_verify_selfcheck_executes_opaque_contract_and_never_reports_official_pass(self):
        before=self.baseline.read_bytes();code,r=self.run_bridge(extra=['--selfcheck'])
        self.assertEqual(code,0);self.assertEqual(r['status'],'executed_uninterpreted');self.assertTrue(r['inputs_unchanged'])
        self.assertTrue(r['child_reaped']);self.assertFalse(r['external_output_content_interpreted'])
        self.assertFalse(r['official_accuracy_verified']);self.assertFalse(r['official_performance_accepted']);self.assertFalse(r['submission_ready'])
        self.assertEqual(self.baseline.read_bytes(),before);self.assertEqual(r['outputs'][0]['sha256'],bridge.sha(b'opaque result bytes'))
    def test_missing_materials_blocks_before_execution(self):
        del self.env['OFFICIAL_TRIDENT_BASELINE'];code,r=self.run_bridge()
        self.assertEqual(code,2);self.assertFalse(r['executed']);self.assertFalse(self.output.exists())
    def test_bench_requires_its_own_script_and_output_contract(self):
        del self.env['OFFICIAL_TRIDENT_BENCH_SCRIPT'];code,r=self.run_bridge('bench')
        self.assertEqual(code,2);self.assertFalse(r['executed']);self.assertIn('BENCH_SCRIPT',r['error'])
    def test_bench_preserves_explicit_arguments_without_generation_injection(self):
        self.env['OFFICIAL_TRIDENT_BENCH_ARGS_JSON']='["--custom-embedding-load","fixture-only"]'
        code,r=self.run_bridge('bench');self.assertEqual(code,0)
        self.assertEqual(r['configuration']['argv'][-2:],['--custom-embedding-load','fixture-only'])
        self.assertNotIn('1024',r['configuration']['argv']);self.assertNotIn('BENCH_MODE',r['configuration'])
    def test_external_nonzero_keeps_log_and_failure(self):
        self.script.write_text("print('native-failure-42',flush=True)\nraise SystemExit(7)\n")
        code,r=self.run_bridge();self.assertEqual(code,2);self.assertEqual(r['external_exit_code'],7)
        self.assertEqual(r['status'],'executed_failed');self.assertIn('native-failure-42',(self.output/'bridge-stdout.log').read_text())
    def test_real_child_timeout_is_reaped(self):
        self.script.write_text("import time\nprint('timeout-child-started',flush=True)\ntime.sleep(5)\n")
        self.env['OFFICIAL_TRIDENT_TIMEOUT_SECONDS']='0.15';code,r=self.run_bridge()
        self.assertEqual(code,2);self.assertEqual(r['status'],'executed_timeout');self.assertTrue(r['child_reaped'])
        self.assertLess(r['execution_wall_seconds'],4)
    def test_output_path_traversal_and_reserved_names_refused(self):
        for name in ['../baseline.txt','bridge-final.json','/absolute-result']:
            with self.subTest(name=name):
                self.env['OFFICIAL_TRIDENT_VERIFY_OUTPUTS_JSON']=json.dumps([name]);code,r=self.run_bridge()
                self.assertEqual(code,2);self.assertFalse(r['executed']);self.assertFalse(self.output.exists())
    def test_existing_output_never_overwritten(self):
        self.output.mkdir();(self.output/'prior.txt').write_text('prior evidence')
        code,r=self.run_bridge();self.assertEqual(code,2);self.assertFalse(r['executed']);self.assertEqual((self.output/'prior.txt').read_text(),'prior evidence')
    def test_input_mutation_is_failure_and_before_hash_is_preserved(self):
        old=bridge.sha(self.baseline.read_bytes())
        self.script.write_text("import os\nfrom pathlib import Path\nPath(os.environ['OFFICIAL_TRIDENT_BASELINE']).write_text('changed by CPU mock')\nPath('result.bin').write_bytes(b'opaque')\n")
        code,r=self.run_bridge();self.assertEqual(code,2);self.assertFalse(r['inputs_unchanged']);self.assertEqual(r['status'],'declared_inputs_changed')
        before=json.loads((self.output/'bridge-inputs-before.json').read_text());record=next(x for x in before if x['path']==str(self.baseline))
        self.assertEqual(record['sha256'],old)
    def test_declared_output_missing_is_failure(self):
        self.script.write_text("print('no declared output produced')\n");code,r=self.run_bridge()
        self.assertEqual(code,2);self.assertEqual(r['status'],'output_contract_failed')
    def test_dry_run_is_pure_plan(self):
        code,r=self.run_bridge(extra=['--dry-run']);self.assertEqual(code,0);self.assertFalse(r['executed']);self.assertFalse(self.output.exists())
    def test_relative_material_environment_paths_are_frozen_before_child_cwd(self):
        self.script.write_text("import os\nfrom pathlib import Path\nassert Path(os.environ['OFFICIAL_TRIDENT_BASELINE']).read_text()=='CPU fixture opaque baseline only'\nassert Path(os.environ['OFFICIAL_TRIDENT_DATA_DIR']).is_dir()\nPath('result.bin').write_bytes(b'opaque result bytes')\n")
        self.env.update(OFFICIAL_TRIDENT_DATA_DIR='./data',OFFICIAL_TRIDENT_QRELS='qrels.txt',
            OFFICIAL_TRIDENT_BASELINE='baseline.txt',OFFICIAL_TRIDENT_OUTPUT_DIR='fresh-output',OFFICIAL_TRIDENT_VERIFY_SCRIPT='external.py')
        previous=Path.cwd()
        try:os.chdir(self.base);code,r=self.run_bridge()
        finally:os.chdir(previous)
        self.assertEqual(code,0);self.assertTrue(all(Path(x).is_absolute() for x in r['child_environment_path_bindings'].values()))
    @unittest.skipUnless(os.name=='posix','POSIX symlink fixture')
    def test_output_symlink_to_input_is_not_accepted(self):
        self.script.write_text("import os\nfrom pathlib import Path\nPath('result.bin').symlink_to(os.environ['OFFICIAL_TRIDENT_BASELINE'])\n")
        code,r=self.run_bridge();self.assertEqual(code,2);self.assertEqual(r['status'],'output_contract_failed')

if __name__=='__main__':unittest.main()
