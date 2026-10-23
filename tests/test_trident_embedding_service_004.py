import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import trident_embedding_runtime_004 as runtime
import trident_embedding_smoke_004 as smoke

TASK=ROOT/"tasks/trident-qwen3vl2b-bw-transformers"


class ServiceTests(unittest.TestCase):
    def test_command_preserves_successful_pooling_route_and_8k(self):
        with mock.patch.dict(os.environ,{},clear=True):config=runtime.configuration(TASK)
        command=config["command"]
        self.assertEqual(command[command.index("--max-model-len")+1],"8192")
        self.assertEqual(command[command.index("--runner")+1],"pooling")
        self.assertEqual(json.loads(command[command.index("--pooler-config")+1]),{"seq_pooling_type":"LAST","use_activation":True})
        self.assertEqual(json.loads(command[command.index("--hf-overrides")+1])["head_dtype"],"float32")
        self.assertFalse(config["http_gpu_verified"])
        self.assertNotIn("--reasoning-parser",command)

    def test_dry_launch_is_metadata_only_and_configuration_restored(self):
        original=runtime.supervisor.configuration
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runtime.main(["launch","--task-dir",str(TASK),"--dry-run"]),0)
        self.assertEqual(json.loads(output.getvalue())["max_len"],8192)
        self.assertIs(runtime.supervisor.configuration,original)
        self.assertNotIn("torch",sys.modules)

    def test_dry_stop_is_plan_only_and_never_calls_stop(self):
        with mock.patch.object(runtime.supervisor,"stop") as stop,contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runtime.main(["stop","--task-dir",str(TASK),"--run-id","a"*32,"--dry-run"]),0)
        stop.assert_not_called()
        result=json.loads(output.getvalue())
        self.assertEqual(result["status"],"plan_only")
        self.assertFalse(result["executed"])
        self.assertFalse(result["process_signalled"])

    def test_relative_model_identity_matches_preflight_and_child_from_other_cwd(self):
        original_cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                expected=str((Path(d)/"relative-checkpoint").resolve())
                def launch(task,dry_run,run_id):
                    # Simulate the server's different cwd after parent preflight.
                    os.chdir(TASK)
                    config=runtime.supervisor.configuration(task)
                    self.assertEqual(config["model_path"],expected)
                    for flag in ("--model","--tokenizer"):
                        self.assertEqual(config["command"][config["command"].index(flag)+1],expected)
                    return 0
                with mock.patch.dict(os.environ,{"TRIDENT_MODEL_PATH":"relative-checkpoint"}),\
                     mock.patch.object(runtime.sys,"platform","linux"),\
                     mock.patch.object(runtime,"verify_checkpoint",return_value={"complete":True}) as verify,\
                     mock.patch.object(runtime,"validate_model_semantics") as semantics,\
                     mock.patch.object(runtime.supervisor,"launch",side_effect=launch):
                    self.assertEqual(runtime.main(["launch","--task-dir",str(TASK)]),0)
                self.assertEqual(verify.call_args.args[0],expected)
                semantics.assert_called_once_with(expected)
            finally:os.chdir(original_cwd)

    def test_missing_official_contract_fails_without_generating_results(self):
        for action in ("verify","bench"):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runtime.main([action,"--task-dir",str(TASK)]),2)
            self.assertFalse(json.loads(output.getvalue())["executed"])

    def test_role_and_assistant_suffix_use_documented_embedding_extension(self):
        for role in ("query","doc"):
            body=smoke.payload("model",role,"text","data:image/png;base64,AA==")
            self.assertEqual(body["messages"][0]["content"][0]["text"],smoke.INSTRUCTIONS[role])
            self.assertEqual([x["type"] for x in body["messages"][1]["content"]],["image_url","text"])
            self.assertEqual(body["messages"][-1]["role"],"assistant")
            self.assertTrue(body["continue_final_message"])
            self.assertTrue(body["add_special_tokens"])

    def valid(self):
        return {"object":"list","model":"trident-qwen3vl-2b","data":[{"index":0,"embedding":[1.]+[0.]*2047}],
                "usage":{"prompt_tokens":20,"total_tokens":20,"completion_tokens":0}}

    def test_embedding_shape_norm_and_usage_reject_false_pass(self):
        self.assertEqual(smoke.response_check(self.valid(),"trident-qwen3vl-2b")["dimension"],2048)
        for change in ("length","nan","norm","usage","generated","boolean_usage"):
            value=self.valid()
            if change=="length":value["data"][0]["embedding"].pop()
            elif change=="nan":value["data"][0]["embedding"][0]=float("nan")
            elif change=="norm":value["data"][0]["embedding"][0]=0
            elif change=="usage":value["usage"]["total_tokens"]=21
            elif change=="generated":value["usage"]["completion_tokens"]=1
            else:value["usage"]["completion_tokens"]=False
            with self.assertRaises(ValueError):smoke.response_check(value,"trident-qwen3vl-2b")

    def test_loopback_only(self):
        for url in ("https://127.0.0.1","http://example.com","http://user:password@127.0.0.1","http://127.0.0.1/?token=x"):
            with self.assertRaises(ValueError):smoke.base_url(url)

    def test_live_local_HTTP_contract_covers_five_true_embedding_requests(self):
        requests=[];valid=self.valid()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def respond(self,value):
                raw=json.dumps(value).encode();self.send_response(200);self.send_header("Content-Length",str(len(raw)))
                self.end_headers();self.wfile.write(raw)
            def do_GET(self):self.respond({"data":[{"id":"trident-qwen3vl-2b"}]} if self.path=="/v1/models" else {})
            def do_POST(self):
                self.assert_endpoint=self.path
                requests.append((self.path,json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.respond(valid)
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d,mock.patch.object(smoke,"prepared_image",return_value=("data:image/png;base64,AA==",[512,512])):
                image=Path(d)/"fixture.png";image.write_bytes(b"fixture")
                log=Path(d)/"smoke.log"
                self.assertEqual(smoke.main(["--base-url",f"http://127.0.0.1:{server.server_port}","--image",str(image),"--log",str(log)]),0)
                self.assertEqual(len(requests),5)
                self.assertTrue(all(endpoint=="/v1/embeddings" for endpoint,_ in requests))
                summary=json.loads(log.read_text().splitlines()[-1])
                self.assertTrue(summary["functional_embedding_api_passed"])
                self.assertFalse(summary["official_accuracy_verified"])
        finally:server.shutdown();server.server_close();thread.join()

    def test_HTTP_400_preserves_bounded_error_body_and_stays_failed(self):
        detail="embedding-tokenizer-contract-detail42"
        raw=json.dumps({"error":{"message":detail}}).encode()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def respond(self,status,value):
                self.send_response(status);self.send_header("Content-Length",str(len(value)));self.end_headers();self.wfile.write(value)
            def do_GET(self):
                value={"data":[{"id":"trident-qwen3vl-2b"}]} if self.path=="/v1/models" else {}
                self.respond(200,json.dumps(value).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.respond(400,raw)
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d,mock.patch.object(smoke,"prepared_image",return_value=("data:image/png;base64,AA==",[512,512])):
                image=Path(d)/"fixture.png";image.write_bytes(b"fixture");log=Path(d)/"smoke.log"
                self.assertEqual(smoke.main(["--base-url",f"http://127.0.0.1:{server.server_port}","--image",str(image),"--log",str(log)]),2)
                rows=[json.loads(line) for line in log.read_text().splitlines()]
                errors=[row for row in rows if "raw_error_body" in row]
                self.assertEqual(len(errors),1)
                self.assertEqual(errors[0]["endpoint"],"/v1/embeddings")
                self.assertEqual(errors[0]["http_status"],400)
                self.assertIn(detail,errors[0]["raw_error_body"])
                self.assertEqual(smoke.base64.b64decode(errors[0]["raw_error_body_base64"]),raw)
                self.assertEqual(errors[0]["raw_error_body_sha256"],smoke.hashlib.sha256(raw).hexdigest())
                self.assertFalse(rows[-1]["functional_embedding_api_passed"])
        finally:server.shutdown();server.server_close();thread.join()


if __name__=="__main__":unittest.main()
