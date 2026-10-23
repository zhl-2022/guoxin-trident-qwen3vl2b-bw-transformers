"""Record bounded Recon functional checks in the captured session003 workspace."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    task = ROOT / "tasks/recon2reason-bw-vllm"
    output = ROOT / "evidence/session003/recon-functional"
    images = output / "images"
    commands = [
        ["bash", str(task / "env_check.sh")],
        [sys.executable, str(ROOT / "scripts/robobrain_heldout_fixtures.py"), "--output-dir", str(images)],
        [sys.executable, str(ROOT / "scripts/recon2reason_smoke.py"),
         "--image", str(images / "heldout_red_square.png"),
         "--image", str(images / "heldout_blue_circle.png"),
         "--base-url", "http://127.0.0.1:30000", "--model", "recon2reason-reasoning-4b",
         "--max-tokens", "256", "--timeout", "120", "--log", str(output / "smoke.jsonl")],
    ]
    if not args.run:
        print(json.dumps({"mode": "plan_only", "commands": commands, "output": str(output),
                          "service_started": False, "executed": False}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    report = {"development_only": True, "official_accuracy": False, "steps": [],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for number, command in enumerate(commands):
        row = {"command": command, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        report["steps"].append(row); save()
        with (output / f"step{number}.log").open("x", encoding="utf-8") as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, timeout=600)
                row["exit_code"] = result.returncode
            except subprocess.TimeoutExpired:
                row["exit_code"] = 124
        row["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(); save()
        if row["exit_code"]:
            break
    for source in (task / "runtime/state.json", task / "runtime/server.log", task / "env.log"):
        if source.is_file():
            name = "state-at-functional-end.json" if source.name == "state.json" else source.name
            shutil.copy2(source, output / name)
    report["steps_completed"] = len(report["steps"]) == 3 and all(row["exit_code"] == 0 for row in report["steps"])
    report["service_stop_requested_by_this_script"] = False
    save()
    print(json.dumps({"output": str(output), "steps_completed": report["steps_completed"]}))
    return 0 if report["steps_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
