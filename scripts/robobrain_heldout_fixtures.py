"""Create new synthetic image sizes and optionally run loopback protocol checks.

These synthetic observations are not an official dataset, metric, or acceptance.
The output directory must be new; images, checksums and raw answers are preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

from PIL import Image, ImageDraw, __version__ as pillow_version

from robobrain_protocol_smoke import (
    NoRedirect, bounded_text, build_request, image_payload, load_configuration,
    local_base_url, request_json, response_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]


def create_fixtures(directory: Path) -> list[dict]:
    specs = [
        {"name": "heldout_red_square", "size": (640, 448),
         "bbox_pixels": (400, 112, 560, 272), "color": (230, 25, 25),
         "shape": "square", "target": "the red square",
         "label": "HELDOUT A - RED SQUARE"},
        {"name": "heldout_blue_circle", "size": (512, 512),
         "bbox_pixels": (64, 280, 208, 424), "color": (25, 75, 230),
         "shape": "circle", "target": "the blue circle",
         "label": "HELDOUT B - BLUE CIRCLE"},
    ]
    fixtures = []
    for spec in specs:
        image = Image.new("RGB", spec["size"], "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), spec["label"], fill="black")
        if spec["shape"] == "square":
            draw.rectangle(spec["bbox_pixels"], fill=spec["color"])
        else:
            draw.ellipse(spec["bbox_pixels"], fill=spec["color"])
        path = directory / (spec["name"] + ".png")
        image.save(path, format="PNG")
        width, height = spec["size"]
        x1, y1, x2, y2 = spec["bbox_pixels"]
        fixtures.append({
            **spec, "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "bbox_xyxy_0_1000": [1000*x1/width, 1000*y1/height,
                                  1000*x2/width, 1000*y2/height],
            "center_xy_0_1000": [500*(x1+x2)/width, 500*(y1+y2)/height],
            "patch_grid_if_no_processor_resize": [1, height//16, width//16],
            "vision_sequence_length_if_no_resize": width*height//256,
            "processor_grid_actually_verified": False,
        })
    return fixtures


def synthetic_coordinate_diagnostic(task: str, text: str, fixture: dict) -> dict:
    expected = fixture["bbox_xyxy_0_1000"]
    if task == "pointing":
        match = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
        if match:
            x, y = map(int, match.groups())
            return {"predicted_xy": [x, y], "point_inside_synthetic_bbox":
                    expected[0] <= x <= expected[2] and expected[1] <= y <= expected[3]}
    if task == "grounding":
        match = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            intersection = max(0, min(x2, expected[2])-max(x1, expected[0])) * max(
                0, min(y2, expected[3])-max(y1, expected[1]))
            union = max(0, x2-x1)*max(0, y2-y1) + (
                expected[2]-expected[0])*(expected[3]-expected[1]) - intersection
            return {"predicted_xyxy": [x1, y1, x2, y2],
                    "synthetic_bbox_iou_unclipped": intersection/union if union > 0 else None}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory only")
    parser.add_argument("--task", type=Path, default=ROOT/"tasks/robobrain-bw-vllm/task.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:30001")
    parser.add_argument("--model", default="robobrain2-5-8b-nv")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--run", action="store_true", help="send seven requests after generation")
    args = parser.parse_args()
    if args.output_dir.exists() or args.timeout <= 0:
        parser.error("output-dir must be new and timeout must be positive")
    base = local_base_url(args.base_url)
    config = load_configuration(args.task)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixtures = create_fixtures(args.output_dir)
    manifest = {"kind": "synthetic_heldout_sizes", "created_at_utc":
                datetime.now(timezone.utc).isoformat(), "pillow": pillow_version,
                "official_accuracy": False, "official_performance": False,
                "semantic_acceptance_verified": False, "fixtures": fixtures,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "task_sha256": hashlib.sha256(args.task.read_bytes()).hexdigest()}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    functional = []
    with (args.output_dir/"heldout.jsonl").open("x", encoding="utf-8") as log:
        def record(case, **values):
            event = {"utc": datetime.now(timezone.utc).isoformat(), "case": case, **values}
            log.write(json.dumps(event, ensure_ascii=False)+"\n")
            log.flush()
            print(json.dumps(event, ensure_ascii=False), flush=True)

        record("scope", dry_run=not args.run, model=args.model, base_url=base,
               official_accuracy=False, semantic_acceptance_verified=False,
               note="Synthetic coordinates are diagnostics; no official metric or trajectory truth.")
        try:
            if args.run:
                models = request_json(opener, base+"/v1/models", args.timeout)
                if args.model not in [item.get("id") for item in models.get("data", [])]:
                    raise ValueError("expected served model is absent")
            payloads = []
            for fixture in fixtures:
                payload, metadata = image_payload(Path(fixture["path"]))
                payloads.append(payload)
                for task in ("general", "grounding", "pointing"):
                    text = ("Briefly describe the colored shape and its location."
                            if task == "general" else fixture["target"])
                    body = build_request(config, task, text, payload, args.model, 256)
                    values = {"task": task, "image": metadata,
                              "expected_bbox_xyxy_0_1000": fixture["bbox_xyxy_0_1000"],
                              "prompt": body["messages"][0]["content"][-1]["text"]}
                    if args.run:
                        diagnostic = response_diagnostics(task, request_json(
                            opener, base+"/v1/chat/completions", args.timeout, body))
                        values.update(diagnostic)
                        values["synthetic_coordinate_diagnostic"] = synthetic_coordinate_diagnostic(
                            task, diagnostic["raw_answer"]["text"], fixture)
                        functional.append(diagnostic["functional_protocol_check_passed"])
                    record(fixture["name"]+"_"+task, **values)
            prompt = "Briefly compare the colored shapes and their locations in the first and second images."
            if args.run:
                body = {"model": args.model, "messages": [{"role": "user", "content":
                        payloads+[{"type": "text", "text": prompt}]}],
                        "temperature": 0, "max_tokens": 256, "stream": False}
                diagnostic = response_diagnostics("general", request_json(
                    opener, base+"/v1/chat/completions", args.timeout, body))
                functional.append(diagnostic["functional_protocol_check_passed"])
                record("two_images", prompt=prompt, **diagnostic)
            else:
                record("two_images", prompt=prompt, image_sha256=[f["sha256"] for f in fixtures])
            passed = len(functional) == 7 and all(functional) if args.run else None
            record("summary", requests_sent=len(functional), functional_protocol_checks_passed=passed,
                   semantic_acceptance_verified=False, official_accuracy=False)
            return 0 if not args.run or passed else 2
        except Exception as exc:
            record("error", error_type=type(exc).__name__, error=bounded_text(str(exc), 2000))
            return 2


if __name__ == "__main__":
    sys.exit(main())
