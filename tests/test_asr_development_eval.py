"""Offline scoring integrity and explicit audio-boundary checks."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("asr_development_eval_under_test", ROOT / "scripts/asr_development_eval.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def wav_bytes(frames=1600, channels=1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels); audio.setsampwidth(2); audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames * channels)
    return buffer.getvalue()


class ASRDevelopmentEvaluationTests(unittest.TestCase):
    def score_pairs(self, references, predictions):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            return evaluation.score(jsonl(root / "ref.jsonl", references), jsonl(root / "pred.jsonl", predictions))

    def test_micro_counts_are_summed_before_dividing_and_order_is_by_id(self):
        refs = [{"id": str(i), "text": "a b"} for i in range(3)]
        preds = [{"id": "2", "text": "a x b"}, {"id": "0", "text": "a c"}, {"id": "1", "text": "a"}]
        report = self.score_pairs(refs, preds)
        totals = report["totals"]["wer"]
        self.assertEqual((totals["substitutions"], totals["deletions"], totals["insertions"]), (1, 1, 1))
        self.assertEqual(totals["reference_units"], 6)
        self.assertEqual(totals["error_rate"], 0.5)
        self.assertEqual([s["id"] for s in report["samples"]], ["0", "1", "2"])
        self.assertEqual(report["scope"], "development_only")
        self.assertFalse(report["threshold_applied"])
        self.assertNotIn("pass", report)
        self.assertNotIn("baseline", report)

    def test_default_identity_preserves_chinese_case_numbers_punctuation_and_unicode(self):
        pairs = [("你好，42A", "你好,四十二a"), ("é", "e\u0301"), ("a b", "ab")]
        refs = [{"id": str(i), "text": ref} for i, (ref, _) in enumerate(pairs)]
        preds = [{"id": str(i), "text": pred} for i, (_, pred) in enumerate(pairs)]
        report = self.score_pairs(refs, preds)
        self.assertEqual(report["policy"]["normalization"], "identity")
        self.assertEqual(report["samples"][0]["wer"]["substitutions"], 1)
        self.assertGreater(report["samples"][0]["cer"]["errors"], 0)
        self.assertEqual(report["samples"][1]["cer"]["error_rate"], 2)
        self.assertEqual(report["samples"][2]["cer"]["deletions"], 1)

    def test_empty_prediction_is_all_deletions_and_rate_can_exceed_one(self):
        report = self.score_pairs([{"id": "a", "text": "a b"}, {"id": "b", "text": "x"}],
                                 [{"id": "a", "text": ""}, {"id": "b", "text": "x y z"}])
        self.assertEqual(report["samples"][0]["wer"]["deletions"], 2)
        self.assertEqual(report["samples"][1]["wer"]["error_rate"], 2)

    def test_empty_or_whitespace_references_are_rejected(self):
        for text in ("", " \t\n", "\u3000"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "references"):
                self.score_pairs([{"id": "a", "text": text}], [{"id": "a", "text": ""}])

    def test_missing_extra_duplicate_and_failed_response_schema_are_rejected(self):
        refs = [{"id": "a", "text": "hello"}]
        cases = [[{"id": "b", "text": "hello"}],
                 [{"id": "a", "text": "hello"}, {"id": "b", "text": "hello"}],
                 [{"id": "a", "text": "hello"}, {"id": "a", "text": "hello"}],
                 [{"id": "a", "text": None}], [{"id": "a", "text": "hello", "error": "timeout"}]]
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                self.score_pairs(refs, predictions)
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            self.score_pairs(refs + refs, [{"id": "a", "text": "hello"}])

    def test_json_duplicate_keys_invalid_constants_blank_lines_and_bom_rejected(self):
        malformed = [b'{"id":"a","id":"b","text":"ok"}\n',
                     b'{"id":"a","text":NaN}\n', b'\n',
                     b'\xef\xbb\xbf{"id":"a","text":"ok"}\n', b'[]\n', b'\xff\n']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            for raw in malformed:
                path.write_bytes(raw)
                with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeError)):
                    evaluation.read_jsonl(path, "predictions")

    def test_legal_unicode_line_separator_is_preserved_and_file_hashes_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            refs = jsonl(root / "ref.jsonl", [{"id": "a", "text": "一\u2028二"}])
            preds = jsonl(root / "pred.jsonl", [{"id": "a", "text": "一\u2028二"}])
            report = evaluation.score(refs, preds)
            self.assertEqual(report["totals"]["cer"]["reference_units"], 3)
            self.assertEqual(report["inputs"]["references"]["sha256"], hashlib.sha256(refs.read_bytes()).hexdigest())
            self.assertNotIn("一", json.dumps(report, ensure_ascii=False))

    def test_cpu_cost_budget_is_checked_before_scoring(self):
        with patch.object(evaluation, "MAX_PAIR_CELLS", 3), patch.object(evaluation, "edit_counts") as scorer:
            with self.assertRaisesRegex(ValueError, "CPU edit-distance"):
                self.score_pairs([{"id": "a", "text": "abcd"}], [{"id": "a", "text": "abcd"}])
            scorer.assert_not_called()

    def test_manifest_default_only_plans_without_opening_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = jsonl(Path(temporary) / "manifest.jsonl", [{"id": "a", "audio": "not-present.wav", "sha256": "0" * 64}])
            with patch.object(evaluation, "check_wav", side_effect=AssertionError("No audio read in plan")):
                report = evaluation.manifest_plan(path)
            self.assertEqual(report["operation"], "manifest_plan_only")
            self.assertFalse(report["audio_files_opened"])

    def test_wav_boundaries_hash_and_truncated_payload_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            for raw, allowed in ((wav_bytes(frames=30 * 16000), True),
                                 (wav_bytes(frames=30 * 16000 + 1), False),
                                 (wav_bytes(frames=0), False), (wav_bytes(channels=2), False),
                                 (wav_bytes(rate=8000), False), (wav_bytes()[:-2], False)):
                audio.write_bytes(raw)
                path = jsonl(root / "manifest.jsonl", [{"id": "a", "audio": "sample.wav", "sha256": hashlib.sha256(raw).hexdigest()}])
                if allowed:
                    self.assertEqual(evaluation.manifest_plan(path, True)["samples"][0]["audio"]["duration_seconds"], 30)
                else:
                    with self.assertRaises((ValueError, wave.Error)):
                        evaluation.manifest_plan(path, True)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                evaluation.check_wav(audio, "0" * 64)

    def test_manifest_id_mismatch_and_remote_audio_locations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            refs = jsonl(root / "refs.jsonl", [{"id": "a", "text": "hello"}])
            preds = jsonl(root / "preds.jsonl", [{"id": "a", "text": "hello"}])
            manifest = jsonl(root / "manifest.jsonl", [{"id": "b", "audio": "b.wav", "sha256": "0" * 64}])
            with self.assertRaisesRegex(ValueError, "Manifest sample ids differ"):
                evaluation.score(refs, preds, manifest)
            for audio in ("https://example.com/a.wav", "//server/share/a.wav", "\\\\server\\share\\a.wav"):
                jsonl(manifest, [{"id": "a", "audio": audio, "sha256": "0" * 64}])
                with self.assertRaisesRegex(ValueError, "local path"):
                    evaluation.manifest_plan(manifest)

    def test_cli_refuses_existing_output_and_never_writes_partial_bad_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            refs = jsonl(root / "refs.jsonl", [{"id": "a", "text": "hello"}])
            preds = jsonl(root / "preds.jsonl", [{"id": "b", "text": "hello"}])
            output = root / "result.json"
            args = ["score", "--references", str(refs), "--predictions", str(preds), "--output", str(output)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(evaluation.main(args), 2)
                self.assertFalse(output.exists())
                jsonl(preds, [{"id": "a", "text": "hello"}])
                self.assertEqual(evaluation.main(args), 0)
                original = output.read_bytes()
                self.assertEqual(evaluation.main(args), 2)
                self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
