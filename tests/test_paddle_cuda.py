import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "comparators"), *sys.path]):
    import paddle_cuda
    import runner


class PaddleCudaTest(unittest.TestCase):
    def test_four_sessions_require_cuda(self):
        provider = "CUDAExecutionProvider"
        sessions = [Mock() for _ in range(4)]
        for session in sessions:
            session.get_providers.return_value = [provider, "CPUExecutionProvider"]
        engine = SimpleNamespace(det_sess=sessions[0], rec=SimpleNamespace(ctc=sessions[1], enc=sessions[2], dec=sessions[3]))
        word = Mock(return_value=engine)
        distribution = Mock(version="0.1.2")
        distribution.read_text.return_value = json.dumps({"vcs_info": {"commit_id": runner.PADDLE_REVISION}})
        fake = {"paddleocr_hebrew": SimpleNamespace(HebrewOCR=SimpleNamespace(word=word)),
                "onnxruntime": SimpleNamespace(get_available_providers=lambda: [provider])}
        with patch.dict(sys.modules, fake), patch.object(runner, "ASSETS", {}), \
                patch.object(runner.metadata, "distribution", return_value=distribution), \
                patch.object(runner.metadata, "version", return_value="1.32.0") as version:
            _, provenance = runner.prepare(Path("assets"), "paddle", "page", paddle_provider=provider)
            version.assert_called_once_with("onnxruntime-gpu")
            self.assertEqual(word.call_args.kwargs["providers"], [provider])
            self.assertEqual(provenance["settings"]["provider"], provider)
            for session in sessions:
                session.disable_fallback.assert_called_once()
            sessions[-1].get_providers.return_value = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "did not start with the CUDA"):
                runner.prepare(Path("assets"), "paddle", "page", paddle_provider=provider)

    def test_order_blanks_and_failure_publication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "page.png"
            image.write_bytes(b"invented fixture")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            rows = [{"id": key, "path": image.name, "sha256": digest} for key in ("a", "b")]
            manifest.write_text(json.dumps(rows))
            pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
            output = root / "predictions.jsonl"
            with patch.object(paddle_cuda, "prepare", return_value=(Mock(side_effect=["שלום\r\n", "\n"]), {})):
                paddle_cuda.predict(manifest, pin, root, output)
            self.assertEqual([json.loads(line) for line in output.read_text().splitlines()],
                             [{"id": "a", "text": "שלום\n"}, {"id": "b", "text": ""}])
            failed = root / "failed.jsonl"
            with patch.object(paddle_cuda, "prepare", return_value=(Mock(side_effect=["ok", RuntimeError()]), {})):
                with self.assertRaisesRegex(ValueError, "valid prediction"):
                    paddle_cuda.predict(manifest, pin, root, failed)
            self.assertFalse(failed.exists())
