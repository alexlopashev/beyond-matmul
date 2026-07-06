import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_check_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "check_torch_frontend_coverage.py"
    spec = importlib.util.spec_from_file_location("check_torch_frontend_coverage", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TorchFrontendCoverageCheckTests(unittest.TestCase):
    def test_supported_rows_have_evidence_references(self):
        check = _load_check_module()
        repo_root = Path(__file__).resolve().parents[1]

        rows = check.parse_coverage_rows(repo_root / "docs" / "torch_frontend_coverage.md")
        errors = check.validate_rows(rows, repo_root)

        supported_rows = {row.pattern for row in rows if row.status == "Supported"}
        self.assertIn("Functional `conv1d`", supported_rows)
        self.assertIn("Exported graph fixed-weight `addmm` and nested linear", supported_rows)
        self.assertEqual(errors, [])

    def test_missing_mapping_for_supported_row_fails(self):
        check = _load_check_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            coverage_doc = repo_root / "coverage.md"
            coverage_doc.write_text(
                "\n".join(
                    [
                        "| Pattern | Status | Captured IR | Notes |",
                        "| --- | --- | --- | --- |",
                        "| Future op | Supported | `DenseOperator` | Accidentally promoted. |",
                    ]
                ),
                encoding="utf-8",
            )
            rows = check.parse_coverage_rows(coverage_doc)

            errors = check.validate_rows(rows, repo_root)

        self.assertIn("supported row has no evidence mapping: Future op", errors)


if __name__ == "__main__":
    unittest.main()
