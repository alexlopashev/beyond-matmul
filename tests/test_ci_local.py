import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class LocalCiScriptTests(unittest.TestCase):
    def test_ci_workflow_publishes_fixed_weight_json_after_demos(self):
        repo_root = Path(__file__).resolve().parents[1]
        workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        coverage_demo = "mise exec -- uv run python examples/torch_coverage_demo.py"
        benchmark_json = (
            "mise exec -- uv run python benchmarks/fixed_weight.py "
            "--json-output docs/results/fixed_weight.json"
        )
        upload_action = "uses: actions/upload-artifact@v4"
        artifact_name = "name: fixed-weight-benchmark-json"
        artifact_path = "path: docs/results/fixed_weight.json"

        self.assertLess(workflow.index(coverage_demo), workflow.index(benchmark_json))
        self.assertLess(workflow.index(benchmark_json), workflow.index(upload_action))
        self.assertIn(artifact_name, workflow)
        self.assertIn(artifact_path, workflow)

    def test_ci_local_generates_fixed_weight_json_artifact(self):
        repo_root = Path(__file__).resolve().parents[1]
        ci_local = (repo_root / "scripts" / "ci_local").read_text(encoding="utf-8")

        self.assertIn(
            '"$MISE_BIN" exec -- uv run python benchmarks/fixed_weight.py '
            "--json-output docs/results/fixed_weight.json",
            ci_local,
        )

    def test_ci_local_resolves_mise_installed_by_bootstrap_outside_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_dir = temp_path / "project"
            home_dir = temp_path / "home"
            scripts_dir = project_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            home_dir.mkdir()

            shutil.copy(repo_root / "scripts" / "ci_local", scripts_dir / "ci_local")
            (scripts_dir / "bootstrap").write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env sh
                    set -eu
                    mkdir -p "$HOME/.local/bin"
                    cat > "$HOME/.local/bin/mise" <<'MISE'
                    #!/usr/bin/env sh
                    printf '%s\\n' "$*" >> "$HOME/mise.log"
                    exit 0
                    MISE
                    chmod +x "$HOME/.local/bin/mise"
                    """
                ),
                encoding="utf-8",
            )
            (scripts_dir / "bootstrap").chmod(stat.S_IRWXU)

            result = subprocess.run(
                ["sh", str(scripts_dir / "ci_local")],
                cwd=project_dir,
                env={
                    "HOME": str(home_dir),
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            mise_calls = (home_dir / "mise.log").read_text(encoding="utf-8").splitlines()
            self.assertIn("exec -- uv sync --locked", mise_calls)
            self.assertIn(
                "exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json",
                mise_calls,
            )


if __name__ == "__main__":
    unittest.main()
