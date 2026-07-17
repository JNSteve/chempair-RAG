"""Guard: every local module server.py imports must be COPY'd into the
Docker image. Added after PR #30 crashed the Railway deploy — the Dockerfile
uses an explicit COPY list, and new modules had not been added to it."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_copies_every_local_module_server_imports():
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    local_modules = sorted(
        {
            module
            for module in re.findall(r"^from (\w+) import", server, re.MULTILINE)
            if (ROOT / f"{module}.py").is_file()
        }
    )
    assert local_modules, "expected server.py to import local modules"

    missing = [
        module for module in local_modules if f"COPY {module}.py" not in dockerfile
    ]
    assert not missing, (
        f"Dockerfile is missing COPY lines for: {missing} — the deploy will "
        "crash at import. Add them to the Dockerfile."
    )


def test_dockerfile_copies_start_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY start.py" in dockerfile


def test_every_dockerfile_copy_source_exists():
    """Reverse guard: a COPY of a deleted file fails the image build before
    Railway even starts the container (bit us when the legacy routing modules
    were removed but their COPY lines stayed behind)."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    sources = re.findall(r"^COPY\s+(\S+)\s+\S+$", dockerfile, re.MULTILINE)
    assert sources, "expected COPY lines in the Dockerfile"

    missing = [source for source in sources if not (ROOT / source).exists()]
    assert not missing, (
        f"Dockerfile COPYs files that do not exist: {missing} — the image "
        "build will fail. Remove or fix these COPY lines."
    )
