"""The image bakes the ONNX weights; these assert it bakes the right ones.

`backend/Dockerfile` downloads the embedding and re-ranking models at build
time so a scale-to-zero host does not re-download ~150MB inside the first
request (DECISIONS.md D-009). The model names live in build args rather than in
the application config, because that layer deliberately sits before `COPY app`
so an application edit does not invalidate the download. That duplication is
the thing worth testing: if a default moves in `app/config.py` and the
Dockerfile is not updated, the image caches weights the app never asks for and
the download silently returns at runtime.
"""

import re
from pathlib import Path

import pytest

from app.config import Settings

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def _build_arg(name: str) -> str:
    match = re.search(rf"^ARG {name}=(.+)$", DOCKERFILE.read_text(), re.MULTILINE)
    assert match, f"{name} is not declared as a build arg in {DOCKERFILE}"
    return match.group(1).strip()


@pytest.mark.parametrize(
    ("build_arg", "setting"),
    [("EMBEDDING_MODEL", "embedding_model"), ("RERANK_MODEL", "rerank_model")],
)
def test_baked_model_matches_config_default(build_arg: str, setting: str) -> None:
    assert _build_arg(build_arg) == Settings.model_fields[setting].default


@pytest.mark.parametrize("variable", ["EMBEDDING_CACHE_DIR", "RERANK_CACHE_DIR"])
def test_image_sets_a_cache_dir(variable: str) -> None:
    """Blank would send the weights to each library's default.

    FlashRank's default is `/tmp`, which on Cloud Run is memory rather than
    disk, so an unset cache dir costs resident memory for no reason.
    """
    assert re.search(rf"^\s+{variable}=\S+", DOCKERFILE.read_text(), re.MULTILINE)


def test_cache_dir_defaults_are_blank() -> None:
    """Blank means "use the library default", which is what local dev wants.

    Only the image overrides them, so a developer's cache is untouched.
    """
    assert Settings.model_fields["embedding_cache_dir"].default == ""
    assert Settings.model_fields["rerank_cache_dir"].default == ""
