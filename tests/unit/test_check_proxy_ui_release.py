import io
import hashlib
import json
from pathlib import Path
from typing import Final
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from scripts.check_proxy_ui_release import ASSETS, check_version, check_wheel, published_wheel


@pytest.mark.parametrize("pinned", ("1.0", "0.9"))
def test_proxy_pin_matches_extras_version(tmp_path: Path, pinned: str) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[project.optional-dependencies]\nproxy = ["litellm-proxy-extras=={pinned}"]\n'
    )
    (tmp_path / "litellm-proxy-extras").mkdir()
    (tmp_path / "litellm-proxy-extras/pyproject.toml").write_text('[project]\nversion = "1.0"\n')
    if pinned == "1.0":
        assert check_version(tmp_path, None) == "1.0"
    else:
        with pytest.raises(ValueError, match="pin"):
            check_version(tmp_path, None)


@pytest.mark.parametrize("contents", (b"current dashboard", b"stale dashboard", None))
def test_release_wheel_must_contain_matching_dashboard(tmp_path: Path, contents: bytes | None) -> None:
    (tmp_path / ASSETS).mkdir(parents=True)
    (tmp_path / ASSETS / "index.html").write_bytes(b"current dashboard")
    buffer: Final = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("litellm_proxy_extras-1.0.dist-info/METADATA", "Name: litellm-proxy-extras\nVersion: 1.0\n")
        if contents is not None:
            archive.writestr("litellm_proxy_extras/ui/index.html", contents)
    if contents == b"current dashboard":
        check_wheel(tmp_path, buffer.getvalue(), "1.0")
    else:
        with pytest.raises(ValueError, match="differs"):
            check_wheel(tmp_path, buffer.getvalue(), "1.0")


@pytest.mark.parametrize("fault", ("none", "digest", "host", "multiple"))
def test_published_artifact_requires_expected_host_and_checksum(fault: str) -> None:
    payload: Final = b"published wheel bytes"
    artifact: Final = {
        "packagetype": "bdist_wheel",
        "url": "https://unexpected.example/wheel" if fault == "host" else "https://files.pythonhosted.org/wheel",
        "digests": {"sha256": "wrong" if fault == "digest" else hashlib.sha256(payload).hexdigest()},
    }
    response: Final = io.BytesIO(json.dumps({"urls": [artifact, artifact] if fault == "multiple" else [artifact]}).encode())
    with patch("urllib.request.urlopen", side_effect=(response, io.BytesIO(payload))):
        if fault == "none":
            assert published_wheel("1.0") == payload
        else:
            with pytest.raises(ValueError, match=r"checksum|host|one published"):
                published_wheel("1.0")


def test_wheel_metadata_must_match_pinned_version(tmp_path: Path) -> None:
    buffer: Final = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("litellm_proxy_extras-1.0.dist-info/METADATA", "Name: litellm-proxy-extras\nVersion: 0.9\n")
    with pytest.raises(ValueError, match="version"):
        check_wheel(tmp_path, buffer.getvalue(), "1.0")
