import io
import hashlib
import json
import sys
from pathlib import Path
from typing import Final
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from scripts.check_proxy_ui_release import ASSETS, check_version, check_wheel, published_wheel
from scripts import check_proxy_ui_release


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
    response: Final = io.BytesIO(
        json.dumps({"urls": [artifact, artifact] if fault == "multiple" else [artifact]}).encode()
    )
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


@pytest.mark.parametrize(("changed", "previous"), ((True, "1.0"), (True, "0.9"), (False, "1.0")))
def test_dashboard_changes_require_a_version_bump(tmp_path: Path, changed: bool, previous: str) -> None:
    (tmp_path / "pyproject.toml").write_text('[project.optional-dependencies]\nproxy = ["litellm-proxy-extras==1.0"]\n')
    (tmp_path / "litellm-proxy-extras").mkdir()
    (tmp_path / "litellm-proxy-extras/pyproject.toml").write_text('[project]\nversion = "1.0"\n')
    with patch(
        "subprocess.check_output",
        side_effect=(str(ASSETS / "index.html") if changed else "", f'[project]\nversion = "{previous}"\n'),
    ):
        if changed and previous == "1.0":
            with pytest.raises(ValueError, match="new proxy extras version"):
                check_version(tmp_path, "base")
        else:
            assert check_version(tmp_path, "base") == "1.0"


@pytest.mark.parametrize("published", (False, True))
@pytest.mark.parametrize("current", (False, True))
def test_release_command_rejects_stale_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool, current: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project.optional-dependencies]\nproxy = ["litellm-proxy-extras==1.0"]\n')
    (tmp_path / ASSETS).mkdir(parents=True)
    (tmp_path / "litellm-proxy-extras/pyproject.toml").write_text('[project]\nversion = "1.0"\n')
    (tmp_path / ASSETS / "index.html").write_bytes(b"current")
    buffer: Final = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("litellm_proxy_extras-1.0.dist-info/METADATA", "Name: litellm-proxy-extras\nVersion: 1.0\n")
        archive.writestr("litellm_proxy_extras/ui/index.html", b"current" if current else b"stale")
    payload: Final = buffer.getvalue()
    wheel: Final = tmp_path / "extras.whl"
    wheel.write_bytes(payload)
    metadata: Final = {
        "urls": [
            {
                "packagetype": "bdist_wheel",
                "url": "https://files.pythonhosted.org/extras.whl",
                "digests": {"sha256": hashlib.sha256(payload).hexdigest()},
            }
        ]
    }
    monkeypatch.setattr(check_proxy_ui_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["check_proxy_ui_release", *(["--published"] if published else ["--extras-wheel", str(wheel)])]
    )
    with patch("urllib.request.urlopen", side_effect=(io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(payload))):
        if current:
            check_proxy_ui_release.main()
            assert "passed for extras 1.0" in capsys.readouterr().out
        else:
            with pytest.raises(ValueError, match="differs"):
                check_proxy_ui_release.main()
