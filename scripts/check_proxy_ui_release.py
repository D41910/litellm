"""Check the dashboard version and the contents of its release wheel."""

import argparse
import hashlib
import io
import json
import subprocess
import tomllib
import urllib.request
from pathlib import Path
from typing import Final
from zipfile import ZipFile

ROOT: Final = Path(__file__).resolve().parents[1]
ASSETS: Final = Path("litellm-proxy-extras/litellm_proxy_extras/ui")


def check_version(root: Path, base: str | None) -> str:
    project: Final = tomllib.loads((root / "pyproject.toml").read_text())
    extras: Final = tomllib.loads((root / "litellm-proxy-extras/pyproject.toml").read_text())
    version: Final[str] = extras["project"]["version"]
    if f"litellm-proxy-extras=={version}" not in project["project"]["optional-dependencies"]["proxy"]:
        raise ValueError("The proxy dependency must pin the current extras version")
    if base is not None:
        changed: Final = subprocess.check_output(
            ["git", "diff", "--name-only", base, "--", str(ASSETS)], cwd=root, text=True
        ).strip()
        previous: Final = tomllib.loads(
            subprocess.check_output(["git", "show", f"{base}:litellm-proxy-extras/pyproject.toml"], cwd=root, text=True)
        )
        if changed and previous["project"]["version"] == version:
            raise ValueError("Dashboard changes require a new proxy extras version and core pin")
    return version


def check_wheel(root: Path, wheel: bytes, version: str) -> None:
    with ZipFile(io.BytesIO(wheel)) as archive:
        metadata: Final = archive.read(f"litellm_proxy_extras-{version}.dist-info/METADATA").decode()
        if f"\nVersion: {version}\n" not in metadata:
            raise ValueError("Extras wheel version does not match the core pin")
        expected: Final = {
            f"litellm_proxy_extras/ui/{path.relative_to(root / ASSETS).as_posix()}": path.read_bytes()
            for path in (root / ASSETS).rglob("*")
            if path.is_file()
        }
        actual: Final = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("litellm_proxy_extras/ui/") and not name.endswith("/")
        }
        if not expected or actual != expected:
            raise ValueError("Extras wheel dashboard differs from the release source")


def published_wheel(version: str) -> bytes:
    with urllib.request.urlopen(f"https://pypi.org/pypi/litellm-proxy-extras/{version}/json", timeout=30) as response:
        metadata: Final = json.load(response)
    wheels: Final = tuple(item for item in metadata["urls"] if item["packagetype"] == "bdist_wheel")
    if len(wheels) != 1:
        raise ValueError("Expected one published universal proxy extras wheel")
    artifact: Final = wheels[0]
    if not artifact["url"].startswith("https://files.pythonhosted.org/"):
        raise ValueError("Unexpected PyPI artifact host")
    with urllib.request.urlopen(artifact["url"], timeout=60) as response:
        data: Final = response.read()
    if hashlib.sha256(data).hexdigest() != artifact["digests"]["sha256"]:
        raise ValueError("Published extras wheel checksum mismatch")
    return data


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    source: Final = parser.add_mutually_exclusive_group()
    source.add_argument("--extras-wheel", type=Path)
    source.add_argument("--published", action="store_true")
    args: Final = parser.parse_args()
    version: Final = check_version(ROOT, args.base)
    if args.published:
        check_wheel(ROOT, published_wheel(version), version)
    elif args.extras_wheel is not None:
        check_wheel(ROOT, args.extras_wheel.read_bytes(), version)
    print(f"Proxy UI release check passed for extras {version}")


if __name__ == "__main__":
    main()
