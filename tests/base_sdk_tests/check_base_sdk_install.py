"""Smoke-check that a base ``pip install litellm`` (no extras) is importable and usable.

Run against a virtualenv that has the built wheel installed with no extras, using
that venv's own interpreter and nothing else. Deliberately stdlib-only: pytest would
pull ``packaging``, ``pluggy`` and ``iniconfig`` into the environment and could mask
the very class of undeclared-dependency bug this guards against.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import traceback
from collections.abc import Callable
from functools import partial
from typing import Final

EXTRAS_ONLY_MODULES = ("fastapi", "uvicorn", "keyring", "mcp", "mcp_types", "httpx2", "httpcore2")
AWS_MODULES: Final = ("boto3", "botocore", "s3transfer", "jmespath")
TOKENIZER_MODULES: Final = ("tokenizers", "huggingface_hub", "hf_xet", "fsspec")


def check_optional_dependencies(profile: str) -> str:
    for modules, expected in (
        (AWS_MODULES, profile in ("aws", "aws,tokenizers", "sdk-extras", "proxy")),
        (TOKENIZER_MODULES, profile in ("tokenizers", "aws,tokenizers", "sdk-extras", "proxy")),
    ):
        for name in modules[:2] if expected else modules:
            present: Final = importlib.util.find_spec(name) is not None
            _require(present == expected, f"{name}: installed={present}, expected={expected} for {profile}")
    return f"optional dependencies match {profile}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_environment_is_base_only() -> str:
    present = tuple(name for name in EXTRAS_ONLY_MODULES if importlib.util.find_spec(name) is not None)
    _require(
        not present,
        f"{', '.join(present)} installed, so this environment is not base-only and the run proves nothing",
    )
    return f"no extras-only packages present ({', '.join(EXTRAS_ONLY_MODULES)})"


def check_import() -> str:
    from importlib.metadata import version

    import litellm

    _require(bool(litellm.__file__), "litellm has no __file__")
    return f"imported litellm {version('litellm')}"


def check_completion() -> str:
    import litellm

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "ping"}],
        mock_response="pong",
    )
    content = response.choices[0].message.content
    _require(content == "pong", f"mock completion returned {content!r}")
    return "mock completion round-trips"


def check_mcp_install_guidance() -> str:
    try:
        import litellm.experimental_mcp_client
    except ImportError as error:
        _require("pip install 'litellm[mcp]'" in str(error), f"missing MCP installation guidance: {error}")
        _require(isinstance(error.__cause__, ModuleNotFoundError), "original missing-dependency cause was lost")
        _require(error.__cause__.name == "mcp", f"unexpected missing dependency: {error.__cause__}")
        return "optional MCP client explains how to install litellm[mcp]"
    raise AssertionError("MCP client imported without the MCP extra")


def check_embedding() -> str:
    import litellm

    response = litellm.embedding(
        model="text-embedding-3-small",
        input=["ping"],
        mock_response=[[0.1, 0.2]],
    )
    _require(len(response.data) == 1, f"mock embedding returned {len(response.data)} rows")
    return "mock embedding round-trips"


def check_streaming() -> str:
    import litellm

    stream: Final = litellm.completion(
        model="gpt-4o", messages=[{"role": "user", "content": "ping"}], mock_response="pong", stream=True
    )
    content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    _require(content == "pong", f"stream returned {content!r}")

    async def async_round_trip() -> None:
        response: Final = await litellm.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "ping"}], mock_response="pong"
        )
        _require(response.choices[0].message.content == "pong", "async completion failed")
        chunks: Final = await litellm.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "ping"}], mock_response="pong", stream=True
        )
        parts: Final = tuple([chunk.choices[0].delta.content or "" async for chunk in chunks if chunk.choices])
        _require("".join(parts) == "pong", "async stream failed")

    asyncio.run(async_round_trip())
    return "sync/async completion and streaming round-trip"


def check_retries() -> str:
    import litellm

    outcomes: Final = iter((False, True))

    def attempt(*, max_retries: int, num_retries: int) -> str:
        if not next(outcomes):
            raise RuntimeError("transient failure")
        return "recovered"

    result: Final = litellm.completion_with_retries(original_function=attempt, num_retries=2)
    _require(result == "recovered", "configured retries did not recover")
    return "SDK retry helper recovers from a transient failure"


def check_search_date_parsing() -> str:
    from litellm.llms.brave.search.transformation import to_yyyy_mm_dd

    _require(to_yyyy_mm_dd("2026-01-02") == "2026-01-02", "search result date parsing failed")
    return "search date parsing works independently of AWS"


def check_bundled_model_metadata() -> str:
    import litellm

    max_input_tokens = litellm.get_model_info("gpt-4o")["max_input_tokens"]
    _require(
        isinstance(max_input_tokens, int) and max_input_tokens > 0,
        f"get_model_info returned max_input_tokens={max_input_tokens!r}",
    )
    prompt_cost, completion_cost = litellm.cost_per_token(model="gpt-4o", prompt_tokens=1000, completion_tokens=1000)
    _require(
        prompt_cost > 0 and completion_cost > 0,
        f"cost_per_token returned ({prompt_cost}, {completion_cost})",
    )
    return f"bundled pricing metadata readable (gpt-4o max_input_tokens={max_input_tokens})"


def check_token_counter() -> str:
    import litellm

    count = litellm.token_counter(model="gpt-4o", text="hello world")
    _require(count > 0, f"token_counter returned {count!r}")
    return f"token_counter returned {count}"


def check_bedrock_credential_resolution() -> str:
    import os
    from unittest import mock

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    non_aws_environ = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    with mock.patch.dict(os.environ, non_aws_environ, clear=True):
        credentials = BaseAWSLLM().get_credentials(
            aws_access_key_id="AKIA-fake-base-sdk-check",
            aws_secret_access_key="fake-secret",
            aws_region_name="us-east-1",
        )
    _require(
        credentials.access_key == "AKIA-fake-base-sdk-check",
        f"get_credentials returned access_key={credentials.access_key!r}",
    )
    return "bedrock credential resolution works with the AWS extra"


def check_aws_install_guidance() -> str:
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    try:
        BaseAWSLLM()._sign_request(
            service_name="bedrock",
            headers={},
            optional_params={"aws_region_name": "us-east-1"},
            request_data={},
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        )
    except ImportError as error:
        _require("litellm[aws]" in str(error), f"missing AWS installation guidance: {error}")
        return "AWS signing explains how to install litellm[aws]"
    raise AssertionError("AWS signing succeeded without the AWS extra")


def check_mantle_bearer_authentication() -> str:
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
    from litellm.llms.bedrock_mantle.common_utils import BedrockMantleAuthMixin

    signer: Final = BedrockMantleAuthMixin()
    signer._aws_signer = BaseAWSLLM()
    headers, body = signer.sign_request(
        headers={},
        optional_params={},
        request_data={"model": "example"},
        api_base="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        api_key="supplied-bearer-token",
    )
    _require(headers["Authorization"] == "Bearer supplied-bearer-token", "bearer token was not preserved")
    _require(body is not None and json.loads(body) == {"model": "example"}, "request body changed")
    return "Mantle bearer authentication does not require AWS credentials"


def check_tokenizer_fallback() -> str:
    import litellm
    from litellm.rust_bridge import tokenizer

    if importlib.util.find_spec("tokenizers") is not None:
        custom: Final = litellm.create_tokenizer(litellm.utils.claude_json_str)
        tokens: Final = litellm.encode(text="hello world", custom_tokenizer=custom)
        _require(litellm.decode(tokens=tokens, custom_tokenizer=custom) == "hello world", "tokenizer round trip failed")
        return "custom Hugging Face tokenizer round-trips"
    try:
        tokenizer._python_huggingface_tokenizer()
    except ImportError as error:
        _require("litellm[tokenizers]" in str(error), f"missing tokenizer installation guidance: {error}")
    else:
        raise AssertionError("Python Hugging Face tokenizer loaded without its extra")
    _require(litellm.token_counter(model="claude-2", text="hello world") > 0, "token counting fallback failed")
    return "missing Python tokenizer explains installation and automatic counting falls back"


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("environment is base-only", check_environment_is_base_only),
    ("import litellm", check_import),
    ("optional MCP installation guidance", check_mcp_install_guidance),
    ("chat completion", check_completion),
    ("embedding", check_embedding),
    ("streaming", check_streaming),
    ("retries", check_retries),
    ("search date parsing", check_search_date_parsing),
    ("bundled model metadata", check_bundled_model_metadata),
    ("token counter", check_token_counter),
    ("Mantle bearer authentication", check_mantle_bearer_authentication),
    ("tokenizer behavior", check_tokenizer_fallback),
)


def _run(check: Callable[[], str]) -> tuple[bool, str]:
    try:
        return True, check()
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("core", "aws", "tokenizers", "aws,tokenizers", "sdk-extras", "proxy"), default="core"
    )
    profile: Final = parser.parse_args().profile
    checks: Final = (
        ("optional dependencies", partial(check_optional_dependencies, profile)),
        *(
            check
            for check in CHECKS
            if profile != "proxy" or check[0] not in ("environment is base-only", "optional MCP installation guidance")
        ),
        (
            "AWS behavior",
            check_bedrock_credential_resolution
            if profile in ("aws", "aws,tokenizers", "sdk-extras", "proxy")
            else check_aws_install_guidance,
        ),
    )
    print(f"base SDK smoke check on {sys.executable}")
    for label, check in checks:
        passed, detail = _run(check)
        if not passed:
            print(f"FAIL  {label}:\n{detail}")
            print(f"SDK installation profile {profile} failed at: {label}")
            return 1
        print(f"PASS  {label}: {detail}")

    print(f"\nall {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
