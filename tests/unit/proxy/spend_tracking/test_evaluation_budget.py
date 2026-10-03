from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI

import litellm
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.internal_call_metadata import (
    EVALUATION_BILLING_OWNER_KEY,
    EVALUATION_BUDGET_RESERVATION_KEY,
    EvaluationBillingOwner,
    evaluation_billing_context,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import Litellm_EntityType, LiteLLM_BudgetTable, LiteLLM_TagTable, LiteLLM_UserTable
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, tag_cache_key
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
)
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_input_cost, estimate_request_max_cost
from litellm.proxy.spend_tracking.evaluation_budget import (
    EvaluationBudgetReservation,
    release_evaluation_budget,
    reserve_evaluation_budget,
)
from litellm.types.router import RetryPolicy

MODEL: Final = "openai/evaluation-budget-fixture"
REQUEST: Final = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "model_max_budget_limiter", _PROXY_VirtualKeyModelMaxBudgetLimiter(cache))
    monkeypatch.setitem(
        litellm.model_cost,
        MODEL,
        {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "max_input_tokens": 1000,
            "max_output_tokens": 1000,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    return cache


async def _owner(scope: str, limit: float) -> EvaluationBillingOwner:
    owner: Final = EvaluationBillingOwner(
        "evaluation-admin",
        {MODEL: {"max_budget": limit, "budget_duration": "1d"}} if scope in ("model", "both") else None,
        max_budget=limit if scope in ("total", "both") else None,
    )
    await proxy_server.user_api_key_cache.async_set_cache(
        key=owner.user_id, value=LiteLLM_UserTable(user_id=owner.user_id, max_budget=owner.max_budget, spend=0.0)
    )
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("total", "model", "both"))
async def test_concurrent_evaluations_reserve_the_creators_remaining_budget(cache: DualCache, scope: str) -> None:
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert estimate is not None and estimate > 0
    owner: Final = await _owner(scope, estimate * 1.5)
    attempts: Final = await asyncio.gather(
        reserve_evaluation_budget(owner, REQUEST, "acompletion"),
        reserve_evaluation_budget(owner, REQUEST, "acompletion"),
        return_exceptions=True,
    )
    admitted: Final = tuple(item for item in attempts if isinstance(item, EvaluationBudgetReservation))
    assert len(admitted) == 1
    assert sum(isinstance(item, litellm.BudgetExceededError) for item in attempts) == 1
    reservation: Final = admitted[0]
    assert (await cache.async_get_cache("spend:user:evaluation-admin") or 0.0) == pytest.approx(
        estimate if scope in ("total", "both") else 0.0
    )
    await release_evaluation_budget(reservation)
    retried: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert retried is not None
    await release_evaluation_budget(retried)
    if reservation.model is not None:
        assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("total", "model"))
async def test_zero_creator_budget_blocks_evaluation(cache: DualCache, scope: str) -> None:
    owner: Final = await _owner(scope, 0.0)
    with pytest.raises(litellm.BudgetExceededError):
        await reserve_evaluation_budget(owner, REQUEST, "acompletion")


@pytest.mark.asyncio
async def test_evaluation_reserves_and_charges_only_creator_without_sampled_tag_budget(
    cache: DualCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner: Final = await _owner("total", 1.0)
    tag: Final = LiteLLM_TagTable(
        tag_name="sampled-tag", spend=0.3, litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0)
    )
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    await proxy_server.user_api_key_cache.async_set_cache(
        key=tag_cache_key(tag.tag_name), value=tag, model_type=LiteLLM_TagTable
    )
    await cache.async_set_cache("spend:user:evaluation-admin", 0.0)
    await cache.async_set_cache("spend:tag:sampled-tag", tag.spend)
    reservation: Final = await reserve_evaluation_budget(owner, {**REQUEST, "tags": [tag.tag_name]}, "acompletion")
    assert reservation is not None
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(estimate)
    assert await cache.async_get_cache("spend:tag:sampled-tag") == pytest.approx(tag.spend)
    await release_evaluation_budget(reservation, actual_cost=0.01)
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(0.01)
    assert await cache.async_get_cache("spend:tag:sampled-tag") == pytest.approx(tag.spend)


def _receipt(
    owner: EvaluationBillingOwner, reservation: EvaluationBudgetReservation, cost: float
) -> Mapping[str, object]:
    return {
        EVALUATION_BILLING_OWNER_KEY: owner,
        EVALUATION_BUDGET_RESERVATION_KEY: reservation,
        "litellm_params": {"metadata": {"user_api_key_user_id": "sampled-user"}},
        "standard_logging_object": {
            "model": MODEL,
            "response_cost": cost,
            "metadata": {"user_api_key_user_id": "sampled-user"},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("model", "both"))
async def test_model_callback_settles_once_and_preserves_recovered_failure_spend(cache: DualCache, scope: str) -> None:
    owner: Final = await _owner(scope, 1.0)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    limiter: Final = proxy_server.model_max_budget_limiter
    await release_evaluation_budget(reservation)
    receipt: Final = _receipt(owner, reservation, 0.003)
    await limiter.async_log_failure_event(receipt, None, None, None)
    await limiter.async_log_success_event(receipt, None, None, None)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.003)
    assert (await cache.async_get_cache("spend:user:evaluation-admin") or 0.0) == pytest.approx(
        0.003 if scope == "both" else 0.0
    )


@pytest.mark.asyncio
async def test_failed_evaluation_keeps_incurred_cost_and_frees_unused_reservation(cache: DualCache) -> None:
    owner: Final = await _owner("both", 1.0)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    await release_evaluation_budget(reservation, actual_cost=0.005)
    await release_evaluation_budget(reservation, actual_cost=0.005)
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(0.005)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_sdk_blocks_concurrent_evaluation_before_dispatch_and_settles_actual_usage(cache: DualCache) -> None:
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert estimate is not None
    owner: Final = await _owner("both", estimate * 1.5)
    litellm.logging_callback_manager.add_litellm_callback(proxy_server.model_max_budget_limiter)
    entered: Final = asyncio.Event()
    complete: Final = asyncio.Event()
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()
    receipts: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()

    async def upstream(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        entered.set()
        await complete.wait()
        return httpx.Response(
            200,
            json={
                "id": "evaluation-response",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    async def capture(kwargs: Mapping[str, object], response: object, start: datetime, end: datetime) -> None:
        receipts.put_nowait(kwargs)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)

        async def completion() -> object:
            return await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                client=client,
                api_key="transport-only",
                num_retries=0,
                success_callback=[capture],
            )

        with evaluation_billing_context(owner):
            pending: Final = asyncio.create_task(completion())
            try:
                await asyncio.wait_for(entered.wait(), 30)
                with pytest.raises(litellm.BudgetExceededError):
                    await asyncio.wait_for(completion(), 5)
            finally:
                complete.set()
                await pending
        receipt: Final = await asyncio.wait_for(receipts.get(), 30)
    reservation: Final = receipt[EVALUATION_BUDGET_RESERVATION_KEY]
    assert isinstance(reservation, EvaluationBudgetReservation) and reservation.model is not None
    actual: Final = 10 * 0.001 + 2 * 0.002
    assert receipt.get("response_cost") == pytest.approx(actual)
    assert reservation.model.settled_cost == pytest.approx(actual)
    await proxy_server.increment_spend_counters(
        token=None, team_id=None, user_id=owner.user_id, response_cost=actual, budget_reservation=reservation.total
    )
    assert requests.qsize() == 1
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(actual)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(actual)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_sdk_evaluation_failure_releases_budget_without_unreserved_retries(
    cache: DualCache, cancelled: bool
) -> None:
    owner: Final = await _owner("both", 1.0)
    entered: Final = asyncio.Event()
    blocked: Final = asyncio.Event()
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    async def upstream(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        entered.set()
        if cancelled:
            await blocked.wait()
        return httpx.Response(500, json={"error": {"message": "upstream failed", "type": "server_error"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client, max_retries=0)
        with evaluation_billing_context(owner):
            pending: Final = asyncio.create_task(
                litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    client=client,
                    api_key="transport-only",
                    num_retries=0,
                    retry_policy=RetryPolicy(InternalServerErrorRetries=2),
                )
            )
            await asyncio.wait_for(entered.wait(), 30)
            if cancelled:
                pending.cancel()
            with pytest.raises(asyncio.CancelledError if cancelled else litellm.InternalServerError):
                await asyncio.wait_for(pending, 5)
    incurred: Final = estimate_request_input_cost(REQUEST, "/chat/completions", None) if cancelled else 0.0
    assert incurred is not None
    model_key: Final = model_budget_spend_cache_key(Litellm_EntityType.USER, owner.user_id, MODEL, "1d")
    assert requests.qsize() == 1
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(incurred)
    assert await cache.async_get_cache(model_key) == pytest.approx(incurred)
