from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.internal_call_metadata import EvaluationBillingOwner
from litellm.proxy._types import Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    ResolvedModelBudget,
    model_budget_spend_cache_key,
    model_budget_start_time_cache_key,
    resolve_model_budget,
)
from litellm.proxy.spend_tracking import budget_reservation
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_input_cost,
    estimate_request_max_cost,
    reserve_budget_for_request,
)
from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes

_NUMBER: Final = TypeAdapter(float)
_METADATA: Final = TypeAdapter(Mapping[str, object])
_REQUEST: Final = TypeAdapter(dict[str, object])
_RECONCILE: Final = TypeAdapter[Callable[[dict[str, object] | None, float | None], Awaitable[object]]](
    Callable[[dict[str, object] | None, float | None], Awaitable[object]]
).validate_python(
    budget_reservation.reconcile_budget_reservation  # pyright: ignore[reportUnknownMemberType]  # legacy reservation parameter is untyped
)


@runtime_checkable
class _NumericCache(Protocol):
    async def async_get_cache(self, key: str) -> object: ...

    async def async_set_cache(self, key: str, value: float, *, ttl: int, nx: bool = False) -> object: ...

    async def async_increment(self, key: str, value: float, *, ttl: int) -> float: ...


_NUMERIC_CACHE: Final = TypeAdapter(_NumericCache, config=ConfigDict(arbitrary_types_allowed=True))


async def _model_window(cache: DualCache, start_key: str, duration: int) -> tuple[float, int]:
    now: Final = time.time()
    redis: Final = cache.redis_cache
    if redis is not None:
        shared: Final = _NUMERIC_CACHE.validate_python(redis)
        await shared.async_set_cache(key=start_key, value=now, ttl=duration, nx=True)
        shared_start: Final = _NUMBER.validate_python(await shared.async_get_cache(key=start_key))
        return shared_start, max(1, math.ceil(duration - (now - shared_start)))
    local: Final = _NUMERIC_CACHE.validate_python(cache.in_memory_cache)
    cached: Final = await local.async_get_cache(start_key)
    if cached is None:
        await local.async_set_cache(key=start_key, value=now, ttl=duration)
    start: Final = _NUMBER.validate_python(now if cached is None else cached)
    return start, max(1, math.ceil(duration - (now - start)))


async def _increment_model(cache: DualCache, key: str, amount: float, ttl: int) -> float:
    backend: Final = _NUMERIC_CACHE.validate_python(cache.redis_cache or cache.in_memory_cache)
    return _NUMBER.validate_python(await backend.async_increment(key=key, value=amount, ttl=ttl))


@dataclass(slots=True)
class EvaluationModelReservation:
    cache: DualCache
    spend_key: str
    start_key: str
    duration: int
    window_start: float
    reserved_cost: float
    settled_cost: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def settle(self, actual_cost: float) -> None:
        async with self.lock:
            if self.settled_cost is not None and actual_cost <= self.settled_cost:
                return
            start, ttl = await _model_window(self.cache, self.start_key, self.duration)
            prior: Final = (
                (self.settled_cost if self.settled_cost is not None else self.reserved_cost)
                if start == self.window_start
                else 0.0
            )
            adjustment: Final = actual_cost - prior
            await _increment_model(self.cache, self.spend_key, adjustment, ttl)
            self.window_start = start
            self.settled_cost = actual_cost


@dataclass(slots=True)
class EvaluationBudgetReservation:
    total: dict[str, object] | None  # mutable-ok: shared reservation is finalized by the spend writer
    model: EvaluationModelReservation | None
    input_cost: float = 0.0
    settled_failure_cost: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def settle_failure(self, actual_cost: float) -> None:
        async with self.lock:
            cost: Final = max(actual_cost, self.settled_failure_cost)
            total: Final = {**self.total, "finalized": False} if self.total is not None else None
            await _RECONCILE(total, cost)
            if self.total is not None:
                self.total["finalized"] = True
            if self.model is not None:
                await self.model.settle(cost)
            self.settled_failure_cost = cost


async def _reserve_model_budget(
    owner: EvaluationBillingOwner,
    resolved: ResolvedModelBudget,
    estimated_cost: float,
    cache: DualCache,
) -> EvaluationModelReservation:
    duration: Final = duration_in_seconds(str(resolved.budget_config.budget_duration))
    spend_key: Final = model_budget_spend_cache_key(
        Litellm_EntityType.USER, owner.user_id, resolved.budget_model, resolved.budget_config.budget_duration
    )
    start_key: Final = model_budget_start_time_cache_key(
        Litellm_EntityType.USER, owner.user_id, resolved.budget_model, resolved.budget_config.budget_duration
    )
    start, ttl = await _model_window(cache, start_key, duration)
    reservation: Final = EvaluationModelReservation(cache, spend_key, start_key, duration, start, estimated_cost)
    current: Final = await _increment_model(cache, spend_key, estimated_cost, ttl)
    limit: Final = _NUMBER.validate_python(resolved.budget_config.max_budget)
    if current - estimated_cost >= limit or current > limit:
        await reservation.settle(0.0)
        raise litellm.BudgetExceededError(
            current_cost=current - estimated_cost,
            max_budget=limit,
            entity_type=Litellm_EntityType.USER.value,
            entity_id=owner.user_id,
        )
    return reservation


async def reserve_evaluation_budget(
    owner: EvaluationBillingOwner, request: Mapping[str, object], call_type: str
) -> EvaluationBudgetReservation | None:
    from litellm.proxy.proxy_server import (
        get_current_spend,
        model_max_budget_limiter,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    metadata: Final = _METADATA.validate_python(request.get("litellm_metadata") or request.get("metadata") or {})
    model: Final = str(metadata.get("model_group") or request["model"])
    resolved: Final = resolve_model_budget(model, owner.user_model_max_budget or {})
    model_budget: Final = (
        resolved
        if resolved is not None
        and resolved.budget_config.max_budget is not None
        and math.isfinite(resolved.budget_config.max_budget)
        and resolved.budget_config.max_budget >= 0
        else None
    )
    total_budget: Final = owner.max_budget is not None and math.isfinite(owner.max_budget)
    if not total_budget and model_budget is None:
        return None
    if total_budget:
        current: Final = await get_current_spend(
            counter_key=f"spend:user:{owner.user_id}", fallback_spend=owner.spend, max_budget=owner.max_budget
        )
        if current >= _NUMBER.validate_python(owner.max_budget):
            raise litellm.BudgetExceededError(
                current_cost=current,
                max_budget=_NUMBER.validate_python(owner.max_budget),
                entity_type=Litellm_EntityType.USER.value,
                entity_id=owner.user_id,
            )
    route: Final = next(route for route, types in API_ROUTE_TO_CALL_TYPES.items() if CallTypes(call_type) in types)
    body: Final = _REQUEST.validate_python(
        {
            **request,
            "metadata": {},
            "litellm_metadata": {},
            "tags": [],
        }
    )
    total: Final = await reserve_budget_for_request(
        request_body=body,
        route=route,
        llm_router=None,
        valid_token=UserAPIKeyAuth(user_id=owner.user_id),
        team_object=None,
        user_object=LiteLLM_UserTable(user_id=owner.user_id, max_budget=owner.max_budget, spend=owner.spend),
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        fail_closed_budget_enforcement=True,
    )
    try:
        estimate: Final = (
            _NUMBER.validate_python(total["reserved_cost"])
            if total is not None
            else estimate_request_max_cost(body, route, llm_router=None)
        )
        if estimate is None:
            raise ValueError("Evaluation budget cannot be checked for an unpriced model")
        model_reservation: Final = (
            await _reserve_model_budget(owner, model_budget, estimate, model_max_budget_limiter.dual_cache)
            if model_budget is not None
            else None
        )
    except BaseException:
        await asyncio.shield(_RECONCILE(total, 0.0))
        raise
    input_cost: Final = (
        _NUMBER.validate_python(total["input_cost"])
        if total is not None
        else estimate_request_input_cost(body, route, None) or 0.0
    )
    return EvaluationBudgetReservation(total, model_reservation, input_cost)


async def release_evaluation_budget(
    reservation: EvaluationBudgetReservation | None, *, cancelled: bool = False, actual_cost: float = 0.0
) -> None:
    if reservation is None:
        return
    incurred: Final = max(reservation.input_cost if cancelled else 0.0, actual_cost)
    await reservation.settle_failure(incurred)
