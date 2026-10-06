"""A Verdix API client that pays for each check via x402, within caps you set."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, TypeVar

import httpx
from x402 import x402ClientSync
from x402.http.utils import (
    decode_payment_required_header,
    decode_payment_response_header,
    encode_payment_signature_header,
)
from x402.mechanisms.evm.exact import register_exact_evm_client
from x402.schemas import PaymentRequired, PaymentRequirements

from langchain_verdix._constants import (
    BASE_MAINNET,
    DEFAULT_API_URL,
    LITE_TIER,
    TIER_LIST_PRICES_USD,
    USDC_BASE,
    USDC_DECIMALS,
    VERDIX_TIERS,
    VerdixLiteTier,
    VerdixTier,
)
from langchain_verdix._version import __version__
from langchain_verdix.errors import VerdixError

VerdixVerdict = Literal["safe", "caution", "danger"]

VerdixLiteVerdict = Literal["no_known_risk", "caution", "danger"]
"""The lite tier's verdicts: its clean answer is `no_known_risk`, never `safe`."""

ADDRESS_PATTERN = re.compile(r"^0x[0-9a-fA-F]{40}$")

# The paid request waits for the facilitator to settle on Base, which takes
# longer than httpx's 5-second default.
DEFAULT_TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class VerdixPayment:
    """A settled x402 payment."""

    transaction: str
    """Settlement transaction hash on Base."""
    network: str
    payer: str | None = None
    """The paying wallet."""


@dataclass(frozen=True)
class VerdixCheckResult:
    """The answer to one address check."""

    address: str
    chain: str
    tier: VerdixTier
    verdict: VerdixVerdict
    """`safe`: none of the checks in `checked` found a risk signal (not a
    guarantee). `caution`: some risk signals, or a check could not complete.
    `danger`: strong risk signals (sanctions, known scam, poisoning lookalike,
    burn address...); do not send funds."""
    risk_score: int
    """0 (no signals) to 100 (highest risk)."""
    reasons: list[str]
    checked: list[str]
    as_of: str
    price_usd: float
    complete: bool
    """False when a data source failed or Verdix runs in degraded mode: the
    verdict is then never `safe`, you are not charged, and you can retry
    (after `retry_after_seconds`, when given)."""
    charged: bool
    """Whether a payment settled for this check."""
    payment: VerdixPayment | None
    retry_after_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerdixLiteCheckResult:
    """The answer to one lite check ($0.01). Its verdict is never `safe`."""

    address: str
    chain: str
    tier: VerdixLiteTier
    verdict: VerdixLiteVerdict
    """`no_known_risk`: the address is on none of the lists lite checks. This
    is not a safety verdict; use the quick tier for one. `caution`: some risk
    signals, or a check could not complete. `danger`: strong risk signals; do
    not send funds."""
    risk_score: int
    reasons: list[str]
    checked: list[str]
    as_of: str
    price_usd: float
    limited_checks: bool
    """Always true: lite runs only some of the checks."""
    checks_performed: list[str]
    not_checked: list[str]
    """Checks lite skipped (address age, caution-only contract checks, and any
    best-effort check that could not finish)."""
    full_check: str
    """The API's pointer to the quick tier, for a check that can say `safe`."""
    complete: bool
    charged: bool
    payment: VerdixPayment | None
    retry_after_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerdixTierQuote:
    """The live price of one tier, from the API's unpaid 402 quote."""

    tier: VerdixTier | VerdixLiteTier
    price_usd: float
    network: str
    asset: str
    pay_to: str
    within_cap: bool


def _usd_to_atomic(usd: float) -> int:
    return int((Decimal(str(usd)) * 10**USDC_DECIMALS).to_integral_value(rounding=ROUND_HALF_UP))


def _atomic_to_usd(amount: int | str) -> float:
    return int(amount) / float(10**USDC_DECIMALS)


def _format_usd(usd: float) -> str:
    return f"${usd:.2f}"


def _format_cap(usd: float) -> str:
    """Like `_format_usd`, but a sub-cent cap (e.g. 0.005) is shown as is, so
    "costs $0.01, above the $0.01 per-call cap" can't happen."""
    return _format_usd(usd) if round(usd, 2) == usd else f"${usd}"


def _check_usd_amount(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value != value  # NaN
        or value in (float("inf"), float("-inf"))
        or value < 0
    ):
        raise VerdixError(f"{name} must be a non-negative number of US dollars")
    return float(value)


def _parse_quote(response: httpx.Response) -> PaymentRequired | None:
    header = response.headers.get("payment-required")
    if not header:
        return None
    try:
        quote = decode_payment_required_header(header)
    except Exception as error:  # noqa: BLE001 - any malformed quote is refused
        raise VerdixError("Verdix sent a payment quote this client cannot read") from error
    # Only x402 v2 quotes carry the per-tier options this client checks.
    return quote if isinstance(quote, PaymentRequired) else None


def _find_tier(
    quote: PaymentRequired | None, tier: VerdixTier | VerdixLiteTier
) -> PaymentRequirements | None:
    accepts = quote.accepts if quote is not None else []
    return next((option for option in accepts if (option.extra or {}).get("tier") == tier), None)


def _assert_acceptable(
    option: PaymentRequirements | None,
    tier: VerdixTier | VerdixLiteTier,
    max_price_per_call_usd: float,
) -> PaymentRequirements:
    """Raises unless `option` is exactly what this client agreed to pay: the
    requested tier, USDC on Base, `exact` scheme, at or under the cap."""
    if option is None or option.scheme != "exact":
        raise VerdixError(f"Verdix offered no payment option for the {tier} tier")
    if option.network != BASE_MAINNET or option.asset.lower() != USDC_BASE:
        raise VerdixError(
            f"Refusing to pay: expected USDC on Base ({BASE_MAINNET}), "
            f"got {option.asset} on {option.network}"
        )
    if int(option.amount) > _usd_to_atomic(max_price_per_call_usd):
        raise VerdixError(
            f"Refusing to pay: the {tier} tier costs {_format_usd(_atomic_to_usd(option.amount))}, "
            f"above the {_format_cap(max_price_per_call_usd)} per-call cap (max_price_per_call_usd)"
        )
    return option


def _payment_from(response: httpx.Response) -> VerdixPayment | None:
    receipt = response.headers.get("payment-response") or response.headers.get("x-payment-response")
    if not receipt:
        return None
    try:
        settled = decode_payment_response_header(receipt)
    except Exception:  # noqa: BLE001 - an unreadable receipt counts as no charge
        return None
    if not settled.success:
        return None
    return VerdixPayment(
        transaction=settled.transaction, network=settled.network, payer=settled.payer
    )


def _to_check_result(
    body: dict[str, Any],
    complete: bool,
    retry_after: str | None,
    payment: VerdixPayment | None,
) -> VerdixCheckResult:
    verdict = body.get("verdict")
    if verdict not in ("safe", "caution", "danger"):
        raise VerdixError("Unexpected answer from Verdix: no verdict")
    try:
        retry_after_seconds: float | None = float(retry_after) if retry_after else None
    except ValueError:
        retry_after_seconds = None
    reasons = body.get("reasons")
    checked = body.get("checked")
    return VerdixCheckResult(
        address=str(body.get("address")),
        chain=str(body.get("chain")),
        tier=body.get("tier"),  # type: ignore[arg-type]
        verdict=verdict,
        risk_score=int(body.get("risk_score") or 0),
        reasons=[str(reason) for reason in reasons] if isinstance(reasons, list) else [],
        checked=[str(check) for check in checked] if isinstance(checked, list) else [],
        as_of=str(body.get("as_of")),
        price_usd=float(body.get("price_usd") or 0),
        complete=complete,
        charged=payment is not None,
        payment=payment,
        retry_after_seconds=retry_after_seconds,
    )


def _to_lite_result(
    body: dict[str, Any],
    complete: bool,
    retry_after: str | None,
    payment: VerdixPayment | None,
) -> VerdixLiteCheckResult:
    verdict = body.get("verdict")
    # Lite never says "safe"; an answer that does is not passed on as one.
    if verdict not in ("no_known_risk", "caution", "danger"):
        raise VerdixError("Unexpected answer from Verdix's lite tier: no lite verdict")
    try:
        retry_after_seconds: float | None = float(retry_after) if retry_after else None
    except ValueError:
        retry_after_seconds = None

    def strings(key: str) -> list[str]:
        value = body.get(key)
        return [str(item) for item in value] if isinstance(value, list) else []

    return VerdixLiteCheckResult(
        address=str(body.get("address")),
        chain=str(body.get("chain")),
        tier=LITE_TIER,
        verdict=verdict,
        risk_score=int(body.get("risk_score") or 0),
        reasons=strings("reasons"),
        checked=strings("checked"),
        as_of=str(body.get("as_of")),
        price_usd=float(body.get("price_usd") or 0),
        limited_checks=True,
        checks_performed=strings("checks_performed"),
        not_checked=strings("not_checked"),
        full_check=str(body.get("full_check") or ""),
        complete=complete,
        charged=payment is not None,
        payment=payment,
        retry_after_seconds=retry_after_seconds,
    )


_Result = TypeVar("_Result", VerdixCheckResult, VerdixLiteCheckResult)


@dataclass
class _Attempt:
    """One check in flight: what was reserved against the budget."""

    tier: VerdixTier | VerdixLiteTier
    reserved: int = 0
    payment_headers: dict[str, str] = field(default_factory=dict)


class VerdixClient:
    """Pays for each Verdix check via x402 from `account`, within the caps you set.

    Args:
        account: The wallet that pays, e.g. `Account.from_key(key)` from
            `eth_account`, or any x402 `ClientEvmSigner`. It only signs the
            x402 payment locally; the key never leaves your process. It needs a
            little USDC on Base mainnet (no ETH: the facilitator pays gas).
        max_price_per_call_usd: Hard cap, in USD, on what a single check may
            cost. A tier whose live price is above it is refused before
            anything is signed.
        max_total_spend_usd: Optional total budget, in USD, for everything this
            client pays over its lifetime. A payment is reserved against it
            before signing and released again if the API reports it was not
            charged.
        api_url: Verdix API base URL. Defaults to https://api.verdixapi.com.
        http_client: Custom `httpx.Client` (for tests, proxies or
            instrumentation).
        async_http_client: Custom `httpx.AsyncClient`, used by the async methods.
        timeout: Request timeout in seconds when the client creates its own
            httpx clients.
    """

    def __init__(
        self,
        *,
        account: Any,
        max_price_per_call_usd: float,
        max_total_spend_usd: float | None = None,
        api_url: str | None = None,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if account is None or not callable(getattr(account, "sign_typed_data", None)):
            raise VerdixError(
                "account must be a signer, e.g. Account.from_key(key) from eth_account"
            )
        self.max_price_per_call_usd = _check_usd_amount(
            "max_price_per_call_usd", max_price_per_call_usd
        )
        self.max_total_spend_usd = (
            None
            if max_total_spend_usd is None
            else _check_usd_amount("max_total_spend_usd", max_total_spend_usd)
        )
        self.api_url = (api_url or DEFAULT_API_URL).rstrip("/")

        self._budget = (
            None if self.max_total_spend_usd is None else _usd_to_atomic(self.max_total_spend_usd)
        )
        self._spent = 0
        self._lock = threading.Lock()

        self._x402 = x402ClientSync(
            payment_requirements_selector=lambda _version, accepts: accepts[0]
        )
        # _prepare_payment makes every check the library's spend controls would
        # (asset allow-list, per-payment cap), against the caller's own cap.
        # Left on, the library's default $1 cap would silently override a
        # higher cap and hide the refusal reason.
        self._x402.set_spend_controls(False)
        register_exact_evm_client(self._x402, account, networks=BASE_MAINNET)

        headers = {"User-Agent": f"langchain-verdix/{__version__}"}
        self._http = http_client or httpx.Client(timeout=timeout, headers=headers)
        self._async_http = async_http_client or httpx.AsyncClient(timeout=timeout, headers=headers)

    @property
    def spent_usd(self) -> float:
        """USD paid (or reserved for an in-flight check) so far by this client."""
        return _atomic_to_usd(self._spent)

    def _tier_url(self, tier: VerdixTier | VerdixLiteTier) -> str:
        return f"{self.api_url}/risk/address/{tier}"

    @staticmethod
    def _validate_address(address: str) -> None:
        if not isinstance(address, str) or not ADDRESS_PATTERN.match(address):
            raise VerdixError("address must be 0x followed by 40 hex characters")

    @classmethod
    def _validate(cls, address: str, tier: VerdixTier) -> None:
        cls._validate_address(address)
        if tier not in VERDIX_TIERS:
            raise VerdixError(f"tier must be one of {', '.join(VERDIX_TIERS)}")

    def _prepare_payment(self, attempt: _Attempt, quote: httpx.Response) -> None:
        """Checks the 402 quote, reserves the price against the budget and signs."""
        payment_required = _parse_quote(quote)
        option = _assert_acceptable(
            _find_tier(payment_required, attempt.tier),
            attempt.tier,
            self.max_price_per_call_usd,
        )
        amount = int(option.amount)
        with self._lock:
            # Reserved under the lock, so concurrent checks cannot overspend.
            if self._budget is not None and self._spent + amount > self._budget:
                raise VerdixError(
                    f"Refusing to pay: the {attempt.tier} tier costs "
                    f"{_format_usd(_atomic_to_usd(amount))} and "
                    f"{_format_usd(_atomic_to_usd(self._budget - self._spent))} of the "
                    f"{_format_usd(_atomic_to_usd(self._budget))} budget "
                    "(max_total_spend_usd) is left"
                )
            self._spent += amount
            attempt.reserved = amount
        try:
            # Only the one vetted option is offered to the signer; the quote's
            # resource and extensions are kept, as the server expects them back.
            assert payment_required is not None
            payload = self._x402.create_payment_payload(
                payment_required.model_copy(update={"accepts": [option]})
            )
        except Exception as error:
            self._release(attempt)
            raise VerdixError(f"Could not sign the x402 payment: {error}") from error
        attempt.payment_headers = {"PAYMENT-SIGNATURE": encode_payment_signature_header(payload)}

    def _release(self, attempt: _Attempt) -> None:
        with self._lock:
            self._spent -= attempt.reserved
            attempt.reserved = 0

    def _finish(
        self,
        attempt: _Attempt,
        response: httpx.Response,
        convert: Callable[[dict[str, Any], bool, str | None, VerdixPayment | None], _Result],
    ) -> _Result:
        payment = _payment_from(response) if attempt.payment_headers else None
        # Any status >= 400, or no settled receipt, means nothing was charged.
        if not response.is_success or payment is None:
            self._release(attempt)

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None
        if not isinstance(body, dict):
            raise VerdixError(f"Unexpected answer from Verdix: HTTP {response.status_code}")
        # 503 carries a "caution" verdict body when an upstream data source
        # failed, and a degraded-mode answer comes back free: both are usable
        # (unpaid) answers resting on incomplete data, not errors.
        if response.is_success or (response.status_code == 503 and body.get("verdict")):
            return convert(
                body,
                response.is_success and not response.headers.get("x-verdix-degraded"),
                response.headers.get("retry-after"),
                payment,
            )
        if response.status_code == 402:
            raise VerdixError("Verdix did not accept the payment (HTTP 402)")
        detail = body.get("detail")
        suffix = f": {detail}" if isinstance(detail, str) else ""
        raise VerdixError(f"Verdix returned HTTP {response.status_code}{suffix}")

    @staticmethod
    def _request_failed(error: Exception) -> VerdixError:
        return VerdixError(f"Verdix request failed: {error}")

    def _check(
        self,
        address: str,
        tier: VerdixTier | VerdixLiteTier,
        convert: Callable[[dict[str, Any], bool, str | None, VerdixPayment | None], _Result],
    ) -> _Result:
        attempt = _Attempt(tier)
        # The URL selects the tier; the body carries only the address.
        body = {"address": address, "chain": "base"}
        try:
            response = self._http.post(self._tier_url(tier), json=body)
            if response.status_code == 402:
                self._prepare_payment(attempt, response)
                response = self._http.post(
                    self._tier_url(tier), json=body, headers=attempt.payment_headers
                )
        except httpx.HTTPError as error:
            # A payment signed before a network failure may still settle, so
            # the reservation is kept (conservative).
            raise self._request_failed(error) from error
        return self._finish(attempt, response, convert)

    async def _acheck(
        self,
        address: str,
        tier: VerdixTier | VerdixLiteTier,
        convert: Callable[[dict[str, Any], bool, str | None, VerdixPayment | None], _Result],
    ) -> _Result:
        attempt = _Attempt(tier)
        body = {"address": address, "chain": "base"}
        try:
            response = await self._async_http.post(self._tier_url(tier), json=body)
            if response.status_code == 402:
                self._prepare_payment(attempt, response)
                response = await self._async_http.post(
                    self._tier_url(tier), json=body, headers=attempt.payment_headers
                )
        except httpx.HTTPError as error:
            raise self._request_failed(error) from error
        return self._finish(attempt, response, convert)

    def check_address(self, address: str, tier: VerdixTier = "standard") -> VerdixCheckResult:
        """Paid address check at the given tier (default `standard`)."""
        self._validate(address, tier)
        return self._check(address, tier, _to_check_result)

    async def acheck_address(
        self, address: str, tier: VerdixTier = "standard"
    ) -> VerdixCheckResult:
        """Async version of `check_address`."""
        self._validate(address, tier)
        return await self._acheck(address, tier, _to_check_result)

    def check_address_lite(self, address: str) -> VerdixLiteCheckResult:
        """Paid lite check ($0.01) at `/risk/address/lite`.

        Its verdict is `no_known_risk`, `caution` or `danger`, never `safe`:
        `no_known_risk` only means the address is on none of lite's lists. Use
        `check_address(address, "quick")` when you need a `safe` verdict.
        """
        self._validate_address(address)
        return self._check(address, LITE_TIER, _to_lite_result)

    async def acheck_address_lite(self, address: str) -> VerdixLiteCheckResult:
        """Async version of `check_address_lite`."""
        self._validate_address(address)
        return await self._acheck(address, LITE_TIER, _to_lite_result)

    def _quote_for(
        self, tier: VerdixTier | VerdixLiteTier, response: httpx.Response
    ) -> VerdixTierQuote:
        option = _find_tier(_parse_quote(response), tier) if response.status_code == 402 else None
        if option is None:
            raise VerdixError(
                f"Unexpected pricing answer from Verdix for the {tier} tier: "
                f"HTTP {response.status_code}"
            )
        return VerdixTierQuote(
            tier=tier,
            price_usd=_atomic_to_usd(option.amount),
            network=option.network,
            asset=option.asset,
            pay_to=option.pay_to,
            within_cap=int(option.amount) <= _usd_to_atomic(self.max_price_per_call_usd),
        )

    def get_pricing(self) -> list[VerdixTierQuote]:
        """Live price of each tier, from the API's unpaid quotes. Nothing is signed."""
        try:
            return [
                self._quote_for(tier, self._http.get(self._tier_url(tier))) for tier in VERDIX_TIERS
            ]
        except httpx.HTTPError as error:
            raise self._request_failed(error) from error

    async def aget_pricing(self) -> list[VerdixTierQuote]:
        """Async version of `get_pricing`."""
        try:
            return [
                self._quote_for(tier, await self._async_http.get(self._tier_url(tier)))
                for tier in VERDIX_TIERS
            ]
        except httpx.HTTPError as error:
            raise self._request_failed(error) from error

    def get_lite_pricing(self) -> VerdixTierQuote:
        """Live price of the lite tier, from its unpaid quote. Nothing is signed.

        Kept apart from `get_pricing`, which lists quick, standard and deep only.
        """
        try:
            return self._quote_for(LITE_TIER, self._http.get(self._tier_url(LITE_TIER)))
        except httpx.HTTPError as error:
            raise self._request_failed(error) from error

    async def aget_lite_pricing(self) -> VerdixTierQuote:
        """Async version of `get_lite_pricing`."""
        try:
            response = await self._async_http.get(self._tier_url(LITE_TIER))
            return self._quote_for(LITE_TIER, response)
        except httpx.HTTPError as error:
            raise self._request_failed(error) from error


def tiers_within_cap(max_price_per_call_usd: float) -> list[VerdixTier]:
    """Tiers whose list price fits under `max_price_per_call_usd`."""
    return [tier for tier in VERDIX_TIERS if TIER_LIST_PRICES_USD[tier] <= max_price_per_call_usd]
