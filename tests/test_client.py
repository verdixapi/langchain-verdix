import asyncio
import math
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from eth_account import Account

from langchain_verdix import VerdixClient, VerdixError, tiers_within_cap
from tests.mock_api import (
    ACCOUNT,
    ADDRESS,
    API_URL,
    PAY_TO,
    TX_HASH,
    MockApi,
    decode,
    option,
    settled_response,
    verdict_body,
)


def make_client(api: MockApi, **overrides) -> VerdixClient:
    return VerdixClient(**{"max_price_per_call_usd": 0.1, **api.client_kwargs(), **overrides})


def test_pays_for_the_requested_tier_at_its_own_url():
    api = MockApi()
    verdix = make_client(api)

    result = verdix.check_address(ADDRESS, "quick")

    assert [(c.method, c.url, c.paid) for c in api.calls] == [
        ("POST", f"{API_URL}/risk/address/quick", False),
        ("POST", f"{API_URL}/risk/address/quick", True),
    ]
    # The URL selects the tier; the body does not repeat it.
    assert api.calls[1].body == {"address": ADDRESS, "chain": "base"}
    assert result.address == ADDRESS
    assert result.tier == "quick"
    assert result.verdict == "safe"
    assert result.risk_score == 5
    assert result.checked == ["ofac", "scam_lists"]
    assert result.as_of == "2026-10-02T00:00:00+00:00"
    assert result.price_usd == 0.02
    assert result.complete is True
    assert result.charged is True
    assert result.payment is not None
    assert result.payment.transaction == TX_HASH
    assert result.payment.network == "eip155:8453"
    assert verdix.spent_usd == 0.02


def test_signs_exactly_the_quoted_terms():
    api = MockApi()
    make_client(api).check_address(ADDRESS, "standard")

    payload = decode(api.calls[1].headers["payment-signature"])
    assert payload["x402Version"] == 2
    assert payload["accepted"]["extra"]["tier"] == "standard"
    # The quote's resource goes back to the server unchanged.
    assert payload["resource"]["url"] == f"{API_URL}/risk/address/standard"
    authorization = payload["payload"]["authorization"]
    assert authorization["from"] == ACCOUNT.address
    assert authorization["to"].lower() == PAY_TO.lower()
    assert authorization["value"] == "100000"
    assert payload["payload"]["signature"].startswith("0x")


def test_defaults_to_the_standard_tier():
    api = MockApi()
    make_client(api).check_address(ADDRESS)
    assert api.calls[0].url == f"{API_URL}/risk/address/standard"


def test_refuses_a_price_above_the_per_call_cap_before_signing():
    api = MockApi()
    verdix = make_client(api, max_price_per_call_usd=0.05)

    with pytest.raises(
        VerdixError,
        match=r"Refusing to pay: the standard tier costs \$0\.10, above the \$0\.05 per-call cap",
    ):
        verdix.check_address(ADDRESS, "standard")
    assert api.paid_calls == []
    assert verdix.spent_usd == 0


def test_pays_above_the_x402_library_default_cap_when_the_caller_allows_it():
    api = MockApi(quote=lambda tier: option(tier, amount="1500000"))
    result = make_client(api, max_price_per_call_usd=2).check_address(ADDRESS, "deep")
    assert result.charged is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"network": "eip155:1"}, "expected USDC on Base"),
        ({"asset": "0x4444444444444444444444444444444444444444"}, "expected USDC on Base"),
        ({"scheme": "upto"}, "no payment option for the quick tier"),
    ],
    ids=["another network", "another asset", "another scheme"],
)
def test_refuses_unexpected_payment_terms(overrides, message):
    api = MockApi(quote=lambda tier: option(tier, **overrides))
    with pytest.raises(VerdixError, match=message):
        make_client(api).check_address(ADDRESS, "quick")
    assert api.paid_calls == []


def test_refuses_a_quote_for_a_different_tier_than_the_url_asked_for():
    api = MockApi(quote=lambda tier: option("quick"))
    with pytest.raises(VerdixError, match="no payment option for the standard tier"):
        make_client(api).check_address(ADDRESS, "standard")
    assert api.paid_calls == []


def test_stops_at_the_total_budget():
    api = MockApi()
    verdix = make_client(api, max_total_spend_usd=0.05)

    verdix.check_address(ADDRESS, "quick")
    verdix.check_address(ADDRESS, "quick")
    with pytest.raises(
        VerdixError,
        match=(
            r"Refusing to pay: the quick tier costs \$0\.02 and \$0\.01 of the \$0\.05 "
            r"budget \(max_total_spend_usd\) is left"
        ),
    ):
        verdix.check_address(ADDRESS, "quick")
    assert len(api.paid_calls) == 2
    assert verdix.spent_usd == 0.04


async def test_reserves_the_budget_for_concurrent_async_checks():
    api = MockApi()
    verdix = make_client(api, max_total_spend_usd=0.02)

    results = await asyncio.gather(
        verdix.acheck_address(ADDRESS, "quick"),
        verdix.acheck_address(ADDRESS, "quick"),
        return_exceptions=True,
    )
    assert sorted(type(r).__name__ for r in results) == ["VerdixCheckResult", "VerdixError"]
    assert len(api.paid_calls) == 1
    assert verdix.spent_usd == 0.02


def test_reserves_the_budget_for_concurrent_threads():
    api = MockApi()
    verdix = make_client(api, max_total_spend_usd=0.1)

    def check(_):
        try:
            return verdix.check_address(ADDRESS, "quick")
        except VerdixError as error:
            return error

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(check, range(10)))
    assert sum(isinstance(r, VerdixError) for r in results) == 5
    assert len(api.paid_calls) == 5
    assert verdix.spent_usd == pytest.approx(0.1)


def test_returns_an_incomplete_unpaid_answer_on_503_and_releases_the_budget():
    api = MockApi(
        paid_response=lambda tier: httpx.Response(
            503, json=verdict_body(tier, "caution"), headers={"retry-after": "60"}
        )
    )
    verdix = make_client(api, max_total_spend_usd=1)

    result = verdix.check_address(ADDRESS, "quick")

    assert result.verdict == "caution"
    assert result.complete is False
    assert result.charged is False
    assert result.payment is None
    assert result.retry_after_seconds == 60
    assert verdix.spent_usd == 0


def test_returns_a_free_degraded_answer_without_signing():
    api = MockApi(
        unpaid_response=lambda tier: httpx.Response(
            200,
            json=verdict_body(tier, "caution"),
            headers={"x-verdix-degraded": "poisoning_watch"},
        )
    )
    verdix = make_client(api)

    result = verdix.check_address(ADDRESS, "quick")

    assert len(api.calls) == 1
    assert result.verdict == "caution"
    assert result.complete is False
    assert result.charged is False
    assert verdix.spent_usd == 0


def test_counts_a_failed_settlement_as_no_charge():
    api = MockApi(paid_response=lambda tier: settled_response(verdict_body(tier), success=False))
    verdix = make_client(api)

    result = verdix.check_address(ADDRESS, "quick")

    assert result.charged is False
    assert verdix.spent_usd == 0


def test_raises_when_the_payment_is_rejected_and_releases_the_budget():
    api = MockApi(paid_response=lambda tier: httpx.Response(402, json={}))
    verdix = make_client(api)
    with pytest.raises(VerdixError, match=r"did not accept the payment \(HTTP 402\)"):
        verdix.check_address(ADDRESS, "quick")
    assert verdix.spent_usd == 0


def test_passes_the_api_error_detail_through():
    api = MockApi(paid_response=lambda tier: httpx.Response(422, json={"detail": "bad chain"}))
    with pytest.raises(VerdixError, match="Verdix returned HTTP 422: bad chain"):
        make_client(api).check_address(ADDRESS, "quick")


def test_keeps_the_reservation_when_the_network_fails_after_signing():
    api = MockApi()
    original = api.handler

    def flaky(request):
        if "payment-signature" in request.headers:
            raise httpx.ConnectError("connection reset")
        return original(request)

    verdix = VerdixClient(
        account=ACCOUNT,
        max_price_per_call_usd=0.1,
        api_url=API_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(flaky)),
    )
    with pytest.raises(VerdixError, match="Verdix request failed: connection reset"):
        verdix.check_address(ADDRESS, "quick")
    # The signed payment may still settle, so it stays counted.
    assert verdix.spent_usd == 0.02


@pytest.mark.parametrize("address", ["", "0x123", "1111111111111111111111111111111111111111", None])
def test_rejects_a_malformed_address_without_any_request(address):
    api = MockApi()
    with pytest.raises(VerdixError, match="address must be 0x followed by 40 hex characters"):
        make_client(api).check_address(address, "quick")
    assert api.calls == []


def test_rejects_an_unknown_tier_without_any_request():
    api = MockApi()
    with pytest.raises(VerdixError, match="tier must be one of quick, standard, deep"):
        make_client(api).check_address(ADDRESS, "premium")
    assert api.calls == []


@pytest.mark.parametrize("cap", [-1, math.nan, math.inf, "0.1", True, None])
def test_rejects_a_bad_per_call_cap(cap):
    with pytest.raises(VerdixError, match="max_price_per_call_usd must be a non-negative number"):
        VerdixClient(account=ACCOUNT, max_price_per_call_usd=cap)


def test_rejects_a_bad_total_budget():
    with pytest.raises(VerdixError, match="max_total_spend_usd must be a non-negative number"):
        VerdixClient(account=ACCOUNT, max_price_per_call_usd=0.1, max_total_spend_usd=-5)


@pytest.mark.parametrize("account", [None, "0xabc", Account.create().key])
def test_requires_a_signer(account):
    with pytest.raises(VerdixError, match="account must be a signer"):
        VerdixClient(account=account, max_price_per_call_usd=0.1)


def test_identifies_itself_in_the_user_agent():
    verdix = VerdixClient(account=ACCOUNT, max_price_per_call_usd=0.1)
    assert verdix._http.headers["user-agent"].startswith("langchain-verdix/")
    assert verdix._async_http.headers["user-agent"].startswith("langchain-verdix/")


def test_get_pricing_reads_the_unpaid_quotes():
    api = MockApi()
    quotes = make_client(api).get_pricing()

    assert [(q.tier, q.price_usd, q.within_cap) for q in quotes] == [
        ("quick", 0.02, True),
        ("standard", 0.1, True),
        ("deep", 0.5, False),
    ]
    assert all(q.pay_to == PAY_TO and q.network == "eip155:8453" for q in quotes)
    assert [(c.method, c.paid) for c in api.calls] == [("GET", False)] * 3


async def test_aget_pricing_reads_the_unpaid_quotes():
    api = MockApi()
    quotes = await make_client(api).aget_pricing()
    assert [q.tier for q in quotes] == ["quick", "standard", "deep"]
    assert api.paid_calls == []


def test_tiers_within_cap():
    assert tiers_within_cap(0.01) == []
    assert tiers_within_cap(0.02) == ["quick"]
    assert tiers_within_cap(0.1) == ["quick", "standard"]
    assert tiers_within_cap(5) == ["quick", "standard", "deep"]
