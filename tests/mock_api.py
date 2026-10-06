"""An in-memory stand-in for the Verdix API: no network, no real payment.

Unpaid requests get a 402 quote in the live API's shape; paid ones (any x402
payment header) get a verdict and a settlement receipt.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from eth_account import Account

ADDRESS = "0x1111111111111111111111111111111111111111"
PAY_TO = "0x2222222222222222222222222222222222222222"
USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
TX_HASH = "0x" + "ab" * 32
API_URL = "https://api.test"

PRICES = {"lite": 10000, "quick": 20000, "standard": 100000, "deep": 500000}

# A throwaway key: signatures are made locally and never reach a network.
ACCOUNT = Account.create()


def encode(value: Any) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()


def decode(value: str) -> Any:
    return json.loads(base64.b64decode(value))


def option(tier: str, **overrides: Any) -> dict[str, Any]:
    return {
        "scheme": "exact",
        "network": "eip155:8453",
        "amount": str(PRICES[tier]),
        "asset": USDC,
        "payTo": PAY_TO,
        "maxTimeoutSeconds": 300,
        "extra": {"name": "USD Coin", "version": "2", "tier": tier},
        **overrides,
    }


def verdict_body(tier: str, verdict: str = "safe") -> dict[str, Any]:
    return {
        "address": ADDRESS,
        "chain": "base",
        "tier": tier,
        "price_usd": PRICES[tier] / 1e6,
        "risk_score": 5 if verdict == "safe" else 60,
        "verdict": verdict,
        "reasons": [] if verdict == "safe" else ["burn_address"],
        "checked": ["ofac", "scam_lists"],
        "as_of": "2026-10-02T00:00:00+00:00",
    }


def lite_body(verdict: str = "no_known_risk") -> dict[str, Any]:
    """The lite tier's answer shape (`LiteRiskResponse`): never "safe"."""
    return {
        "address": ADDRESS,
        "chain": "base",
        "tier": "lite",
        "price_usd": PRICES["lite"] / 1e6,
        "risk_score": 5 if verdict == "no_known_risk" else 90,
        "verdict": verdict,
        "reasons": [] if verdict == "no_known_risk" else ["deployer_flagged"],
        "checked": ["ofac", "scam_lists", "poisoning_watch", "burn_list"],
        "as_of": "2026-10-06T00:00:00+00:00",
        "limited_checks": True,
        "checks_performed": [
            "ofac",
            "scam_lists",
            "poisoning_watch",
            "burn_list",
            "phishing_token",
            "deployer_flagged",
        ],
        "not_checked": ["address_age", "flash_loan_contracts", "unverified_contracts"],
        "full_check": "Limited checks only: ... use POST /risk/address/quick.",
    }


def settled_response(body: Any, status: int = 200, success: bool = True) -> httpx.Response:
    receipt = {
        "success": success,
        "transaction": TX_HASH if success else "",
        "network": "eip155:8453",
        "payer": ACCOUNT.address,
    }
    return httpx.Response(status, json=body, headers={"payment-response": encode(receipt)})


@dataclass
class Call:
    method: str
    url: str
    paid: bool
    body: Any
    headers: httpx.Headers


@dataclass
class MockApi:
    """Records every request; answers like the live API unless overridden."""

    quote: Callable[[str], dict[str, Any]] = option
    paid_response: Callable[[str], httpx.Response] | None = None
    unpaid_response: Callable[[str], httpx.Response | None] | None = None
    calls: list[Call] = field(default_factory=list)

    @property
    def paid_calls(self) -> list[Call]:
        return [call for call in self.calls if call.paid]

    def handler(self, request: httpx.Request) -> httpx.Response:
        tier = request.url.path.rsplit("/", 1)[-1]
        paid = "payment-signature" in request.headers or "x-payment" in request.headers
        content = request.content.decode()
        self.calls.append(
            Call(
                request.method,
                str(request.url),
                paid,
                json.loads(content) if content else None,
                request.headers,
            )
        )
        if not paid:
            if request.method == "POST" and self.unpaid_response:
                answer = self.unpaid_response(tier)
                if answer is not None:
                    return answer
            quote = {
                "x402Version": 2,
                "error": "Payment required",
                "resource": {
                    "url": str(request.url),
                    "description": "test",
                    "mimeType": "application/json",
                },
                "accepts": [self.quote(tier)],
            }
            return httpx.Response(402, json={}, headers={"payment-required": encode(quote)})
        if self.paid_response:
            return self.paid_response(tier)
        if tier == "lite":
            return settled_response(lite_body())
        return settled_response(verdict_body(tier))

    def http_client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def async_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def client_kwargs(self) -> dict[str, Any]:
        return {
            "account": ACCOUNT,
            "api_url": API_URL,
            "http_client": self.http_client(),
            "async_http_client": self.async_http_client(),
        }
