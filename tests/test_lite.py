"""The lite tier ($0.01, never "safe"): client methods and `check_address_risk_lite`."""

import json
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_verdix import (
    VERDIX_TIERS,
    VerdixAddressRiskTool,
    VerdixClient,
    VerdixError,
    VerdixLiteAddressRiskTool,
    VerdixLiteCheckResult,
    tiers_within_cap,
)
from tests.mock_api import (
    ACCOUNT,
    ADDRESS,
    API_URL,
    PAY_TO,
    TX_HASH,
    MockApi,
    decode,
    lite_body,
    option,
    settled_response,
)
from tests.test_tools import ToolCallingFakeModel, tier_enum

LITE_URL = f"{API_URL}/risk/address/lite"


def make_client(api: MockApi, **overrides: Any) -> VerdixClient:
    return VerdixClient(**{"max_price_per_call_usd": 0.1, **api.client_kwargs(), **overrides})


def make_lite_tool(api: MockApi, **options: Any) -> VerdixLiteAddressRiskTool:
    cap = options.pop("max_price_per_call_usd", 0.01)
    client = VerdixClient(
        max_price_per_call_usd=cap,
        max_total_spend_usd=options.pop("max_total_spend_usd", None),
        **api.client_kwargs(),
    )
    return VerdixLiteAddressRiskTool(max_price_per_call_usd=cap, client=client, **options)


def lite_call(args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {"type": "tool_call", "id": call_id, "name": "check_address_risk_lite", "args": args}


# --- client --------------------------------------------------------------------------------


def test_pays_for_lite_at_its_own_url():
    api = MockApi()
    verdix = make_client(api)

    result = verdix.check_address_lite(ADDRESS)

    assert [(c.method, c.url, c.paid) for c in api.calls] == [
        ("POST", LITE_URL, False),
        ("POST", LITE_URL, True),
    ]
    # Like the other single-tier URLs, no tier field goes in the body.
    assert api.calls[1].body == {"address": ADDRESS, "chain": "base"}
    assert isinstance(result, VerdixLiteCheckResult)
    assert result.tier == "lite"
    assert result.verdict == "no_known_risk"
    assert result.limited_checks is True
    assert "deployer_flagged" in result.checks_performed
    assert result.not_checked == ["address_age", "flash_loan_contracts", "unverified_contracts"]
    assert "/risk/address/quick" in result.full_check
    assert result.price_usd == 0.01
    assert result.complete is True
    assert result.charged is True
    assert result.payment is not None and result.payment.transaction == TX_HASH
    assert verdix.spent_usd == 0.01


def test_signs_exactly_the_lite_quote():
    api = MockApi()
    make_client(api).check_address_lite(ADDRESS)

    payload = decode(api.calls[1].headers["payment-signature"])
    assert payload["accepted"]["extra"]["tier"] == "lite"
    assert payload["resource"]["url"] == LITE_URL
    authorization = payload["payload"]["authorization"]
    assert authorization["from"] == ACCOUNT.address
    assert authorization["to"].lower() == PAY_TO.lower()
    assert authorization["value"] == "10000"


@pytest.mark.parametrize("verdict", ["caution", "danger"])
def test_passes_lite_risk_verdicts_through(verdict):
    api = MockApi(paid_response=lambda tier: settled_response(lite_body(verdict)))
    result = make_client(api).check_address_lite(ADDRESS)
    assert result.verdict == verdict
    assert result.charged is True


def test_never_passes_on_a_safe_answer_from_lite():
    body = {**lite_body(), "verdict": "safe"}
    api = MockApi(paid_response=lambda tier: settled_response(body))
    with pytest.raises(VerdixError, match="lite tier: no lite verdict"):
        make_client(api).check_address_lite(ADDRESS)


def test_refuses_a_cap_below_the_lite_price_before_signing():
    api = MockApi()
    verdix = make_client(api, max_price_per_call_usd=0.005)
    with pytest.raises(
        VerdixError,
        match=r"Refusing to pay: the lite tier costs \$0\.01, above the \$0\.005 per-call cap",
    ):
        verdix.check_address_lite(ADDRESS)
    assert api.paid_calls == []
    assert verdix.spent_usd == 0


def test_refuses_a_lite_url_quote_for_another_tier():
    api = MockApi(quote=lambda tier: option("quick"))
    with pytest.raises(VerdixError, match="no payment option for the lite tier"):
        make_client(api).check_address_lite(ADDRESS)
    assert api.paid_calls == []


def test_lite_503_is_an_incomplete_unpaid_answer():
    api = MockApi(
        paid_response=lambda tier: httpx.Response(
            503, json=lite_body("caution"), headers={"retry-after": "30"}
        )
    )
    verdix = make_client(api, max_total_spend_usd=1)

    result = verdix.check_address_lite(ADDRESS)

    assert result.verdict == "caution"
    assert result.complete is False
    assert result.charged is False
    assert result.retry_after_seconds == 30
    assert verdix.spent_usd == 0


def test_lite_degraded_answer_is_free_and_not_signed():
    api = MockApi(
        unpaid_response=lambda tier: httpx.Response(
            200, json=lite_body("caution"), headers={"x-verdix-degraded": "poisoning_watch"}
        )
    )
    verdix = make_client(api)

    result = verdix.check_address_lite(ADDRESS)

    assert len(api.calls) == 1
    assert result.complete is False
    assert result.charged is False
    assert verdix.spent_usd == 0


def test_lite_rejects_a_malformed_address_without_any_request():
    api = MockApi()
    with pytest.raises(VerdixError, match="address must be 0x followed by 40 hex characters"):
        make_client(api).check_address_lite("0xdead")
    assert api.calls == []


async def test_acheck_address_lite():
    api = MockApi()
    result = await make_client(api).acheck_address_lite(ADDRESS)
    assert result.verdict == "no_known_risk"
    assert [c.url for c in api.paid_calls] == [LITE_URL]


def test_lite_and_full_checks_share_one_budget():
    api = MockApi()
    verdix = make_client(api, max_total_spend_usd=0.03)

    verdix.check_address_lite(ADDRESS)
    verdix.check_address(ADDRESS, "quick")
    with pytest.raises(VerdixError, match=r"budget \(max_total_spend_usd\)"):
        verdix.check_address_lite(ADDRESS)
    assert verdix.spent_usd == pytest.approx(0.03)


def test_get_lite_pricing_reads_the_unpaid_lite_quote():
    api = MockApi()
    quote = make_client(api, max_price_per_call_usd=0.005).get_lite_pricing()

    assert (quote.tier, quote.price_usd, quote.within_cap) == ("lite", 0.01, False)
    assert quote.pay_to == PAY_TO
    assert [(c.method, c.url, c.paid) for c in api.calls] == [("GET", LITE_URL, False)]


async def test_aget_lite_pricing():
    api = MockApi()
    quote = await make_client(api).aget_lite_pricing()
    assert (quote.tier, quote.within_cap) == ("lite", True)
    assert api.paid_calls == []


def test_existing_tiers_and_pricing_leave_lite_out():
    assert VERDIX_TIERS == ("quick", "standard", "deep")
    assert tiers_within_cap(0.01) == []
    api = MockApi()
    assert [q.tier for q in make_client(api).get_pricing()] == ["quick", "standard", "deep"]
    assert all(not c.url.endswith("/lite") for c in api.calls)
    with pytest.raises(VerdixError, match="tier must be one of quick, standard, deep"):
        make_client(api).check_address(ADDRESS, "lite")  # type: ignore[arg-type]


# --- tool ----------------------------------------------------------------------------------


def test_lite_tool_takes_only_an_address_and_never_promises_safe():
    tool = make_lite_tool(MockApi())

    assert tool.name == "check_address_risk_lite"
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert list(parameters["properties"]) == ["address"]
    assert parameters["required"] == ["address"]
    assert 'NEVER answers "safe"' in tool.description
    assert "$0.01" in tool.description
    assert "check_address_risk, quick tier" in tool.description


def test_lite_tool_answers_with_no_known_risk_and_advice():
    api = MockApi()
    message = make_lite_tool(api).invoke(lite_call({"address": ADDRESS}))

    assert isinstance(message, ToolMessage)
    content = json.loads(message.content)
    assert content["verdict"] == "no_known_risk"
    assert content["limited_checks"] is True
    assert content["charged"] is True
    assert "NOT a safety verdict" in content["advice"]
    assert message.artifact["payment"]["transaction"] == TX_HASH
    assert [c.url for c in api.paid_calls] == [LITE_URL]


async def test_lite_tool_works_async():
    api = MockApi(paid_response=lambda tier: settled_response(lite_body("danger")))
    message = await make_lite_tool(api).ainvoke(lite_call({"address": ADDRESS}))
    content = json.loads(message.content)
    assert content["verdict"] == "danger"
    assert content["advice"].startswith("Do not send funds")


def test_lite_tool_returns_an_error_to_the_model_instead_of_safe():
    body = {**lite_body(), "verdict": "safe"}
    api = MockApi(paid_response=lambda tier: settled_response(body))
    message = make_lite_tool(api).invoke(lite_call({"address": ADDRESS}))
    assert message.status == "error"
    assert "safe" not in json.dumps(message.artifact or {})


def test_lite_tool_refuses_a_cap_below_its_price():
    with pytest.raises(VerdixError, match=r"below the lite tier's price \(\$0\.01\)"):
        VerdixLiteAddressRiskTool(account=ACCOUNT, max_price_per_call_usd=0.005)


def test_lite_tool_requires_a_per_call_cap_and_a_signer():
    with pytest.raises(VerdixError, match="max_price_per_call_usd is required"):
        VerdixLiteAddressRiskTool(account=ACCOUNT)
    with pytest.raises(VerdixError, match="account must be a signer"):
        VerdixLiteAddressRiskTool(max_price_per_call_usd=0.01)


def test_lite_tool_does_not_expose_the_account():
    tool = VerdixLiteAddressRiskTool(account=ACCOUNT, max_price_per_call_usd=0.01)
    assert ACCOUNT.key.hex() not in repr(tool)
    assert "account" not in tool.model_dump()


def test_lite_tool_sends_a_malformed_address_back_as_an_error():
    api = MockApi()
    message = make_lite_tool(api).invoke(lite_call({"address": "0xdead"}))
    assert message.status == "error"
    assert api.calls == []


def test_full_tool_is_unchanged_next_to_lite():
    tool = VerdixAddressRiskTool(
        max_price_per_call_usd=1, client=make_client(MockApi(), max_price_per_call_usd=1)
    )
    assert tool.name == "check_address_risk"
    assert tier_enum(tool) == ["quick", "standard", "deep"]
    assert "lite" not in tool.description


def test_an_agent_can_hold_both_tools_on_one_budget():
    api = MockApi()
    shared = make_client(api, max_total_spend_usd=0.1)
    model = ToolCallingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "check_address_risk_lite",
                            "args": {"address": ADDRESS},
                        }
                    ],
                ),
                AIMessage(content="No known risk, but that is not a safety verdict."),
            ]
        )
    )
    tools = [
        VerdixAddressRiskTool(max_price_per_call_usd=0.1, client=shared),
        VerdixLiteAddressRiskTool(max_price_per_call_usd=0.1, client=shared),
    ]
    agent = create_agent(model, tools=tools)

    result = agent.invoke({"messages": [{"role": "user", "content": f"Send 5 USDC to {ADDRESS}"}]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_messages] == ["check_address_risk_lite"]
    assert json.loads(tool_messages[0].content)["verdict"] == "no_known_risk"
    assert [c.url for c in api.paid_calls] == [LITE_URL]
    assert shared.spent_usd == 0.01


# --- LangChain's standard tool tests -------------------------------------------------------


class _LiteParams:
    @property
    def tool_constructor(self) -> type[VerdixLiteAddressRiskTool]:
        return VerdixLiteAddressRiskTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {
            "max_price_per_call_usd": 0.01,
            "client": VerdixClient(max_price_per_call_usd=0.01, **MockApi().client_kwargs()),
        }

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"address": ADDRESS}


class TestVerdixLiteToolUnit(_LiteParams, ToolsUnitTests):
    pass


class TestVerdixLiteToolStandardInvoke(_LiteParams, ToolsIntegrationTests):
    pass
