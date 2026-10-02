import json
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_verdix import VerdixAddressRiskTool, VerdixClient, VerdixError
from tests.mock_api import ACCOUNT, ADDRESS, API_URL, TX_HASH, MockApi


def make_tool(api: MockApi, **options: Any) -> VerdixAddressRiskTool:
    cap = options.pop("max_price_per_call_usd", 0.1)
    client = VerdixClient(
        max_price_per_call_usd=cap,
        max_total_spend_usd=options.pop("max_total_spend_usd", None),
        **api.client_kwargs(),
    )
    return VerdixAddressRiskTool(max_price_per_call_usd=cap, client=client, **options)


def tool_call(args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {"type": "tool_call", "id": call_id, "name": "check_address_risk", "args": args}


def tier_enum(tool: VerdixAddressRiskTool) -> list[str]:
    schema = convert_to_openai_tool(tool)["function"]["parameters"]
    tier = schema["properties"]["tier"]
    # Pydantic writes a single allowed value as "const".
    return tier["enum"] if "enum" in tier else [tier["const"]]


def test_shows_the_model_only_the_tiers_within_the_cap():
    tool = make_tool(MockApi(), max_price_per_call_usd=0.1)

    assert tool.name == "check_address_risk"
    assert tier_enum(tool) == ["quick", "standard"]
    assert "- quick ($0.02)" in tool.description
    assert "- standard ($0.10) [default]" in tool.description
    assert "- deep" not in tool.description
    assert "BEFORE sending it funds" in tool.description


def test_defaults_to_the_cheapest_tier_when_standard_is_out_of_reach():
    tool = make_tool(MockApi(), max_price_per_call_usd=0.05)
    assert tier_enum(tool) == ["quick"]
    assert tool.default_tier == "quick"
    assert "- quick ($0.02) [default]" in tool.description


def test_offers_every_tier_under_a_high_cap():
    tool = make_tool(MockApi(), max_price_per_call_usd=1)
    assert tier_enum(tool) == ["quick", "standard", "deep"]


def test_refuses_a_cap_below_the_cheapest_tier():
    with pytest.raises(VerdixError, match=r"below the cheapest tier \(\$0\.02\)"):
        VerdixAddressRiskTool(account=ACCOUNT, max_price_per_call_usd=0.01)


def test_requires_a_per_call_cap():
    with pytest.raises(VerdixError, match="max_price_per_call_usd is required"):
        VerdixAddressRiskTool(account=ACCOUNT)


def test_requires_an_account_when_no_client_is_given():
    with pytest.raises(VerdixError, match="account must be a signer"):
        VerdixAddressRiskTool(max_price_per_call_usd=0.1)


def test_honours_allowed_tiers_and_default_tier():
    tool = make_tool(
        MockApi(), max_price_per_call_usd=1, allowed_tiers=["deep", "quick"], default_tier="quick"
    )
    assert tier_enum(tool) == ["quick", "deep"]
    assert "- quick ($0.02) [default]" in tool.description


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"allowed_tiers": []}, "allowed_tiers must be a non-empty list"),
        ({"allowed_tiers": ["premium"]}, "allowed_tiers must be a non-empty list"),
        ({"default_tier": "deep"}, 'default_tier "deep" is not in the allowed tiers'),
    ],
)
def test_rejects_inconsistent_tier_options(options, message):
    with pytest.raises(VerdixError, match=message):
        make_tool(MockApi(), **options)


def test_does_not_expose_the_account():
    tool = VerdixAddressRiskTool(account=ACCOUNT, max_price_per_call_usd=0.1)
    assert ACCOUNT.key.hex() not in repr(tool)
    assert "account" not in tool.model_dump()


def test_answers_a_tool_call_with_the_verdict_and_advice():
    api = MockApi()
    message = make_tool(api).invoke(tool_call({"address": ADDRESS, "tier": "quick"}))

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_1"
    content = json.loads(message.content)
    assert content["verdict"] == "safe"
    assert content["tier"] == "quick"
    assert content["charged"] is True
    assert "not a guarantee" in content["advice"]
    # The artifact carries the same answer for the developer's code.
    assert message.artifact["payment"]["transaction"] == TX_HASH
    assert [c.url for c in api.paid_calls] == [f"{API_URL}/risk/address/quick"]


def test_uses_the_default_tier_when_the_model_gives_none():
    api = MockApi()
    make_tool(api).invoke({"address": ADDRESS})
    assert api.calls[0].url == f"{API_URL}/risk/address/standard"


async def test_works_async():
    api = MockApi()
    message = await make_tool(api).ainvoke(tool_call({"address": ADDRESS, "tier": "standard"}))
    assert json.loads(message.content)["tier"] == "standard"
    assert len(api.paid_calls) == 1


def test_a_tier_outside_the_cap_never_reaches_the_api():
    api = MockApi()
    message = make_tool(api, max_price_per_call_usd=0.05).invoke(
        tool_call({"address": ADDRESS, "tier": "deep"})
    )
    assert message.status == "error"
    assert api.calls == []


def test_a_malformed_address_goes_back_to_the_model_as_an_error():
    api = MockApi()
    message = make_tool(api).invoke(tool_call({"address": "0xdead"}))
    assert message.status == "error"
    assert api.calls == []


def test_a_budget_refusal_goes_back_to_the_model_as_an_error():
    api = MockApi()
    tool = make_tool(api, max_total_spend_usd=0.02)

    tool.invoke(tool_call({"address": ADDRESS, "tier": "quick"}))
    message = tool.invoke(tool_call({"address": ADDRESS, "tier": "quick"}, "call_2"))

    assert message.status == "error"
    assert "budget (max_total_spend_usd)" in message.content
    assert len(api.paid_calls) == 1


class ToolCallingFakeModel(GenericFakeChatModel):
    """Replays scripted messages; accepts tools like a real chat model."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFakeModel":
        return self


def test_runs_inside_a_langchain_agent():
    api = MockApi()
    model = ToolCallingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "check_address_risk",
                            "args": {"address": ADDRESS, "tier": "quick"},
                        }
                    ],
                ),
                AIMessage(content="The address looks clean; please confirm the amount."),
            ]
        )
    )
    agent = create_agent(model, tools=[make_tool(api)])

    result = agent.invoke({"messages": [{"role": "user", "content": f"Send 50 USDC to {ADDRESS}"}]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["verdict"] == "safe"
    assert result["messages"][-1].content.startswith("The address looks clean")
    assert len(api.paid_calls) == 1
