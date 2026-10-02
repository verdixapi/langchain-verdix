"""The `check_address_risk` LangChain tool."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from langchain_verdix._constants import TIER_LIST_PRICES_USD, VERDIX_TIERS, VerdixTier
from langchain_verdix.client import VerdixCheckResult, VerdixClient, tiers_within_cap
from langchain_verdix.errors import VerdixError

TIER_GUIDANCE: dict[VerdixTier, str] = {
    "quick": (
        "small, routine transfers (under ~$100) to an address the user has used before. "
        "Sanctions, scam/phishing lists, address-poisoning lookalikes, burn addresses, "
        "contract/deployer signals and a fast address-age read."
    ),
    "standard": (
        "transfers of ~$100-$1,000, or any address that came from a message, a transaction "
        "history or a copy-paste. Everything in quick, plus deeper on-chain behaviour analysis "
        "(transfer history, activity patterns)."
    ),
    "deep": (
        "large transfers (over ~$1,000), first-time counterparties or contract interactions. "
        "Currently runs the same checks as standard; further counterparty checks land here first."
    ),
}

ADVICE: dict[str, str] = {
    "safe": (
        'No risk signals were found by the checks listed in "checked". This is not a guarantee: '
        "still confirm the full address and amount with the user before sending."
    ),
    "caution": (
        "Risk signals were found, or a check could not complete. Do not send automatically: "
        "show the reasons to the user and send only if they explicitly confirm."
    ),
    "danger": "Do not send funds to this address. Show the reasons to the user.",
}


def _describe(tiers: list[VerdixTier], default_tier: VerdixTier) -> str:
    tier_lines = "\n".join(
        f"- {tier} (${TIER_LIST_PRICES_USD[tier]:.2f})"
        f"{' [default]' if tier == default_tier else ''}: {TIER_GUIDANCE[tier]}"
        for tier in tiers
    )
    return "\n".join(
        [
            "Screen an EVM address on Base BEFORE sending it funds or approving it, and get a",
            'verdict: "safe", "caution" or "danger". Each call is paid in USDC from the',
            "agent's wallet, so call it once per destination address, not repeatedly.",
            "",
            "Choose the tier by how much is at stake:",
            tier_lines,
            "",
            'Verdicts: "safe" means no risk signals were found by the listed checks (not a',
            'guarantee). "caution" means risk signals or incomplete data: ask the user before',
            'sending. "danger" means do not send. Follow the "advice" field of the result.',
        ]
    )


def _input_schema(tiers: list[VerdixTier], default_tier: VerdixTier) -> type[BaseModel]:
    # Only the allowed tiers appear in the schema, so the model cannot ask for
    # one above the cap.
    tier_type: Any = Literal[tuple(tiers)]
    return create_model(
        "CheckAddressRiskInput",
        address=(
            str,
            Field(
                pattern=r"^0x[0-9a-fA-F]{40}$",
                description="The full destination address (0x followed by 40 hex characters).",
            ),
        ),
        tier=(
            tier_type,
            Field(
                default=default_tier,
                description=f"Check depth; pick by transfer size. Defaults to {default_tier}.",
            ),
        ),
    )


class VerdixAddressRiskTool(BaseTool):
    """Screens an EVM address on Base before an agent sends it funds.

    Each call is paid via x402 in USDC on Base from `account`, under caps the
    model cannot change. The tool returns the verdict as JSON for the model,
    and the full `VerdixCheckResult` as the tool message's `artifact`.

    Setup:
        ```bash
        pip install -U langchain-verdix
        ```

    Instantiate:
        ```python
        from eth_account import Account
        from langchain_verdix import VerdixAddressRiskTool

        tool = VerdixAddressRiskTool(
            account=Account.from_key(os.environ["AGENT_PRIVATE_KEY"]),
            max_price_per_call_usd=0.10,  # never pay more than $0.10 for one check
            max_total_spend_usd=1.00,  # and at most $1 in total
        )
        ```

    Invoke:
        ```python
        tool.invoke({"address": "0x000000000000000000000000000000000000dEaD", "tier": "quick"})
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "check_address_risk"
    description: str = ""
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    # Refusals (price above the cap, budget used up, bad address...) go back to
    # the model as the tool's answer instead of ending the run.
    handle_tool_error: bool | str | Any = True
    handle_validation_error: bool | str | Any = True

    account: Any = Field(default=None, exclude=True, repr=False)
    """The paying wallet, e.g. `Account.from_key(key)`. Not needed if `client` is given."""
    max_price_per_call_usd: float
    """Hard cap, in USD, on what one check may cost."""
    max_total_spend_usd: float | None = None
    """Optional total budget, in USD, for this tool instance."""
    allowed_tiers: list[VerdixTier] | None = None
    """Tiers the model may choose. Defaults to every tier within the cap."""
    default_tier: VerdixTier | None = None
    """Tier used when the model gives none. Defaults to `standard` if allowed,
    otherwise the cheapest allowed tier."""
    api_url: str | None = None
    """Verdix API base URL. Defaults to https://api.verdixapi.com."""
    client: VerdixClient | None = Field(default=None, exclude=True)
    """A ready `VerdixClient` (its caps then apply); built from the fields above if omitted."""

    @model_validator(mode="before")
    @classmethod
    def _setup(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if "max_price_per_call_usd" not in values:
            raise VerdixError("max_price_per_call_usd is required")
        cap = values["max_price_per_call_usd"]

        tiers = values.get("allowed_tiers")
        if tiers is None:
            tiers = tiers_within_cap(cap) if isinstance(cap, (int, float)) else []
            if not tiers:
                raise VerdixError(
                    f"max_price_per_call_usd (${cap}) is below the cheapest tier "
                    f"(${TIER_LIST_PRICES_USD['quick']:.2f})"
                )
        elif not tiers or any(tier not in VERDIX_TIERS for tier in tiers):
            raise VerdixError(
                f"allowed_tiers must be a non-empty list of {', '.join(VERDIX_TIERS)}"
            )
        tiers = [tier for tier in VERDIX_TIERS if tier in tiers]

        default_tier: VerdixTier = values.get("default_tier") or (
            "standard" if "standard" in tiers else tiers[0]
        )
        if default_tier not in tiers:
            raise VerdixError(f'default_tier "{default_tier}" is not in the allowed tiers')

        if values.get("client") is None:
            values["client"] = VerdixClient(
                account=values.get("account"),
                max_price_per_call_usd=cap,
                max_total_spend_usd=values.get("max_total_spend_usd"),
                api_url=values.get("api_url"),
            )
        values["allowed_tiers"] = tiers
        values["default_tier"] = default_tier
        values.setdefault("description", _describe(tiers, default_tier))
        values.setdefault("args_schema", _input_schema(tiers, default_tier))
        return values

    def _resolve_tier(self, tier: VerdixTier | None) -> VerdixTier:
        chosen = tier or self.default_tier
        assert chosen is not None and self.allowed_tiers is not None
        if chosen not in self.allowed_tiers:
            raise ToolException(
                f"tier must be one of {', '.join(self.allowed_tiers)} (the tiers within your caps)"
            )
        return chosen

    @staticmethod
    def _output(result: VerdixCheckResult) -> tuple[str, dict[str, Any]]:
        answer = {**result.to_dict(), "advice": ADVICE[result.verdict]}
        return json.dumps(answer), answer

    def _run(
        self,
        address: str,
        tier: VerdixTier | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        assert self.client is not None
        try:
            result = self.client.check_address(address, self._resolve_tier(tier))
        except VerdixError as error:
            raise ToolException(str(error)) from error
        return self._output(result)

    async def _arun(
        self,
        address: str,
        tier: VerdixTier | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        assert self.client is not None
        try:
            result = await self.client.acheck_address(address, self._resolve_tier(tier))
        except VerdixError as error:
            raise ToolException(str(error)) from error
        return self._output(result)
