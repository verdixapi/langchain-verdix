"""LangChain's standard tool tests, run against the mocked API."""

from typing import Any

from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_verdix import VerdixAddressRiskTool, VerdixClient
from tests.mock_api import ADDRESS, MockApi


class _Params:
    @property
    def tool_constructor(self) -> type[VerdixAddressRiskTool]:
        return VerdixAddressRiskTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {
            "max_price_per_call_usd": 0.1,
            "client": VerdixClient(max_price_per_call_usd=0.1, **MockApi().client_kwargs()),
        }

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"address": ADDRESS, "tier": "quick"}


class TestVerdixToolUnit(_Params, ToolsUnitTests):
    pass


class TestVerdixToolStandardInvoke(_Params, ToolsIntegrationTests):
    pass
