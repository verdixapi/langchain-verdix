"""One real, paid Verdix check through the LangChain tool (no model needed).

Pays REAL USDC on Base mainnet ($0.02, quick tier) from AGENT_PRIVATE_KEY.

    AGENT_PRIVATE_KEY=0x... python examples/check_before_send.py [address]
"""

import json
import os
import sys

from eth_account import Account

from langchain_verdix import VerdixAddressRiskTool

address = sys.argv[1] if len(sys.argv) > 1 else "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

tool = VerdixAddressRiskTool(
    account=Account.from_key(os.environ["AGENT_PRIVATE_KEY"]),
    max_price_per_call_usd=0.02,  # quick tier only
    max_total_spend_usd=0.02,  # one check
)

message = tool.invoke(
    {
        "type": "tool_call",
        "id": "example",
        "name": tool.name,
        "args": {"address": address, "tier": "quick"},
    }
)
# A refusal or API error comes back as the message content, with no artifact.
print(json.dumps(message.artifact, indent=2) if message.artifact else message.content)
