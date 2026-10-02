# langchain-verdix

A [LangChain](https://python.langchain.com) tool that screens an EVM address on Base **before your agent sends it funds**, and answers `safe`, `caution` or `danger` with reasons.

It checks OFAC sanctions, scam, phishing and exploit lists, live **address-poisoning lookalikes**, burn addresses, contract and deployer signals, and on-chain behaviour. You don't need an API key or an account. Each check is paid per call via [x402](https://x402.org), in USDC on Base, from your agent's own wallet, under caps you set.

## Install

```bash
pip install -U langchain-verdix
```

Payments are real USDC on Base mainnet; there is no testnet mode.

## Usage

```python
import os

from eth_account import Account
from langchain.agents import create_agent
from langchain_verdix import VerdixAddressRiskTool

check_address_risk = VerdixAddressRiskTool(
    account=Account.from_key(os.environ["AGENT_PRIVATE_KEY"]),
    max_price_per_call_usd=0.10,  # never pay more than $0.10 for one check
    max_total_spend_usd=1.00,  # and at most $1 in total
)

agent = create_agent("openai:gpt-5-mini", tools=[check_address_risk])
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Send 250 USDC on Base to 0x000000000000000000000000000000000000dEaD.",
            }
        ]
    }
)
print(result["messages"][-1].content)
```

Any chat model that supports tool calling works, as does LangGraph (`ToolNode`) or calling the tool directly:

```python
check_address_risk.invoke(
    {"address": "0x000000000000000000000000000000000000dEaD", "tier": "quick"}
)
```

The model gets one tool, `check_address_risk(address, tier?)`, and sees this result as JSON:

```jsonc
{
  "verdict": "danger", // "safe" | "caution" | "danger"
  "risk_score": 90,
  "reasons": ["burn_address"],
  "checked": ["ofac", "scam_lists", "poisoning_watch", "burn_list", "..."],
  "advice": "Do not send funds to this address. Show the reasons to the user.",
  "tier": "standard",
  "price_usd": 0.1,
  "complete": true,
  "charged": true,
  "payment": { "transaction": "0x...", "network": "eip155:8453", "payer": "0x..." },
  "retry_after_seconds": null,
  "address": "0x000000000000000000000000000000000000dEaD",
  "chain": "base",
  "as_of": "2026-10-02T12:00:00+00:00"
}
```

The same dict is the `artifact` of the `ToolMessage`, so your code can read the payment's transaction hash without parsing the content.

### What the verdicts mean

| Verdict | Meaning | What the agent should do |
|---|---|---|
| `safe` | None of the checks in `checked` found a risk signal. **Not a guarantee.** | Still confirm the full address and amount with the user. |
| `caution` | Risk signals were found, or a data source could not be reached (`complete: false`). | Show the reasons and send only if the user explicitly confirms. |
| `danger` | Strong risk signals: sanctioned, known scam, poisoning lookalike, burn address... | Do not send. |

The tool description tells the model the same thing, and every result carries an `advice` string.

### Tiers

The model picks a tier by what is at stake. It can only pick the tiers that fit under `max_price_per_call_usd`: the others are left out of the tool's description and input schema.

| Tier | List price | When |
|---|---|---|
| `quick` | $0.02 | Small, routine transfers (under ~$100) to an address used before. Sanctions, scam lists, poisoning lookalikes, burn addresses, contract/deployer signals and a fast address-age read. |
| `standard` (default) | $0.10 | ~$100-$1,000, or any address that came from a message, a transaction history or a copy-paste. Adds deeper on-chain behaviour analysis. |
| `deep` | $0.50 | Over ~$1,000, first-time counterparties, contract interactions. Currently the same checks as `standard`; new counterparty checks land here first. |

Each tier has its own URL (`https://api.verdixapi.com/risk/address/{tier}`) with a single price, so the tier the model asks for is exactly the tier that is paid.

## Options

```python
VerdixAddressRiskTool(
    account=...,  # required: an eth_account account (or any x402 ClientEvmSigner)
    max_price_per_call_usd=0.10,  # required: refuse any check costing more, before signing
    max_total_spend_usd=1.00,  # optional: total budget for this tool instance
    allowed_tiers=["quick", "standard"],  # optional (default: all tiers within the cap)
    default_tier="standard",  # optional: used when the model gives none
    api_url=None,  # optional: default https://api.verdixapi.com
)
```

## Payments and safety

- The wallet needs a little **USDC on Base mainnet**. It needs no ETH: x402 payments are gasless signatures that the facilitator settles.
- The private key only signs the payment locally. It is never sent anywhere.
- A payment is signed only for **USDC on Base**, only for the tier requested, and only at or under `max_price_per_call_usd`. Anything else is refused before signing.
- Before each payment is signed, its price is reserved against `max_total_spend_usd`, so parallel tool calls (threads or asyncio) cannot overspend. The reservation is released when the API reports no charge.
- **You are not charged when a data source fails.** The API then answers `caution` with `complete: false` (and `retry_after_seconds`), and never `safe`.
- The model cannot change the caps: they are fixed when the tool is created.
- A refusal (price over the cap, budget used up, malformed address) goes back to the model as the tool's error message instead of ending the run.

## Without a model

```python
from langchain_verdix import VerdixClient

verdix = VerdixClient(account=account, max_price_per_call_usd=0.10)

pricing = verdix.get_pricing()  # free: reads the unpaid 402 quotes
result = verdix.check_address("0x...", tier="quick")  # or: await verdix.acheck_address(...)
if result.verdict != "safe":
    ...  # stop, or ask the user
```

The errors to expect are `VerdixError`s: a malformed address, a price over your cap, an exhausted budget or an unexpected API answer.

## Development

```bash
pip install -e ".[test]"
pytest            # offline: mocked API, throwaway key, sockets disabled, no payment
```

`examples/check_before_send.py` makes one real paid check.

## License

MIT
