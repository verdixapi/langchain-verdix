from typing import Literal

DEFAULT_API_URL = "https://api.verdixapi.com"

# CAIP-2 id of Base mainnet, the only network Verdix accepts payment on.
BASE_MAINNET = "eip155:8453"

# Native USDC on Base (lowercase), the only asset Verdix accepts.
USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

USDC_DECIMALS = 6

VerdixTier = Literal["quick", "standard", "deep"]

VERDIX_TIERS: tuple[VerdixTier, ...] = ("quick", "standard", "deep")

# List prices in USD, used to pick which tiers fit under your cap and to
# describe them to the model. The API's own 402 quote is authoritative: a
# tier is never paid for if its live price is above max_price_per_call_usd.
TIER_LIST_PRICES_USD: dict[VerdixTier, float] = {
    "quick": 0.02,
    "standard": 0.10,
    "deep": 0.50,
}
