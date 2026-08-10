# Replay Assumptions and Non-Goals

This repository intentionally does **not** claim exact exchange replication.

## Implemented behavior

The replay kernel implements the following mechanics used by the project:

- buy quantities are positive and sell quantities are negative;
- historical ask volumes are normalized to negative quantities;
- marketable buys consume historical asks at or below their limit;
- marketable sells consume historical bids at or above their limit;
- executions occur at the historical resting-book price;
- submitted buy and sell quantities are checked independently against the absolute position limit;
- if one submitted side would breach the limit, all orders on that side are rejected.

The golden tests lock down these semantics for the local replay kernel.

## Deliberately not modelled

- passive queue position;
- fills of resting orders from future bot flow;
- persistence of unfilled orders after a replay tick;
- latency;
- conversions;
- hidden liquidity;
- exchange priority beyond visible price ordering;
- mechanisms that cannot be reconstructed from a historical snapshot.

Because of these omissions, results are **historical replay estimates**, not exact competition P&L.

## Position limits

The CLI uses a fallback position limit of 50 only when no explicit override is supplied.

This is a demo assumption, not a claim about a specific Prosperity round.

Research runs should pass:

```text
--position-limit PRODUCT=LIMIT