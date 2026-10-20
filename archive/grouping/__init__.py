"""
Archived. Superseded by `placement/`, which absorbed the four modules it depended on
(`profiles`, `graph`, and slivers of `analytic` and `multilevel`).

Kept importable for one reason: `analytic.AnalyticCost` is the model
`placement.cost.PackingCost` is pinned to by an equality test over random partitions.
That cross-check is what makes the new model's numbers trustworthy, so the reference
implementation stays reachable rather than being deleted.

The package `__init__` is deliberately bare -- the old one re-exported modules that no
longer live here. Import the submodule you want directly.
"""
