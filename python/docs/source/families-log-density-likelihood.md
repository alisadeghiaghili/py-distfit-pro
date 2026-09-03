(veridist-families-log-density-likelihood)=
# Evaluated families, scalar log density, and streaming log likelihood

## Closed five-family registry

`FAMILY_REGISTRY` is immutable metadata for exactly five evaluated families:
`normal`, `gamma`, `weibull_min`, `lognormal`, and `gumbel_right`. It fixes
canonical names, aliases, parameter names, validation roles, and the currently
available scalar `logpdf` operation. It is neither a generic distribution
registry nor a fitting, inference, goodness-of-fit, or ranking interface.

For the contract-level API surface, see
<a class="veridist-api-url" href="api.html#veridist-api">veridist.statistics.log_density</a>.

| Family | Canonical parameters |
| --- | --- |
| `normal` | `mu`, `sigma > 0` |
| `gamma` | `shape > 0`, `scale > 0` |
| `weibull_min` | `shape > 0`, `scale > 0` |
| `lognormal` | `mu_log`, `sigma_log > 0` |
| `gumbel_right` | `location`, `scale > 0` |

## Scalar log-density evaluation

`evaluate_log_density` evaluates one built-in finite scalar observation after
canonical parameter validation. A successful result contains a finite binary64
log-density. Data-domain failures are closed values: `nonfinite_observation`,
`support_violation`, `nonfinite_log_density`, or `numerical_overflow`.
Programmer misuse of a family identity or parameter set raises rather than
being converted into a data result. This is not an array API, a CDF/PPF API,
censoring likelihood, fit, inference, goodness-of-fit test, or model ranking.

```python
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics.log_density import LogDensityFailure, evaluate_log_density

assert FAMILY_REGISTRY.resolve("gaussian").id is FamilyId.NORMAL
assert evaluate_log_density(FamilyId.NORMAL, 0.0, mu=0.0, sigma=1.0).log_density < 0.0
failure = evaluate_log_density(FamilyId.GAMMA, 0.0, shape=2.0, scale=1.0)
assert isinstance(failure, LogDensityFailure)
assert failure.code.value == "support_violation"
```

## Exact-state streaming log likelihood

`reduce_log_likelihood_chunks` consumes an iterable of ragged observation
chunks once. It evaluates each scalar through the same closed evaluator and
accumulates each successful binary64 result as an exact integer number of
subnormal units. The final sum is rounded to binary64 once, at finalization;
it is not a chunk-wise floating-point sum.

```{math}
\operatorname{LL} = \operatorname{round}_{binary64}\left(\sum_i \log f(x_i)\right)
```

The immutable `LogLikelihoodState` supports compatible exact-state merging.
One state has an explicit unsigned-64 observation limit (`2^64 - 1`); its
maximum absolute exact integer total has a documented 2162-bit bound. A scalar
failure stops reduction and reports only the number of successful observations
before it. An empty chunk sequence succeeds with count zero and total `0.0`.

```python
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

success = reduce_log_likelihood_chunks(
    FamilyId.NORMAL, ((0.0, 1.0), (), (-1.0,)), mu=0.0, sigma=1.0
)
assert success.observation_count == 3
empty = reduce_log_likelihood_chunks(FamilyId.NORMAL, ((),), mu=0.0, sigma=1.0)
assert empty.observation_count == 0
failed = reduce_log_likelihood_chunks(FamilyId.GAMMA, ((1.0,), (0.0,)), shape=2.0, scale=1.0)
assert failed.code.value == "scalar_evaluation_failure"
assert failed.processed_count == 1
```

## Evidence and limits

`LLR-06` retains generated-stream evidence for normal-zero scalar reduction at
10k/100k/1m observations and three chunk sizes. It records one outer iterator
acquisition, every yielded observation, bitwise final-total agreement with an
independent exact-unit oracle, and the 2162-bit algorithmic bound. Elapsed time
and `tracemalloc` values are descriptive only.

This evidence is scoped to the tested scalar normal stream and exact reducer.
It does not establish fitting, inference, goodness-of-fit, ranking, arrays,
censoring, generic out-of-core processing, throughput, process-memory/RSS
bounds, another data adapter, or cross-platform performance.
