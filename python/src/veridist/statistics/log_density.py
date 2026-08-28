"""Finite scalar log-density evaluation for the closed family registry."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import exp, isfinite, lgamma, log, pi
from types import MappingProxyType
from typing import Final, TypeAlias, cast

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, FamilySpec, Operation


class LogDensityErrorCode(StrEnum):
    """Closed, locale-neutral scalar evaluation failure codes."""

    NONFINITE_OBSERVATION = "nonfinite_observation"
    SUPPORT_VIOLATION = "support_violation"
    NONFINITE_LOG_DENSITY = "nonfinite_log_density"
    NUMERICAL_OVERFLOW = "numerical_overflow"
    NUMERICAL_UNDERFLOW = "numerical_underflow"


@dataclass(frozen=True, slots=True)
class LogDensitySuccess:
    """A finite log-density with no retained observation or parameter values."""

    family: FamilyId
    log_density: float

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        if type(self.log_density) is not float or not isfinite(self.log_density):
            raise ValueError("log_density must be a finite built-in float")

    def to_json(self) -> str:
        """Serialize the safe success surface deterministically."""

        return json.dumps(
            {"family": self.family.value, "log_density": self.log_density, "outcome": "success"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class LogDensityFailure:
    """A typed failure that deliberately contains no raw input data."""

    family: FamilyId
    code: LogDensityErrorCode

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        if type(self.code) is not LogDensityErrorCode:
            raise TypeError("code must be a LogDensityErrorCode")

    def to_json(self) -> str:
        """Serialize only the stable failure identity deterministically."""

        return json.dumps(
            {"code": self.code.value, "family": self.family.value, "outcome": "failure"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


LogDensityResult: TypeAlias = LogDensitySuccess | LogDensityFailure
_Evaluator: TypeAlias = Callable[[float, Mapping[str, float]], float]
_LOG_SQRT_TWO_PI: Final = 0.5 * log(2.0 * pi)


class _NumericalOverflow(Exception):
    """Internal signal for a non-representable required intermediate."""


def evaluate_log_density(
    family: FamilyId, observation: object, /, **parameters: object
) -> LogDensityResult:
    """Evaluate one exact-observation log-density under closed metadata contracts.

    Programmer misuse of family identity or canonical parameter names raises
    ``TypeError``/``ValueError``.  Data-domain and binary64 numerical failures
    are returned as typed, locale-neutral values.
    """

    if type(family) is not FamilyId:
        raise TypeError("family must be a FamilyId")
    specification = FAMILY_REGISTRY.families[family]
    validated = specification.validate_parameters(**parameters)
    numeric_observation = _validate_observation(observation)
    if numeric_observation is None:
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_OBSERVATION)
    if specification.fixed_location == 0.0 and numeric_observation <= 0.0:
        return LogDensityFailure(family, LogDensityErrorCode.SUPPORT_VIOLATION)
    try:
        candidate = _DISPATCH[family](numeric_observation, validated)
    except (OverflowError, _NumericalOverflow):
        return LogDensityFailure(family, LogDensityErrorCode.NUMERICAL_OVERFLOW)
    except ValueError:
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
    if not isfinite(candidate):
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
    return LogDensitySuccess(family, candidate)


def _validate_observation(observation: object) -> float | None:
    if type(observation) not in (int, float):
        return None
    try:
        numeric = float(cast(int | float, observation))
    except OverflowError:
        return None
    return numeric if isfinite(numeric) else None


def _normal(observation: float, parameters: Mapping[str, float]) -> float:
    centered = _finite_intermediate(observation - parameters["mu"])
    z = _finite_intermediate(centered / parameters["sigma"])
    quadratic = _finite_intermediate(z * z)
    return -_LOG_SQRT_TWO_PI - log(parameters["sigma"]) - 0.5 * quadratic


def _gamma(observation: float, parameters: Mapping[str, float]) -> float:
    shape = parameters["shape"]
    scale = parameters["scale"]
    log_observation = _finite_intermediate(log(observation))
    quotient = _finite_intermediate(observation / scale)
    log_gamma = _finite_intermediate(lgamma(shape))
    scale_term = _finite_intermediate(shape * log(scale))
    return (shape - 1.0) * log_observation - quotient - log_gamma - scale_term


def _weibull_min(observation: float, parameters: Mapping[str, float]) -> float:
    shape = parameters["shape"]
    scale = parameters["scale"]
    log_ratio = _finite_intermediate(log(observation) - log(scale))
    powered_ratio = exp(shape * log_ratio)
    return log(shape) - log(scale) + (shape - 1.0) * log_ratio - powered_ratio


def _lognormal(observation: float, parameters: Mapping[str, float]) -> float:
    log_observation = log(observation)
    centered = _finite_intermediate(log_observation - parameters["mu_log"])
    z = _finite_intermediate(centered / parameters["sigma_log"])
    quadratic = _finite_intermediate(z * z)
    return -log_observation - log(parameters["sigma_log"]) - _LOG_SQRT_TWO_PI - 0.5 * quadratic


def _gumbel_right(observation: float, parameters: Mapping[str, float]) -> float:
    scale = parameters["scale"]
    centered = _finite_intermediate(observation - parameters["location"])
    z = _finite_intermediate(centered / scale)
    return -log(scale) - z - exp(-z)


def _finite_intermediate(value: float) -> float:
    if not isfinite(value):
        raise _NumericalOverflow
    return value


_DISPATCH: Final[Mapping[FamilyId, _Evaluator]] = MappingProxyType(
    {
        FamilyId.NORMAL: _normal,
        FamilyId.GAMMA: _gamma,
        FamilyId.WEIBULL_MIN: _weibull_min,
        FamilyId.LOGNORMAL: _lognormal,
        FamilyId.GUMBEL_RIGHT: _gumbel_right,
    }
)


def _verify_registry_dispatch(
    registry: Mapping[FamilyId, FamilySpec], dispatch: Mapping[FamilyId, _Evaluator]
) -> None:
    """Fail deterministically if advertised scalar capability loses parity."""

    if set(registry) != set(dispatch):
        raise RuntimeError("log-density dispatch must exactly match the family registry")
    if any(not spec.supports(Operation.LOGPDF) for spec in registry.values()):
        raise RuntimeError("log-density dispatch cannot exceed advertised registry capability")


_verify_registry_dispatch(FAMILY_REGISTRY.families, _DISPATCH)


__all__ = [
    "LogDensityErrorCode",
    "LogDensityFailure",
    "LogDensityResult",
    "LogDensitySuccess",
    "evaluate_log_density",
]
