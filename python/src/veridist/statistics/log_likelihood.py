"""Exact-state streaming reduction of finite scalar binary64 log densities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from typing import Final, TypeAlias

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, ParameterRole
from veridist.statistics.log_density import LogDensityFailure, _evaluate_validated_log_density

MAX_OBSERVATION_COUNT: Final = (1 << 64) - 1
"""The explicit unsigned-64 observation cap for one reducer state."""

_UNITS_DENOMINATOR: Final = 1 << 1074
_MAX_CONTRIBUTION_UNITS: Final = ((1 << 53) - 1) << 2045
MAX_TOTAL_UNITS: Final = MAX_OBSERVATION_COUNT * _MAX_CONTRIBUTION_UNITS
"""Maximum absolute exact unit total under the declared count cap (2162 bits)."""

_ParameterSignature: TypeAlias = tuple[tuple[str, str], ...]


class LogLikelihoodErrorCode(StrEnum):
    """Closed, locale-neutral streaming log-likelihood failure codes."""

    SCALAR_EVALUATION_FAILURE = "scalar_evaluation_failure"
    OBSERVATION_LIMIT_EXCEEDED = "observation_limit_exceeded"
    FINAL_TOTAL_NOT_REPRESENTABLE = "final_total_not_representable"


@dataclass(frozen=True, slots=True)
class LogLikelihoodState:
    """A compatible exact sum in subnormal binary64 units, with no observations."""

    family: FamilyId
    parameter_signature: _ParameterSignature
    observation_count: int
    total_units: int

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        _validate_signature(self.family, self.parameter_signature)
        if type(self.observation_count) is not int:
            raise TypeError("observation_count must be a built-in integer")
        if not 0 <= self.observation_count <= MAX_OBSERVATION_COUNT:
            raise ValueError("observation_count is outside the unsigned-64 reducer bound")
        if type(self.total_units) is not int:
            raise TypeError("total_units must be a built-in integer")
        maximum_for_count = self.observation_count * _MAX_CONTRIBUTION_UNITS
        if abs(self.total_units) > maximum_for_count:
            raise ValueError("total_units exceeds the declared count-specific state bound")
        if self.observation_count == 0 and self.total_units != 0:
            raise ValueError("empty reducer state must have zero total units")

    @classmethod
    def empty(cls, family: FamilyId, /, **parameters: object) -> LogLikelihoodState:
        """Create the neutral state after one canonical parameter validation."""

        validated = _validate_family_and_parameters(family, parameters)
        return cls(family, _signature_from_validated(validated), 0, 0)

    def add_log_density(self, log_density: float) -> LogLikelihoodState:
        """Add one successful finite binary64 scalar evaluator output exactly."""

        if type(log_density) is not float or not isfinite(log_density):
            raise ValueError("log_density must be a finite built-in float")
        if self.observation_count == MAX_OBSERVATION_COUNT:
            raise _ObservationLimitExceeded
        numerator, denominator = log_density.as_integer_ratio()
        units = numerator * (_UNITS_DENOMINATOR // denominator)
        return LogLikelihoodState(
            self.family,
            self.parameter_signature,
            self.observation_count + 1,
            self.total_units + units,
        )

    def merge(self, other: LogLikelihoodState) -> LogLikelihoodState:
        """Merge compatible exact states; integer addition is order-independent."""

        if type(other) is not LogLikelihoodState:
            raise TypeError("other must be a LogLikelihoodState")
        if self.family is not other.family:
            raise ValueError("cannot merge states for different families")
        if self.parameter_signature != other.parameter_signature:
            raise ValueError("cannot merge states with different canonical parameters")
        if self.observation_count > MAX_OBSERVATION_COUNT - other.observation_count:
            raise _ObservationLimitExceeded
        return LogLikelihoodState(
            self.family,
            self.parameter_signature,
            self.observation_count + other.observation_count,
            self.total_units + other.total_units,
        )

    def finalize(self) -> float:
        """Correctly round the exact rational unit total once to binary64."""

        try:
            result = float(Fraction(self.total_units, _UNITS_DENOMINATOR))
        except OverflowError as error:
            raise _FinalTotalNotRepresentable from error
        if not isfinite(result):
            raise _FinalTotalNotRepresentable
        return result


@dataclass(frozen=True, slots=True)
class LogLikelihoodSuccess:
    """A finite final total with only compatibility and count facts retained."""

    family: FamilyId
    parameter_signature: _ParameterSignature
    observation_count: int
    total_log_likelihood: float

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        _validate_signature(self.family, self.parameter_signature)
        if (
            type(self.observation_count) is not int
            or not 0 <= self.observation_count <= MAX_OBSERVATION_COUNT
        ):
            raise ValueError("observation_count is outside the unsigned-64 reducer bound")
        if type(self.total_log_likelihood) is not float or not isfinite(self.total_log_likelihood):
            raise ValueError("total_log_likelihood must be a finite built-in float")

    def to_json(self) -> str:
        """Serialize stable, locale-neutral success facts without raw observations."""

        return json.dumps(
            {
                "family": self.family.value,
                "observation_count": self.observation_count,
                "outcome": "success",
                "total_log_likelihood": self.total_log_likelihood,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class LogLikelihoodFailure:
    """A closed failure with no partial total or raw observation value."""

    family: FamilyId
    parameter_signature: _ParameterSignature
    code: LogLikelihoodErrorCode
    processed_count: int

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        _validate_signature(self.family, self.parameter_signature)
        if type(self.code) is not LogLikelihoodErrorCode:
            raise TypeError("code must be a LogLikelihoodErrorCode")
        if (
            type(self.processed_count) is not int
            or not 0 <= self.processed_count <= MAX_OBSERVATION_COUNT
        ):
            raise ValueError("processed_count is outside the unsigned-64 reducer bound")

    def to_json(self) -> str:
        """Serialize only closed failure facts; count is incomplete on early failure."""

        return json.dumps(
            {
                "code": self.code.value,
                "family": self.family.value,
                "outcome": "failure",
                "processed_count": self.processed_count,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


LogLikelihoodResult: TypeAlias = LogLikelihoodSuccess | LogLikelihoodFailure


class _ObservationLimitExceeded(Exception):
    """Internal count-boundary signal without observation data."""


class _FinalTotalNotRepresentable(Exception):
    """Internal finalization-boundary signal without the exact total."""


def reduce_log_likelihood_chunks(
    family: FamilyId, chunks: Iterable[Iterable[object]], /, **parameters: object
) -> LogLikelihoodResult:
    """Reduce ragged chunks in one pass, without materializing source or chunks.

    Parameters are validated before iteration.  `processed_count` on a failure
    counts only successful observations preceding the terminal failure and is
    deliberately not a complete-input count.
    """

    validated = _validate_family_and_parameters(family, parameters)
    state = LogLikelihoodState(family, _signature_from_validated(validated), 0, 0)
    if not isinstance(chunks, Iterable):
        raise TypeError("chunks must be an iterable of observation iterables")
    for chunk in chunks:
        if not isinstance(chunk, Iterable):
            raise TypeError("each chunk must be an iterable of observations")
        for observation in chunk:
            try:
                evaluated = _evaluate_validated_log_density(family, validated, observation)
                if type(evaluated) is LogDensityFailure:
                    return LogLikelihoodFailure(
                        family,
                        state.parameter_signature,
                        LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE,
                        state.observation_count,
                    )
                state = state.add_log_density(evaluated.log_density)
            except _ObservationLimitExceeded:
                return LogLikelihoodFailure(
                    family,
                    state.parameter_signature,
                    LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED,
                    state.observation_count,
                )
    try:
        total = state.finalize()
    except _FinalTotalNotRepresentable:
        return LogLikelihoodFailure(
            family,
            state.parameter_signature,
            LogLikelihoodErrorCode.FINAL_TOTAL_NOT_REPRESENTABLE,
            state.observation_count,
        )
    return LogLikelihoodSuccess(family, state.parameter_signature, state.observation_count, total)


def _validate_family_and_parameters(
    family: FamilyId, parameters: Mapping[str, object]
) -> Mapping[str, float]:
    if type(family) is not FamilyId:
        raise TypeError("family must be a FamilyId")
    return FAMILY_REGISTRY.families[family].validate_parameters(**parameters)


def _signature_from_validated(validated: Mapping[str, float]) -> _ParameterSignature:
    return tuple((name, value.hex()) for name, value in validated.items())


def _validate_signature(family: FamilyId, signature: object) -> None:
    specification = FAMILY_REGISTRY.families[family]
    expected_names = tuple(parameter.name for parameter in specification.parameters)
    if type(signature) is not tuple or len(signature) != len(expected_names):
        raise TypeError("parameter_signature must match canonical parameter names")
    names: list[str] = []
    for item, parameter in zip(signature, specification.parameters, strict=True):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("parameter_signature must contain name/float-hex pairs")
        name, value = item
        if type(name) is not str or type(value) is not str:
            raise TypeError("parameter_signature must contain built-in strings")
        try:
            decoded = float.fromhex(value)
        except ValueError as error:
            raise ValueError("parameter_signature contains invalid float hex") from error
        if not isfinite(decoded):
            raise ValueError("parameter_signature must contain finite values")
        names.append(name)
        if parameter.role is ParameterRole.POSITIVE and decoded <= 0.0:
            raise ValueError("parameter_signature violates a positive parameter contract")
    if tuple(names) != expected_names:
        raise ValueError("parameter_signature names are not canonical")
    canonical = tuple((name, float.fromhex(value).hex()) for name, value in signature)
    if signature != canonical:
        raise ValueError("parameter_signature must use canonical float hex values")


__all__ = [
    "MAX_OBSERVATION_COUNT",
    "MAX_TOTAL_UNITS",
    "LogLikelihoodErrorCode",
    "LogLikelihoodFailure",
    "LogLikelihoodResult",
    "LogLikelihoodState",
    "LogLikelihoodSuccess",
    "reduce_log_likelihood_chunks",
]
