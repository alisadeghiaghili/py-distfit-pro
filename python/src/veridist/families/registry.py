"""Immutable metadata contracts for evaluated distribution families.

This module deliberately declares only family identity, parameters, and the
single future log-density operation.  It performs no numerical evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Final, cast


class FamilyId(StrEnum):
    """Canonical identifiers for the closed evaluated-family set."""

    NORMAL = "normal"
    GAMMA = "gamma"
    WEIBULL_MIN = "weibull_min"
    LOGNORMAL = "lognormal"
    GUMBEL_RIGHT = "gumbel_right"


class Operation(StrEnum):
    """Operations that a family may explicitly support."""

    LOGPDF = "logpdf"


class ParameterRole(StrEnum):
    """Closed numerical validation roles for canonical parameters."""

    FINITE = "finite"
    POSITIVE = "positive"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One immutable parameter name and its numerical boundary contract."""

    name: str
    role: ParameterRole

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("parameter name must be a non-empty built-in string")
        if type(self.role) is not ParameterRole:
            raise TypeError("parameter role must be a ParameterRole")

    def validate(self, value: object) -> float:
        """Validate a built-in finite real value for this parameter."""

        if type(value) not in (int, float):
            raise TypeError(f"{self.name} must be a built-in real number")
        numeric = float(cast(int | float, value))
        if not isfinite(numeric):
            raise ValueError(f"{self.name} must be finite")
        if self.role is ParameterRole.POSITIVE and numeric <= 0.0:
            raise ValueError(f"{self.name} must be positive")
        return numeric


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Immutable operation-level metadata for exactly one family."""

    id: FamilyId
    aliases: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    fixed_location: float | None
    operations: frozenset[Operation]

    def __post_init__(self) -> None:
        if type(self.id) is not FamilyId:
            raise TypeError("family id must be a FamilyId")
        aliases_are_invalid = type(self.aliases) is not tuple or any(
            type(alias) is not str or not alias for alias in self.aliases
        )
        if aliases_are_invalid:
            raise TypeError("aliases must be a tuple of non-empty built-in strings")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("family aliases must be unique")
        if self.id.value in self.aliases:
            raise ValueError("a family alias cannot repeat its canonical id")
        if type(self.parameters) is not tuple or not self.parameters:
            raise TypeError("parameters must be a non-empty tuple")
        if any(type(parameter) is not ParameterSpec for parameter in self.parameters):
            raise TypeError("parameters must contain ParameterSpec values")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        if self.fixed_location is not None:
            if type(self.fixed_location) not in (int, float):
                raise TypeError("fixed location must be a built-in real number or None")
            if float(self.fixed_location) != 0.0:
                raise ValueError("the evaluated zero-location families fix location at zero")
        if type(self.operations) is not frozenset or not self.operations:
            raise TypeError("operations must be a non-empty frozenset")
        if any(type(operation) is not Operation for operation in self.operations):
            raise TypeError("operations must contain Operation values")

    @property
    def free_parameter_count(self) -> int:
        """Return the declared parameter count, never a caller mapping length."""

        return len(self.parameters)

    def supports(self, operation: Operation) -> bool:
        """Return whether this declared metadata contract supports an operation."""

        if type(operation) is not Operation:
            raise TypeError("operation must be an Operation")
        return operation in self.operations

    def validate_parameters(self, **values: object) -> Mapping[str, float]:
        """Validate the exact canonical parameter set without evaluating a density."""

        expected_names = tuple(parameter.name for parameter in self.parameters)
        if set(values) != set(expected_names):
            raise TypeError("parameter keys must equal the canonical parameter tuple")
        validated = {
            parameter.name: parameter.validate(values[parameter.name])
            for parameter in self.parameters
        }
        return MappingProxyType(validated)


class FamilyRegistry:
    """Closed, deterministic lookup for immutable family specifications."""

    __slots__ = ("_families", "_lookup", "_ordered")

    def __init__(self, families: tuple[FamilySpec, ...]) -> None:
        if type(families) is not tuple or not families:
            raise TypeError("families must be a non-empty tuple of FamilySpec values")
        if any(type(family) is not FamilySpec for family in families):
            raise TypeError("families must contain FamilySpec values")
        family_mapping: dict[FamilyId, FamilySpec] = {}
        lookup: dict[str, FamilySpec] = {}
        for family in families:
            if family.id in family_mapping:
                raise ValueError("canonical family id collision")
            family_mapping[family.id] = family
            for name in (family.id.value, *family.aliases):
                if name in lookup:
                    raise ValueError("alias collision")
                lookup[name] = family
        self._families = MappingProxyType(family_mapping)
        self._lookup = MappingProxyType(lookup)
        self._ordered = families

    @property
    def families(self) -> Mapping[FamilyId, FamilySpec]:
        """Expose the canonical mapping without mutation capability."""

        return self._families

    def __iter__(self) -> Iterator[FamilyId]:
        """Iterate canonical identifiers in declared, stable order."""

        return iter(family.id for family in self._ordered)

    def list(self) -> tuple[FamilySpec, ...]:
        """Return the immutable specifications in declared, stable order."""

        return self._ordered

    def resolve(self, name: str) -> FamilySpec:
        """Resolve an exact canonical ID or explicitly declared alias."""

        if type(name) is not str:
            raise TypeError("family name must be a built-in string")
        try:
            return self._lookup[name]
        except KeyError as error:
            raise ValueError("unknown evaluated family") from error


_LOGPDF: Final = frozenset({Operation.LOGPDF})
_FAMILY_SPECS: Final = (
    FamilySpec(
        FamilyId.NORMAL,
        ("gaussian",),
        (
            ParameterSpec("mu", ParameterRole.FINITE),
            ParameterSpec("sigma", ParameterRole.POSITIVE),
        ),
        None,
        _LOGPDF,
    ),
    FamilySpec(
        FamilyId.GAMMA,
        (),
        (
            ParameterSpec("shape", ParameterRole.POSITIVE),
            ParameterSpec("scale", ParameterRole.POSITIVE),
        ),
        0.0,
        _LOGPDF,
    ),
    FamilySpec(
        FamilyId.WEIBULL_MIN,
        ("weibull",),
        (
            ParameterSpec("shape", ParameterRole.POSITIVE),
            ParameterSpec("scale", ParameterRole.POSITIVE),
        ),
        0.0,
        _LOGPDF,
    ),
    FamilySpec(
        FamilyId.LOGNORMAL,
        (),
        (
            ParameterSpec("mu_log", ParameterRole.FINITE),
            ParameterSpec("sigma_log", ParameterRole.POSITIVE),
        ),
        0.0,
        _LOGPDF,
    ),
    FamilySpec(
        FamilyId.GUMBEL_RIGHT,
        ("gumbel",),
        (
            ParameterSpec("location", ParameterRole.FINITE),
            ParameterSpec("scale", ParameterRole.POSITIVE),
        ),
        None,
        _LOGPDF,
    ),
)

FAMILY_REGISTRY: Final = FamilyRegistry(_FAMILY_SPECS)


def list_families() -> tuple[FamilySpec, ...]:
    """List evaluated family metadata in canonical registry order."""

    return FAMILY_REGISTRY.list()


__all__ = [
    "FAMILY_REGISTRY",
    "FamilyId",
    "FamilyRegistry",
    "FamilySpec",
    "Operation",
    "ParameterRole",
    "ParameterSpec",
    "list_families",
]
