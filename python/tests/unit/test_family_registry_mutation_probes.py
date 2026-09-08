"""Mutation probes for closed evaluated-family metadata contracts."""

from __future__ import annotations

import unittest

from veridist.families.registry import (
    FAMILY_REGISTRY,
    FamilyId,
    FamilyRegistry,
    FamilySpec,
    Operation,
    ParameterRole,
    ParameterSpec,
)


def _valid_family(**changes: object) -> FamilySpec:
    values: dict[str, object] = {
        "id": FamilyId.NORMAL,
        "aliases": (),
        "parameters": (ParameterSpec("scale", ParameterRole.POSITIVE),),
        "fixed_location": None,
        "planned_operations": frozenset({Operation.LOGPDF}),
        "available_operations": frozenset(),
    }
    values.update(changes)
    return FamilySpec(**values)  # type: ignore[arg-type]


class FamilyRegistryMutationProbeTests(unittest.TestCase):
    """Kill boundary mutants rather than merely executing the happy path."""

    def test_parameter_spec_rejects_invalid_names_and_roles(self) -> None:
        for name in ("", 1):
            with self.subTest(name=name):
                with self.assertRaises(TypeError):
                    ParameterSpec(name, ParameterRole.FINITE)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ParameterSpec("value", "finite")  # type: ignore[arg-type]

    def test_family_spec_rejects_all_structural_mutants(self) -> None:
        invalid_cases = (
            {"id": "normal"},
            {"aliases": ["alias"]},
            {"aliases": ("alias", "alias")},
            {"aliases": ("normal",)},
            {"parameters": ()},
            {"parameters": (object(),)},
            {
                "parameters": (
                    ParameterSpec("same", ParameterRole.FINITE),
                    ParameterSpec("same", ParameterRole.POSITIVE),
                )
            },
            {"fixed_location": "0"},
            {"fixed_location": 1.0},
            {"planned_operations": set()},
            {"planned_operations": frozenset({object()})},
            {"available_operations": set()},
            {"available_operations": frozenset({object()})},
            {
                "planned_operations": frozenset(),
                "available_operations": frozenset({Operation.LOGPDF}),
            },
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    _valid_family(**changes)

    def test_registry_constructor_rejects_non_tuple_empty_and_invalid_items(self) -> None:
        for families in ([], (), (object(),)):
            with self.subTest(families=families):
                with self.assertRaises(TypeError):
                    FamilyRegistry(families)  # type: ignore[arg-type]

    def test_wrong_operation_type_is_not_silently_supported(self) -> None:
        with self.assertRaises(TypeError):
            FAMILY_REGISTRY.resolve("normal").supports("logpdf")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            FAMILY_REGISTRY.resolve("normal").plans("logpdf")  # type: ignore[arg-type]

    def test_registry_rejects_each_identity_collision_class_exactly(self) -> None:
        gamma = _valid_family(id=FamilyId.GAMMA)
        cases = (
            (
                "duplicate-canonical",
                (_valid_family(), _valid_family(aliases=("bell",))),
                "canonical family id collision",
            ),
            (
                "canonical-versus-alias",
                (_valid_family(aliases=("gamma",)), gamma),
                "alias collision",
            ),
            (
                "alias-versus-alias",
                (
                    _valid_family(aliases=("shared",)),
                    _valid_family(id=FamilyId.GAMMA, aliases=("shared",)),
                ),
                "alias collision",
            ),
            (
                "normalized-aliases",
                (
                    _valid_family(aliases=("a_b",)),
                    _valid_family(id=FamilyId.GAMMA, aliases=("ab",)),
                ),
                "alias normalization collision",
            ),
        )
        for name, families, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"^{message}$"):
                    FamilyRegistry(families)

    def test_registry_resolve_is_exact_closed_and_preserves_spec_identity(self) -> None:
        normal = _valid_family(aliases=("bell",))
        gamma = _valid_family(id=FamilyId.GAMMA, aliases=("shape_scale",))
        registry = FamilyRegistry((normal, gamma))
        self.assertIs(registry.resolve("normal"), normal)
        self.assertIs(registry.resolve("bell"), normal)
        self.assertIs(registry.resolve("gamma"), gamma)
        self.assertIs(registry.resolve("shape_scale"), gamma)
        self.assertEqual(tuple(registry), (FamilyId.NORMAL, FamilyId.GAMMA))
        self.assertEqual(registry.list(), (normal, gamma))

        class StringSubclass(str):
            pass

        for name in ("", " normal", "normal ", "NORMAL", "unknown"):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "^unknown evaluated family$"
            ):
                registry.resolve(name)
        for value in (StringSubclass("normal"), b"normal", None, FamilyId.NORMAL):
            with self.subTest(value=value), self.assertRaisesRegex(
                TypeError, "^family name must be a built-in string$"
            ):
                registry.resolve(value)  # type: ignore[arg-type]
