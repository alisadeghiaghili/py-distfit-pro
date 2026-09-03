"""Dependency-free validation and render checks for the documentation tree."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

SUPPORTED_DIRECTIONS = {"en": "ltr", "fa": "rtl", "de": "ltr"}


def _is_machine_literal(value: str) -> bool:
    """Return whether a gettext message is an intentionally stable API token."""

    return bool(re.fullmatch(r"`[^`\r\n]+`(?:,\s*`[^`\r\n]+`)*", value)) or value in {
        r"\operatorname{LL}",
        r"\operatorname{LL} = \operatorname{round}_{binary64}\left(\sum_i \log f(x_i)\right)",
    }


def direction_for(locale: str) -> str:
    """Return the declared writing direction; reject undeclared fallback locales."""

    try:
        return SUPPORTED_DIRECTIONS[locale]
    except KeyError as error:
        raise ValueError(f"unsupported documentation locale: {locale}") from error


def _decode_po_string(value: str) -> str:
    decoded = ast.literal_eval(value)
    if not isinstance(decoded, str):
        raise ValueError(f"invalid PO string: {value}")
    return decoded


def _parse_po(path: Path) -> dict[str, str]:
    """Parse the msgid/msgstr subset emitted by Sphinx for singular messages."""

    entries: dict[str, str] = {}
    msgid: list[str] = []
    msgstr: list[str] = []
    active: list[str] | None = None

    def flush() -> None:
        nonlocal msgid, msgstr, active
        source = "".join(msgid)
        translation = "".join(msgstr)
        if source:
            if source in entries:
                raise ValueError(f"duplicate msgid in locale catalog: {source!r}")
            entries[source] = translation
        msgid = []
        msgstr = []
        active = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            flush()
        elif line.startswith("#"):
            continue
        elif line.startswith("msgid "):
            if msgid or msgstr:
                flush()
            active = msgid
            active.append(_decode_po_string(line[6:]))
        elif line.startswith("msgstr "):
            active = msgstr
            active.append(_decode_po_string(line[7:]))
        elif line.startswith('"') and active is not None:
            active.append(_decode_po_string(line))
        else:
            raise ValueError(f"unsupported PO syntax in {path}:{line_number}: {raw_line}")
    flush()
    return entries


def validate_parity(docs_root: Path) -> dict[str, Any]:
    """Validate page/message IDs and complete non-fallback FA/DE catalogs."""

    manifest_path = docs_root / "i18n" / "parity-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported parity manifest schema")
    if manifest.get("canonical_locale") != "en":
        raise ValueError("English must be the canonical documentation locale")
    if manifest.get("required_locales") != ["en", "fa", "de"]:
        raise ValueError("required locales must be exactly en, fa, de")

    pages = manifest.get("pages", [])
    messages = manifest.get("messages", [])
    page_ids = [page["id"] for page in pages]
    message_ids = [message["id"] for message in messages]
    if len(page_ids) != len(set(page_ids)) or len(message_ids) != len(set(message_ids)):
        raise ValueError("manifest page and message IDs must be unique")

    source_root = docs_root / "source"
    for page in pages:
        source_text = (source_root / page["source"]).read_text(encoding="utf-8")
        if f"({page['anchor']})=" not in source_text:
            raise ValueError(f"missing stable anchor for {page['id']}")
    for message in messages:
        source_text = (source_root / message["page"]).read_text(encoding="utf-8")
        normalized_source = re.sub(r"\s+", " ", source_text)
        if message["source"] not in normalized_source:
            raise ValueError(f"canonical message is absent for {message['id']}")

    missing: dict[str, list[str]] = {}
    fallbacks: dict[str, list[str]] = {}
    localized = [locale for locale in manifest["required_locales"] if locale != "en"]
    for locale in localized:
        catalog_entries: dict[str, str] = {}
        catalog_root = docs_root / "locales" / locale / "LC_MESSAGES"
        for catalog in sorted(catalog_root.glob("*.po")):
            for source, translation in _parse_po(catalog).items():
                if source in catalog_entries:
                    raise ValueError(f"duplicate msgid across {locale} catalogs: {source!r}")
                catalog_entries[source] = translation
        for message in messages:
            message_id = message["id"]
            source = message["source"]
            translation = catalog_entries.get(source, "")
            if not translation.strip():
                missing.setdefault(locale, []).append(message_id)
            elif translation.strip() == source.strip() and not _is_machine_literal(source):
                fallbacks.setdefault(locale, []).append(message_id)

    return {
        "locales": localized,
        "page_count": len(pages),
        "message_count": len(messages),
        "missing": missing,
        "fallbacks": fallbacks,
    }


def _html_files(output_directory: Path) -> list[Path]:
    files = sorted(output_directory.rglob("*.html"))
    if not files:
        raise ValueError(f"no rendered HTML files in {output_directory}")
    return files


def assert_rendered_direction(output_directory: Path, locale: str) -> None:
    """Require every rendered page to declare its locale and direction."""

    direction = direction_for(locale)
    for html_file in _html_files(output_directory):
        content = html_file.read_text(encoding="utf-8")
        html_tag = re.search(r"<html\b[^>]*>", content, flags=re.IGNORECASE)
        if html_tag is None:
            raise ValueError(f"missing html element in {html_file}")
        tag = html_tag.group(0)
        if re.search(rf'\blang=["\']{re.escape(locale)}["\']', tag) is None:
            raise ValueError(f"missing lang={locale} in {html_file}")
        if re.search(rf'\bdir=["\']{direction}["\']', tag) is None:
            raise ValueError(f"missing dir={direction} in {html_file}")


def assert_rtl_smoke(output_directory: Path) -> None:
    """Require the Persian render to load RTL CSS and preserve a code sample."""

    contents = [path.read_text(encoding="utf-8") for path in _html_files(output_directory)]
    if not any("rtl.css" in content for content in contents):
        raise ValueError("Persian render does not load rtl.css")
    if not any(re.search(r"<(code|pre)\b", content, flags=re.IGNORECASE) for content in contents):
        raise ValueError("Persian RTL smoke has no rendered code block")


def postprocess_rendered_html(app: Any, exception: BaseException | None) -> None:
    """Sphinx build-finished hook that makes direction explicit in static HTML."""

    if exception is not None or getattr(app.builder, "format", None) != "html":
        return
    locale = app.config.language or "en"
    direction = direction_for(locale)
    for html_file in _html_files(Path(app.outdir)):
        content = html_file.read_text(encoding="utf-8")

        def replace_tag(match: re.Match[str]) -> str:
            tag = match.group(0)
            tag = re.sub(r"\s+dir=([\"\']).*?\1", "", tag, flags=re.IGNORECASE)
            if re.search(r"\slang=", tag, flags=re.IGNORECASE) is None:
                tag = tag[:-1] + f' lang="{locale}">'
            return tag[:-1] + f' dir="{direction}">'

        updated, count = re.subn(
            r"<html\b[^>]*>", replace_tag, content, count=1, flags=re.IGNORECASE
        )
        if count != 1:
            raise ValueError(f"cannot apply direction to {html_file}")
        html_file.write_text(updated, encoding="utf-8")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="validate translation parity")
    check_parser.add_argument("docs_root", nargs="?", type=Path, default=Path(__file__).parent)
    render_parser = subparsers.add_parser("render", help="validate rendered locale HTML")
    render_parser.add_argument("output_directory", type=Path)
    render_parser.add_argument("locale", choices=tuple(SUPPORTED_DIRECTIONS))
    arguments = parser.parse_args()

    if arguments.command == "check":
        report = validate_parity(arguments.docs_root)
        if report["missing"] or report["fallbacks"]:
            raise SystemExit(json.dumps(report, ensure_ascii=False, sort_keys=True))
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    assert_rendered_direction(arguments.output_directory, arguments.locale)
    if arguments.locale == "fa":
        assert_rtl_smoke(arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
