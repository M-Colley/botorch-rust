"""Certify complementary broad/equivalence reports for one frozen Python release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_FINGERPRINT_KEYS = (
    "native_sha256",
    "python_sources_sha256",
    "helper_sha256",
    "source_revision",
    "reference_revision",
)


def _ids(value: Any, label: str, errors: list[str]) -> Counter[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        errors.append(f"{label} must be a list of nonempty case IDs.")
        return Counter()
    result = Counter(value)
    if any(count != 1 for count in result.values()):
        errors.append(f"{label} contains duplicate IDs.")
    return result


def _audit_phase(
    report: dict[str, Any], phase: str, errors: list[str]
) -> dict[str, Any]:
    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{phase}: {message}")

    check(report.get("phase") == phase, "phase label differs from its input role.")
    check(
        type(report.get("exit_code")) is int and report["exit_code"] == 0,
        "pytest did not exit successfully.",
    )
    for name in (
        "phases_valid",
        "case_ids_exactly_once",
        "fingerprints_stable",
        "tracked_trees_clean",
    ):
        check(report.get(name) is True, f"{name} is not true.")
    check(
        report.get("duplicate_selected_ids") == [],
        "duplicate_selected_ids is not empty.",
    )
    check(report.get("phase_errors") == [], "phase_errors is not empty.")
    selected = _ids(report.get("selected"), f"{phase}.selected", errors)
    deselected = _ids(report.get("deselected"), f"{phase}.deselected", errors)
    check(bool(selected), "no tests were selected.")
    check(
        not (selected.keys() & deselected.keys()),
        "selected and deselected IDs overlap.",
    )
    skipped_collection = _ids(
        report.get("collection_skips"), f"{phase}.collection_skips", errors
    )
    cases = report.get("cases")
    if not isinstance(cases, dict):
        check(False, "cases must be a mapping.")
        cases = {}
    check(
        cases.keys() == selected.keys(),
        "case records do not match selected IDs exactly.",
    )
    counts: Counter[str] = Counter()
    incomplete, bad_phases = [], []
    for nodeid, case in cases.items():
        if not isinstance(case, dict):
            incomplete.append(nodeid)
            continue
        outcome = case.get("outcome")
        if outcome not in ("passed", "skipped"):
            incomplete.append(nodeid)
        else:
            counts[outcome] += 1
        events = case.get("reports")
        if not isinstance(events, dict):
            bad_phases.append(nodeid)
            continue
        setup, call, teardown = (
            events.get(name, []) for name in ("setup", "call", "teardown")
        )
        if (
            set(events) - {"setup", "call", "teardown"}
            or not all(isinstance(value, list) for value in (setup, call, teardown))
            or len(setup) != 1
            or len(teardown) != 1
            or len(call) != (1 if setup == ["passed"] else 0)
            or any(
                value not in ("passed", "skipped")
                for value in (*setup, *call, *teardown)
            )
            or (
                outcome == "passed"
                and (setup, call, teardown) != (["passed"], ["passed"], ["passed"])
            )
            or (outcome == "skipped" and "skipped" not in (*setup, *call, *teardown))
        ):
            bad_phases.append(nodeid)
    check(not incomplete, f"cases failed or did not complete: {incomplete[:8]}.")
    check(
        not bad_phases,
        f"execution phases are missing, repeated or inconsistent: {bad_phases[:8]}.",
    )
    declared = report.get("totals")
    check(
        isinstance(declared, dict)
        and all(type(value) is int and value >= 0 for value in declared.values())
        and Counter(declared) == counts,
        "declared totals do not equal the completed case ledger.",
    )
    before, after = report.get("fingerprints_before"), report.get("fingerprints_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        check(False, "fingerprints_before/after must be mappings.")
        before, after = {}, {}
    check(before == after, "pre/post fingerprints differ.")
    for key in _FINGERPRINT_KEYS:
        check(
            isinstance(before.get(key), str) and bool(before[key]),
            f"missing {key} fingerprint.",
        )
        check(
            report.get(key) == before.get(key),
            f"top-level {key} differs from its fingerprint.",
        )
    for key in ("source_tree_status", "reference_tree_status"):
        check(before.get(key) == after.get(key) == "", f"{key} is missing or dirty.")
    return {
        "selected": selected,
        "deselected": deselected,
        "collection_skips": skipped_collection,
        "totals": counts,
        "fingerprints": before,
    }


def certify(broad: dict[str, Any], equivalence: dict[str, Any]) -> dict[str, Any]:
    """Validate individual ledgers and their exact complementary case partition."""
    errors: list[str] = []
    first = _audit_phase(broad, "broad", errors)
    second = _audit_phase(equivalence, "equivalence", errors)
    if first["selected"] != second["deselected"]:
        errors.append("Broad selected IDs do not equal equivalence deselected IDs.")
    if second["selected"] != first["deselected"]:
        errors.append("Equivalence selected IDs do not equal broad deselected IDs.")
    if first["selected"].keys() & second["selected"].keys():
        errors.append("The two selected phases overlap.")
    if first["collection_skips"] != second["collection_skips"]:
        errors.append("Collection skips differ between phases.")
    if first["fingerprints"] != second["fingerprints"]:
        errors.append(
            "The phases verified different source, reference, helper or installed artifacts."
        )
    for key in (
        "package_path",
        "native_path",
        "python_executable",
        "platform",
        "python",
        "torch",
        "cuda_device",
        "environment",
    ):
        if key not in broad or key not in equivalence or broad[key] != equivalence[key]:
            errors.append(
                f"Runtime/provenance field {key} is missing or differs between phases."
            )
    canonical = sorted(first["selected"].keys() | first["deselected"].keys())
    executed = first["selected"] + second["selected"]
    if executed != Counter(canonical):
        errors.append(
            "The combined execution ledger is not exactly one run per canonical case ID."
        )
    counts = first["totals"] + second["totals"]
    return {
        "valid": not errors,
        "errors": errors,
        "canonical_case_count": len(canonical),
        "canonical_case_ids_sha256": hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "case_ids_exactly_once": not errors,
        "phase_totals": {
            "broad": dict(first["totals"]),
            "equivalence": dict(second["totals"]),
        },
        "totals": dict(counts),
        "collection_skips": sorted(first["collection_skips"]),
        "fingerprints": first["fingerprints"],
        **{key: first["fingerprints"].get(key) for key in _FINGERPRINT_KEYS},
        **{
            key: broad.get(key)
            for key in (
                "package_path",
                "native_path",
                "python_executable",
                "torch",
                "cuda_device",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sources = {"broad": args.broad.resolve(), "equivalence": args.equivalence.resolve()}
    destination = args.report.resolve()
    if destination in sources.values():
        parser.error("--report must not overwrite either phase report.")
    metadata: dict[str, Any] = {}
    try:
        reports = {}
        for phase, path in sources.items():
            data = path.read_bytes()
            report = json.loads(data)
            if not isinstance(report, dict):
                raise ValueError(f"{phase} report must be a JSON object.")
            reports[phase] = report
            metadata[phase] = {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        certificate = certify(reports["broad"], reports["equivalence"])
    except (OSError, ValueError, TypeError) as error:
        certificate = {"valid": False, "errors": [str(error)]}
    certificate.update(
        {
            "verified_utc": datetime.now(UTC).isoformat(),
            "phase_reports": metadata,
            "pair_verifier_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(destination),
                "valid": certificate["valid"],
                "totals": certificate.get("totals"),
                "errors": certificate["errors"],
            }
        )
    )
    return int(not certificate["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
