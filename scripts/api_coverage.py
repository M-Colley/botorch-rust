#!/usr/bin/env python3
"""Report which parts of BoTorch's public API botorch_rs implements.

Walks every module of BoTorch (and of ``botorch_community``) and collects the
public classes and functions defined there: names without a leading underscore
whose ``__module__`` is that module, so re-exports are counted once. Each name
is looked up in botorch_rs under the same module path or an enclosing package
(botorch_rs re-exports names in package ``__init__`` files). For names that
exist in both, the report lists the parameters of BoTorch's signature (of
``__init__`` for classes) that botorch_rs does not accept, and for classes the
public methods and properties BoTorch defines that botorch_rs lacks.

The result is written as Markdown (``docs/api_coverage.md`` in the
repository)::

    PYTHONPATH=/path/to/botorch:botorch_pyo3/python python scripts/api_coverage.py \\
        --out docs/api_coverage.md
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import botorch_rs

# Test utilities and private helpers rather than library API.
SKIP_MODULES = (
    "botorch.utils.testing",
    "botorch.utils.test_helpers",
    "botorch_community.utils.testing",
)


@dataclass
class Entry:
    module: str
    name: str
    kind: str  # "class" or "function"
    ported: bool = False
    missing_params: list[str] = field(default_factory=list)
    missing_members: list[str] = field(default_factory=list)


def walk(package: str) -> tuple[list[ModuleType], dict[str, str]]:
    """Import every module of ``package``; return them and the import errors."""
    root = importlib.import_module(package)
    modules, errors = [root], {}
    for info in pkgutil.walk_packages(root.__path__, f"{package}."):
        if info.name.startswith(SKIP_MODULES) or any(
            part.startswith("_") for part in info.name.split(".")
        ):
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except Exception as e:  # optional dependencies of some modules
            errors[info.name] = f"{type(e).__name__}: {e}".splitlines()[0][:120]
    return modules, errors


def public_objects(module: ModuleType) -> list[tuple[str, Any, str]]:
    out = []
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj):
            out.append((name, obj, "class"))
        elif inspect.isfunction(obj):
            out.append((name, obj, "function"))
    return out


def counterpart(module: str, name: str) -> Any | None:
    """The object of the same name in botorch_rs, at the same module path or in
    an enclosing package."""
    parts = module.split(".")
    parts[0] = "botorch_rs"
    for k in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        obj = getattr(mod, name, None)
        if obj is not None and not isinstance(obj, ModuleType):
            return obj
    return None


def parameters(obj: Any) -> tuple[set[str], bool] | None:
    """Named parameters of a callable (of ``__init__`` for classes) and whether
    it accepts ``**kwargs``; None if there is no signature."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    names, var_kw = set(), False
    for p in sig.parameters.values():
        if p.kind is p.VAR_KEYWORD:
            var_kw = True
        elif p.kind is not p.VAR_POSITIONAL and p.name != "self":
            names.add(p.name)
    return names, var_kw


def botorch_members(cls: type) -> set[str]:
    """Public methods and properties that BoTorch's own classes define for
    ``cls``: not those inherited from torch, GPyTorch or ``object``, and not
    overrides of ``torch.nn.Module``'s API (``forward``, ``load_state_dict``,
    ...), which botorch_rs does not provide."""
    import torch

    module_api = set(dir(torch.nn.Module))
    out = set()
    for klass in cls.__mro__:
        if not klass.__module__.startswith(("botorch.", "botorch_community.")):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in module_api:
                continue
            if inspect.isfunction(value) or isinstance(
                value, (property, classmethod, staticmethod)
            ):
                out.add(name)
    return out


def analyze(package: str) -> tuple[list[Entry], dict[str, str]]:
    modules, errors = walk(package)
    entries = []
    for module in modules:
        for name, obj, kind in public_objects(module):
            entry = Entry(module.__name__, name, kind)
            rs = counterpart(module.__name__, name) if package == "botorch" else None
            if rs is not None:
                entry.ported = True
                bt_params, rs_params = parameters(obj), parameters(rs)
                if bt_params and rs_params and not rs_params[1]:
                    entry.missing_params = sorted(bt_params[0] - rs_params[0])
                if kind == "class":
                    entry.missing_members = sorted(
                        m for m in botorch_members(obj) if not hasattr(rs, m)
                    )
            entries.append(entry)
    return entries, errors


def area_of(module: str) -> str:
    """``botorch.acquisition.multi_objective.logei`` -> ``acquisition.multi_objective``."""
    parts = module.split(".")[1:]
    if not parts:
        return "(top level)"
    pkg = importlib.import_module(".".join(module.split(".")[:2]))
    if len(parts) >= 2 and hasattr(pkg, "__path__"):
        sub = importlib.import_module(".".join(module.split(".")[:3]))
        if hasattr(sub, "__path__"):
            return ".".join(parts[:2])
    return parts[0]


def git_revision(path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def report(entries: list[Entry], community: list[Entry], errors: dict[str, str]) -> str:
    import botorch

    rev = git_revision(botorch.__file__)
    lines = [
        "# BoTorch API coverage of botorch-rust",
        "",
        "Generated by `scripts/api_coverage.py` on "
        f"{datetime.date.today().isoformat()} from BoTorch {botorch.__version__}"
        + (f" (commit {rev})" if rev else "")
        + f" and botorch_rs {botorch_rs.__version__}. It lists every public class and "
        "function defined in a BoTorch module (re-exports counted once) and whether "
        "botorch_rs provides it under the same path. For names that exist in both, it "
        "lists the parameters of BoTorch's signature (`__init__` for classes) that "
        "botorch_rs does not accept and the public methods and properties of BoTorch's "
        "class that botorch_rs lacks (botorch_rs models and acquisition functions are not "
        "`torch.nn.Module`s, so `forward`, state dicts and buffers are not listed). A name "
        "being present does not mean every "
        "configuration is supported: see the README for the supported options.",
        "",
        "## Summary",
        "",
        "| Area | Public classes and functions | In botorch_rs | Share |",
        "|---|---|---|---|",
    ]
    by_area: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        by_area[area_of(e.module)].append(e)
    total = ported = 0
    for area in sorted(by_area):
        es = by_area[area]
        n, k = len(es), sum(e.ported for e in es)
        total, ported = total + n, ported + k
        lines.append(f"| `{area}` | {n} | {k} | {100 * k / n:.0f}% |")
    lines.append(
        f"| **all of `botorch`** | **{total}** | **{ported}** | **{100 * ported / total:.0f}%** |"
    )
    lines.append(f"| `botorch_community` | {len(community)} | 0 | 0% |")
    lines += ["", "## Ported, with differences", ""]
    lines.append(
        "Parameters and members of BoTorch's API that the botorch_rs counterpart does not have."
    )
    lines.append("")
    lines.append("| Name | Missing parameters | Missing methods and properties |")
    lines.append("|---|---|---|")
    for e in sorted(entries, key=lambda e: (e.module, e.name)):
        if e.ported and (e.missing_params or e.missing_members):
            params = ", ".join(f"`{p}`" for p in e.missing_params) or ""
            members = ", ".join(f"`{m}`" for m in e.missing_members) or ""
            lines.append(f"| `{e.module}.{e.name}` | {params} | {members} |")
    lines += ["", "## Not ported", ""]
    for area in sorted(by_area):
        missing = [e for e in by_area[area] if not e.ported]
        if not missing:
            continue
        lines += [f"### `{area}` ({len(missing)} of {len(by_area[area])})", ""]
        by_module: dict[str, list[str]] = defaultdict(list)
        for e in missing:
            by_module[e.module].append(e.name)
        for module in sorted(by_module):
            names = ", ".join(f"`{n}`" for n in sorted(by_module[module]))
            lines.append(f"- `{module}`: {names}")
        lines.append("")
    lines += [
        "## `botorch_community`",
        "",
        "Contributed models and acquisition functions; none are ported.",
        "",
    ]
    by_module = defaultdict(list)
    for e in community:
        by_module[e.module].append(e.name)
    for module in sorted(by_module):
        lines.append(f"- `{module}`: " + ", ".join(f"`{n}`" for n in sorted(by_module[module])))
    if errors:
        lines += [
            "",
            "## Modules that could not be imported",
            "",
            "Their names are not counted above.",
            "",
        ]
        lines += [f"- `{m}`: {err}" for m, err in sorted(errors.items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/api_coverage.md"))
    parser.add_argument("--json-out", type=Path, help="Write a machine-readable API gap inventory.")
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.0,
        help="Fail if public symbol availability is below this percentage (0 to 100).",
    )
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Fail for missing symbols, parameters, members or unimportable BoTorch modules. "
        "Passing this check still requires separate numerical parity and performance validation.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.minimum_coverage <= 100.0:
        parser.error("--minimum-coverage must lie between 0 and 100")
    warnings.simplefilter("ignore")
    entries, errors = analyze("botorch")
    try:
        community, community_errors = analyze("botorch_community")
    except ImportError as e:
        community, community_errors = [], {"botorch_community": str(e)}
    errors.update(community_errors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(entries, community, errors), encoding="utf-8")
    n, k = len(entries), sum(e.ported for e in entries)
    print(f"{k} of {n} public BoTorch classes and functions are in botorch_rs; wrote {args.out}")
    missing = [e for e in entries if not e.ported or e.missing_params or e.missing_members]
    core_errors = {name: error for name, error in errors.items() if name.startswith("botorch.")}
    if args.json_out is not None:
        import botorch

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "date": datetime.date.today().isoformat(),
                    "reference_version": botorch.__version__,
                    "reference_commit": git_revision(botorch.__file__),
                    "rust_version": botorch_rs.__version__,
                    "public_symbols": n,
                    "available_symbols": k,
                    "availability_percent": 100 * k / n if n else 0,
                    "api_gaps": [asdict(e) for e in missing],
                    "import_errors": errors,
                    "note": "Symbol availability is not proof of numerical parity, supported "
                    "configurations, autograd compatibility or performance.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    sys.stdout.flush()
    if not n or 100 * k < args.minimum_coverage * n:
        print("Public symbol availability is below --minimum-coverage.", file=sys.stderr)
        return 1
    if args.require_full and (missing or core_errors):
        print(
            f"Full API coverage is not achieved: {len(missing)} symbols have API gaps; "
            f"{len(core_errors)} BoTorch modules could not be imported.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
