from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import PermittedUse, ProviderProfile, Redistribution


CONFIG_FILES = (
    "providers.yaml",
    "ttl_table.yaml",
    "price_table.yaml",
)


@dataclass(frozen=True)
class ConfigBundle:
    """Resolved configuration loaded from `config/` YAML files."""

    providers: list[ProviderProfile]
    ttl_table: dict[str, dict[str, Any]]
    price_table: dict[str, Any]
    config_dir: Path


def _candidate_dirs(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    here = Path.cwd()
    out = [here / "config"]
    for parent in here.parents:
        out.append(parent / "config")
    # Fall back to the install/repo root relative to THIS file, so the config
    # is found even when the process was launched from an unrelated cwd (e.g.
    # `streamlit run` started from the user's home directory). src layout:
    # <root>/src/optagent/config_loader.py  ->  parents[2] == <root>.
    pkg_file = Path(__file__).resolve()
    out.append(pkg_file.parent / "config")  # bundled inside the package
    for up in (2, 1, 3):
        try:
            out.append(pkg_file.parents[up] / "config")
        except IndexError:
            pass
    return out


def find_config_dir(explicit: Path | None = None) -> Path:
    """Locate the first directory upwards from cwd that contains the config files."""

    for d in _candidate_dirs(explicit):
        if all((d / name).exists() for name in CONFIG_FILES):
            return d
    raise FileNotFoundError(
        f"no config directory found containing {list(CONFIG_FILES)} starting from {Path.cwd()}"
    )


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_provider_profiles(path: Path) -> list[ProviderProfile]:
    raw = _load_yaml(path)
    if not isinstance(raw, dict) or "profiles" not in raw:
        raise ValueError(f"{path}: expected top-level 'profiles' list")
    profiles: list[ProviderProfile] = []
    for entry in raw["profiles"]:
        profiles.append(
            ProviderProfile(
                id=entry["id"],
                permitted_use=PermittedUse(entry["permitted_use"]),
                redistribution=Redistribution(entry["redistribution"]),
                entitlement_required=entry.get("entitlement_required", False),
                rate_limit_qpm=entry.get("rate_limit_qpm"),
                attribution_string=entry.get("attribution_string"),
                terms_url=entry["terms_url"],
                profile_version=str(entry["profile_version"]),
                required_notices=tuple(entry.get("required_notices") or ()),
            )
        )
    return profiles


def load_ttl_table(path: Path) -> dict[str, dict[str, Any]]:
    raw = _load_yaml(path)
    if not isinstance(raw, dict) or "data_types" not in raw:
        raise ValueError(f"{path}: expected top-level 'data_types' mapping")
    return raw["data_types"]


def load_price_table(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(f"{path}: expected top-level 'models' mapping")
    return raw


def load_bundle(config_dir: Path | None = None) -> ConfigBundle:
    base = find_config_dir(config_dir)
    return ConfigBundle(
        providers=load_provider_profiles(base / "providers.yaml"),
        ttl_table=load_ttl_table(base / "ttl_table.yaml"),
        price_table=load_price_table(base / "price_table.yaml"),
        config_dir=base,
    )
