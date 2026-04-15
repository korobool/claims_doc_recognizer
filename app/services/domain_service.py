"""Domain description registry.

A "domain" is a free-form knowledge/skill description that contextualizes the
LLM's behavior for a particular application vertical (health insurance, motor
insurance, real estate, etc.). The active domain's description is injected into
LLM system prompts for both document extraction and schema generation so the
same codebase can serve multiple verticals without rebuilding.

Storage mirrors `document_schemas.py`:
    - One YAML file per domain under app/config/domains/
    - A single-line `active_domain.txt` holds the currently active domain_id
    - An in-memory cache is invalidated on save/delete and by the frontend on
      explicit reload

Each YAML file has three top-level keys:
    domain_id:    short identifier, must match the filename stem
    display_name: human-readable label
    description:  the free-form knowledge block that is injected into prompts
"""

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DOMAINS_DIR = Path(__file__).parent.parent / "config" / "domains"
ACTIVE_DOMAIN_FILE = Path(__file__).parent.parent / "config" / "active_domain.txt"
DEFAULT_DOMAIN_ID = "health_insurance"


@dataclass
class Domain:
    domain_id: str
    display_name: str
    description: str
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return {
            "domain_id": self.domain_id,
            "display_name": self.display_name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict, source_file: Optional[str] = None) -> "Domain":
        return cls(
            domain_id=(data.get("domain_id") or "").strip(),
            display_name=(data.get("display_name") or "").strip() or "Untitled Domain",
            description=data.get("description") or "",
            source_file=source_file,
        )


_domains_cache: Optional[Dict[str, Domain]] = None


def _load_domain_from_yaml(filepath: Path) -> Optional[Domain]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            print(f"[Domain] {filepath.name}: root must be a YAML mapping, skipping")
            return None
        domain = Domain.from_dict(data, source_file=str(filepath))
        if not domain.domain_id:
            print(f"[Domain] {filepath.name}: missing domain_id, skipping")
            return None
        return domain
    except Exception as e:
        print(f"[Domain] Error loading {filepath}: {e}")
        return None


def _load_all_domains() -> Dict[str, Domain]:
    domains: Dict[str, Domain] = {}
    if not DOMAINS_DIR.exists():
        DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
        return domains

    for yaml_file in sorted(DOMAINS_DIR.glob("*.yaml")):
        domain = _load_domain_from_yaml(yaml_file)
        if domain:
            domains[domain.domain_id] = domain
            print(f"[Domain] Loaded: {domain.domain_id} ({domain.display_name})")

    # Safety net: if everything was deleted / empty, synthesize a neutral
    # fallback so prompt composition never crashes on a missing description.
    if not domains:
        domains["general"] = Domain(
            domain_id="general",
            display_name="General",
            description=(
                "You are a general-purpose document processor. Be precise, "
                "preserve original formatting when uncertain, and extract "
                "what the schema asks for."
            ),
        )
    return domains


def get_domains() -> Dict[str, Domain]:
    global _domains_cache
    if _domains_cache is None:
        _domains_cache = _load_all_domains()
    return _domains_cache


def reload_domains() -> Dict[str, Domain]:
    global _domains_cache
    _domains_cache = None
    return get_domains()


def get_domain(domain_id: str) -> Optional[Domain]:
    return get_domains().get(domain_id)


def list_domains() -> List[Dict[str, str]]:
    """Metadata-only listing suitable for the sidebar."""
    return [
        {
            "domain_id": d.domain_id,
            "display_name": d.display_name,
            "source_file": d.source_file,
            "description_length": len(d.description),
        }
        for d in get_domains().values()
    ]


def save_domain_yaml(domain_id: str, yaml_content: str) -> Domain:
    """Persist YAML content to disk and reload. Raises ValueError on bad input."""
    if not yaml_content.strip():
        raise ValueError("Empty YAML body")
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")

    actual_id = (data.get("domain_id") or domain_id or "").strip()
    if not actual_id:
        raise ValueError("Missing 'domain_id' in YAML")

    filepath = DOMAINS_DIR / f"{actual_id}.yaml"
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    reload_domains()
    saved = get_domain(actual_id)
    if not saved:
        raise ValueError(f"Domain '{actual_id}' failed to reload after save")
    return saved


def delete_domain(domain_id: str) -> bool:
    filepath = DOMAINS_DIR / f"{domain_id}.yaml"
    if not filepath.exists():
        return False
    os.remove(filepath)
    reload_domains()
    # If we just deleted the active domain, fall back.
    if get_active_domain_id() == domain_id:
        remaining = list(get_domains().keys())
        set_active_domain_id(remaining[0] if remaining else DEFAULT_DOMAIN_ID)
    return True


def get_active_domain_id() -> str:
    try:
        if ACTIVE_DOMAIN_FILE.exists():
            value = ACTIVE_DOMAIN_FILE.read_text(encoding="utf-8").strip()
            if value:
                return value
    except Exception as e:
        print(f"[Domain] Error reading active_domain.txt: {e}")
    return DEFAULT_DOMAIN_ID


def set_active_domain_id(domain_id: str) -> None:
    ACTIVE_DOMAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_DOMAIN_FILE.write_text(f"{domain_id}\n", encoding="utf-8")


def get_active_domain() -> Domain:
    """Return the currently active Domain, falling back cleanly if it was
    deleted or the pointer is stale."""
    domains = get_domains()
    active_id = get_active_domain_id()
    if active_id in domains:
        return domains[active_id]
    # Fall back to whatever is available.
    if DEFAULT_DOMAIN_ID in domains:
        return domains[DEFAULT_DOMAIN_ID]
    return next(iter(domains.values()))


def get_active_domain_description() -> str:
    """Convenience for prompt builders: returns just the text block."""
    return get_active_domain().description
