import fnmatch
from typing import Iterable

from .models import EvidenceItem, severity_at_least


def path_allowed(rel_path: str, include_patterns: Iterable[str], exclude_patterns: Iterable[str]) -> bool:
    include = [p for p in include_patterns if p]
    exclude = [p for p in exclude_patterns if p]
    if include and not any(fnmatch.fnmatch(rel_path, pattern) for pattern in include):
        return False
    if exclude and any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude):
        return False
    return True


def apply_min_severity(item: EvidenceItem, min_severity: str) -> EvidenceItem:
    item.findings = [f for f in item.findings if severity_at_least(f.severity, min_severity)]
    return item
