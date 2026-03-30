from collections import Counter
from typing import Dict, List

from .models import EvidenceItem


def build_summary(items: List[EvidenceItem]) -> Dict[str, Dict[str, int]]:
    by_module = Counter()
    by_severity = Counter()
    for item in items:
        for finding in item.findings:
            by_module[finding.module] += 1
            by_severity[finding.severity] += 1
    return {
        "by_module": dict(sorted(by_module.items())),
        "by_severity": dict(by_severity),
    }
