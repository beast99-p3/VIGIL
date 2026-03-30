from dataclasses import dataclass, field
from typing import List, Optional

SEVERITY_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}


@dataclass
class Finding:
    module: str
    severity: str
    summary: str
    details: Optional[str] = None
    confidence: Optional[float] = None
    detector: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class EvidenceItem:
    path: str
    rel_path: str
    findings: List[Finding] = field(default_factory=list)
    ocr_snippet: Optional[str] = None
    qr_data: Optional[str] = None
    vision_objects: List[str] = field(default_factory=list)
    vision_labels: List[str] = field(default_factory=list)
    geoint_link: Optional[str] = None
    entropy: Optional[float] = None
    stego_payload: Optional[str] = None
    stego_message: Optional[str] = None
    stego_reason: Optional[str] = None
    stego_artifact_path: Optional[str] = None
    stego_artifact_thumbnail: Optional[str] = None
    stego_ocr: Optional[str] = None
    yolo_labels: List[str] = field(default_factory=list)

    @property
    def top_severity(self) -> str:
        if not self.findings:
            return "Low"
        return max(self.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 0)).severity


def severity_at_least(severity: str, minimum: str) -> bool:
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(minimum, 0)
