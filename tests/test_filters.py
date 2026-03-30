from vigil.filters import path_allowed
from vigil.models import EvidenceItem, Finding
from vigil.filters import apply_min_severity


def test_path_allowed_include_exclude():
    assert path_allowed("a/b/file.png", ["**/*.png"], [])
    assert not path_allowed("a/b/file.jpg", ["**/*.png"], [])
    assert not path_allowed("a/tmp/file.png", ["**/*.png"], ["**/tmp/*"])


def test_apply_min_severity_keeps_high_only():
    item = EvidenceItem(path="x", rel_path="x")
    item.findings = [
        Finding(module="A", severity="Low", summary="low"),
        Finding(module="B", severity="High", summary="high"),
    ]
    apply_min_severity(item, "Medium")
    assert len(item.findings) == 1
    assert item.findings[0].severity == "High"
