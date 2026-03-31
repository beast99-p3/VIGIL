import json
from pathlib import Path

import pytest

from vigil.scan import run_scan


@pytest.mark.integration
def test_scan_smoke_generates_reports(tmp_path: Path):
    root = Path("Evidence_Dump")
    if not root.exists():
        pytest.skip("Evidence_Dump not available")

    html_report = tmp_path / "report.html"
    json_report = tmp_path / "report.json"

    run_scan(
        root=str(root),
        report_path=str(html_report),
        report_json_path=str(json_report),
        bip39_path=None,
        yolo_model="nonexistent.pt",
        yolo_conf=0.15,
        yolo_imgsz=640,
        vision_score=0.4,
        use_vision=False,
        include_patterns=["**/*canvas*.png", "**/*tracker*.jpg", "**/*.jpeg"],
        exclude_patterns=[],
        min_severity="Low",
        workers=1,
        use_cache=False,
        debug=False,
    )

    assert html_report.exists()
    assert json_report.exists()

    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "items" in payload
    assert isinstance(payload["items"], list)
