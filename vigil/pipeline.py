import os
from typing import Callable, Dict

from .models import EvidenceItem


def analyze_file(
	path: str,
	report_path: str,
	status: Dict[str, bool],
	bip39_words: set,
	vision_enabled: bool,
	vision_score: float,
	yolo_model_instance,
	yolo_conf: float,
	yolo_imgsz: int,
	debug: bool,
	print_status: Callable,
	red_color,
	green_color,
	callbacks: Dict[str, Callable],
) -> EvidenceItem:
	rel_path = os.path.relpath(path, os.path.dirname(report_path))
	item = EvidenceItem(path=path, rel_path=rel_path)
	image_bgr = callbacks["load_image_cv2"](path)

	ocr_snippet, ocr_findings = callbacks["run_ocr"](image_bgr, bip39_words, status["ocr"])
	item.ocr_snippet = ocr_snippet
	item.findings.extend(ocr_findings)

	qr_data, qr_findings = callbacks["run_qr"](image_bgr)
	item.qr_data = qr_data
	item.findings.extend(qr_findings)

	if vision_enabled:
		vision_qr, vision_objects, vision_labels, vision_findings = callbacks["run_vision"](
			path,
			min_score=vision_score,
			debug=debug,
		)
		if vision_qr and not item.qr_data:
			item.qr_data = vision_qr
		item.vision_objects = vision_objects
		item.vision_labels = vision_labels
		item.findings.extend(vision_findings)
	else:
		labels, yolo_findings = callbacks["run_yolo"](yolo_model_instance, path, yolo_conf, yolo_imgsz)
		item.yolo_labels = labels
		item.findings.extend(yolo_findings)

	geoint_link, geoint_findings = callbacks["run_geoint"](path)
	item.geoint_link = geoint_link
	item.findings.extend(geoint_findings)

	entropy, entropy_findings = callbacks["run_entropy"](image_bgr if status["stego"] else None, path)
	item.entropy = entropy
	item.findings.extend(entropy_findings)

	stego_payload, stego_message, stego_reason, stego_findings, stego_artifact = callbacks["run_stego_decode"](path)
	item.stego_payload = stego_payload
	item.stego_message = stego_message
	item.stego_reason = stego_reason
	item.findings.extend(stego_findings)
	item.stego_artifact_path = stego_artifact

	if stego_artifact:
		item.stego_artifact_thumbnail = callbacks["image_to_thumbnail"](stego_artifact)
		ocr_text = callbacks["ocr_image_path"](stego_artifact)
		item.stego_ocr = ocr_text
		if ocr_text and not item.stego_message:
			item.stego_message = ocr_text[:300]
			item.stego_reason = (
				"Hidden text extracted and decoded from embedded file: "
				f"{os.path.basename(stego_artifact)}"
			)

	if stego_findings:
		for finding in stego_findings:
			print_status(
				"[CRITICAL]",
				f"{path} -> {finding.summary} (see report)",
				red_color,
			)

	if item.findings:
		print_status("[FOUND]", f"{path} -> {item.top_severity}", red_color)
	else:
		print_status("[CLEAN]", path, green_color)

	if debug:
		print_status(
			"[DEBUG]",
			f"OCR:{bool(ocr_findings)} QR:{bool(qr_data)} VISION:{len(item.vision_objects)} "
			f"LABELS:{len(item.vision_labels)} YOLO:{len(item.yolo_labels)} GEO:{bool(geoint_link)} "
			f"STEGO:{entropy} DECODE:{bool(stego_payload)}",
		)
		if item.vision_labels and not item.vision_objects:
			print_status("[DEBUG]", f"Vision labels: {', '.join(item.vision_labels[:6])}")

	return item
