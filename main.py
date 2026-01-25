import argparse
import base64
import io
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ExifTags

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:  # pragma: no cover
    pytesseract = None
    TesseractNotFoundError = Exception

try:
    from pyzbar import pyzbar
except Exception:  # pragma: no cover
    pyzbar = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    from exif import Image as ExifImage
except Exception:  # pragma: no cover
    ExifImage = None

try:
    from skimage.measure import shannon_entropy
except Exception:  # pragma: no cover
    shannon_entropy = None

try:
    from colorama import Fore, Style, init as colorama_init
except Exception:  # pragma: no cover
    Fore = Style = None
    colorama_init = None

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except Exception:  # pragma: no cover
    Environment = FileSystemLoader = select_autoescape = None


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
KEYWORDS = {"confidential", "password", "login", "seed", "recovery"}
DEFAULT_BIP39 = {
    "abandon",
    "ability",
    "able",
    "about",
    "above",
    "absent",
    "absorb",
    "abstract",
    "absurd",
    "abuse",
    "access",
    "accident",
    "account",
    "accuse",
    "achieve",
    "acid",
    "acoustic",
    "acquire",
    "across",
    "act",
    "action",
    "actor",
    "actress",
    "actual",
    "adapt",
    "add",
    "addict",
    "address",
    "adjust",
    "admit",
    "adult",
    "advance",
    "advice",
    "aerobic",
    "affair",
    "afford",
    "afraid",
    "again",
    "age",
    "agent",
    "agree",
    "ahead",
    "aim",
    "air",
    "airport",
    "aisle",
    "alarm",
    "album",
    "alcohol",
    "alert",
    "alien",
    "all",
    "alley",
    "allow",
    "almost",
    "alone",
    "alpha",
    "already",
    "also",
    "alter",
    "always",
    "amateur",
    "amazing",
    "among",
    "amount",
    "amused",
    "analyst",
    "anchor",
    "ancient",
    "anger",
    "angle",
    "angry",
    "animal",
    "ankle",
    "announce",
    "annual",
    "another",
    "answer",
    "antenna",
    "antique",
    "anxiety",
    "any",
    "apart",
    "apology",
    "appear",
    "apple",
    "approve",
    "april",
    "arch",
    "arctic",
    "area",
    "arena",
    "argue",
    "arm",
    "armed",
    "armor",
    "army",
    "around",
    "arrange",
    "arrest",
    "arrive",
    "arrow",
    "art",
    "artefact",
    "artist",
    "artwork",
    "ask",
    "aspect",
    "assault",
    "asset",
    "assist",
    "assume",
    "asthma",
    "athlete",
    "atom",
    "attack",
    "attend",
    "attitude",
    "attract",
    "auction",
    "audit",
    "august",
    "aunt",
    "author",
    "auto",
    "autumn",
    "average",
    "avocado",
    "avoid",
    "awake",
    "aware",
    "away",
    "awesome",
    "awful",
    "awkward",
    "axis",
    "baby",
    "bachelor",
    "bacon",
    "badge",
    "bag",
    "balance",
    "balcony",
    "ball",
    "bamboo",
    "banana",
    "banner",
    "bar",
    "barely",
    "bargain",
    "barrel",
    "base",
    "basic",
    "basket",
    "battle",
    "beach",
    "bean",
    "beauty",
    "because",
    "become",
    "beef",
    "before",
    "begin",
    "behave",
    "behind",
    "believe",
    "below",
    "belt",
    "bench",
    "benefit",
    "best",
    "betray",
    "better",
    "between",
    "beyond",
    "bicycle",
    "bid",
    "bike",
    "bind",
    "biology",
    "bird",
    "birth",
    "bitter",
    "black",
    "blade",
    "blame",
    "blanket",
    "blast",
    "bleak",
    "bless",
    "blind",
    "blood",
    "blossom",
    "blouse",
    "blue",
    "blur",
    "blush",
    "board",
    "boat",
    "body",
    "boil",
    "bomb",
    "bone",
    "bonus",
    "book",
    "boost",
    "border",
    "boring",
    "borrow",
    "boss",
    "bottom",
    "bounce",
    "box",
    "boy",
    "bracket",
    "brain",
    "brand",
    "brass",
    "brave",
    "bread",
    "breeze",
    "brick",
    "bridge",
    "brief",
    "bright",
    "bring",
    "brisk",
    "broccoli",
    "broken",
    "bronze",
    "broom",
    "brother",
    "brown",
    "brush",
    "bubble",
    "buddy",
    "budget",
    "buffalo",
    "build",
    "bulb",
    "bulk",
    "bullet",
    "bundle",
    "bunker",
    "burden",
    "burger",
    "burst",
    "bus",
    "business",
    "busy",
    "butter",
    "buyer",
    "buzz",
}


SEVERITY_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
YOLO_TARGETS = {"cell phone", "laptop", "credit card", "handbag", "backpack"}


@dataclass
class Finding:
    module: str
    severity: str
    summary: str
    details: Optional[str] = None


@dataclass
class EvidenceItem:
    path: str
    rel_path: str
    findings: List[Finding] = field(default_factory=list)
    ocr_snippet: Optional[str] = None
    qr_data: Optional[str] = None
    geoint_link: Optional[str] = None
    entropy: Optional[float] = None
    yolo_labels: List[str] = field(default_factory=list)

    @property
    def top_severity(self) -> str:
        if not self.findings:
            return "Low"
        return max(self.findings, key=lambda f: SEVERITY_ORDER[f.severity]).severity


def normalize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def load_bip39(path: Optional[str]) -> set:
    if not path:
        return set(DEFAULT_BIP39)
    words = set()
    try:
        if path.startswith("http://") or path.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(path, timeout=10) as response:
                for line in response.read().decode("utf-8").splitlines():
                    word = line.strip().lower()
                    if word:
                        words.add(word)
        else:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
    except FileNotFoundError:
        print(f"[WARN] BIP39 list not found at {path}. Using default mini-list.")
        return set(DEFAULT_BIP39)
    except Exception:
        print(f"[WARN] Failed to load BIP39 list from {path}. Using default mini-list.")
        return set(DEFAULT_BIP39)
    return words


def load_image_cv2(path: str):
    if not cv2:
        return None
    return cv2.imread(path)


def configure_tesseract():
    if not pytesseract:
        return
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path


def run_ocr(image_bgr, bip39_words: set, enabled: bool) -> Tuple[Optional[str], List[Finding]]:
    findings: List[Finding] = []
    if not enabled or image_bgr is None or not pytesseract:
        return None, findings
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        text = pytesseract.image_to_string(gray) or ""
    except TesseractNotFoundError:
        return None, findings
    except Exception:
        return None, findings
    words = normalize_words(text)
    bip_hits = [w for w in words if w in bip39_words]
    keyword_hits = [w for w in words if w in KEYWORDS]
    if len(set(bip_hits)) >= 5 or keyword_hits:
        summary = "OCR match"
        details = None
        if keyword_hits:
            details = f"Keyword(s): {', '.join(sorted(set(keyword_hits)))}"
        elif bip_hits:
            details = f"BIP39 words: {', '.join(sorted(set(bip_hits))[:8])}"
        findings.append(
            Finding(
                module="OCR",
                severity="Critical",
                summary=summary,
                details=details,
            )
        )
    snippet = text.strip().replace("\n", " ")
    return (snippet[:200] if snippet else None), findings


def looks_like_crypto(data: str) -> bool:
    if data.startswith("bc1") or data.startswith("1") or data.startswith("3"):
        return True
    if re.fullmatch(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}", data):
        return True
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", data):
        return True
    return False


def looks_like_url(data: str) -> bool:
    return bool(re.match(r"^(https?://|www\.)", data.lower()))


def run_qr(image_bgr) -> Tuple[Optional[str], List[Finding]]:
    findings: List[Finding] = []
    payloads: List[str] = []
    if image_bgr is None:
        return None, findings
    if pyzbar:
        decoded = pyzbar.decode(image_bgr)
        for entry in decoded:
            data = entry.data.decode("utf-8", errors="ignore")
            if data:
                payloads.append(data)
    if not payloads and cv2:
        try:
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(image_bgr)
            if data:
                payloads.append(data)
        except Exception:
            pass
    if not payloads:
        return None, findings
    payload = payloads[0]
    severity = "High" if (looks_like_url(payload) or looks_like_crypto(payload)) else "Medium"
    details = "URL" if looks_like_url(payload) else "Crypto/Other" if looks_like_crypto(payload) else "QR data"
    findings.append(
        Finding(
            module="QR",
            severity=severity,
            summary="QR code found",
            details=f"{details}: {payload[:200]}",
        )
    )
    if len(payloads) > 1:
        findings.append(
            Finding(
                module="QR",
                severity="Low",
                summary="Multiple QR payloads",
                details=f"Count: {len(payloads)}",
            )
        )
    return payload, findings


def load_yolo(model_path: str):
    if not YOLO:
        return None
    try:
        return YOLO(model_path)
    except Exception:
        return None


def run_yolo(model, image_path: str, conf: float, imgsz: int) -> Tuple[List[str], List[Finding]]:
    findings: List[Finding] = []
    labels: List[str] = []
    if not model:
        return labels, findings
    try:
        results = model(image_path, verbose=False, conf=conf, imgsz=imgsz)
    except Exception:
        return labels, findings
    for result in results:
        if not result.names:
            continue
        for cls_id in result.boxes.cls.tolist() if result.boxes is not None else []:
            name = result.names.get(int(cls_id))
            if not name:
                continue
            labels.append(name)
            if name in YOLO_TARGETS:
                findings.append(
                    Finding(
                        module="YOLO",
                        severity="Medium",
                        summary="Object detected",
                        details=name,
                    )
                )
    return sorted(set(labels)), findings


def dms_to_decimal(dms: Tuple, ref: str) -> Optional[float]:
    try:
        degrees = float(dms[0].numerator) / float(dms[0].denominator)
        minutes = float(dms[1].numerator) / float(dms[1].denominator)
        seconds = float(dms[2].numerator) / float(dms[2].denominator)
    except Exception:
        try:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
        except Exception:
            return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in {"S", "W"}:
        decimal *= -1
    return round(decimal, 6)


def extract_gps_from_exif(path: str) -> Optional[str]:
    if ExifImage:
        try:
            with open(path, "rb") as handle:
                exif_img = ExifImage(handle)
            if not exif_img.has_exif:
                return None
            lat = getattr(exif_img, "gps_latitude", None)
            lat_ref = getattr(exif_img, "gps_latitude_ref", None)
            lon = getattr(exif_img, "gps_longitude", None)
            lon_ref = getattr(exif_img, "gps_longitude_ref", None)
            if lat and lon and lat_ref and lon_ref:
                lat_dec = dms_to_decimal(lat, lat_ref)
                lon_dec = dms_to_decimal(lon, lon_ref)
                if lat_dec is None or lon_dec is None:
                    return None
                return f"https://maps.google.com/?q={lat_dec},{lon_dec}"
        except Exception:
            return None
    try:
        image = Image.open(path)
        exif_data = image._getexif() or {}
    except Exception:
        return None
    gps_info = None
    for tag, value in exif_data.items():
        if ExifTags.TAGS.get(tag) == "GPSInfo":
            gps_info = value
            break
    if not gps_info:
        return None
    def gps_value(key):
        return gps_info.get(key)
    lat = gps_value(2)
    lat_ref = gps_value(1)
    lon = gps_value(4)
    lon_ref = gps_value(3)
    if not (lat and lon and lat_ref and lon_ref):
        return None
    lat_dec = dms_to_decimal(lat, lat_ref)
    lon_dec = dms_to_decimal(lon, lon_ref)
    if lat_dec is None or lon_dec is None:
        return None
    return f"https://maps.google.com/?q={lat_dec},{lon_dec}"


def run_geoint(path: str) -> Tuple[Optional[str], List[Finding]]:
    findings: List[Finding] = []
    link = extract_gps_from_exif(path)
    if link:
        findings.append(
            Finding(
                module="GEOINT",
                severity="Medium",
                summary="Location data found",
                details=link,
            )
        )
    return link, findings


def trailing_bytes_count(path: str) -> int:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except Exception:
        return 0
    lower = path.lower()
    if lower.endswith(".png"):
        signature = b"\x00\x00\x00\x00IEND\xaeB`\x82"
        index = data.rfind(signature)
        if index != -1:
            end = index + len(signature)
            return max(0, len(data) - end)
    if lower.endswith((".jpg", ".jpeg")):
        eoi = b"\xff\xd9"
        index = data.rfind(eoi)
        if index != -1:
            end = index + len(eoi)
            return max(0, len(data) - end)
    if lower.endswith(".bmp") and len(data) >= 6:
        declared = int.from_bytes(data[2:6], byteorder="little", signed=False)
        if declared and len(data) > declared:
            return len(data) - declared
    return 0


def run_entropy(image_bgr, path: str) -> Tuple[Optional[float], List[Finding]]:
    findings: List[Finding] = []
    trailing = trailing_bytes_count(path)
    if trailing > 1024:
        findings.append(
            Finding(
                module="STEGO",
                severity="Medium",
                summary="Trailing data detected",
                details=f"Trailing bytes: {trailing}",
            )
        )
    if image_bgr is None or not shannon_entropy:
        return None, findings
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    entropy = float(shannon_entropy(gray))
    if entropy > 7.6:
        findings.append(
            Finding(
                module="STEGO",
                severity="Medium",
                summary="High entropy image",
                details=f"Entropy: {entropy:.2f}",
            )
        )
    return entropy, findings


def image_to_thumbnail(path: str, max_size: int = 240) -> Optional[str]:
    try:
        image = Image.open(path)
        image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def print_status(label: str, message: str, color: Optional[str] = None):
    if color and Fore and Style:
        print(f"{color}{label}{Style.RESET_ALL} {message}")
    else:
        print(f"{label} {message}")


def collect_images(root: str) -> List[str]:
    files = []
    for base, _, names in os.walk(root):
        for name in names:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(base, name))
    return sorted(files)


def preflight_checks(yolo_model: str) -> Dict[str, bool]:
    status = {
        "ocr": bool(pytesseract),
        "qr": bool(pyzbar),
        "yolo": bool(YOLO),
        "geoint": True,
        "stego": bool(shannon_entropy),
    }
    if status["ocr"]:
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            status["ocr"] = False
    if status["yolo"]:
        try:
            _ = YOLO(yolo_model)
        except Exception:
            status["yolo"] = False
    return status


def print_preflight(status: Dict[str, bool], yolo_model: str):
    print_status("[CHECK]", f"OCR available: {status['ocr']}")
    print_status("[CHECK]", f"QR available: {status['qr']}")
    print_status("[CHECK]", f"YOLO available ({yolo_model}): {status['yolo']}")
    print_status("[CHECK]", f"GEOINT available: {status['geoint']}")
    print_status("[CHECK]", f"STEGO available: {status['stego']}")
    if not status["ocr"]:
        print_status("[WARN]", "OCR disabled: install Tesseract and add to PATH.")
    if not status["qr"]:
        print_status("[WARN]", "QR disabled: install pyzbar and zbar.")
    if not status["yolo"]:
        print_status("[WARN]", "YOLO disabled: model failed to load.")
    if not status["stego"]:
        print_status("[WARN]", "Stego disabled: install scikit-image.")


def run_scan(
    root: str,
    report_path: str,
    bip39_path: Optional[str],
    yolo_model: str,
    yolo_conf: float,
    yolo_imgsz: int,
    debug: bool,
):
    if colorama_init:
        colorama_init()
    status = preflight_checks(yolo_model)
    print_preflight(status, yolo_model)
    bip39_words = load_bip39(bip39_path)
    yolo_model_instance = load_yolo(yolo_model) if status["yolo"] else None
    files = collect_images(root)
    print_status("[INFO]", f"Scanning {len(files)} image(s) in {root}")
    items: List[EvidenceItem] = []
    for path in files:
        rel_path = os.path.relpath(path, os.path.dirname(report_path))
        item = EvidenceItem(path=path, rel_path=rel_path)
        image_bgr = load_image_cv2(path)

        ocr_snippet, ocr_findings = run_ocr(image_bgr, bip39_words, status["ocr"])
        item.ocr_snippet = ocr_snippet
        item.findings.extend(ocr_findings)

        qr_data, qr_findings = run_qr(image_bgr if status["qr"] else None)
        item.qr_data = qr_data
        item.findings.extend(qr_findings)

        labels, yolo_findings = run_yolo(yolo_model_instance, path, yolo_conf, yolo_imgsz)
        item.yolo_labels = labels
        item.findings.extend(yolo_findings)

        geoint_link, geoint_findings = run_geoint(path)
        item.geoint_link = geoint_link
        item.findings.extend(geoint_findings)

        entropy, entropy_findings = run_entropy(image_bgr if status["stego"] else None, path)
        item.entropy = entropy
        item.findings.extend(entropy_findings)

        if item.findings:
            print_status("[FOUND]", f"{path} -> {item.top_severity}", Fore.RED if Fore else None)
        else:
            print_status("[CLEAN]", path, Fore.GREEN if Fore else None)
        if debug:
            print_status(
                "[DEBUG]",
                f"OCR:{bool(ocr_findings)} QR:{bool(qr_findings)} YOLO:{len(labels)} GEO:{bool(geoint_link)} STEGO:{entropy}",
            )

        items.append(item)

    generate_report(report_path, root, items)
    print_status("[DONE]", f"Report saved to {report_path}", Fore.CYAN if Fore else None)


def generate_report(report_path: str, root: str, items: List[EvidenceItem]):
    if not Environment:
        print_status("[WARN]", "Jinja2 not installed, skipping HTML report.")
        return
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    threats = sum(1 for item in items if item.findings)
    locations = sum(1 for item in items if item.geoint_link)
    for item in items:
        item.thumbnail = image_to_thumbnail(item.path)
    html = template.render(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        root=root,
        files_scanned=len(items),
        threats=threats,
        locations=locations,
        items=items,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project V.I.G.I.L. - Image Intelligence Scanner")
    parser.add_argument(
        "path",
        nargs="?",
        default="Evidence_Dump",
        help="Folder to scan (default: Evidence_Dump)",
    )
    parser.add_argument(
        "--report",
        default="report.html",
        help="Output HTML report path",
    )
    parser.add_argument(
        "--bip39",
        default=None,
        help="Path to BIP39 wordlist text file",
    )
    parser.add_argument(
        "--yolo",
        default="yolov8n.pt",
        help="YOLO model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.15,
        help="YOLO confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=960,
        help="YOLO inference size (default: 960)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-image module debug info",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.path):
        print_status("[ERROR]", f"Path not found: {args.path}")
        return 1
    configure_tesseract()
    run_scan(
        args.path,
        args.report,
        args.bip39,
        args.yolo,
        args.yolo_conf,
        args.yolo_imgsz,
        args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
