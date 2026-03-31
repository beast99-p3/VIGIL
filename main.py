import argparse
import os

from vigil.scan import (
    run_scan,
    load_env,
    ensure_inline_credentials,
    configure_tesseract,
    print_status,
    Fore,
)


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
        "--report-json",
        default="report.json",
        help="Output JSON report path (default: report.json)",
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
        "--vision-score",
        type=float,
        default=0.4,
        help="Vision confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Google Vision API when available (default: true)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob include filter relative to scan root (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob exclude filter relative to scan root (repeatable)",
    )
    parser.add_argument(
        "--min-severity",
        choices=["Low", "Medium", "High", "Critical"],
        default="Low",
        help="Only keep findings at or above this severity in reports",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for image analysis (default: 1)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached results by file hash (default: true)",
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
    load_env()
    ensure_inline_credentials()
    configure_tesseract()
    run_scan(
        args.path,
        args.report,
        args.report_json,
        args.bip39,
        args.yolo,
        args.yolo_conf,
        args.yolo_imgsz,
        args.vision_score,
        args.vision,
        args.include,
        args.exclude,
        args.min_severity,
        max(1, args.workers),
        args.cache,
        args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
