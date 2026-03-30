# Project V.I.G.I.L. - Visual Inspection, GEOINT & Intelligence Locator

## Milestones (Golden Path)

1. **Phase 1: Skeleton + OCR**
   - Run: `python main.py Evidence_Dump`
   - Check: password screenshot and seed phrase image flagged as `Critical`.
2. **Phase 2: Visuals (QR + YOLO)**
   - Check: QR code flagged `High`; phone/laptop/credit card flagged `Medium`.
3. **Phase 3: Special Ops (GEOINT + Stego)**
   - Check: `tracker.jpg` shows Google Maps link; high-noise image flagged.
4. **Phase 4: Report**
   - Open `report.html` and confirm badges, thumbnails, and details.
5. **Phase 5: Polish**
   - Console output shows `[FOUND]` vs `[CLEAN]`.

## Setup

```bash
pip install -r requirements.txt
```

Install Tesseract OCR:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

## Usage

```bash
python main.py Evidence_Dump --report report.html
```

Optional:
- `--bip39 path/to/bip39.txt`
- `--yolo yolov8n.pt`
- `--vision` / `--no-vision`
- `--vision-score 0.4`
- `--report-json report.json`
- `--include "**/*.png"` (repeatable)
- `--exclude "**/archive/*"` (repeatable)
- `--min-severity Medium`
- `--workers 4`
- `--cache` / `--no-cache`

Example with analyst filters and machine-readable export:

```bash
python main.py Evidence_Dump --report report.html --report-json report.json --min-severity Medium --exclude "**/tmp/*" --workers 4
```

## Project Structure (Refactor Foundation)

- `main.py`: orchestration and CLI
- `vigil/models.py`: shared data models (`Finding`, `EvidenceItem`)
- `vigil/filters.py`: path and severity filters
- `vigil/reporting.py`: summary aggregation helpers
- `vigil/pipeline.py`: staging point for further scanner decomposition

## Reporting Enhancements

- HTML report now includes:
   - Severity/module summary counters
   - Confidence and evidence context per finding
   - Incident timeline
   - Critical-only one-click filter
- JSON report contains full findings and summary metadata for automation/SIEM ingestion.

## Performance Enhancements

- Optional threaded scanning (`--workers N`)
- Duplicate detection by file hash in a scan
- Persistent scan cache in `.vigil_cache.json` (content-hash keyed)

## Development

Run tests:

```bash
pytest -q
```

Run integration smoke test only:

```bash
pytest -q -m integration
```

## Package CLI

Run via package module:

```bash
python -m vigil Evidence_Dump --report report.html --report-json report.json
```

Install editable and use command directly:

```bash
pip install -e .
vigil Evidence_Dump --report report.html --report-json report.json
```

## Notes

- The HTML report embeds thumbnails (no separate assets).
- For a full BIP39 list, you can pass a local file or the official wordlist URL:
  https://raw.githubusercontent.com/bitcoin/bips/refs/heads/master/bip-0039/english.txt

## Google Vision API (QR + Objects)

1. Create a Google Cloud project with Vision API enabled.
2. Create a service account key (JSON).
3. Set environment variable:

```bash
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

Then run with `--vision` (default on). If not configured, the app falls back to QR/YOLO.

You can also set it via `.env`:

```
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

If you want to paste the JSON directly into `.env`, use:

```
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}
```

For safety with multiline values, you can base64-encode it:

```
GOOGLE_APPLICATION_CREDENTIALS_JSON=base64:PASTE_BASE64_HERE
```
