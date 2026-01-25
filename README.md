# Project V.I.G.I.L.

Image intelligence scanner for hackathon demos: OCR + QR + YOLO + GEOINT + Stego entropy.

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

## Notes

- The HTML report embeds thumbnails (no separate assets).
- For a full BIP39 list, you can pass a local file or the official wordlist URL:
  https://raw.githubusercontent.com/bitcoin/bips/refs/heads/master/bip-0039/english.txt
