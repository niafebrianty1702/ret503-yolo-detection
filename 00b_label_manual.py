"""Label manual bounding box untuk satu kelas (default: botol).

Dipakai kalau auto-label program 00 kurang akurat (mis. botol bening).
Jalankan program 00 dulu (supaya dataset_yolo/classes.txt ada), lalu:

  python 00b_label_manual.py --sumber dataset --kelas botol

Kontrol untuk setiap gambar (gambar diperbesar agar mudah digambar):
  - tarik mouse untuk membuat kotak, lalu tekan ENTER atau SPASI untuk menyimpan
  - tekan C lalu ENTER untuk MELEWATI gambar (gambar tidak dipakai)
  - tekan ESC saat menggambar tidak berpengaruh; tutup dengan Ctrl+C untuk berhenti,
    progres tersimpan dan bisa dilanjutkan (gambar yang sudah dilabeli dilewati)
Butuh layar GUI (jendela OpenCV).
"""
import argparse
import shutil
from pathlib import Path

import cv2

from util_yolo import baca_kelas, daftar_gambar, tulis_label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sumber", default="dataset")
    ap.add_argument("--kelas", default="botol")
    ap.add_argument("--keluaran", default="dataset_yolo")
    ap.add_argument("--jumlah", type=int, default=200)
    ap.add_argument("--zoom", type=float, default=4.0, help="pembesaran gambar saat menggambar")
    ap.add_argument("--ulang", action="store_true", help="label ulang termasuk yang sudah ada")
    a = ap.parse_args()

    out = Path(a.keluaran)
    kelas = baca_kelas(out)
    if a.kelas not in kelas:
        raise SystemExit(f"[ERROR] kelas '{a.kelas}' tidak ada di {out}/classes.txt: {kelas}")
    idx = kelas.index(a.kelas)
    (out / "images").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    gambar = daftar_gambar(Path(a.sumber) / a.kelas)[: a.jumlah]
    if not gambar:
        raise SystemExit(f"[ERROR] tidak ada gambar di {Path(a.sumber) / a.kelas}")

    if a.ulang:  # hapus label lama kelas ini
        for p in list((out / "images").glob(f"{a.kelas}_*")) + list((out / "labels").glob(f"{a.kelas}_*")):
            p.unlink()

    simpan = lewat = sudah = 0
    try:
        for n, p in enumerate(gambar, 1):
            nama = f"{a.kelas}_{p.stem}"
            lbl = out / "labels" / f"{nama}.txt"
            if lbl.exists():
                sudah += 1
                continue
            im = cv2.imread(str(p))
            H, W = im.shape[:2]
            besar = cv2.resize(im, None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_CUBIC)
            judul = f"[{n}/{len(gambar)}] {p.name} | tarik kotak lalu ENTER | C = lewati"
            x, y, w, h = cv2.selectROI(judul, besar, showCrosshair=False)
            cv2.destroyAllWindows()
            if w < 4 or h < 4:  # kosong / dibatalkan -> lewati
                lewat += 1
                for e in (out / "images").glob(f"{nama}.*"):
                    e.unlink()
                continue
            x1, y1, x2, y2 = x / a.zoom, y / a.zoom, (x + w) / a.zoom, (y + h) / a.zoom
            tulis_label(lbl, [(idx, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)])
            shutil.copy2(p, out / "images" / f"{nama}{p.suffix.lower()}")
            simpan += 1
    except KeyboardInterrupt:
        print("\nDihentikan. Progres tersimpan.")
    finally:
        cv2.destroyAllWindows()

    print(f"Selesai: {simpan} dilabeli, {lewat} dilewati, {sudah} sudah ada sebelumnya.")
    print("Lanjut: python3 01_cek_anotasi.py")


if __name__ == "__main__":
    main()
