"""Tahap 1 - memeriksa anotasi YOLO: pasangan gambar-label, format, rentang nilai,
jumlah per kelas, lalu menyimpan SEMUA gambar ber-bbox ke <folder>/cek_visual/
beserta satu gambar ringkasan (_ringkasan.jpg).

  python 01_cek_anotasi.py dataset_yolo
  python 01_cek_anotasi.py dataset_yolo --tampil   (buka jendela gambar, perlu layar GUI)
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from util_yolo import baca_kelas, baca_label, daftar_gambar


def buat_ringkasan(daftar, kolom=4, baris=4, ukuran=(240, 180)):
    """Susun beberapa gambar ber-bbox (diambil merata) menjadi satu grid."""
    if not daftar:
        return None
    n = min(len(daftar), kolom * baris)
    langkah = len(daftar) / n
    pilih = [cv2.resize(daftar[int(i * langkah)], ukuran) for i in range(n)]
    while len(pilih) % kolom:
        pilih.append(np.zeros((ukuran[1], ukuran[0], 3), np.uint8))
    return np.vstack([np.hstack(pilih[i:i + kolom]) for i in range(0, len(pilih), kolom)])


def main(folder, tampil=False):
    folder = Path(folder)
    kelas = baca_kelas(folder)
    gambar = daftar_gambar(folder / "images")
    masalah, per_kelas, per_kelas_gbr = [], Counter(), Counter()
    tanpa_label = 0
    prev = folder / "cek_visual"
    prev.mkdir(exist_ok=True)
    gambar_bbox = []

    for p in gambar:
        txt = folder / "labels" / f"{p.stem}.txt"
        if not txt.exists():
            masalah.append(f"{p.name}: label tidak ada")
            continue
        try:
            kotak = baca_label(txt)
        except ValueError as e:
            masalah.append(str(e))
            continue
        if not kotak:
            tanpa_label += 1
        for c, x, y, w, h in kotak:
            if not 0 <= c < len(kelas):
                masalah.append(f"{txt.name}: id kelas {c} di luar rentang")
            if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                masalah.append(f"{txt.name}: nilai bbox tidak valid ({x},{y},{w},{h})")
            if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                masalah.append(f"{txt.name}: bbox keluar gambar")
            per_kelas[c] += 1
        for c in {k[0] for k in kotak}:
            per_kelas_gbr[c] += 1
        if kotak:
            im = cv2.imread(str(p))
            H, W = im.shape[:2]
            for c, x, y, w, h in kotak:
                x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
                x2, y2 = int((x + w / 2) * W), int((y + h / 2) * H)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(im, kelas[c], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(prev / p.name), im)
            gambar_bbox.append(im)

    label_tanpa_gambar = [t.name for t in (folder / "labels").glob("*.txt")
                          if not any((folder / "images" / f"{t.stem}{e}").exists() for e in (".jpg", ".jpeg", ".png", ".bmp"))]
    for n in label_tanpa_gambar:
        masalah.append(f"{n}: gambar tidak ada")

    print(f"Folder        : {folder}")
    print(f"Jumlah gambar : {len(gambar)} | tanpa objek: {tanpa_label}")
    print("Per kelas     :")
    for i, k in enumerate(kelas):
        print(f"  [{i}] {k:<10} objek={per_kelas[i]:>4} gambar={per_kelas_gbr[i]:>4}")
    if masalah:
        print(f"\n{len(masalah)} MASALAH:")
        for m in masalah[:30]:
            print("  -", m)
        sys.exit(1)
    print(f"\nOK: semua anotasi valid. {len(gambar_bbox)} gambar ber-bbox tersimpan di {prev}/")
    ringkasan = buat_ringkasan(gambar_bbox)
    if ringkasan is not None:
        cv2.imwrite(str(prev / "_ringkasan.jpg"), ringkasan)
        print(f"Gambar ringkasan: {prev / '_ringkasan.jpg'}")
        if tampil:
            cv2.imshow("Cek anotasi (tekan tombol apa saja untuk menutup)", ringkasan)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(args[0] if args else "dataset_yolo", tampil="--tampil" in sys.argv)
