"""Fungsi bantu bersama untuk program 00-06."""
from pathlib import Path

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp"}


def daftar_gambar(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EKSTENSI)


def baca_kelas(folder_dataset):
    f = Path(folder_dataset) / "classes.txt"
    if not f.exists():
        raise SystemExit(f"[ERROR] {f} tidak ditemukan. Jalankan program 00 dulu.")
    return [k.strip() for k in f.read_text(encoding="utf-8").splitlines() if k.strip()]


def baca_label(path_txt):
    """Kembalikan list (kelas, xc, yc, w, h) dari satu file label YOLO."""
    hasil = []
    if not Path(path_txt).exists():
        return hasil
    for no, baris in enumerate(Path(path_txt).read_text().splitlines(), 1):
        if not baris.strip():
            continue
        bagian = baris.split()
        if len(bagian) != 5:
            raise ValueError(f"{path_txt}:{no} harus 5 kolom, dapat {len(bagian)}")
        hasil.append((int(bagian[0]), *map(float, bagian[1:])))
    return hasil


def tulis_label(path_txt, kotak):
    with open(path_txt, "w", encoding="utf-8") as f:
        for c, x, y, w, h in kotak:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
