"""Tahap 1 - menyiapkan data.

Membuat dataset_yolo/ (images + labels format YOLO + classes.txt) dari folder
dataset berisi subfolder per kelas (botol/, box/, ...).

Dataset yang dipakai TIDAK punya anotasi bounding box, jadi label dibuat otomatis:
  * kelas yang ada di --peta-coco (default botol=39 -> 'bottle' COCO) dideteksi
    dengan model YOLO pretrained; gambar tanpa deteksi dibuang.
  * kelas lain (mis. box/kardus yang memenuhi frame) diberi bbox hampir penuh.
  * --tanpa-model : semua kelas pakai bbox hampir penuh (tanpa PyTorch).

Contoh:
  python 00_ambil_subset_coco.py --sumber dataset --kelas botol box --jumlah 200
  python 00_ambil_subset_coco.py --sumber dataset --kelas botol box --batas box=20
"""
import argparse
import random
import shutil
from pathlib import Path

from util_yolo import daftar_gambar, tulis_label

KELAS_ABAIKAN = {"train", "val", "test"}  # split lama di dalam zip, abaikan


def parse_kv(daftar, tipe=int):
    d = {}
    for item in daftar or []:
        k, v = item.split("=")
        d[k] = tipe(v)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sumber", default="dataset")
    ap.add_argument("--kelas", nargs="+", help="nama subfolder kelas (default: semua)")
    ap.add_argument("--jumlah", type=int, default=200, help="maks gambar per kelas")
    ap.add_argument("--batas", nargs="*", help="batas per kelas, mis. box=20 (untuk bikin tidak seimbang)")
    ap.add_argument("--peta-coco", nargs="*", default=["botol=39"], help="kelas=id_COCO untuk auto-label model")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.02, help="margin bbox penuh (fraksi)")
    ap.add_argument("--tanpa-model", action="store_true")
    ap.add_argument("--keluaran", default="dataset_yolo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sumber = Path(a.sumber)
    if not sumber.is_dir():
        raise SystemExit(f"[ERROR] folder {sumber} tidak ada")
    kelas = a.kelas or sorted(d.name for d in sumber.iterdir() if d.is_dir() and d.name not in KELAS_ABAIKAN)
    batas = parse_kv(a.batas)
    peta = {} if a.tanpa_model else parse_kv(a.peta_coco)

    model = None
    if peta:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("[ERROR] ultralytics belum terpasang (pip install ultralytics) atau pakai --tanpa-model")
        model = YOLO(a.model)

    out = Path(a.keluaran)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    (out / "classes.txt").write_text("\n".join(kelas) + "\n", encoding="utf-8")

    rng = random.Random(a.seed)
    m = a.margin
    ringkas = []
    for idx, k in enumerate(kelas):
        gambar = daftar_gambar(sumber / k)
        rng.shuffle(gambar)
        gambar = gambar[: batas.get(k, a.jumlah)]
        ok = dibuang = 0
        for p in gambar:
            if k in peta:
                r = model.predict(str(p), conf=a.conf, classes=[peta[k]], verbose=False)[0]
                if len(r.boxes) == 0:
                    dibuang += 1
                    continue
                b = r.boxes[r.boxes.conf.argmax()]
                x, y, w, h = b.xywhn[0].tolist()
            else:
                x, y, w, h = 0.5, 0.5, 1 - 2 * m, 1 - 2 * m
            nama = f"{k}_{p.stem}"
            shutil.copy2(p, out / "images" / f"{nama}{p.suffix.lower()}")
            tulis_label(out / "labels" / f"{nama}.txt", [(idx, x, y, w, h)])
            ok += 1
        ringkas.append((k, ok, dibuang))

    print(f"Dataset YOLO dibuat di: {out}/")
    for k, ok, d in ringkas:
        sumber_lbl = "auto-label model" if k in peta else "bbox penuh"
        print(f"  {k:<10} {ok:>4} gambar  ({sumber_lbl}; tanpa deteksi dibuang: {d})")


if __name__ == "__main__":
    main()
