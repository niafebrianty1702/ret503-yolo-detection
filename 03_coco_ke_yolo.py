"""Konversi COCO JSON -> YOLO, lalu uji konversi dua arah (round-trip).

Kalau folder gambar asli (default: folder JSON-nya) ada label YOLO aslinya,
hasil balik dibandingkan dengan label asli (toleransi 1e-4).

  python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
"""
import json
import shutil
import sys
from pathlib import Path

from util_yolo import baca_label, tulis_label


def main(json_path, keluar):
    json_path = Path(json_path)
    asal = json_path.parent
    coco = json.loads(json_path.read_text(encoding="utf-8"))
    kat = sorted(coco["categories"], key=lambda c: c["id"])
    id_ke_idx = {c["id"]: i for i, c in enumerate(kat)}
    imgs = {i["id"]: i for i in coco["images"]}

    keluar = Path(keluar)
    if keluar.exists():
        shutil.rmtree(keluar)
    (keluar / "labels").mkdir(parents=True)
    (keluar / "images").mkdir()
    (keluar / "classes.txt").write_text("\n".join(c["name"] for c in kat) + "\n", encoding="utf-8")

    per_gambar = {i: [] for i in imgs}
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        per_gambar[a["image_id"]].append((id_ke_idx[a["category_id"]], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))

    for iid, kotak in per_gambar.items():
        nama = Path(imgs[iid]["file_name"]).stem
        tulis_label(keluar / "labels" / f"{nama}.txt", kotak)
        src = asal / "images" / imgs[iid]["file_name"]
        if src.exists():
            shutil.copy2(src, keluar / "images" / src.name)

    # uji round-trip
    asli = asal / "labels"
    if not asli.exists():
        print("Label asli tidak ditemukan, round-trip tidak diuji.")
        return
    beda = 0
    for iid in imgs:
        nama = Path(imgs[iid]["file_name"]).stem
        a, b = baca_label(asli / f"{nama}.txt"), baca_label(keluar / "labels" / f"{nama}.txt")
        if len(a) != len(b) or any(x[0] != y[0] or any(abs(p - q) > 1e-4 for p, q in zip(x[1:], y[1:])) for x, y in zip(a, b)):
            beda += 1
            print("  BEDA:", nama)
    print(f"{len(imgs)} gambar -> {keluar}/")
    print("Round-trip YOLO->COCO->YOLO: " + ("LULUS, label identik" if beda == 0 else f"GAGAL, {beda} file berbeda"))
    if beda:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
