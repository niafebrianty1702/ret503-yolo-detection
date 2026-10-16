"""Konversi YOLO -> COCO JSON.

  python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json
"""
import json
import sys
from pathlib import Path

import cv2

from util_yolo import baca_kelas, baca_label, daftar_gambar


def main(folder, keluar):
    folder = Path(folder)
    kelas = baca_kelas(folder)
    coco = {
        "info": {"description": "Hasil konversi YOLO -> COCO"},
        "images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": k} for i, k in enumerate(kelas)],
    }
    ann_id = 1
    for img_id, p in enumerate(daftar_gambar(folder / "images"), 1):
        H, W = cv2.imread(str(p)).shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name, "width": W, "height": H})
        for c, x, y, w, h in baca_label(folder / "labels" / f"{p.stem}.txt"):
            bw, bh = w * W, h * H
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": c + 1,
                "bbox": [round((x - w / 2) * W, 2), round((y - h / 2) * H, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0,
            })
            ann_id += 1
    Path(keluar).write_text(json.dumps(coco, indent=1), encoding="utf-8")
    print(f"{len(coco['images'])} gambar, {len(coco['annotations'])} anotasi -> {keluar}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
