"""Tahap 3-5 - melatih dan membandingkan tiga eksperimen YOLO.

  E1 pretrained : yolo26n.pt   (baseline, transfer learning dari COCO)
  E2 scratch    : yolo26n.yaml (arsitektur sama, bobot acak)
  E3 tuning     : yolo26n.pt + SATU perubahan via --ubah kunci=nilai

Data, epoch, ukuran input, dan seed sama untuk ketiganya. Model dengan mAP50-95
val tertinggi diuji SEKALI di data test.

  python 06_latih_bandingkan.py --nim 4222101001 --nama "Budi Santoso"
  python 06_latih_bandingkan.py --nim ... --nama ... --ubah lr0=0.001
"""
import argparse
import ast
import csv
import hashlib
import os
import platform
import socket
import time
from datetime import datetime
from pathlib import Path


def parse_ubah(s):
    if s.count("=") != 1:
        raise SystemExit("[ERROR] --ubah harus satu pasangan kunci=nilai, mis. lr0=0.001")
    k, v = s.split("=")
    try:
        v = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    return k.strip(), v


def sidik_jari(dataset):
    """SHA-256 dari daftar nama file + ukuran + isi label (penanda unik dataset)."""
    h = hashlib.sha256()
    for p in sorted(Path(dataset).rglob("*")):
        if p.is_file() and p.name != "data.yaml":
            h.update(str(p.relative_to(dataset)).encode())
            h.update(str(p.stat().st_size).encode())
            if p.suffix == ".txt":
                h.update(p.read_bytes())
    return h.hexdigest()[:16]


def metrik(m, nama_kelas):
    return {
        "map50_95": float(m.box.map), "map50": float(m.box.map50),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_kelas": {nama_kelas[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nim", required=True)
    ap.add_argument("--nama", required=True)
    ap.add_argument("--data", default="dataset_v1/data.yaml")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ubah", default="lr0=0.001", help="satu perubahan untuk E3, format kunci=nilai")
    ap.add_argument("--device", default=None, help="mis. 0 atau cpu (default: otomatis)")
    ap.add_argument("--proyek", default="runs_praktik")
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch

    data = Path(a.data)
    if not data.exists():
        raise SystemExit(f"[ERROR] {data} tidak ada. Jalankan program 05 dulu.")
    device = a.device if a.device is not None else (0 if torch.cuda.is_available() else "cpu")
    kunci, nilai = parse_ubah(a.ubah)

    umum = dict(data=str(data), epochs=a.epoch, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
                deterministic=True, device=device, workers=0, project=str(Path(a.proyek).resolve()),
                exist_ok=True, plots=False, verbose=False)
    eksperimen = [
        ("E1_pretrained", "yolo26n.pt", {}),
        ("E2_scratch", "yolo26n.yaml", {"pretrained": False}),
        ("E3_tuning", "yolo26n.pt", {kunci: nilai}),
    ]

    mulai = datetime.now()
    sidik = sidik_jari(data.parent)
    print("=" * 64)
    print(f"Mahasiswa : {a.nama} ({a.nim})")
    print(f"Waktu     : {mulai:%Y-%m-%d %H:%M:%S}")
    print(f"Komputer  : {socket.gethostname()} | {platform.platform()} | device={device}")
    print(f"Dataset   : {data.parent} | sidik jari={sidik}")
    print(f"Pengaturan: epoch={a.epoch} imgsz={a.imgsz} batch={a.batch} seed={a.seed} | E3 ubah {kunci}={nilai}")
    print("=" * 64)

    hasil = []
    for nama, bobot, tambahan in eksperimen:
        print(f"\n>>> {nama} ({bobot} {tambahan or ''})")
        t0 = time.time()
        model = YOLO(bobot)
        model.train(name=nama, **umum, **tambahan)
        best = Path(model.trainer.best)  # lokasi best.pt sebenarnya
        model = YOLO(str(best))
        m = model.val(data=str(data), split="val", imgsz=a.imgsz, device=device, workers=0, plots=False, verbose=False)
        r = metrik(m, model.names)
        r.update(nama=nama, bobot=str(best), detik=time.time() - t0)
        hasil.append(r)
        print(f"    val mAP50-95={r['map50_95']:.4f} mAP50={r['map50']:.4f} ({r['detik']:.0f} dtk)")

    terbaik = max(hasil, key=lambda r: r["map50_95"])
    model = YOLO(terbaik["bobot"])
    mt = model.val(data=str(data), split="test", imgsz=a.imgsz, device=device, workers=0, plots=False, verbose=False)
    tes = metrik(mt, model.names)

    e1, e2, e3 = hasil
    baris = []
    baris.append("HASIL PRAKTIK")
    baris.append(f"Mahasiswa : {a.nama} ({a.nim}) | {mulai:%Y-%m-%d %H:%M:%S} | {socket.gethostname()} | sidik jari dataset {sidik}")
    baris.append("")
    baris.append(f"{'Eksperimen':<16}{'mAP50-95 val':>14}{'mAP50 val':>12}{'Precision':>11}{'Recall':>9}{'Waktu(dtk)':>12}")
    for r in hasil:
        baris.append(f"{r['nama']:<16}{r['map50_95']:>14.4f}{r['map50']:>12.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['detik']:>12.0f}")
    baris.append("")
    baris.append(f"Manfaat transfer learning (E1-E2): {e1['map50_95'] - e2['map50_95']:+.4f} mAP50-95")
    baris.append(f"Hasil tuning {kunci}={nilai} (E3-E1): {e3['map50_95'] - e1['map50_95']:+.4f} mAP50-95")
    baris.append("")
    baris.append(f"Model terbaik (val): {terbaik['nama']}")
    baris.append(f"Uji test (sekali) : mAP50-95={tes['map50_95']:.4f} mAP50={tes['map50']:.4f} P={tes['precision']:.4f} R={tes['recall']:.4f}")
    baris.append("mAP50-95 per kelas (val, model terbaik):")
    for k, v in terbaik["per_kelas"].items():
        baris.append(f"  {k:<12}{v:.4f}")
    teks = "\n".join(baris)
    print("\n" + teks)

    Path(f"hasil_praktik_{a.nim}.txt").write_text(teks + "\n", encoding="utf-8")
    with open(f"hasil_praktik_{a.nim}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nim", "nama", "waktu", "komputer", "sidik_dataset", "eksperimen", "map50_95_val", "map50_val", "precision", "recall", "detik"])
        for r in hasil:
            w.writerow([a.nim, a.nama, f"{mulai:%Y-%m-%d %H:%M:%S}", socket.gethostname(), sidik, r["nama"],
                        f"{r['map50_95']:.4f}", f"{r['map50']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['detik']:.0f}"])
        w.writerow([a.nim, a.nama, f"{mulai:%Y-%m-%d %H:%M:%S}", socket.gethostname(), sidik,
                    f"TEST({terbaik['nama']})", f"{tes['map50_95']:.4f}", f"{tes['map50']:.4f}", f"{tes['precision']:.4f}", f"{tes['recall']:.4f}", ""])
    print(f"\nTersimpan: hasil_praktik_{a.nim}.txt dan .csv")


if __name__ == "__main__":
    main()
