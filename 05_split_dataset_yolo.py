"""Tahap 2 - membagi dataset_yolo menjadi train/val/test (stratified per kelas)
dan membuat data.yaml.

  python 05_split_dataset_yolo.py dataset_yolo dataset_v1
  opsi: --rasio 0.7 0.2 0.1 --seed 0
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from util_yolo import baca_kelas, baca_label, daftar_gambar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sumber")
    ap.add_argument("tujuan")
    ap.add_argument("--rasio", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if abs(sum(a.rasio) - 1) > 1e-6:
        raise SystemExit("[ERROR] rasio harus berjumlah 1")

    src, dst = Path(a.sumber), Path(a.tujuan)
    kelas = baca_kelas(src)
    kelompok = defaultdict(list)  # kelas utama gambar -> daftar gambar
    for p in daftar_gambar(src / "images"):
        kotak = baca_label(src / "labels" / f"{p.stem}.txt")
        kelompok[kotak[0][0] if kotak else -1].append(p)

    if dst.exists():
        shutil.rmtree(dst)
    rng = random.Random(a.seed)
    hitung = defaultdict(lambda: defaultdict(int))
    for c, lst in sorted(kelompok.items()):
        rng.shuffle(lst)
        n = len(lst)
        n_tr = round(n * a.rasio[0])
        n_va = round(n * a.rasio[1])
        bagian = {"train": lst[:n_tr], "val": lst[n_tr:n_tr + n_va], "test": lst[n_tr + n_va:]}
        for nama, items in bagian.items():
            (dst / "images" / nama).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / nama).mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.copy2(p, dst / "images" / nama / p.name)
                lbl = src / "labels" / f"{p.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, dst / "labels" / nama / lbl.name)
                hitung[nama][c] += 1

    yaml = f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    yaml += f"nc: {len(kelas)}\nnames:\n" + "".join(f"  {i}: {k}\n" for i, k in enumerate(kelas))
    (dst / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"Dataset terbagi di {dst}/ (seed={a.seed})")
    for nama in ("train", "val", "test"):
        rinci = ", ".join(f"{kelas[c]}={n}" for c, n in sorted(hitung[nama].items()) if c >= 0)
        print(f"  {nama:<5} {sum(hitung[nama].values()):>4} gambar  ({rinci})")
    print(f"data.yaml -> {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
