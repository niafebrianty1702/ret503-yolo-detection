# Praktik YOLO: botol vs box (RET503)

Pipeline tahap 1–6: siapkan data → cek anotasi → konversi YOLO↔COCO → split → latih & bandingkan 3 eksperimen.

## Setup (VS Code, terminal di folder ini)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Ekstrak `dataset.zip` sehingga ada folder `dataset/` (isi: `botol/`, `box/`) di folder ini.
Subfolder `train/` dan `val/` bawaan zip diabaikan; pembagian dibuat ulang di tahap 2.

## Urutan perintah

```bash
python 00_ambil_subset_coco.py --sumber dataset --kelas botol box --jumlah 200 --batas box=25
python 01_cek_anotasi.py dataset_yolo
python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json
python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
python 05_split_dataset_yolo.py dataset_yolo dataset_v1
python 06_latih_bandingkan.py --nim NIM_KAMU --nama "Nama Kamu"
```

## Catatan penting

- **Dataset tidak punya bbox.** Label dibuat otomatis di program 00: botol dideteksi model
  YOLO pretrained (kelas COCO `bottle`, id 39), box/kardus diberi bbox hampir penuh karena
  kardus memenuhi frame. Cek hasilnya di `dataset_yolo/cek_visual/` sebelum lanjut.
  Tanpa PyTorch/model: tambahkan `--tanpa-model` (semua kelas bbox penuh).
- `--batas box=25` sengaja membuat kelas tidak seimbang (botol 50 vs box 25), sesuai
  anjuran slide agar efek ketidakseimbangan terlihat di hasil per kelas. Ubah sesuai kebutuhan.
- Program 06: E1 pretrained, E2 scratch, E3 tuning dengan satu perubahan lewat
  `--ubah kunci=nilai` (default `lr0=0.001`, mis. `--ubah imgsz=320` atau `--ubah optimizer=SGD`).
  Data, epoch (default 30), imgsz, dan seed sama untuk ketiganya. Tambah `--epoch` kalau kuat.
- Keluaran: `hasil_praktik_<NIM>.txt` dan `.csv` (berisi identitas, waktu, komputer,
  dan sidik jari dataset).
- Jumlah kelas di slide 3, dataset ini 2. Kalau tugas wajib 3 kelas, tambahkan folder kelas
  ketiga di `dataset/` dan sebutkan di `--kelas`.

## Push ke GitHub

```bash
git init
git add .
git commit -m "Praktik YOLO botol vs box"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```
`.gitignore` sudah mengecualikan `*.pt`, `runs_praktik/`, dan `.venv/`.
