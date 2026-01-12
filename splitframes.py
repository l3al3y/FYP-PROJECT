import cv2
import os
import sys

# =================================================================
# --- TETAPAN KONFIGURASI PENGGUNA (Sila Ubah Suai) ---
# =================================================================

# 1. LALUAN PENUH ke fail video.
VIDEO_PATH = r'D:\Robo PSM\maggi cukup rasa.mp4'

# 2. Simpan 1 gambar setiap 'N' bingkai. FRAME_SKIP = 1 bermakna tiada skip.
FRAME_SKIP = 10 

# 3. Folder utama untuk menyimpan semua output bingkai.
ROOT_OUTPUT_DIR = 'dataset_frames' 

# =================================================================
# --- KOD UTAMA (Lebih Mantap) ---
# =================================================================

def extract_frames(video_path: str, root_output_dir: str, frame_skip: int):
    """
    Mengekstrak bingkai daripada video, memastikan nama fail yang konsisten dan output yang bersih.
    """
    
    # 1. Penyediaan Nama & Laluan
    base_name = os.path.basename(video_path)
    # Gantikan ruang dengan garis bawah untuk nama produk (lebih sesuai untuk ML)
    product_name_raw = os.path.splitext(base_name)[0]
    product_name = product_name_raw.replace(' ', '_').replace('-', '_') 
    
    # Guna os.path.join secara konsisten untuk laluan silang platform
    output_folder = os.path.join(root_output_dir, product_name)

    # Cipta folder output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Folder output '{output_folder}' telah dicipta.")
    else:
        print(f"📁 Folder output '{output_folder}' sedia ada. Menyimpan di dalamnya.")

    # 2. Buka Fail Video & Dapatkan Maklumat
    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print(f"❌ Ralat: Tidak dapat membuka fail video '{video_path}'")
        sys.exit(1) # Keluar dengan kod ralat
    
    # Dapatkan maklumat video untuk keterangan
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    print(f"\n🎥 Memulakan proses pengekstrakan untuk: {base_name}")
    print(f"   -> Jumlah Bingkai Video: {total_frames}")
    print(f"   -> Kadar Bingkai (FPS): {fps:.2f}")
    print(f"   -> Kadar Langkauan (Skip Rate): 1 setiap {frame_skip} bingkai")

    # 3. Loop Pengekstrakan
    success, image = vidcap.read()
    count = 0
    saved_frame_count = 0

    while success:
        
        if count % frame_skip == 0:
            
            # Cipta nama fail: product_name_0001.jpg
            image_name = f"{product_name}_{saved_frame_count:04d}.jpg"
            image_path = os.path.join(output_folder, image_name)
            
            # Simpan bingkai
            cv2.imwrite(image_path, image)
            
            saved_frame_count += 1

        # Baca bingkai seterusnya
        success, image = vidcap.read()
        count += 1
        
        # Kemas kini status (optional)
        if count % 1000 == 0:
            print(f"   Memproses bingkai: {count}/{total_frames}...")

    vidcap.release()
    
    print("\n✅ Pengekstrakan selesai.")
    print(f"   -> Jumlah Bingkai Diproses: {count}")
    print(f"   -> Jumlah Imej Disimpan: {saved_frame_count}")
    print(f"   -> Imej disimpan dalam: '{output_folder}'")


if __name__ == '__main__':
    # 1. Semak Laluan Video
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ RALAT: Laluan video tidak wujud: {VIDEO_PATH}")
        sys.exit(1)
        
    # 2. Jalankan Fungsi
    extract_frames(VIDEO_PATH, ROOT_OUTPUT_DIR, FRAME_SKIP)