import cv2
import time
import threading
import numpy as np
import sqlite3
import datetime
import pygame
import os
import tempfile
from gtts import gTTS
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = "/home/psm/smart_checkout/best.pt"
CONF_THRESHOLD = 0.8
DB_FILE = "checkout_data.db"
SKIP_FRAMES = 1  # 0 = Process every frame. 1 = Process every 2nd frame (Higher FPS)

# Product Database
PRODUCT_DB = {
    "Maggi":            {"name": "Maggi Kari", "price": 4.50},
    "Maggi Cukup Rasa": {"name": "Maggi Cukup Rasa",      "price": 1.00},
    "Roti":             {"name": "Roti Gardenia",  "price": 0.90},
    "Tisu":             {"name": "Tisu",   "price": 12.90},
    "Ubat Gigi":        {"name": "Colgate Paste",   "price": 18.50}
}

# ==============================================================================
# RPI 5 HARDWARE MONITORING
# ==============================================================================
def get_cpu_temp():
    """Reads the RPi 5 CPU temperature from system files."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0.0

class FPSMeter:
    """Calculates smoothed FPS."""
    def __init__(self):
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0
        
    def update(self):
        self.curr_time = time.time()
        delta = self.curr_time - self.prev_time
        if delta > 0:
            self.fps = 1 / delta
        self.prev_time = self.curr_time
        return self.fps

# ==============================================================================
# VIDEO STREAM OPTIMIZATION (THREADED)
# ==============================================================================
class VideoStream:
    """
    Reads frames in a separate thread to prevent blocking the main loop.
    Crucial for RPi 5 to keep UI responsive during YOLO inference.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Set specific resolution for RPi Camera
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stopped = True
                break
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==============================================================================
# DATABASE MANAGER
# ==============================================================================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS transactions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT, item_name TEXT, price REAL, confidence REAL)''')
        conn.commit()

def log_transaction(item_name, price, confidence):
    threading.Thread(target=_log_worker, args=(item_name, price, confidence), daemon=True).start()

def _log_worker(item_name, price, confidence):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO transactions (timestamp, item_name, price, confidence) VALUES (?, ?, ?, ?)",
                      (timestamp, item_name, price, float(confidence)))
            conn.commit()
        print(f"📝 Logged: {item_name}")
    except Exception as e:
        print(f"❌ DB Error: {e}")

# ==============================================================================
# UTILITIES
# ==============================================================================
class VoiceAssistant:
    def __init__(self):
        try:
            pygame.mixer.init()
            self.lock = threading.Lock()
        except Exception as e:
            print(f"❌ Audio Init Error: {e}")
        
    def _speak_thread(self, text):
        with self.lock:
            path = None
            try:
                tts = gTTS(text=text, lang='ms')
                fd, path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                tts.save(path)
                
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
            except Exception as e:
                print(f"❌ Speak Error: {e}")
            finally:
                if path and os.path.exists(path):
                    try: os.remove(path)
                    except: pass

    def speak(self, text):
        print(f"🔊 Speaking: {text}")
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def stop(self):
        try: pygame.mixer.music.stop()
        except: pass

class ShoppingCart:
    def __init__(self):
        self.items = [] 
    
    def add(self, key):
        if key in PRODUCT_DB:
            item = PRODUCT_DB[key]
            self.items.append(item)
            return item
        return None

    def remove_at_index(self, index):
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None

    def get_total(self):
        return sum(item['price'] for item in self.items)
    
    def clear(self):
        self.items = []

def play_beep():
    def _beep():
        try:
            if not pygame.mixer.get_init(): pygame.mixer.init()
            # Optimized simple beep
            sample_rate = 44100
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = 0.5 * np.sign(np.sin(2 * np.pi * 2500 * t))
            audio = (wave * 32767).astype(np.int16)
            audio = np.repeat(audio.reshape(-1, 1), 2, axis=1)
            sound = pygame.sndarray.make_sound(audio)
            sound.play()
            time.sleep(duration + 0.05)
        except Exception as e:
            print(f"❌ Beep Error: {e}")
    threading.Thread(target=_beep, daemon=True).start()

# ==============================================================================
# MAIN SYSTEM
# ==============================================================================
def main():
    print("🚀 Starting RPi 5 Checkout System...")
    
    # Init Hardware / DB
    try: pygame.mixer.pre_init(44100, -16, 2, 512)
    except: pass
    pygame.mixer.init()
    init_db()
    
    # Load Model (Optimized for CPU)
    print("🧠 Loading YOLO...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Start Video Stream (Threaded)
    vs = VideoStream().start()
    time.sleep(2.0) # Warmup

    voice = VoiceAssistant()
    cart = ShoppingCart()
    voice.speak("SELAMAT DATANG, SILA LETAKKAN BARANG ANDA DI KAMERA.")

    # State Variables
    is_scanning_locked = False 
    last_scan_completion_time = 0 
    scan_delay = 2.0
    verification_mode = False
    entered_pin = ""
    click_action = None 
    target_delete_index = -1
    
    # Monitoring
    fps_meter = FPSMeter()
    frame_count = 0
    current_temp = 0.0
    last_temp_check = 0

    # Mouse Callback
    def mouse_handler(event, x, y, flags, param):
        nonlocal click_action, target_delete_index
        if event == cv2.EVENT_LBUTTONDOWN:
            if 150 <= x <= 280 and 400 <= y <= 430:
                click_action = "PAY"
            
            # Delete logic
            start_index = max(0, len(cart.items) - 8)
            visible_count = min(len(cart.items), 8)
            base_y = 90
            for i in range(visible_count):
                btn_y_center = base_y - 5
                if 270 <= x <= 295 and (btn_y_center - 10) <= y <= (btn_y_center + 10):
                    click_action = "DELETE_ITEM"
                    target_delete_index = start_index + i
                    break
                base_y += 30

    cv2.namedWindow("Smart Checkout Terminal", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("Smart Checkout Terminal", mouse_handler)

    try:
        while True:
            # 1. READ FRAME (Non-blocking)
            frame = vs.read()
            if frame is None: break

            # 2. MONITORING UPDATES
            current_time = time.time()
            fps = fps_meter.update()
            
            # Check temp every 1 second only (save CPU)
            if current_time - last_temp_check > 1.0:
                current_temp = get_cpu_temp()
                last_temp_check = current_time

            # 3. LOGIC HANDLERS
            if click_action == "DELETE_ITEM" and not verification_mode:
                if len(cart.items) > 0:
                    voice.speak("Pengurus diperlukan.")
                    verification_mode = True
                    entered_pin = ""
                else:
                    voice.speak("Kosong.")
                click_action = None
                
            elif click_action == "PAY" and not verification_mode:
                if len(cart.items) > 0:
                    voice.speak(f"Jumlah Bayaran RM {cart.get_total():.2f} Ringgit. Terima kasih,Sila Datang Lagi.")
                    cart.clear()
                    is_scanning_locked = False 
                else:
                    voice.speak("Kosong.")
                click_action = None

            # 4. INFERENCE (Run on every Nth frame to boost FPS)
            annotated_frame = frame.copy()
            
            if not verification_mode:
                # OPTIMIZATION: Skip frames if needed
                if frame_count % (SKIP_FRAMES + 1) == 0:
                    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=320) # Lower imgsz for speed if needed
                    
                    if results[0].boxes:
                        annotated_frame = results[0].plot()
                        cls_id = int(results[0].boxes.cls[0])
                        detected_class = model.names[cls_id]
                        conf_val = float(results[0].boxes.conf[0])

                        if detected_class in PRODUCT_DB:
                            if not is_scanning_locked and (current_time - last_scan_completion_time > scan_delay):
                                added_item = cart.add(detected_class)
                                play_beep()
                                log_transaction(added_item['name'], added_item['price'], conf_val)
                                voice.speak(f"{added_item['name']}, {added_item['price']:.2f}")
                                
                                is_scanning_locked = True
                                cv2.rectangle(annotated_frame, (0,0), (640,480), (0,255,0), 10)
                
                # Draw locking status even if not inferencing this frame
                if is_scanning_locked:
                     cv2.putText(annotated_frame, "CLEAR AREA", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                     # Check if area is clear (simple timeout logic for demo, or re-run detection)
                     # In real usage, you'd check next frame. Here we assume user clears it.
                     # We only unlock if we don't see an object, but since we skip frames,
                     # we rely on the next detection cycle.
                     
                     # Simple logic: If locked, and enough time passed, we assume it might be clear 
                     # BUT better logic is: If we detect NOTHING, we unlock.
                     # For this code, let's just use the timer to force re-check
                     pass 
                
                # Logic to unlock (simplified for threaded performance)
                if is_scanning_locked and results and not results[0].boxes:
                     # No boxes detected -> Unlock
                     is_scanning_locked = False
                     last_scan_completion_time = current_time

            else:
                annotated_frame = cv2.GaussianBlur(frame, (21, 21), 0)
                cv2.putText(annotated_frame, "MANAGER PIN:", (50, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
                cv2.putText(annotated_frame, entered_pin, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # 5. UI COMPOSITION
            ui_h, ui_w = 480, 940 # 300 + 640
            canvas = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
            canvas[:, :300] = (240, 240, 240) # Sidebar
            
            # Draw Sidebar Info
            cv2.putText(canvas, "RPI 5 CHECKOUT", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)
            
            # --- MONITORING INFO ---
            # Color logic: Green if cool, Red if hot (>70C)
            temp_color = (0, 150, 0) if current_temp < 70 else (0, 0, 255)
            cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
            cv2.putText(canvas, f"TEMP: {current_temp:.1f}C", (120, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)
            # -----------------------

            # Draw Cart
            y_offset = 80
            total = cart.get_total()
            start_index = max(0, len(cart.items) - 8)
            for i, item in enumerate(cart.items[start_index:]):
                cv2.putText(canvas, item['name'][:18], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
                cv2.putText(canvas, f"{item['price']:.2f}", (210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
                # Delete X
                cv2.rectangle(canvas, (270, y_offset - 10), (290, y_offset + 5), (0,0,255), -1)
                cv2.putText(canvas, "X", (274, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                y_offset += 30

            # Pay Button
            cv2.rectangle(canvas, (150, 400), (280, 430), (0, 200, 0), -1)
            cv2.putText(canvas, "PAY", (195, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(canvas, f"TOT: RM {total:.2f}", (20, 420), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)

            # Combine Camera
            canvas[:, 300:] = annotated_frame
            cv2.imshow("Smart Checkout Terminal", canvas)
            
            frame_count += 1
            
            # INPUT HANDLING
            key = cv2.waitKey(1) & 0xFF
            if not verification_mode:
                if key == ord('q'): break
            else:
                if key == 13: # ENTER
                    if entered_pin == "1234":
                        if target_delete_index != -1:
                            rem = cart.remove_at_index(target_delete_index)
                            if rem: voice.speak(f"Padam {rem['name']}")
                            target_delete_index = -1
                        verification_mode = False
                    else:
                        voice.speak("Salah.")
                        entered_pin = ""
                elif key == 27: verification_mode = False
                elif 48 <= key <= 57: 
                    if len(entered_pin) < 4: entered_pin += chr(key)
                elif key == 8: entered_pin = entered_pin[:-1]

    finally:
        print("🛑 Shutting down...")
        vs.stop()
        cv2.destroyAllWindows()
        voice.stop()

if __name__ == "__main__":
    main()
