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
MODEL_PATH = "runs/detect/smart_checkout_model/weights/best.pt"
CONF_THRESHOLD = 0.8
DB_FILE = "checkout_data.db"

# Product Database: Map YOLO class names to Display Name and Price (MYR)
# Keys must match the model's class names EXACTLY (Case Sensitive)
PRODUCT_DB = {
    "Maggi":            {"name": "Maggi Curry 5pk", "price": 5.90},
    "Maggi Cukup Rasa": {"name": "Cukup Rasa",      "price": 4.50},
    "Roti":             {"name": "Gardenia Bread",  "price": 3.20},
    "Tisu":             {"name": "Facial Tissue",   "price": 12.90},
    "Ubat Gigi":        {"name": "Colgate Paste",   "price": 18.50}
}

# ==============================================================================
# DATABASE MANAGER
# ==============================================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  item_name TEXT,
                  price REAL,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def log_transaction(item_name, price, confidence):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO transactions (timestamp, item_name, price, confidence) VALUES (?, ?, ?, ?)",
                  (timestamp, item_name, price, float(confidence)))
        conn.commit()
        conn.close()
        print(f"📝 Logged to DB: {item_name} ({confidence:.2f})")
    except Exception as e:
        print(f"❌ DB Error: {e}")

# ==============================================================================
# UTILITIES: VOICE & CART
# ==============================================================================

class VoiceAssistant:
    def __init__(self):
        try:
            pygame.mixer.init()
            self.lock = threading.Lock()
        except Exception as e:
            print(f"❌ Audio Init Error: {e}")
        
    def _speak_thread(self, text):
        # Use a lock to ensure only one speech thread manages the mixer at a time
        with self.lock:
            try:
                tts = gTTS(text=text, lang='ms')
                fd, path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                tts.save(path)
                
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                pygame.mixer.music.unload() # Crucial for Windows to release the file
                try:
                    os.remove(path)
                except:
                    pass
            except Exception as e:
                print(f"❌ Audio Error: {e}")

    def speak(self, text):
        print(f"🔊 Speaking (Native MS): {text}")
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def stop(self):
        try:
            pygame.mixer.music.stop()
        except:
            pass

class ShoppingCart:
    def __init__(self):
        self.items = [] # List of dicts
    
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
    # Robust beep using pygame
    def _beep():
        try:
            # Ensure mixer is initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
                
            frequency = 2500  # Hz
            duration = 0.15   # seconds
            sample_rate = 44100
            
            # Generate square wave (louder/clearer than sine for beeps)
            n_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, n_samples, False)
            
            # Square wave: 1.0 for sin > 0, -1.0 for sin < 0
            wave = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
            
            # Convert to 16-bit signed integers (standard for pygame)
            audio = (wave * 32767).astype(np.int16)
            
            # Stereo duplication if needed (numpy broadcasting)
            # Some mixers default to stereo, so let's make it 2-channel just in case
            audio = np.repeat(audio.reshape(n_samples, 1), 2, axis=1)
            
            sound = pygame.sndarray.make_sound(audio)
            sound.play()
            
            # Wait for length of beep so thread doesn't die too fast? 
            # Not strictly necessary for fire-and-forget, but good practice if we want to avoid cutting off
            time.sleep(duration + 0.05) 
            
        except Exception as e:
            print(f"❌ Beep Error: {e}")
            
    threading.Thread(target=_beep, daemon=True).start()

# ==============================================================================
# MAIN SYSTEM
# ==============================================================================
def main():
    print("🚀 Starting Retail Self-Checkout...")
    
    # Initialize Audio Mixer Globally (Channels=2, Buffer=512 for low latency)
    try:
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
    except Exception as e:
        print(f"Warning: Audio mixer init failed: {e}")

    init_db() # Initialize Database
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    voice = VoiceAssistant()
    cart = ShoppingCart()
    voice.speak("Selamat datang ke Kaunter Pembayaran Pintar. Sila tunjukkan barang anda pada kamera.")

    is_scanning_locked = False 
    last_scanned_item_name = ""
    last_scan_completion_time = 0  # Timestamp when the last item was cleared from view
    scan_delay = 1.5               # Seconds to wait after clearing before next scan
    
    # Verification State
    verification_mode = False
    entered_pin = ""
    verification_message = "ENTER PIN: "
    
    # Interaction State
    click_action = None 
    target_delete_index = -1 # Index in cart.items to delete

    def mouse_handler(event, x, y, flags, param):
        nonlocal click_action, target_delete_index
        if event == cv2.EVENT_LBUTTONDOWN:
            # CHECK PAY BUTTON: x(150-280), y(400-430)
            if 150 <= x <= 280 and 400 <= y <= 430:
                click_action = "PAY"
            
            # CHECK DELETE BUTTONS (Red 'X' next to items)
            # Items start at y=90, spaced by 30px.
            # We show up to 8 items.
            start_index = max(0, len(cart.items) - 8)
            visible_count = min(len(cart.items), 8)
            
            base_y = 90
            # Check each visible row
            for i in range(visible_count):
                row_y = base_y - 20 # Top of the text line approx
                # Button area: x(270-295), y(row_y to row_y+20)
                # Text is drawn at baseline 'base_y'. So button should be around base_y - 15
                btn_y_center = base_y - 5
                if 270 <= x <= 295 and (btn_y_center - 10) <= y <= (btn_y_center + 10):
                    click_action = "DELETE_ITEM"
                    target_delete_index = start_index + i
                    break
                
                base_y += 30

    cv2.namedWindow("Smart Checkout Terminal")
    cv2.setMouseCallback("Smart Checkout Terminal", mouse_handler)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("❌ Camera read failed or disconnected.")
                break
            frame = cv2.resize(frame, (640, 480))
            current_time = time.time()
            
            # Handle Mouse Actions
            if click_action == "DELETE_ITEM" and not verification_mode:
                if len(cart.items) > 0:
                    voice.speak("Kelulusan pengurus diperlukan untuk memadam item. Masukkan pin.")
                    verification_mode = True
                    entered_pin = ""
                else:
                    voice.speak("Troli kosong.")
                click_action = None # Consumed, but we keep target_delete_index
                
            elif click_action == "PAY" and not verification_mode:
                if len(cart.items) > 0:
                    final_amount = cart.get_total()
                    voice.speak(f"Pembayaran berjaya. Jumlah adalah {final_amount:.2f} Ringgit. Terima kasih.")
                    cart.clear()
                    is_scanning_locked = False 
                else:
                    voice.speak("Troli kosong.")
                click_action = None

            if not verification_mode:
                results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
                annotated_frame = results[0].plot()
                detected_class = None
                conf_val = 0.0
                
                if results[0].boxes:
                    cls_id = int(results[0].boxes.cls[0])
                    detected_class = model.names[cls_id]
                    conf_val = float(results[0].boxes.conf[0])

                # Logic with Delay
                if detected_class:
                    # Check if we are ready to scan (not locked and delay has passed)
                    if not is_scanning_locked and (current_time - last_scan_completion_time > scan_delay):
                        if detected_class in PRODUCT_DB:
                            added_item = cart.add(detected_class)
                            play_beep() # BEEP!
                            
                            # Log to Database
                            log_transaction(added_item['name'], added_item['price'], conf_val)

                            # Enhanced Audio: Read Name and Price
                            price_val = added_item['price']
                            voice_text = f"Ditambah {added_item['name']}. Harga, {price_val:.2f} Ringgit."
                            voice.speak(voice_text)
                            
                            is_scanning_locked = True
                            last_scanned_item_name = detected_class
                            cv2.rectangle(annotated_frame, (0,0), (640,480), (0,255,0), 10)
                    elif is_scanning_locked:
                        cv2.putText(annotated_frame, "PLEASE CLEAR AREA", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        # Waiting for scan_delay to finish
                        cv2.putText(annotated_frame, "SYSTEM READY IN SEC...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                else:
                    if is_scanning_locked:
                        is_scanning_locked = False
                        last_scanned_item_name = ""
                        last_scan_completion_time = current_time # Start delay timer now
            else:
                # In Verification Mode - Show blurred or darkened frame
                annotated_frame = cv2.GaussianBlur(frame, (21, 21), 0)
                cv2.putText(annotated_frame, "MANAGER APPROVAL REQUIRED", (50, 200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2)
                cv2.putText(annotated_frame, verification_message + entered_pin, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.putText(annotated_frame, "Press ESC to Cancel", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            ui_height = 480
            ui_width = 300 + 640
            canvas = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)
            canvas[:, :300] = (240, 240, 240) 
            cv2.putText(canvas, "MY SMART SHOP", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)
            cv2.putText(canvas, "---------------------", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)
            
            y_offset = 90
            total = cart.get_total()
            
            # Calculate which items to show
            start_index = max(0, len(cart.items) - 8)
            recent_items = cart.items[start_index:]
            
            for i, item in enumerate(recent_items):
                # Name
                cv2.putText(canvas, item['name'], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                # Price
                price_str = f"{item['price']:.2f}"
                cv2.putText(canvas, price_str, (210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                
                # DELETE BUTTON [X]
                # Draw a small red box at x=270
                cv2.rectangle(canvas, (270, y_offset - 10), (290, y_offset + 5), (0,0,255), -1)
                cv2.putText(canvas, "X", (274, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
                
                y_offset += 30
                
            # Global Buttons (Only PAY now)
            cv2.rectangle(canvas, (150, 400), (280, 430), (0, 200, 0), -1)
            cv2.putText(canvas, "PAY", (195, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            cv2.putText(canvas, "TOTAL:", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
            cv2.putText(canvas, f"RM {total:.2f}", (140, 465), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,100,0), 2)
            canvas[:, 300:] = annotated_frame
            cv2.imshow("Smart Checkout Terminal", canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if not verification_mode:
                if key == ord('q'): break
            else:
                if key == 13: # ENTER
                    if entered_pin == "1234":
                        if target_delete_index != -1:
                            removed = cart.remove_at_index(target_delete_index)
                            if removed: 
                                voice.speak(f"Dibuang {removed['name']}")
                                # Reset target
                                target_delete_index = -1
                        verification_mode = False
                    else:
                        voice.speak("Akses ditolak. PIN salah.")
                        entered_pin = ""
                elif key == 27: # ESC
                    voice.speak("Dibatalkan.")
                    verification_mode = False
                    target_delete_index = -1
                elif 48 <= key <= 57: # 0-9
                    if len(entered_pin) < 4: entered_pin += chr(key)
                elif key == 8: # Backspace
                    entered_pin = entered_pin[:-1]
    
    finally:
        # 5. CLEANUP
        print("🛑 Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        voice.stop()

if __name__ == "__main__":
    main()
