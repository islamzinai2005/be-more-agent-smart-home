# =========================================================================
#  Be More Agent 🤖
#  A Local, Offline-First AI Agent for Raspberry Pi
#
#  Copyright (c) 2026 brenpoly
#  Licensed under the MIT License
#  Source: https://github.com/brenpoly/be-more-agent
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
#  This project is a generic framework and includes no copyrighted assets.
# =========================================================================

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import subprocess
import random
import re
import sys
import select
import traceback
import atexit
import datetime
import warnings
import wave
import struct 
import unicodedata

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal 

# --- AI ENGINES ---
import openwakeword
from openwakeword.model import Model
import ollama 

# --- WEB SEARCH (Using your working import) ---
from ddgs import DDGS 
from home_control import (
    light_on,
    light_off,
    open_window,
    close_window,
    open_door,
    close_door,
)

# Optional STT fallback (used when whisper.cpp is not available)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel  # type: ignore
except Exception:
    FasterWhisperModel = None

# Optional TTS fallback (Windows SAPI)
try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None 

DEFAULT_CONFIG = {
    "text_model": "gemma3:1b",
    "vision_model": "moondream",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "input_device_name": None,
    "fullscreen": True,
    "chat_memory": True,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "stt_model": "small",
    "stt_language": "fr",
    "stt_beam_size": 5,
    "normalize_mic_audio": True
}

# LLM SETTINGS
OLLAMA_OPTIONS = {
    'keep_alive': '-1',     
    'num_thread': 4,
    'temperature': 0.7,     
    'top_k': 40,
    'top_p': 0.9
}

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
VISION_MODEL = CURRENT_CONFIG["vision_model"]
INPUT_DEVICE_NAME = CURRENT_CONFIG.get("input_device_name")

class BotStates:
    IDLE = "idle"             
    LISTENING = "listening"   
    THINKING = "thinking"     
    SPEAKING = "speaking"     
    ERROR = "error"           
    CAPTURING = "capturing" 
    WARMUP = "warmup"       

# --- SYSTEM PROMPT ---
BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a Raspberry Pi.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic.

INSTRUCTIONS:
- If the user asks for a physical action (time, search, photo), output JSON.
- If the user just wants to chat, reply with NORMAL TEXT.

### EXAMPLES ###

User: What time is it?
You: {"action": "get_time", "value": "now"}

User: Hello!
You: Hi! I am ready to help!

User: Search for news about robots.
You: {"action": "search_web", "value": "robots news"}

User: What do you see right now?
You: {"action": "capture_image", "value": "environment"}

### END EXAMPLES ###
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

# Sound Directories
greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"

# =========================================================================
# 2. GUI CLASS
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480 
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300 

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        start_fullscreen = CURRENT_CONFIG.get("fullscreen", os.name != 'nt')
        if os.name == 'nt' and "fullscreen" not in CURRENT_CONFIG:
            start_fullscreen = False

        master.attributes('-fullscreen', bool(start_fullscreen))
        if not start_fullscreen:
            master.geometry("1000x700")
            try:
                master.state('normal')
            except Exception:
                pass
        master.bind('<Escape>', self.exit_fullscreen)
        
        # Inputs
        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)
        
        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0 
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        
        self.last_ptt_time = 0 
        self.ptt_event = threading.Event()       
        self.recording_active = threading.Event() 
        self.interrupted = threading.Event() 
        
        self.tts_queue = []          
        self.tts_queue_lock = threading.Lock() 
        self.tts_thread = None       
        self.tts_active = threading.Event()
        self.current_audio_process = None 

        self._is_exiting = False
        self._stt_model = None
        self._tts_engine = None
        self.shutdown_event = threading.Event()
        self.main_thread = None
        self._input_settings_cache = {}
        
        # --- WAKE WORD INITIALIZATION ---
        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                # Some Windows wheels of openwakeword don't bundle the feature models
                # (melspectrogram/embedding/vad). Download them if missing.
                try:
                    openwakeword.utils.download_models(model_names=["__features_only__"])
                except Exception as e:
                    print(f"[WARN] Could not download OpenWakeWord feature models: {e}")

                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded.", flush=True)
                except TypeError:
                    # Backwards compatibility with older openwakeword APIs
                    self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded (Legacy API).", flush=True)
            except Exception as e:
                print(f"[CRITICAL] Failed to load model: {e}")
        else:
            print(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")

        # GUI Setup
        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility) 
        
        self.overlay_label = tk.Label(master, bg='black')
        self.overlay_label.bind('<Button-1>', self.toggle_hud_visibility)
        
        self.response_text = tk.Text(master, height=6, width=60, wrap=tk.WORD, 
                                     state=tk.DISABLED, bg="#ffffff", fg="#000000", font=('Arial', 12)) 
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")
        
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)

        self.load_animations()
        self.update_animation() 
        
        self.main_thread = threading.Thread(target=self.safe_main_execution, daemon=True)
        self.main_thread.start()

    # --- HELPERS ---

    def extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return None
        except: return None

    def safe_exit(self):
        if getattr(self, "_is_exiting", False):
            return
        self._is_exiting = True
        self.shutdown_event.set()
        self.ptt_event.set()

        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except: pass

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear() 
        
        self.save_chat_history()

        try:
            if self._tts_engine is not None:
                self._tts_engine.stop()
        except Exception:
            pass
        
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except: pass

        try:
            if self.tts_thread and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=1)
        except Exception:
            pass

        try:
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=1)
        except Exception:
            pass

        try:
            self.master.quit()
            self.master.destroy()
        except Exception:
            pass
        return

    def _get_available_ollama_models(self):
        try:
            data = ollama.list()
            models = []
            if hasattr(data, "models"):
                models = getattr(data, "models") or []
            elif isinstance(data, dict):
                models = data.get("models", [])

            names = []
            for m in models:
                if isinstance(m, str):
                    names.append(m)
                elif hasattr(m, "model") and getattr(m, "model"):
                    names.append(getattr(m, "model"))
                elif isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if name:
                        names.append(name)
            return names
        except Exception:
            return []

    def _resolve_ollama_model(self, preferred: str) -> str:
        names = self._get_available_ollama_models()
        if not names:
            return preferred
        if preferred in names:
            return preferred
        # family-aware fallbacks
        if preferred.startswith("gemma3"):
            for n in names:
                if n.startswith("gemma3"):
                    return n
        if preferred.startswith("gemma"):
            for n in names:
                if n.startswith("gemma"):
                    return n

        for candidate in ("gemma:latest", "gemma", "gemma3:4b", "gemma:2b", "llama3", "llama3.1"):
            if candidate in names:
                return candidate
        return names[0]

    def _get_input_device_candidates(self):
        try:
            devices = sd.query_devices()
        except Exception:
            return []

        target = INPUT_DEVICE_NAME
        target_name = None
        if isinstance(target, int):
            try:
                target_name = sd.query_devices(target, 'input')['name']
            except Exception:
                target_name = None
        elif isinstance(target, str) and target.strip():
            target_name = target.strip()

        candidates = []
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] <= 0:
                continue
            hostapi_name = ''
            try:
                hostapi_name = sd.query_hostapis(dev['hostapi'])['name']
            except Exception:
                pass
            candidates.append({
                'index': idx,
                'name': dev['name'],
                'hostapi': hostapi_name,
                'samplerate': int(dev['default_samplerate']) if dev.get('default_samplerate') else 48000,
            })

        def score(item):
            name = item['name']
            hostapi = item['hostapi']
            score = 0
            if target_name:
                if name == target_name:
                    score += 100
                elif target_name.lower() in name.lower() or name.lower() in target_name.lower():
                    score += 50
            if isinstance(target, int) and item['index'] == target:
                score += 120

            # On Windows, callback recording is often more stable with DirectSound/MME
            # than with WASAPI/WDM-KS on some drivers.
            if 'DirectSound' in hostapi:
                score += 40
            elif 'MME' in hostapi:
                score += 30
            elif 'WASAPI' in hostapi:
                score += 20
            elif 'WDM-KS' in hostapi:
                score += 10

            return score

        candidates.sort(key=score, reverse=True)

        # de-duplicate while preserving priority
        seen = set()
        ordered = []
        for item in candidates:
            if item['index'] in seen:
                continue
            seen.add(item['index'])
            ordered.append(item)
        return ordered

    def _get_working_input_settings(self, purpose='record'):
        cached = self._input_settings_cache.get(purpose)
        if cached:
            return cached

        candidates = self._get_input_device_candidates()
        if not candidates:
            return (INPUT_DEVICE_NAME, 48000)

        preferred_rates = []
        for item in candidates:
            if item['samplerate'] not in preferred_rates:
                preferred_rates.append(item['samplerate'])
        for sr in (48000, 44100, 16000):
            if sr not in preferred_rates:
                preferred_rates.append(sr)

        dtype = 'int16' if purpose == 'wakeword' else None
        channels = 1

        def can_open_stream(device_index, sr):
            try:
                kwargs = {
                    'device': device_index,
                    'samplerate': sr,
                    'channels': channels,
                }
                if dtype is not None:
                    kwargs['dtype'] = dtype

                if purpose == 'record':
                    def _noop_callback(indata, frames, time_info, status):
                        return None
                    kwargs['callback'] = _noop_callback

                stream = sd.InputStream(**kwargs)
                stream.start()
                sd.sleep(50)
                stream.stop()
                stream.close()
                return True
            except Exception:
                return False

        for item in candidates:
            for sr in preferred_rates:
                try:
                    kwargs = {
                        'device': item['index'],
                        'samplerate': sr,
                        'channels': channels,
                    }
                    if dtype is not None:
                        kwargs['dtype'] = dtype
                    sd.check_input_settings(**kwargs)
                    if can_open_stream(item['index'], sr):
                        settings = (item['index'], sr)
                        self._input_settings_cache[purpose] = settings
                        print(f"[AUDIO] Using input device {item['index']}: {item['name']} ({item['hostapi']}) @ {sr} Hz for {purpose}.", flush=True)
                        return settings
                except Exception:
                    continue

        fallback = (candidates[0]['index'], candidates[0]['samplerate'])
        self._input_settings_cache[purpose] = fallback
        return fallback
        
    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError: pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5: 
            return 
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            print("[PTT] Toggle OFF", flush=True)
            self.recording_active.clear() 
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self.recording_active.set() 
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state == BotStates.SPEAKING or self.current_state == BotStates.THINKING:
            self.interrupted.set()
            self.thinking_sound_active.clear()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            if self.current_audio_process:
                try: self.current_audio_process.terminate()
                except: pass
            self.set_state(BotStates.IDLE, "Interrupted.")

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"] 
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                if state in self.animations.get("idle", []):
                     self.animations[state] = self.animations["idle"]
                else:
                    # Blue screen fallback
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color='#0000FF')
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            if len(frames) > 1:
                self.current_frame_index = random.randint(1, len(frames) - 1)
            else:
                self.current_frame_index = 0 
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        self.background_label.config(image=frames[self.current_frame_index])
        
        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg="", cam_path=None):
        def _update():
            if msg: print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg: self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except: pass
            else:
                self.overlay_label.place_forget()
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            if newline: 
                self.response_text.insert(tk.END, text + "\n")
            else: 
                self.response_text.insert(tk.END, text)
            
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
            
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def update_text_stream():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, chunk)
            self.response_text.see(tk.END) 
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, update_text_stream)

    def handle_home_command(self, text):
        normalized_text = unicodedata.normalize("NFKD", text.lower())
        normalized_text = "".join(ch for ch in normalized_text if not unicodedata.combining(ch))
        normalized_text = re.sub(r"[^a-z\s]", " ", normalized_text)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

        command_map = {
            "turn on the light": (light_on, "Lumière allumée."),
            "turn off the light": (light_off, "Lumière éteinte."),
            "open the window": (open_window, "Fenêtre ouverte."),
            "close the window": (close_window, "Fenêtre fermée."),
            "open the door": (open_door, "Porte ouverte."),
            "close the door": (close_door, "Porte fermée."),
            "allume la lumiere": (light_on, "Lumière allumée."),
            "allumer la lumiere": (light_on, "Lumière allumée."),
            "allume la lampe": (light_on, "Lumière allumée."),
            "allumer la lampe": (light_on, "Lumière allumée."),
            "eteins la lumiere": (light_off, "Lumière éteinte."),
            "eteindre la lumiere": (light_off, "Lumière éteinte."),
            "eteins la lampe": (light_off, "Lumière éteinte."),
            "eteindre la lampe": (light_off, "Lumière éteinte."),
            "ouvre la fenetre": (open_window, "Fenêtre ouverte."),
            "ouvrir la fenetre": (open_window, "Fenêtre ouverte."),
            "ouvre la baie": (open_window, "Fenêtre ouverte."),
            "ouvrir la baie": (open_window, "Fenêtre ouverte."),
            "ferme la fenetre": (close_window, "Fenêtre fermée."),
            "fermer la fenetre": (close_window, "Fenêtre fermée."),
            "ferme la baie": (close_window, "Fenêtre fermée."),
            "fermer la baie": (close_window, "Fenêtre fermée."),
            "ouvre la porte": (open_door, "Porte ouverte."),
            "ouvrir la porte": (open_door, "Porte ouverte."),
            "ferme la porte": (close_door, "Porte fermée."),
            "fermer la porte": (close_door, "Porte fermée."),
            "ouvre le portail": (open_door, "Porte ouverte."),
            "ouvrir le portail": (open_door, "Porte ouverte."),
            "ferme le portail": (close_door, "Porte fermée."),
            "fermer le portail": (close_door, "Porte fermée."),
        }

        for trigger, (action_func, confirmation_text) in command_map.items():
            if trigger in normalized_text:
                action_func()
                self.thinking_sound_active.clear()
                self.set_state(BotStates.SPEAKING, confirmation_text)
                self.append_to_text("BOT: ", newline=False)
                self.append_to_text(confirmation_text, newline=True)
                with self.tts_queue_lock:
                    self.tts_queue.append(confirmation_text)
                self.wait_for_tts()
                self.set_state(BotStates.IDLE, "Ready")
                return True

        return False

    # =========================================================================
    # 3. ACTION ROUTER
    # =========================================================================
    
    def execute_action_and_get_result(self, action_data):
        raw_action = action_data.get("action", "").lower().strip()
        value = action_data.get("value") or action_data.get("query")
        
        VALID_TOOLS = {
            "get_time", "search_web", "capture_image"
        }
        
        ALIASES = {
            "google": "search_web", "browser": "search_web", "news": "search_web",         
            "search_news": "search_web", "look": "capture_image", "see": "capture_image", 
            "check_time": "get_time"
        }

        action = ALIASES.get(raw_action, raw_action)
        print(f"ACTION: {raw_action} -> {action}", flush=True)

        if action not in VALID_TOOLS:
            if value and isinstance(value, str) and len(value.split()) > 1:
                return f"CHAT_FALLBACK::{value}"
            return "INVALID_ACTION"

        if action == "get_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."
        
        elif action == "search_web":
            print(f"Searching web for: {value}...", flush=True)
            try:
                # 'us-en' region is often more stable for CLI queries
                with DDGS() as ddgs:
                    results = []
                    # 1. News search
                    try:
                        results = list(ddgs.news(value, region='us-en', max_results=1))
                        if results: 
                            print(f"[DEBUG] Found News: {results[0].get('title')}", flush=True)
                    except Exception as e: 
                        print(f"[DEBUG] News Search Error: {e}", flush=True)
                    
                    # 2. Text fallback
                    if not results:
                        print("[DEBUG] No news found, trying text search...", flush=True)
                        try: 
                            results = list(ddgs.text(value, region='us-en', max_results=1))
                            if results: 
                                print(f"[DEBUG] Found Text: {results[0].get('title')}", flush=True)
                        except Exception as e:
                             print(f"[DEBUG] Text Search Error: {e}", flush=True)

                    if results:
                        r = results[0]
                        # Safe get
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('snippet', 'No Body'))
                        return f"SEARCH RESULTS for '{value}':\nTitle: {title}\nSnippet: {body[:300]}"
                    else: 
                        print(f"[DEBUG] Search returned 0 results.", flush=True)
                        return "SEARCH_EMPTY"
            except Exception as e:
                print(f"[DEBUG] Connection/Library Error: {e}", flush=True)
                return "SEARCH_ERROR"
        
        elif action == "capture_image":
             return "IMAGE_CAPTURE_TRIGGERED"

        return None

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self.tts_active.set()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
            while not self.shutdown_event.is_set():
                trigger_source = self.detect_wake_word_or_ptt()
                if self.shutdown_event.is_set() or trigger_source is None:
                    break
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue

                self.set_state(BotStates.LISTENING, "I'm listening!")
                
                audio_file = None
                if trigger_source == "PTT":
                    audio_file = self.record_voice_ptt()
                else:
                    audio_file = self.record_voice_adaptive()
                
                if not audio_file: 
                    self.set_state(BotStates.IDLE, "Heard nothing.")
                    continue
                
                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue
                
                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                if self.handle_home_command(user_text):
                    continue
                self.chat_and_respond(user_text, img_path=None)
                    
        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        global TEXT_MODEL
        TEXT_MODEL = self._resolve_ollama_model(TEXT_MODEL)
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as e:
            print(f"Failed to load {TEXT_MODEL}: {e}", flush=True)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("[INFO] PTT mode: click the app window, press Enter to START recording, speak into the microphone, then press Enter again to STOP.", flush=True)
        print("[INFO] Typing text in the PowerShell terminal is ignored by the assistant.", flush=True)
        print("Models loaded.", flush=True)

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting...")
        self.ptt_event.clear()
        
        if self.oww_model: self.oww_model.reset()

        if self.oww_model is None:
            while not self.shutdown_event.is_set() and not self.ptt_event.wait(timeout=0.1):
                pass
            if self.shutdown_event.is_set():
                return None
            self.ptt_event.clear()
            return "PTT"

        CHUNK_SIZE = 1280
        OWW_SAMPLE_RATE = 16000

        input_device, native_rate = self._get_working_input_settings('wakeword')
            
        use_resampling = (native_rate != OWW_SAMPLE_RATE)
        input_rate = native_rate if use_resampling else OWW_SAMPLE_RATE
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE

        try:
            with sd.InputStream(samplerate=input_rate, channels=1, dtype='int16', 
                                blocksize=input_chunk_size, device=input_device) as stream:
                while True:
                    if self.shutdown_event.is_set():
                        return None
                    if self.ptt_event.is_set():
                        self.ptt_event.clear()
                        return "PTT"

                    # On Windows, select() works only with sockets; stdin triggers WinError 10038.
                    if os.name != 'nt':
                        try:
                            rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                            if rlist:
                                sys.stdin.readline()
                                return "CLI"
                        except Exception:
                            pass

                    data, _ = stream.read(input_chunk_size)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    if use_resampling:
                         audio_data = scipy.signal.resample(audio_data, CHUNK_SIZE).astype(np.int16)

                    prediction = self.oww_model.predict(audio_data)
                    for mdl in self.oww_model.prediction_buffer.keys():
                        if list(self.oww_model.prediction_buffer[mdl])[-1] > WAKE_WORD_THRESHOLD:
                            self.oww_model.reset() 
                            return "WAKE"
        except Exception as e:
            if self.shutdown_event.is_set():
                return None
            print(f"Wake Word Stream Error: {e}")
            while not self.shutdown_event.is_set() and not self.ptt_event.wait(timeout=0.1):
                pass
            return None if self.shutdown_event.is_set() else "PTT"

    def record_voice_adaptive(self, filename="input.wav"):
        print("Recording (Adaptive)...", flush=True)
        time.sleep(0.5) 
        input_device, samplerate = self._get_working_input_settings('record')

        silence_threshold = 0.006
        silence_duration = 1.5
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05 
        chunk_size = int(samplerate * chunk_duration)
        
        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())  
            recorded_chunks += 1
            if recorded_chunks < 5: return 
            if volume_norm < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks: silence_started = True
            else: silent_chunks = 0

        try:
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, 
                                device=input_device, blocksize=chunk_size): 
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception as e: return None 
        
        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
        time.sleep(0.5)
        input_device, samplerate = self._get_working_input_settings('record')

        buffer = []
        def callback(indata, frames, time_info, status): buffer.append(indata.copy())
        
        try:
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=input_device):
                while self.recording_active.is_set(): sd.sleep(50)
        except Exception as e:
            self.recording_active.clear()
            print(f"PTT Recording Error: {e}", flush=True)
            return None

        self.recording_active.clear()
        if not buffer:
            print("PTT Recording Error: no audio captured. Make sure the app window is focused and speak into the selected microphone.", flush=True)
            return None
            
        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer: return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

        if CURRENT_CONFIG.get("normalize_mic_audio", True):
            peak = float(np.max(np.abs(audio_data))) if len(audio_data) else 0.0
            if peak > 0:
                target_peak = 0.85
                gain = min(target_peak / peak, 8.0)
                audio_data = np.clip(audio_data * gain, -1.0, 1.0)

        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        try:
            stt_model_name = CURRENT_CONFIG.get("stt_model", "small")
            stt_language = CURRENT_CONFIG.get("stt_language", "fr")
            stt_beam_size = int(CURRENT_CONFIG.get("stt_beam_size", 5))

            # 1) Prefer whisper.cpp when present (default Raspberry Pi setup)
            whisper_cli = "./whisper.cpp/build/bin/whisper-cli"
            if os.name == 'nt' and os.path.exists(whisper_cli + ".exe"):
                whisper_cli = whisper_cli + ".exe"
            whisper_model = "./whisper.cpp/models/ggml-base.en.bin"

            if os.path.exists(whisper_cli) and os.path.exists(whisper_model):
                result = subprocess.run(
                    [whisper_cli, "-m", whisper_model, "-l", "en", "-t", "4", "-f", filename],
                    capture_output=True, text=True
                )
                transcription_lines = result.stdout.strip().split('\n')
                if transcription_lines and transcription_lines[-1].strip():
                    last_line = transcription_lines[-1].strip()
                    if ']' in last_line:
                        transcription = last_line.split("]")[1].strip()
                    else:
                        transcription = last_line
                else:
                    transcription = ""
                print(f"Heard: '{transcription}'", flush=True)
                return transcription.strip()

            # 2) Cross-platform fallback (Windows friendly)
            if FasterWhisperModel is None:
                print("Transcription Error: whisper.cpp not found and faster-whisper is not installed.")
                print("Install: python -m pip install faster-whisper", flush=True)
                return ""

            if self._stt_model is None:
                print(f"[STT] Initializing faster-whisper model '{stt_model_name}' (first run may download and take 1-3 minutes)...", flush=True)
                self._stt_model = FasterWhisperModel(stt_model_name, device="cpu", compute_type="int8")

            lang = None if str(stt_language).lower() in ("auto", "", "none") else stt_language
            print(f"[STT] Running transcription (language={lang or 'auto'}, beam={stt_beam_size})...", flush=True)
            segments, _info = self._stt_model.transcribe(
                filename,
                language=lang,
                beam_size=stt_beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400}
            )
            transcription = " ".join((seg.text or "").strip() for seg in segments).strip()
            print(f"Heard: '{transcription}'", flush=True)
            return transcription
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        try:
            subprocess.run(["rpicam-still", "-t", "500", "-n", "--width", "640", "--height", "480", "-o", BMO_IMAGE_FILE], check=True)
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = Image.open(BMO_IMAGE_FILE)
                img = img.rotate(rotation, expand=True) 
                img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            print(f"Camera Error: {e}")
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def chat_and_respond(self, text, img_path=None):
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            with self.tts_queue_lock: 
                self.tts_queue.append("Okay. Memory wiped.")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        model_to_use = VISION_MODEL if img_path else TEXT_MODEL
        self.set_state(BotStates.THINKING, "Thinking...", cam_path=img_path)
        
        messages = []
        if img_path:
            messages = [{"role": "user", "content": text, "images": [img_path]}]
        else:
            user_msg = {"role": "user", "content": text}
            messages = self.permanent_memory + self.session_memory + [user_msg]
        
        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()
        
        full_response_buffer = ""
        sentence_buffer = "" 
        
        try:
            stream = ollama.chat(model=model_to_use, messages=messages, stream=True, options=OLLAMA_OPTIONS)
            
            is_action_mode = False
            
            for chunk in stream:
                if self.interrupted.is_set(): break 
                content = chunk['message']['content']
                full_response_buffer += content
                
                if '{"' in content or "action:" in content.lower():
                    is_action_mode = True
                    self.thinking_sound_active.clear()
                    continue 

                if is_action_mode: continue

                self.thinking_sound_active.clear()
                if self.current_state != BotStates.SPEAKING:
                    self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                    self.append_to_text("BOT: ", newline=False)

                self._stream_to_text(content)
                
                sentence_buffer += content
                if any(punct in content for punct in ".!?\n"):
                    clean_sentence = sentence_buffer.strip()
                    if clean_sentence and re.search(r'[a-zA-Z0-9]', clean_sentence):
                        with self.tts_queue_lock: self.tts_queue.append(clean_sentence)
                    sentence_buffer = ""

            if is_action_mode:
                action_data = self.extract_json_from_text(full_response_buffer)
                if action_data:
                    tool_result = self.execute_action_and_get_result(action_data)

                    if tool_result and tool_result.startswith("CHAT_FALLBACK::"):
                        chat_text = tool_result.split("::", 1)[1]
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(chat_text, newline=True)
                        with self.tts_queue_lock: self.tts_queue.append(chat_text)
                        self.session_memory.append({"role": "assistant", "content": chat_text})
                        self.wait_for_tts()
                        self.set_state(BotStates.IDLE, "Ready")
                        return

                    if tool_result == "IMAGE_CAPTURE_TRIGGERED":
                        new_img_path = self.capture_image()
                        if new_img_path:
                            self.chat_and_respond(text, img_path=new_img_path)
                            return 

                    elif tool_result == "INVALID_ACTION":
                        fallback_text = "I am not sure how to do that."
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock: self.tts_queue.append(fallback_text)

                    elif tool_result == "SEARCH_EMPTY":
                        fallback_text = "I searched, but I couldn't find any news about that."
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock: self.tts_queue.append(fallback_text)

                    elif tool_result == "SEARCH_ERROR":
                        fallback_text = "I cannot reach the internet right now."
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock: self.tts_queue.append(fallback_text)

                    elif tool_result:
                        summary_prompt = [
                            {"role": "system", "content": "Summarize this result in one short sentence."},
                            {"role": "user", "content": f"RESULT: {tool_result}\nUser Question: {text}"}
                        ]
                        
                        self.set_state(BotStates.THINKING, "Reading...")
                        self.thinking_sound_active.set()
                        
                        final_resp = ollama.chat(model=model_to_use, messages=summary_prompt, stream=False, options=OLLAMA_OPTIONS)
                        final_text = final_resp['message']['content']
                        
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                        
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(final_text, newline=True)
                        with self.tts_queue_lock: self.tts_queue.append(final_text)
                        self.session_memory.append({"role": "assistant", "content": final_text})
            else:
                self.append_to_text("")
                self.session_memory.append({"role": "assistant", "content": full_response_buffer}) 
            
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
                
        except Exception as e:
            print(f"LLM Error: {e}")
            self.set_state(BotStates.ERROR, "Brain Freeze!")

    def wait_for_tts(self):
        while (self.tts_queue or self.tts_active.is_set()) and not self.shutdown_event.is_set():
            if self.interrupted.is_set(): break
            time.sleep(0.1)

    def _tts_worker(self):
        while not self.shutdown_event.is_set() or self.tts_queue:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue: 
                    text = self.tts_queue.pop(0)
                    self.tts_active.set() 
            if text: 
                self.speak(text)
                self.tts_active.clear() 
            else: time.sleep(0.05)

    def _speak_windows_tts_fallback(self, text):
        escaped = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speak.Volume = 100; "
            "$speak.Rate = 0; "
            f"$speak.Speak('{escaped}')"
        )

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.returncode == 0
        except Exception:
            return False

    def speak(self, text):
        if self.shutdown_event.is_set():
            return
        clean = re.sub(r"[^\w\s,.!?:-]", "", text)
        if not clean.strip(): return
        
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")

        piper_bin = "./piper/piper"
        if os.name == 'nt' and os.path.exists(piper_bin + ".exe"):
            piper_bin = piper_bin + ".exe"

        if not os.path.exists(piper_bin) or not os.path.exists(voice_model):
            if os.name == 'nt':
                print(f"[TTS WINDOWS] '{clean}'", flush=True)
                if self._speak_windows_tts_fallback(clean):
                    return

            if pyttsx3 is None:
                print("Audio Error: Piper not found and pyttsx3 not installed.")
                return
            try:
                print(f"[TTS FALLBACK] '{clean}'", flush=True)
                if self._tts_engine is None:
                    self._tts_engine = pyttsx3.init()
                    try:
                        self._tts_engine.setProperty('volume', 1.0)
                    except Exception:
                        pass
                self._tts_engine.say(clean)
                self._tts_engine.runAndWait()
            except Exception as e:
                print(f"Audio Error: {e}")
            return

        print(f"[PIPER SPEAKING] '{clean}'", flush=True)
        
        try:
            self.current_audio_process = subprocess.Popen(
                [piper_bin, "--model", voice_model, "--output-raw"], 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            self.current_audio_process.stdin.write(clean.encode() + b'\n')
            self.current_audio_process.stdin.close() 

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except:
                native_rate = 48000 

            PIPER_RATE = 22050
            use_native_rate = False
            
            try:
                sd.check_output_settings(device=None, samplerate=PIPER_RATE)
            except:
                use_native_rate = True

            with sd.RawOutputStream(samplerate=native_rate if use_native_rate else PIPER_RATE, 
                                    channels=1, dtype='int16', 
                                    device=None, latency='low', blocksize=2048) as stream:
                while True:
                    if self.interrupted.is_set() or self.shutdown_event.is_set(): break
                    data = self.current_audio_process.stdout.read(4096)
                    if not data: break 
                    
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        self.current_volume = np.max(np.abs(audio_chunk))
                        if use_native_rate:
                            num_samples = int(len(audio_chunk) * (native_rate / PIPER_RATE))
                            audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.int16)
                        stream.write(audio_chunk.tobytes())
                    else:
                        self.current_volume = 0
                time.sleep(0.5) 
                    
        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.current_volume = 0 
            if self.current_audio_process:
                if self.current_audio_process.stdout: self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None: self.current_audio_process.terminate()
                self.current_audio_process = None

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set() and not self.shutdown_event.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound: self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set() or self.shutdown_event.is_set(): return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path): return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except:
                native_rate = 48000 

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait() 
        except: pass

    def load_chat_history(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f: return json.load(f)
            except: pass
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        conv = full[1:]
        if len(conv) > 10: conv = conv[-10:]
        with open(MEMORY_FILE, "w") as f: 
            json.dump([full[0]] + conv, f, indent=4)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting...", flush=True)
        try:
            app.safe_exit()
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
        try:
            app.safe_exit()
        except Exception:
            pass
