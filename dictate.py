#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper.
Hold the hotkey to record, release to transcribe and copy to clipboard.
"""

import argparse
import configparser
import subprocess
import tempfile
import threading
import signal
import sys
import os
import select
from pathlib import Path

import evdev
from evdev import ecodes
from faster_whisper import WhisperModel

__version__ = "0.1.0"

CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"

# Maps config key names to evdev KEY_ constant suffixes
_KEY_NAME_MAP = {
    "scroll_lock": "SCROLLLOCK",
    "caps_lock": "CAPSLOCK",
    "num_lock": "NUMLOCK",
    "page_up": "PAGEUP",
    "page_down": "PAGEDOWN",
    "print_screen": "SYSRQ",
    "insert": "INSERT",
    "delete": "DELETE",
    "home": "HOME",
    "end": "END",
    "left": "LEFT",
    "right": "RIGHT",
    "up": "UP",
    "down": "DOWN",
    "enter": "ENTER",
    "return": "ENTER",
    "space": "SPACE",
    "backspace": "BACKSPACE",
    "tab": "TAB",
    "esc": "ESC",
    "escape": "ESC",
    "ctrl": "LEFTCTRL",
    "ctrl_l": "LEFTCTRL",
    "ctrl_r": "RIGHTCTRL",
    "alt": "LEFTALT",
    "alt_l": "LEFTALT",
    "alt_r": "RIGHTALT",
    "shift": "LEFTSHIFT",
    "shift_l": "LEFTSHIFT",
    "shift_r": "RIGHTSHIFT",
    "super": "LEFTMETA",
    "super_l": "LEFTMETA",
    "super_r": "RIGHTMETA",
    "pause": "PAUSE",
    "menu": "COMPOSE",
}

# Modifier names that accept either left or right physical key
_MODIFIER_GROUPS = {
    "ctrl":    frozenset({ecodes.KEY_LEFTCTRL,  ecodes.KEY_RIGHTCTRL}),
    "ctrl_l":  frozenset({ecodes.KEY_LEFTCTRL}),
    "ctrl_r":  frozenset({ecodes.KEY_RIGHTCTRL}),
    "shift":   frozenset({ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT}),
    "shift_l": frozenset({ecodes.KEY_LEFTSHIFT}),
    "shift_r": frozenset({ecodes.KEY_RIGHTSHIFT}),
    "alt":     frozenset({ecodes.KEY_LEFTALT,   ecodes.KEY_RIGHTALT}),
    "alt_l":   frozenset({ecodes.KEY_LEFTALT}),
    "alt_r":   frozenset({ecodes.KEY_RIGHTALT}),
    "super":   frozenset({ecodes.KEY_LEFTMETA,  ecodes.KEY_RIGHTMETA}),
    "super_l": frozenset({ecodes.KEY_LEFTMETA}),
    "super_r": frozenset({ecodes.KEY_RIGHTMETA}),
}


def detect_display_server():
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


DISPLAY_SERVER = detect_display_server()


def load_config():
    config = configparser.ConfigParser()
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "key": "ctrl+shift+space",
        "auto_type": "true",
        "copy_to_clipboard": "false",
        "notifications": "true",
    }
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)
    return {
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        "initial_prompt": config.get("whisper", "initial_prompt", fallback=None),
        "hotwords": config.get("whisper", "hotwords", fallback=None),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "copy_to_clipboard": config.getboolean("behavior", "copy_to_clipboard", fallback=False),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
    }


CONFIG = load_config()


def _resolve_key_code(name):
    """Resolve a single key name to an evdev code."""
    evdev_name = _KEY_NAME_MAP.get(name, name.upper())
    key_const = f"KEY_{evdev_name}"
    if hasattr(ecodes, key_const):
        return getattr(ecodes, key_const)
    return None


def parse_hotkey(key_str):
    """Parse a key combo such as 'ctrl+shift+space' into (modifier_groups, trigger_code, display_name).

    The last token is the trigger key; all preceding tokens are modifiers.
    Each modifier group is a frozenset of evdev codes — any one of them satisfies that modifier.
    """
    parts = [p.strip().lower() for p in key_str.split("+")]
    trigger_name = parts[-1]
    modifier_names = parts[:-1]

    trigger_code = _resolve_key_code(trigger_name)
    if trigger_code is None:
        print(f"Unknown trigger key: {trigger_name!r}, defaulting to space")
        trigger_code = ecodes.KEY_SPACE
        trigger_name = "space"

    modifier_groups = []
    valid_modifier_names = []
    for mod in modifier_names:
        if mod in _MODIFIER_GROUPS:
            modifier_groups.append(_MODIFIER_GROUPS[mod])
            valid_modifier_names.append(mod)
        else:
            code = _resolve_key_code(mod)
            if code is not None:
                modifier_groups.append(frozenset({code}))
                valid_modifier_names.append(mod)
            else:
                print(f"Unknown modifier: {mod!r}, ignoring")

    display_name = "+".join(valid_modifier_names + [trigger_name])
    return modifier_groups, trigger_code, display_name


def find_keyboards():
    """Return all input devices that look like keyboards."""
    keyboards = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            keys = caps.get(ecodes.EV_KEY, [])
            if ecodes.KEY_A in keys:
                keyboards.append(dev)
        except (PermissionError, OSError):
            continue
    return keyboards


MODEL_SIZE = CONFIG["model"]
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
INITIAL_PROMPT = CONFIG["initial_prompt"]
HOTWORDS = CONFIG["hotwords"]
AUTO_TYPE = CONFIG["auto_type"]
COPY_TO_CLIPBOARD = CONFIG["copy_to_clipboard"]
NOTIFICATIONS = CONFIG["notifications"]
MODIFIER_GROUPS, HOTKEY_CODE, HOTKEY_NAME = parse_hotkey(CONFIG["key"])


def _detect_wayland_typer():
    """Return the first available Wayland typing tool, or None."""
    for cmd in ["ydotool", "wtype"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
            return cmd
    return None


WAYLAND_TYPER = _detect_wayland_typer() if DISPLAY_SERVER == "wayland" else None


class Dictation:
    def __init__(self):
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        self.held_keys = set()

        print(f"Loading Whisper model ({MODEL_SIZE})...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            self.model_loaded.set()
            print(f"Model loaded. Ready for dictation!")
            print(f"Hold [{HOTKEY_NAME}] to record, release to transcribe.")
            print("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            print(f"Failed to load model: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                print("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def _modifiers_held(self):
        """Return True if every required modifier group has at least one key held."""
        return all(
            any(k in self.held_keys for k in group)
            for group in MODIFIER_GROUPS
        )

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        if not NOTIFICATIONS:
            return
        subprocess.run(
            [
                "notify-send",
                "-a", "SoupaWhisper",
                "-i", icon,
                "-t", str(timeout),
                "-h", "string:x-canonical-private-synchronous:soupawhisper",
                title,
                message
            ],
            capture_output=True
        )

    def start_recording(self):
        if self.recording or self.model_error:
            return

        self.recording = True
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        self.record_process = subprocess.Popen(
            [
                "arecord",
                "-f", "S16_LE",
                "-r", "16000",
                "-c", "1",
                "-t", "wav",
                self.temp_file.name
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("Recording...")
        self.notify("Recording...", f"Release {HOTKEY_NAME} when done", "audio-input-microphone", 30000)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        print("Transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 30000)

        self.model_loaded.wait()

        if self.model_error:
            print("Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        try:
            segments, info = self.model.transcribe(
                self.temp_file.name,
                beam_size=5,
                vad_filter=True,
                initial_prompt=INITIAL_PROMPT,
                hotwords=HOTWORDS,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            if text:
                if COPY_TO_CLIPBOARD:
                    self._copy_to_clipboard(text)
                if AUTO_TYPE:
                    self._type_text(text)
                print(f"Copied: {text}")
                self.notify("Copied!", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                print("No speech detected")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            print(f"Error: {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)

    def _copy_to_clipboard(self, text):
        if DISPLAY_SERVER == "wayland":
            subprocess.run(["wl-copy"], input=text.encode(), check=False)
        else:
            process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            process.communicate(input=text.encode())

    def _type_text(self, text):
        if DISPLAY_SERVER == "wayland":
            if WAYLAND_TYPER == "wtype":
                subprocess.run(["wtype", text], check=False)
            elif WAYLAND_TYPER == "ydotool":
                subprocess.run(["ydotool", "type", "--", text], check=False)
        else:
            subprocess.run(["xdotool", "type", "--clearmodifiers", text])

    def stop(self):
        print("\nExiting...")
        self.running = False
        os._exit(0)

    def run(self):
        keyboards = find_keyboards()
        if not keyboards:
            print("Error: No keyboard devices found.")
            print("Make sure you are in the 'input' group:")
            print("  sudo usermod -aG input $USER  (then log out and back in)")
            sys.exit(1)

        print(f"Monitoring {len(keyboards)} keyboard device(s)...")

        while self.running:
            try:
                r, _, _ = select.select(keyboards, [], [], 0.1)
            except (ValueError, OSError):
                keyboards = find_keyboards()
                continue

            for dev in r:
                try:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue
                        if event.value == 1:    # key down
                            self.held_keys.add(event.code)
                            if event.code == HOTKEY_CODE and self._modifiers_held():
                                self.start_recording()
                        elif event.value == 0:  # key up
                            if event.code == HOTKEY_CODE and self.recording:
                                threading.Thread(target=self.stop_recording, daemon=True).start()
                            self.held_keys.discard(event.code)
                except OSError:
                    pass


def check_dependencies():
    missing = []

    if subprocess.run(["which", "arecord"], capture_output=True).returncode != 0:
        missing.append(("arecord", "alsa-utils"))

    if DISPLAY_SERVER == "wayland":
        if subprocess.run(["which", "wl-copy"], capture_output=True).returncode != 0:
            missing.append(("wl-copy", "wl-clipboard"))
        if AUTO_TYPE and WAYLAND_TYPER is None:
            missing.append(("wtype or ydotool", "wtype  # or: sudo apt install ydotool"))
    else:
        if subprocess.run(["which", "xclip"], capture_output=True).returncode != 0:
            missing.append(("xclip", "xclip"))
        if AUTO_TYPE:
            if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
                missing.append(("xdotool", "xdotool"))

    if missing:
        print("Missing dependencies:")
        for cmd, pkg in missing:
            print(f"  {cmd} - install with: sudo apt install {pkg}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SoupaWhisper - Push-to-talk voice dictation"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.parse_args()

    print(f"SoupaWhisper v{__version__}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Display server: {DISPLAY_SERVER}")

    check_dependencies()

    dictation = Dictation()

    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
