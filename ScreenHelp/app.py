"""
app.py - Main CustomTkinter application window for ScreenHelp

Layout:
  ┌─────────────────────────────────────────────────┐
  │  🤖 ScreenHelp        [Dark 🌙]  [⚙ Settings]  │  ← Header
  ├─────────────────────────────────────────────────┤
  │  [Screenshot 300×200]                           │
  │  Question: [____________text entry___] [Ask]    │
  ├─────────────────────────────────────────────────┤
  │  AI Response (scrollable CTkTextbox)            │
  │                              [📋 Copy Response] │
  ├─────────────────────────────────────────────────┤
  │  Provider: [OpenAI ▼]   Shortcut: Ctrl+Shift+S │  ← Footer / status
  └─────────────────────────────────────────────────┘
"""

import threading
import time
import logging
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from config import load_config, save_config, get_api_key, get_model
from capture import capture_fullscreen, image_to_bytes, image_to_thumbnail
from ai_client import AIClient
from hotkey import HotkeyManager
from invisibility import set_window_invisible

logger = logging.getLogger(__name__)


# ── Settings Toplevel ──────────────────────────────────────────────────────────

class SettingsWindow(ctk.CTkToplevel):
    """
    Modal-like settings panel.
    Lets the user configure API keys, provider, hotkey, and theme.
    """

    def __init__(self, parent: "ScreenHelpApp"):
        super().__init__(parent)
        self._parent = parent
        self._config = dict(parent.config)  # work on a copy

        self.title("ScreenHelp — Settings")
        self.geometry("500x720")
        self.resizable(False, True)
        self.grab_set()  # make it modal
        self.focus_set()

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 6}

        # Bottom action buttons (pinned)
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="bottom", pady=12, padx=16, fill="x")

        ctk.CTkButton(btn_frame, text="Save", command=self._save, width=110).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_frame, text="Cancel", command=self.destroy,
            fg_color="gray40", hover_color="gray30", width=90
        ).pack(side="right")

        self._status_label = ctk.CTkLabel(btn_frame, text="", text_color="green")
        self._status_label.pack(side="left")

        # Scrollable container for settings options
        self._scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._scroll.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # ── Display & Automation ───────────────────────────────────────────────
        ctk.CTkLabel(self._scroll, text="Display & Automation", font=ctk.CTkFont(weight="bold")).pack(**pad, anchor="w")

        # Opacity slider
        op_frame = ctk.CTkFrame(self._scroll, fg_color="transparent")
        op_frame.pack(padx=16, pady=2, anchor="w", fill="x")
        ctk.CTkLabel(op_frame, text="Window Opacity:").pack(side="left", padx=(0, 8))
        self._settings_opacity_slider = ctk.CTkSlider(
            op_frame,
            from_=0.35,
            to=1.0,
            number_of_steps=13,
            command=self._on_settings_opacity_slide,
            width=180,
        )
        current_op = float(self._config.get("opacity", 0.80))
        self._settings_opacity_slider.set(current_op)
        self._settings_opacity_slider.pack(side="left", padx=(0, 8))
        self._settings_opacity_label = ctk.CTkLabel(op_frame, text=f"{int(current_op * 100)}%", font=ctk.CTkFont(weight="bold"))
        self._settings_opacity_label.pack(side="left")

        # Checkboxes
        self._settings_pin_var = ctk.BooleanVar(value=bool(self._config.get("always_on_top", True)))
        ctk.CTkCheckBox(
            self._scroll,
            text="Keep window floating always on top (📌 Pinned)",
            variable=self._settings_pin_var,
        ).pack(padx=16, pady=4, anchor="w")

        self._settings_auto_var = ctk.BooleanVar(value=bool(self._config.get("auto_analyze", True)))
        ctk.CTkCheckBox(
            self._scroll,
            text="⚡ Auto-Solve upon screenshot without any click",
            variable=self._settings_auto_var,
        ).pack(padx=16, pady=4, anchor="w")

        # ── Provider ────────────────────────────────────────────────────────────
        ctk.CTkLabel(self._scroll, text="AI Provider", font=ctk.CTkFont(weight="bold")).pack(**pad, anchor="w")
        self._provider_var = ctk.StringVar(value=self._config.get("provider", "gemini"))
        provider_menu = ctk.CTkOptionMenu(
            self._scroll,
            values=["gemini", "llama", "openai", "anthropic"],
            variable=self._provider_var,
            command=self._on_provider_change,
            width=220,
        )
        provider_menu.pack(**pad, anchor="w")

        # ── API Keys ────────────────────────────────────────────────────────────
        ctk.CTkLabel(self._scroll, text="API Keys", font=ctk.CTkFont(weight="bold")).pack(**pad, anchor="w")

        self._key_entries: dict[str, ctk.CTkEntry] = {}
        for provider, label in [
            ("gemini", "Google Gemini API Key"),
            ("llama", "Llama API Key (Groq: gsk_… / or type 'ollama')"),
            ("openai", "OpenAI API Key"),
            ("anthropic", "Anthropic API Key"),
        ]:
            ctk.CTkLabel(self._scroll, text=label).pack(padx=16, pady=(4, 0), anchor="w")
            entry = ctk.CTkEntry(self._scroll, width=420, show="•", placeholder_text="Paste your key here…")
            entry.pack(padx=16, pady=(0, 4))
            key_field = f"{provider}_api_key"
            existing = self._config.get(key_field, "")
            if existing:
                entry.insert(0, existing)
            self._key_entries[provider] = entry

        # ── Models ──────────────────────────────────────────────────────────────
        ctk.CTkLabel(self._scroll, text="Models", font=ctk.CTkFont(weight="bold")).pack(**pad, anchor="w")

        models = self._config.get("model", {})
        self._model_entries: dict[str, ctk.CTkEntry] = {}
        for provider, default_model in [
            ("gemini", "gemini-flash-latest"),
            ("llama", "llama-4-scout-17b-16e-instruct"),
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-5-sonnet-20241022"),
        ]:
            ctk.CTkLabel(self._scroll, text=f"{provider.capitalize()} model").pack(padx=16, pady=(4, 0), anchor="w")
            entry = ctk.CTkEntry(self._scroll, width=420, placeholder_text=default_model)
            entry.pack(padx=16, pady=(0, 4))
            entry.insert(0, models.get(provider, default_model))
            self._model_entries[provider] = entry

        # ── Hotkey ──────────────────────────────────────────────────────────────
        ctk.CTkLabel(self._scroll, text="Global Hotkey", font=ctk.CTkFont(weight="bold")).pack(**pad, anchor="w")
        self._hotkey_entry = ctk.CTkEntry(self._scroll, width=220, placeholder_text="e.g. ctrl+shift+s")
        self._hotkey_entry.insert(0, self._config.get("hotkey", "ctrl+shift+s"))
        self._hotkey_entry.pack(**pad, anchor="w")

        # ── Theme ───────────────────────────────────────────────────────────────
        ctk.CTkLabel(self._scroll, text="Theme", font=ctk.CTkFont(weight="bold")).pack(**pad, anchor="w")
        self._theme_var = ctk.StringVar(value=self._config.get("theme", "dark"))
        ctk.CTkOptionMenu(
            self._scroll,
            values=["dark", "light", "system"],
            variable=self._theme_var,
            width=220,
        ).pack(**pad, anchor="w")

    def _on_settings_opacity_slide(self, val: float) -> None:
        self._settings_opacity_label.configure(text=f"{int(val * 100)}%")

    def _on_provider_change(self, value: str) -> None:
        self._config["provider"] = value

    def _save(self) -> None:
        # Gather values
        self._config["provider"] = self._provider_var.get()
        self._config["theme"] = self._theme_var.get()
        self._config["hotkey"] = self._hotkey_entry.get().strip() or "ctrl+shift+s"
        self._config["opacity"] = round(float(self._settings_opacity_slider.get()), 2)
        self._config["always_on_top"] = self._settings_pin_var.get()
        self._config["auto_analyze"] = self._settings_auto_var.get()

        for provider, entry in self._key_entries.items():
            val = entry.get().strip()
            if val:
                self._config[f"{provider}_api_key"] = val

        models = self._config.get("model", {})
        for provider, entry in self._model_entries.items():
            val = entry.get().strip()
            if val:
                models[provider] = val
        self._config["model"] = models

        # Persist
        save_config(self._config)

        # Apply changes to the running app
        self._parent.apply_config(self._config)

        self._status_label.configure(text="✔ Saved!", text_color="green")
        self.after(1000, self.destroy)


# ── Main Application Window ────────────────────────────────────────────────────

class ScreenHelpApp(ctk.CTk):
    """
    The main ScreenHelp application window.
    Manages layout, hotkey binding, screenshot display, and AI streaming.
    """

    MIN_WIDTH = 700
    MIN_HEIGHT = 560
    THUMB_SIZE = (300, 200)

    def __init__(self):
        super().__init__()

        # ── Load config & apply theme ──────────────────────────────────────────
        self.config = load_config()
        ctk.set_appearance_mode(self.config.get("theme", "dark"))
        ctk.set_default_color_theme("blue")

        # ── Opacity & Always-on-top ────────────────────────────────────────────
        self._opacity = float(self.config.get("opacity", 0.80))
        self.attributes("-alpha", self._opacity)
        self._always_on_top = bool(self.config.get("always_on_top", True))
        self.attributes("-topmost", self._always_on_top)

        # ── State ──────────────────────────────────────────────────────────────
        self._screenshot: Optional[Image.Image] = None  # full-res PIL image
        self._screenshot_bytes: Optional[bytes] = None  # PNG bytes for AI
        self._thumb_ctk: Optional[ctk.CTkImage] = None  # CTkImage for display
        self._ai_thread: Optional[threading.Thread] = None
        self._settings_win: Optional[SettingsWindow] = None

        # ── Window setup ───────────────────────────────────────────────────────
        self.title("ScreenHelp")
        self.geometry("820x620")
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Build UI ───────────────────────────────────────────────────────────
        self._build_ui()

        # ── Apply screen-share invisibility ────────────────────────────────────
        # Schedule after mainloop starts so the HWND is available
        self.after(500, self._apply_invisibility)

        # ── Start hotkey listener ──────────────────────────────────────────────
        self._hotkey_manager = HotkeyManager(
            callback=self._hotkey_triggered,
            hotkey=self.config.get("hotkey", "ctrl+shift+s"),
        )
        self._hotkey_manager.start()

        self._set_status("Ready — press Ctrl+Shift+S to auto-capture & solve")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all UI widgets."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_main_area()
        self._build_footer()

    def _build_header(self) -> None:
        """Top bar: logo, ghost transparency slider, auto-solve, pin, theme, settings."""
        header = ctk.CTkFrame(self, corner_radius=0, height=52)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        # Logo / title
        logo_label = ctk.CTkLabel(
            header,
            text="🤖  ScreenHelp",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        logo_label.grid(row=0, column=0, padx=14, pady=8, sticky="w")

        # Center HUD controls: Ghost Opacity slider, Auto-Solve switch, Pin button
        hud_frame = ctk.CTkFrame(header, fg_color="transparent")
        hud_frame.grid(row=0, column=1, padx=4, pady=6)

        ctk.CTkLabel(hud_frame, text="Ghost:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 2))
        self._opacity_slider = ctk.CTkSlider(
            hud_frame,
            from_=0.35,
            to=1.0,
            number_of_steps=13,
            command=self._on_opacity_change,
            width=90,
        )
        self._opacity_slider.set(self._opacity)
        self._opacity_slider.pack(side="left", padx=(0, 4))

        self._opacity_label = ctk.CTkLabel(
            hud_frame,
            text=f"{int(self._opacity * 100)}%",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=36,
        )
        self._opacity_label.pack(side="left", padx=(0, 8))

        # Auto-Solve switch
        self._auto_var = ctk.BooleanVar(value=bool(self.config.get("auto_analyze", True)))
        self._auto_switch = ctk.CTkSwitch(
            hud_frame,
            text="⚡ Auto-Solve",
            variable=self._auto_var,
            command=self._on_auto_toggle,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._auto_switch.pack(side="left", padx=(0, 8))

        # Pin / Float toggle
        self._pin_btn = ctk.CTkButton(
            hud_frame,
            text="📌 Pinned" if self._always_on_top else "📍 Float",
            width=76,
            height=28,
            fg_color="#1f6aa5" if self._always_on_top else "gray40",
            command=self._toggle_pin,
        )
        self._pin_btn.pack(side="left")

        # Right-side buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=12, pady=6, sticky="e")

        # Theme toggle
        self._theme_btn = ctk.CTkButton(
            btn_frame,
            text=self._theme_icon(),
            width=100,
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side="left", padx=(0, 8))

        # Settings
        ctk.CTkButton(
            btn_frame,
            text="⚙  Settings",
            width=95,
            command=self._open_settings,
        ).pack(side="left")

    def _build_main_area(self) -> None:
        """Middle area: screenshot panel (left) + response panel (right/bottom)."""
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # ── Left column: screenshot + question ────────────────────────────────
        left = ctk.CTkFrame(main, width=320)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_propagate(False)

        # Screenshot thumbnail
        self._thumb_label = ctk.CTkLabel(
            left,
            text="No screenshot yet.\nPress Ctrl+Shift+S\nor click Capture below.",
            width=300,
            height=200,
            fg_color=("gray85", "gray25"),
            corner_radius=8,
        )
        self._thumb_label.pack(padx=10, pady=(12, 6))

        # Manual capture button
        ctk.CTkButton(
            left,
            text="📷  Capture Screen",
            command=self._manual_capture,
            width=200,
        ).pack(pady=(0, 10))

        # Divider label
        ctk.CTkLabel(left, text="Question", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=12
        )

        # Question entry
        self._question_entry = ctk.CTkEntry(
            left,
            placeholder_text="Ask something about the screenshot…",
            width=290,
            height=38,
        )
        self._question_entry.pack(padx=10, pady=(4, 6))
        self._question_entry.bind("<Return>", lambda _e: self._ask())

        # Ask button
        self._ask_btn = ctk.CTkButton(
            left, text="Ask AI  →", command=self._ask, width=200
        )
        self._ask_btn.pack(pady=(0, 10))

        # ── Right column: AI response ──────────────────────────────────────────
        right = ctk.CTkFrame(main)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right, text="AI Response", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self._response_box = ctk.CTkTextbox(
            right,
            wrap="word",
            font=ctk.CTkFont(size=13),
            activate_scrollbars=True,
        )
        self._response_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self._response_box.configure(state="disabled")

        # Copy button
        ctk.CTkButton(
            right,
            text="📋  Copy Response",
            command=self._copy_response,
            width=160,
        ).pack(anchor="e", padx=12, pady=(0, 10))

    def _build_footer(self) -> None:
        """Bottom status bar with provider selector and hotkey hint."""
        footer = ctk.CTkFrame(self, corner_radius=0, height=40)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(2, weight=1)

        # Provider selector label
        ctk.CTkLabel(footer, text="Provider:").grid(
            row=0, column=0, padx=(14, 4), pady=8
        )

        # Provider dropdown
        self._provider_var = ctk.StringVar(value=self.config.get("provider", "gemini"))
        self._provider_menu = ctk.CTkOptionMenu(
            footer,
            values=["gemini", "llama", "openai", "anthropic"],
            variable=self._provider_var,
            command=self._on_provider_change,
            width=130,
        )
        self._provider_menu.grid(row=0, column=1, padx=(0, 20), pady=6)

        # Status label (centre, expands)
        self._status_label = ctk.CTkLabel(
            footer,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray70"),
        )
        self._status_label.grid(row=0, column=2, sticky="ew", padx=8)

        # Hotkey hint (right-aligned)
        self._hotkey_hint = ctk.CTkLabel(
            footer,
            text=f"Shortcut: {self.config.get('hotkey', 'ctrl+shift+s').upper()}",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60"),
        )
        self._hotkey_hint.grid(row=0, column=3, padx=(0, 14), pady=8)

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _theme_icon(self) -> str:
        mode = ctk.get_appearance_mode().lower()
        return "Light Mode ☀️" if mode == "dark" else "Dark Mode 🌙"

    def _toggle_theme(self) -> None:
        current = ctk.get_appearance_mode().lower()
        new_theme = "light" if current == "dark" else "dark"
        ctk.set_appearance_mode(new_theme)
        self._theme_btn.configure(text=self._theme_icon())
        self.config["theme"] = new_theme
        save_config(self.config)

    # ── HUD Controls (Opacity, Pin, Auto-Solve) ────────────────────────────────

    def _on_opacity_change(self, val: float) -> None:
        self._opacity = round(float(val), 2)
        self.attributes("-alpha", self._opacity)
        self._opacity_label.configure(text=f"{int(self._opacity * 100)}%")
        self.config["opacity"] = self._opacity
        save_config(self.config)

    def _toggle_pin(self) -> None:
        self._always_on_top = not self._always_on_top
        self.attributes("-topmost", self._always_on_top)
        self._pin_btn.configure(
            text="📌 Pinned" if self._always_on_top else "📍 Float",
            fg_color="#1f6aa5" if self._always_on_top else "gray40",
        )
        self.config["always_on_top"] = self._always_on_top
        save_config(self.config)

    def _on_auto_toggle(self) -> None:
        self.config["auto_analyze"] = self._auto_var.get()
        save_config(self.config)

    # ── Screenshot & Capture ───────────────────────────────────────────────────

    def _hotkey_triggered(self) -> None:
        """
        Called from the hotkey listener thread.
        Hides the window FIRST, captures, then re-shows.
        Uses after() to safely interact with Tkinter from a non-main thread.
        """
        self.after(0, self._do_capture_flow)

    def _manual_capture(self) -> None:
        """Triggered by the 'Capture Screen' button in the UI."""
        self._do_capture_flow()

    def _do_capture_flow(self) -> None:
        """
        Full capture flow:
          1. Withdraw the window so it doesn't appear in the screenshot
          2. Small delay to let compositing flush
          3. Capture
          4. Restore window
          5. Display thumbnail
          6. Automatically trigger AI question solving (if auto-solve is enabled)
        """
        self._set_status("Capturing screen…")
        logger.info("[app] Screen capture initiated...")
        self.withdraw()          # hide window
        self.update_idletasks()

        def _capture_and_restore():
            time.sleep(0.20)    # wait for window to fully hide from compositor
            img = capture_fullscreen()
            # Restore window on main thread
            self.after(0, lambda: self._on_capture_done(img))

        t = threading.Thread(target=_capture_and_restore, daemon=True)
        t.start()

    def _on_capture_done(self, img: Optional[Image.Image]) -> None:
        """Called on the main thread after a screenshot is taken."""
        self.deiconify()   # show window again
        self.lift()
        self.focus_force()

        # Re-apply transparency and topmost
        self.attributes("-alpha", self._opacity)
        if self._always_on_top:
            self.attributes("-topmost", True)

        if img is None:
            logger.error("[app] Screen capture failed — got None.")
            self._set_status("Capture failed. Check console for details.")
            return

        logger.info(f"[app] Screen captured ({img.width}x{img.height}).")
        self._screenshot = img
        self._screenshot_bytes = image_to_bytes(img, fmt="PNG")

        # Display thumbnail
        thumb = image_to_thumbnail(img, max_size=self.THUMB_SIZE)
        self._thumb_ctk = ctk.CTkImage(
            light_image=thumb,
            dark_image=thumb,
            size=thumb.size,
        )
        self._thumb_label.configure(image=self._thumb_ctk, text="")

        # ⚡ Zero-Click Automatic Solving!
        if self._auto_var.get():
            self._set_status("⚡ Auto-detecting question & solving...")
            logger.info("[app] Auto-solve active: triggering analysis immediately.")
            self.after(60, self._ask)
        else:
            self._set_status("Screenshot captured — enter a question or click Ask AI")

    # ── AI Query ───────────────────────────────────────────────────────────────

    def _ask(self) -> None:
        """Validate inputs and start an AI streaming thread."""
        if self._screenshot_bytes is None:
            self._set_response("⚠️  No screenshot yet. Press Ctrl+Shift+S or click 'Capture Screen' first.")
            return

        provider = self._provider_var.get()
        api_key = get_api_key(self.config, provider)

        if not api_key:
            self._set_response(
                f"⚠️  No API key configured for '{provider}'.\n\n"
                "Click ⚙ Settings at the top right to enter your API key."
            )
            return

        # Prevent multiple simultaneous requests
        if self._ai_thread and self._ai_thread.is_alive():
            self._set_status("Already processing… please wait.")
            return

        question = self._question_entry.get().strip()
        model = get_model(self.config, provider)

        self._clear_response()
        prompt_preview = question if question else "⚡ Auto-solving question on screen..."
        self._set_status(f"Analyzing ({provider}): {prompt_preview[:40]}")
        logger.info(f"[app] Querying {provider} ({model})...")
        self._ask_btn.configure(state="disabled")

        self._ai_thread = threading.Thread(
            target=self._stream_ai,
            args=(provider, api_key, model, self._screenshot_bytes, question),
            daemon=True,
        )
        self._ai_thread.start()

    def _stream_ai(
        self,
        provider: str,
        api_key: str,
        model: str,
        image_bytes: bytes,
        question: str,
    ) -> None:
        """
        Runs in a background thread.
        Streams tokens from the AI and pushes each chunk to the textbox
        via self.after() (thread-safe Tkinter update).
        """
        try:
            logger.info(f"[app] Connecting to {provider} vision API...")
            client = AIClient(provider=provider, api_key=api_key, model=model)
            chunk_count = 0
            for chunk in client.stream_response(image_bytes, question):
                chunk_count += 1
                # Schedule UI update on the main thread
                self.after(0, lambda c=chunk: self._append_response(c))
            logger.info(f"[app] Response streaming completed ({chunk_count} chunks).")
            self.after(0, self._on_stream_done)
        except RuntimeError as exc:
            logger.error(f"[app] AI RuntimeError: {exc}")
            self.after(0, lambda: self._on_stream_error(str(exc)))
        except Exception as exc:
            logger.error(f"[app] AI Unexpected exception: {exc}")
            self.after(0, lambda: self._on_stream_error(f"Unexpected error: {exc}"))

    def _on_stream_done(self) -> None:
        self._set_status("Done ✔")
        self._ask_btn.configure(state="normal")

    def _on_stream_error(self, message: str) -> None:
        provider = self._provider_var.get()
        if "401" in message or "invalid_api_key" in message.lower() or "Invalid API Key" in message:
            help_msg = (
                f"❌ API Authentication Error (401 - Unauthorized)\n\n"
                f"The API key for '{provider}' was not accepted by the service.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 How to get an instant working key:\n\n"
                f"Option 1: Groq Cloud (Free Llama 4 Vision, Fast)\n"
                f"  1. Go to: https://console.groq.com/keys\n"
                f"  2. Click 'Create API Key' (starts with 'gsk_...')\n"
                f"  3. In ScreenHelp, click ⚙ Settings → paste into 'Llama API Key'\n\n"
                f"Option 2: Google Gemini (Free & High Vision Accuracy)\n"
                f"  1. Go to: https://aistudio.google.com/\n"
                f"  2. Click 'Get API key'\n"
                f"  3. In ScreenHelp, click ⚙ Settings → paste into 'Google Gemini API Key'\n"
                f"  4. Select 'gemini' in Provider dropdown\n\n"
                f"Option 3: Local Ollama (100% Free & Offline)\n"
                f"  1. Run in terminal: ollama pull llama3.2-vision\n"
                f"  2. In ScreenHelp Settings, enter 'ollama' as the key\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self._append_response(help_msg)
        else:
            self._append_response(f"❌ Error: {message}")

        self._set_status("Error — see response box for details.")
        self._ask_btn.configure(state="normal")

    # ── Response Box Helpers ───────────────────────────────────────────────────

    def _set_response(self, text: str) -> None:
        self._response_box.configure(state="normal")
        self._response_box.delete("1.0", "end")
        self._response_box.insert("end", text)
        self._response_box.configure(state="disabled")

    def _clear_response(self) -> None:
        self._response_box.configure(state="normal")
        self._response_box.delete("1.0", "end")
        self._response_box.configure(state="disabled")

    def _append_response(self, text: str) -> None:
        self._response_box.configure(state="normal")
        self._response_box.insert("end", text)
        self._response_box.see("end")  # auto-scroll
        self._response_box.configure(state="disabled")

    def _copy_response(self) -> None:
        content = self._response_box.get("1.0", "end").strip()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            self._set_status("Response copied to clipboard ✔")

    # ── Status Bar ────────────────────────────────────────────────────────────

    def _set_status(self, message: str) -> None:
        self._status_label.configure(text=message)

    # ── Settings ──────────────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        if self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.focus()
            return
        self._settings_win = SettingsWindow(self)

    def apply_config(self, new_config: dict) -> None:
        """Called by SettingsWindow after Save to apply live changes."""
        self.config = new_config

        # Apply theme
        ctk.set_appearance_mode(new_config.get("theme", "dark"))
        self._theme_btn.configure(text=self._theme_icon())

        # Apply opacity
        new_opacity = float(new_config.get("opacity", 0.80))
        self._opacity = new_opacity
        self.attributes("-alpha", new_opacity)
        if hasattr(self, "_opacity_slider"):
            self._opacity_slider.set(new_opacity)
        if hasattr(self, "_opacity_label"):
            self._opacity_label.configure(text=f"{int(new_opacity * 100)}%")

        # Apply always-on-top
        new_pin = bool(new_config.get("always_on_top", True))
        self._always_on_top = new_pin
        self.attributes("-topmost", new_pin)
        if hasattr(self, "_pin_btn"):
            self._pin_btn.configure(
                text="📌 Pinned" if new_pin else "📍 Float",
                fg_color="#1f6aa5" if new_pin else "gray40",
            )

        # Apply auto_analyze
        new_auto = bool(new_config.get("auto_analyze", True))
        if hasattr(self, "_auto_var"):
            self._auto_var.set(new_auto)

        # Apply hotkey
        new_hotkey = new_config.get("hotkey", "ctrl+shift+s")
        if new_hotkey != self._hotkey_manager.current_hotkey:
            self._hotkey_manager.update(new_hotkey)
            self._hotkey_hint.configure(text=f"Shortcut: {new_hotkey.upper()}")

        # Sync provider dropdown
        self._provider_var.set(new_config.get("provider", "gemini"))

        self._set_status("Settings saved and applied.")

    # ── Provider Change ───────────────────────────────────────────────────────

    def _on_provider_change(self, value: str) -> None:
        self.config["provider"] = value
        save_config(self.config)

    # ── Window Invisibility ───────────────────────────────────────────────────

    def _apply_invisibility(self) -> None:
        """
        Apply WDA_EXCLUDEFROMCAPTURE so ScreenHelp is invisible to Zoom/Teams.
        Must run after mainloop starts (so the HWND exists).
        """
        try:
            hwnd = self.winfo_id()
            ok = set_window_invisible(hwnd)
            if ok:
                logger.info("[app] Screen-share invisibility applied successfully.")
            else:
                logger.info("[app] Screen-share invisibility not applied (see invisibility.py logs).")
        except Exception as exc:
            logger.warning(f"[app] Could not apply invisibility: {exc}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Gracefully clean up before exit."""
        self._hotkey_manager.stop()
        save_config(self.config)
        self.destroy()
