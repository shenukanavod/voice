#!/usr/bin/env python3
"""
Voice Authentication Desktop Application

A modern desktop GUI for the voice authentication system using tkinter.
Features:
- Voice enrollment
- Real-time verification
- User management
"""

import io
import json
import os
import sys
import threading
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional

import librosa
import numpy as np
import pyaudio

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Import voice authentication components
from app.audio.preprocessing import AudioPreprocessor
from app.config import settings
from app.models.anti_spoofing import AntiSpoofingSystem
from app.security.mongodb_profile_manager import (
    MongoDBVoiceProfileManager,  # MongoDB storage
)
from voice_monitoring import VoiceMonitor
from admin_panel import AdminPanel
from db import VoiceDatabase
from db_logger import setup_database_logging


class VoiceAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Biometric Auth System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0a1628")

        # Initialize database and logging
        self.db_instance = None
        self.db_logger = None
        try:
            self.db_instance = VoiceDatabase(settings.MONGODB_URL)
            self.db_logger = setup_database_logging(self.db_instance, "desktop_app", console_output=True)
            print("✅ Database logging initialized")
        except Exception as e:
            print(f"⚠️  Database logging initialization failed: {e}")
            print("⚠️  Continuing with console logging only")

        # Initialize voice authentication components
        try:
            self.audio_processor = AudioPreprocessor()
            # Choose storage based on configuration toggle
            if getattr(settings, "USE_MONGODB", False):
                try:
                    self.profile_manager = MongoDBVoiceProfileManager(
                        model_path="models/speaker_embedding_model.pth"
                    )  # MongoDB storage with trained CNN+LSTM model
                    print("✅ Using MongoDB for storage")
                except Exception as db_error:
                    print(f"⚠️  MongoDB connection failed: {str(db_error)[:200]}")
                    print("⚠️  Falling back to local file-based storage...")
                    from app.security.cnn_profile_manager import CNNVoiceProfileManager

                    self.profile_manager = CNNVoiceProfileManager(
                        model_path="models/speaker_embedding_model.pth"
                    )
                    print("✅ Using local file storage")
            else:
                from app.security.cnn_profile_manager import CNNVoiceProfileManager

                self.profile_manager = CNNVoiceProfileManager(
                    model_path="models/speaker_embedding_model.pth"
                )
                print("✅ Using local file storage (MongoDB disabled)")
            
            # Initialize anti-spoofing system
            self.anti_spoofing = AntiSpoofingSystem()
            
            # Load trained anti-spoofing model if available
            antispoofing_model_path = "models/anti_spoofing_model.pth"
            if os.path.exists(antispoofing_model_path):
                self.anti_spoofing.load_cnn_model(antispoofing_model_path)
                print("✅ Anti-spoofing model loaded successfully")
            else:
                print("⚠️  Anti-spoofing model not found - using default detection")
        except Exception as e:
            print(f"❌ Error initializing components: {str(e)}")
            messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize voice authentication:\n\n{str(e)}\n\nPlease check configuration.",
            )
            raise

        # Audio recording variables
        self.is_recording = False
        self.audio_data = []
        self.audio_stream = None
        self.pyaudio_instance = None
        self.audio_device_index = None  # Will be auto-detected
        self.current_mode = None  # "enrollment" or "verification"

        # Multi-sample registration
        self.registration_samples = []
        self.current_sample_number = 0
        self.total_samples_needed = 3

        # Current user
        self.current_user = None
        
        # Voice monitoring
        self.current_monitor = None
        self.current_user_embedding = None

        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.setup_audio()

    def setup_styles(self):
        """Setup modern styling for the application."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors with dark navy theme (matching mouse dynamics UI)
        style.configure(
            "Title.TLabel",
            font=("Arial", 20, "bold"),
            background="#0a1628",
            foreground="#ffffff",
        )
        style.configure(
            "Header.TLabel",
            font=("Arial", 12, "bold"),
            background="#1a2d4d",
            foreground="#ffffff",
        )
        style.configure(
            "Status.TLabel",
            font=("Arial", 10),
            background="#1a2d4d",
            foreground="#00d4ff",
        )

        # Main action button styles - cyan blue theme
        style.configure(
            "Register.TButton",
            font=("Arial", 14, "bold"),
            background="#00d4ff",
            foreground="#0a1628",
        )
        style.map(
            "Register.TButton",
            background=[("active", "#00b8e6"), ("pressed", "#0099cc")],
        )

        style.configure(
            "Login.TButton",
            font=("Arial", 14, "bold"),
            background="#00d4ff",
            foreground="#0a1628",
        )
        style.map(
            "Login.TButton", background=[("active", "#00b8e6"), ("pressed", "#0099cc")]
        )

        # Button styles
        style.configure(
            "Record.TButton",
            font=("Arial", 10, "bold"),
            background="#00d4ff",
            foreground="#0a1628",
        )
        style.map(
            "Record.TButton", background=[("active", "#00b8e6"), ("pressed", "#0099cc")]
        )

        style.configure(
            "Action.TButton",
            font=("Arial", 10),
            background="#10b981",
            foreground="#ffffff",
        )
        style.map(
            "Action.TButton", background=[("active", "#059669"), ("pressed", "#047857")]
        )

        # Frame styles
        style.configure(
            "TLabelframe", background="#1a2d4d", foreground="#00d4ff", borderwidth=2
        )
        style.configure(
            "TLabelframe.Label",
            font=("Arial", 11, "bold"),
            background="#1a2d4d",
            foreground="#00d4ff",
        )
        style.configure("TFrame", background="#1a2d4d")

    def create_widgets(self):
        """Create and layout all GUI widgets with multiple pages."""
        # Container for all pages
        self.pages_container = tk.Frame(self.root, bg="#0a1628")
        self.pages_container.pack(expand=True, fill="both")

        # Create all pages
        self.create_landing_page()
        self.create_registration_page()
        self.create_login_page()

        # Show landing page by default
        self.show_page("landing")

    def create_landing_page(self):
        """Create the landing page."""
        self.landing_page = tk.Frame(self.pages_container, bg="#0a1628")

        # Center container with border (matching mouse dynamics style)
        center_container = tk.Frame(
            self.landing_page,
            bg="#1a2d4d",
            relief=tk.SOLID,
            bd=1,
            highlightbackground="#2a4d6d",
            highlightthickness=1,
        )
        center_container.place(relx=0.5, rely=0.5, anchor="center")

        # Inner padding frame
        inner_frame = tk.Frame(center_container, bg="#1a2d4d")
        inner_frame.pack(padx=60, pady=40)

        # Title
        title_label = tk.Label(
            inner_frame,
            text="Voice Biometric Auth",
            font=("Segoe UI", 24, "bold"),
            bg="#1a2d4d",
            fg="#ffffff",
        )
        title_label.pack(pady=(0, 25))

        # Subtitle
        subtitle_label = tk.Label(
            inner_frame,
            text="Choose an action:",
            font=("Segoe UI", 11),
            bg="#1a2d4d",
            fg="#c0c0c0",
        )
        subtitle_label.pack(pady=(0, 25))

        # Buttons Frame
        buttons_frame = tk.Frame(inner_frame, bg="#1a2d4d")
        buttons_frame.pack(pady=0)

        # Register User Button (Admin)
        register_choice_btn = tk.Button(
            buttons_frame,
            text="Enrollment",
            font=("Segoe UI", 11, "bold"),
            bg="#00bfff",
            fg="#0a1628",
            activebackground="#00a8e6",
            activeforeground="#0a1628",
            relief=tk.FLAT,
            bd=0,
            padx=35,
            pady=10,
            cursor="hand2",
            command=lambda: self.show_page("registration"),
        )
        register_choice_btn.pack(side="left", padx=8)

        # Login Button (Users)
        login_choice_btn = tk.Button(
            buttons_frame,
            text="Verification",
            font=("Segoe UI", 11, "bold"),
            bg="#00bfff",
            fg="#0a1628",
            activebackground="#00a8e6",
            activeforeground="#0a1628",
            relief=tk.FLAT,
            bd=0,
            padx=35,
            pady=10,
            cursor="hand2",
            command=lambda: self.show_page("login"),
        )
        login_choice_btn.pack(side="left", padx=8)
        
        # Admin Panel Button
        admin_btn = tk.Button(
            buttons_frame,
            text="👤 Admin Panel",
            font=("Segoe UI", 11, "bold"),
            bg="#ff8800",
            fg="white",
            activebackground="#ffaa00",
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            padx=35,
            pady=10,
            cursor="hand2",
            command=self.open_admin_panel,
        )
        admin_btn.pack(side="left", padx=8)

    def create_registration_page(self):
        """Create the registration page."""
        self.registration_page = tk.Frame(self.pages_container, bg="#0a1628")

        # Title
        title_label = tk.Label(
            self.registration_page,
            text="Enrollment",
            font=("Segoe UI", 28, "bold"),
            bg="#0a1628",
            fg="#ffffff",
        )
        title_label.pack(pady=(50, 15))

        # Subtitle
        subtitle_label = tk.Label(
            self.registration_page,
            text="Enter the user ID (e.g., student number) and record a continuous session of voice",
            font=("Segoe UI", 10),
            bg="#0a1628",
            fg="#8899aa",
        )
        subtitle_label.pack(pady=(0, 10))

        subtitle_label2 = tk.Label(
            self.registration_page,
            text="samples to create the behavioral template.",
            font=("Segoe UI", 10),
            bg="#0a1628",
            fg="#8899aa",
        )
        subtitle_label2.pack(pady=(0, 35))

        # User Number Input Frame
        input_frame = tk.Frame(
            self.registration_page, bg="#1a2d4d", relief=tk.SOLID, bd=1
        )
        input_frame.pack(pady=15, padx=80, fill="x")

        user_label = tk.Label(
            input_frame,
            text="User ID",
            font=("Segoe UI", 11, "bold"),
            bg="#1a2d4d",
            fg="#ffffff",
        )
        user_label.pack(pady=(18, 8))

        self.user_id_var = tk.StringVar()
        self.reg_user_entry = tk.Entry(
            input_frame,
            textvariable=self.user_id_var,
            font=("Segoe UI", 14),
            width=25,
            bg="#0d1c2f",
            fg="#ffffff",
            insertbackground="#00d4ff",
            relief=tk.FLAT,
            bd=2,
            justify="center",
        )
        self.reg_user_entry.pack(pady=(0, 18), padx=25)

        # Start Registration Button
        start_reg_btn = tk.Button(
            self.registration_page,
            text="Start Recording",
            font=("Segoe UI", 12, "bold"),
            bg="#00d4ff",
            fg="#0a1628",
            activebackground="#00b8e6",
            activeforeground="#0a1628",
            relief=tk.FLAT,
            bd=0,
            padx=45,
            pady=12,
            cursor="hand2",
            command=self.register_user,
        )
        start_reg_btn.pack(pady=25)

        # Status Frame
        self.reg_status_frame = tk.Frame(
            self.registration_page, bg="#1a2d4d", relief=tk.SOLID, bd=1
        )
        self.reg_status_frame.pack(pady=18, padx=80, fill="x")

        self.reg_status_var = tk.StringVar(value="Ready to record")
        reg_status_label = tk.Label(
            self.reg_status_frame,
            textvariable=self.reg_status_var,
            font=("Segoe UI", 10),
            bg="#1a2d4d",
            fg="#00d4ff",
            pady=12,
        )
        reg_status_label.pack()

        self.reg_recording_var = tk.StringVar(value="")
        reg_recording_label = tk.Label(
            self.reg_status_frame,
            textvariable=self.reg_recording_var,
            font=("Segoe UI", 10, "bold"),
            bg="#1a2d4d",
            fg="#ff6b6b",
        )
        reg_recording_label.pack()

        # Info label
        info_label = tk.Label(
            self.registration_page,
            text="Last recorded events: 0",
            font=("Segoe UI", 9),
            bg="#0a1628",
            fg="#8899aa",
        )
        info_label.pack(pady=10)

        # Back Button
        back_btn = tk.Button(
            self.registration_page,
            text="◀  Back",
            font=("Segoe UI", 10),
            bg="#1a2d4d",
            fg="#8899aa",
            activebackground="#2a3d5d",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=25,
            pady=10,
            cursor="hand2",
            command=lambda: self.show_page("landing"),
        )
        back_btn.pack(pady=15)

    def create_login_page(self):
        """Create the login page."""
        self.login_page = tk.Frame(self.pages_container, bg="#0a1628")

        # Title
        title_label = tk.Label(
            self.login_page,
            text="Verification",
            font=("Segoe UI", 28, "bold"),
            bg="#0a1628",
            fg="#ffffff",
        )
        title_label.pack(pady=(50, 15))

        # Subtitle
        subtitle_label = tk.Label(
            self.login_page,
            text="Enter your user ID and speak naturally to verify your identity",
            font=("Segoe UI", 10),
            bg="#0a1628",
            fg="#8899aa",
        )
        subtitle_label.pack(pady=(0, 35))

        # User Number Input Frame
        input_frame = tk.Frame(self.login_page, bg="#1a2d4d", relief=tk.SOLID, bd=1)
        input_frame.pack(pady=15, padx=80, fill="x")

        user_label = tk.Label(
            input_frame,
            text="User ID",
            font=("Segoe UI", 11, "bold"),
            bg="#1a2d4d",
            fg="#ffffff",
        )
        user_label.pack(pady=(18, 8))

        self.login_user_id_var = tk.StringVar()
        self.login_user_entry = tk.Entry(
            input_frame,
            textvariable=self.login_user_id_var,
            font=("Segoe UI", 14),
            width=25,
            bg="#0d1c2f",
            fg="#ffffff",
            insertbackground="#00d4ff",
            relief=tk.FLAT,
            bd=2,
            justify="center",
        )
        self.login_user_entry.pack(pady=(0, 18), padx=25)

        # Start Login Button
        start_login_btn = tk.Button(
            self.login_page,
            text="Start Verification",
            font=("Segoe UI", 12, "bold"),
            bg="#00d4ff",
            fg="#0a1628",
            activebackground="#00b8e6",
            activeforeground="#0a1628",
            relief=tk.FLAT,
            bd=0,
            padx=45,
            pady=12,
            cursor="hand2",
            command=self.login_user,
        )
        start_login_btn.pack(pady=25)

        # Status Frame
        self.login_status_frame = tk.Frame(
            self.login_page, bg="#1a2d4d", relief=tk.SOLID, bd=1
        )
        self.login_status_frame.pack(pady=18, padx=80, fill="x")

        self.login_status_var = tk.StringVar(value="Ready to verify")
        login_status_label = tk.Label(
            self.login_status_frame,
            textvariable=self.login_status_var,
            font=("Segoe UI", 10),
            bg="#1a2d4d",
            fg="#00d4ff",
            pady=12,
        )
        login_status_label.pack()

        self.login_recording_var = tk.StringVar(value="")
        login_recording_label = tk.Label(
            self.login_status_frame,
            textvariable=self.login_recording_var,
            font=("Segoe UI", 10, "bold"),
            bg="#1a2d4d",
            fg="#ff6b6b",
        )
        login_recording_label.pack()

        # Info label
        info_label = tk.Label(
            self.login_page,
            text="Last recorded events: 0",
            font=("Segoe UI", 9),
            bg="#0a1628",
            fg="#8899aa",
        )
        info_label.pack(pady=10)

        # Back Button
        back_btn = tk.Button(
            self.login_page,
            text="◀  Back",
            font=("Segoe UI", 10),
            bg="#1a2d4d",
            fg="#8899aa",
            activebackground="#2a3d5d",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=25,
            pady=10,
            cursor="hand2",
            command=lambda: self.show_page("landing"),
        )
        back_btn.pack(pady=15)

    def show_page(self, page_name):
        """Show the specified page and hide others."""
        # Hide all pages
        self.landing_page.pack_forget()
        self.registration_page.pack_forget()
        self.login_page.pack_forget()

        # Show requested page
        if page_name == "landing":
            self.landing_page.pack(expand=True, fill="both")
        elif page_name == "registration":
            self.registration_page.pack(expand=True, fill="both")
            self.reg_user_entry.focus()
        elif page_name == "login":
            self.login_page.pack(expand=True, fill="both")
            self.login_user_entry.focus()

    def log(self, message: str, user_id: str = None, level: str = "INFO"):
        """Add a message to the activity log and database."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Save to database if available
        if self.db_instance:
            try:
                self.db_instance.save_log(
                    level=level,
                    message=message,
                    module="desktop_app",
                    user_id=user_id
                )
            except Exception as e:
                # Don't fail the app if database logging fails
                print(f"⚠️  Database logging error: {e}")

    def setup_audio(self):
        """Initialize audio recording components and detect available microphones."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()

            # Detect and list all available audio input devices
            device_count = self.pyaudio_instance.get_device_count()
            available_devices = []

            self.log(f"Scanning {device_count} audio devices...")

            for i in range(device_count):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    # Check if device has input channels (is a microphone)
                    if device_info["maxInputChannels"] > 0:
                        available_devices.append(
                            {
                                "index": i,
                                "name": device_info["name"],
                                "channels": device_info["maxInputChannels"],
                                "sample_rate": int(device_info["defaultSampleRate"]),
                                "is_default": i
                                == self.pyaudio_instance.get_default_input_device_info()[
                                    "index"
                                ],
                            }
                        )
                        self.log(
                            f"Found microphone: {device_info['name']} (Index: {i})"
                        )
                except Exception as e:
                    continue

            if not available_devices:
                raise Exception("No microphone devices found")

            # Try to use default input device first
            try:
                default_device = self.pyaudio_instance.get_default_input_device_info()
                self.audio_device_index = default_device["index"]
                self.log(f"✓ Using default microphone: {default_device['name']}")
            except:
                # If no default, use the first available microphone
                self.audio_device_index = available_devices[0]["index"]
                self.log(f"✓ Using microphone: {available_devices[0]['name']}")

            # Test the selected microphone
            self._test_microphone()

            self.log("Audio system initialized successfully")

        except Exception as e:
            self.log(f"Error initializing audio: {str(e)}")
            messagebox.showerror(
                "Audio Error", f"Failed to initialize audio system: {str(e)}"
            )

    def _test_microphone(self):
        """Test if the selected microphone is working."""
        try:
            # Open and immediately close a test stream
            test_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=settings.SAMPLE_RATE,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=1024,
            )
            test_stream.close()
            self.log(f"✓ Microphone test successful")
            return True
        except Exception as e:
            self.log(f"⚠ Microphone test failed: {str(e)}")
            # Try to find another working microphone
            return self._find_working_microphone()

    def _find_working_microphone(self):
        """Find a working microphone if default fails."""
        device_count = self.pyaudio_instance.get_device_count()

        for i in range(device_count):
            try:
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    # Try to open this device
                    test_stream = self.pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=settings.SAMPLE_RATE,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024,
                    )
                    test_stream.close()

                    # If successful, use this device
                    self.audio_device_index = i
                    self.log(f"✓ Switched to working microphone: {device_info['name']}")
                    return True
            except:
                continue

        self.log("✗ No working microphone found")
        return False

    def update_status(self, status: str):
        """Update the system status display based on current mode."""
        if self.current_mode == "enrollment":
            self.reg_status_var.set(status)
        elif self.current_mode == "verification":
            self.login_status_var.set(status)
        self.root.update_idletasks()

    def update_recording_status(self, status: str):
        """Update the recording status display based on current mode."""
        if self.current_mode == "enrollment":
            self.reg_recording_var.set(status)
        elif self.current_mode == "verification":
            self.login_recording_var.set(status)
        self.root.update_idletasks()

    def register_user(self):
        """Register new user with 3 voice samples for better accuracy (ADMIN ONLY)."""
        user_id = self.user_id_var.get().strip()

        if not user_id:
            messagebox.showerror("Error", "Please enter a User Number first!")
            self.user_id_entry.focus()
            return

        # Check if user already exists in MongoDB
        if self.profile_manager.user_exists(user_id):
            response = messagebox.askyesno(
                "User Exists",
                f"User {user_id} is already registered!\n\nDo you want to re-register with new voice samples?",
            )
            if not response:
                return

        # Reset for new registration
        self.registration_samples = []
        self.current_sample_number = 1

        self.log(f"👤 ADMIN: Starting registration for User: {user_id}", user_id=user_id)
        self.log(
            f"ℹ️ Admin will record {self.total_samples_needed} voice samples from the user", user_id=user_id
        )
        self.update_status(
            f"🔴 Recording Sample 1/{self.total_samples_needed}... User should speak now!"
        )

        # Start recording first sample
        self.current_mode = "enrollment"
        self.start_recording("enrollment")

    def login_user(self):
        """Login user with voice verification."""
        user_id = self.login_user_id_var.get().strip()

        if not user_id:
            messagebox.showerror("Error", "Please enter your User Number first!")
            self.login_user_entry.focus()
            return

        self.log(f"🔐 Starting login verification for User: {user_id}", user_id=user_id)
        self.update_status("🔴 Recording... Speak now!")

        # Start recording
        self.current_mode = "verification"
        self.start_recording("verification")

    def toggle_enrollment_recording(self):
        """Toggle enrollment recording on/off."""
        if not self.is_recording:
            user_id = self.user_id_var.get().strip()
            if not user_id:
                messagebox.showerror("Error", "Please enter a User ID first")
                return

            self.start_recording("enrollment")
            self.enroll_button.config(text="⏹️ Stop Recording")
        else:
            self.stop_recording()
            self.enroll_button.config(text="🎙️ Record for Enrollment")

    def toggle_verification_recording(self):
        """Toggle verification recording on/off."""
        if not self.is_recording:
            user_id = self.user_id_var.get().strip()
            if not user_id:
                messagebox.showerror("Error", "Please enter a User ID first")
                return

            self.start_recording("verification")
            self.verify_button.config(text="⏹️ Stop Recording")
        else:
            self.stop_recording()
            self.verify_button.config(text="🔍 Record for Verification")

    def start_recording(self, purpose: str):
        """Start audio recording."""
        try:
            self.is_recording = True
            self.audio_data = []

            # Audio parameters - optimized for speech capture
            chunk = 2048  # Larger chunk for better stability
            format = pyaudio.paInt16
            channels = 1
            rate = settings.SAMPLE_RATE

            # Close any existing stream first
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass

            self.audio_stream = self.pyaudio_instance.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=self.audio_device_index,  # Use detected device
                frames_per_buffer=chunk,
                stream_callback=None,
            )

            self.update_recording_status("🔴 Recording... (3 seconds)")
            self.log(f"Started recording for {purpose} - SPEAK NOW!")

            # Start recording in a separate thread with auto-stop after 3 seconds
            self.recording_thread = threading.Thread(
                target=self._record_audio_timed, args=(3.0,)
            )
            self.recording_thread.daemon = True
            self.recording_thread.start()

        except Exception as e:
            self.log(f"Error starting recording: {str(e)}")
            messagebox.showerror(
                "Recording Error",
                f"Failed to start recording: {str(e)}\n\nPlease check your microphone connection.",
            )
            self.is_recording = False

    def _record_audio_timed(self, duration: float):
        """Record audio for a specific duration."""
        try:
            import time

            start_time = time.time()
            chunk_size = 2048

            self.log(f"🎙️ Recording for {duration} seconds - PLEASE SPEAK NOW!")

            while self.is_recording and (time.time() - start_time) < duration:
                try:
                    data = self.audio_stream.read(
                        chunk_size, exception_on_overflow=False
                    )
                    self.audio_data.append(data)

                    # Update countdown
                    remaining = duration - (time.time() - start_time)
                    if remaining > 0:
                        self.update_recording_status(
                            f"🔴 Recording... ({remaining:.1f}s) - SPEAK NOW!"
                        )
                except Exception as read_error:
                    self.log(f"⚠️ Read error (continuing): {str(read_error)}")
                    continue

            self.log(f"✅ Recording complete - captured {len(self.audio_data)} chunks")

            # Auto-stop after duration
            self.root.after(0, self.stop_recording)

        except Exception as e:
            self.log(f"❌ Error during recording: {str(e)}")
            self.root.after(0, self.stop_recording)

    def _record_audio(self):
        """Record audio in a separate thread."""
        try:
            while self.is_recording:
                data = self.audio_stream.read(1024, exception_on_overflow=False)
                self.audio_data.append(data)
        except Exception as e:
            self.log(f"Error during recording: {str(e)}")

    def stop_recording(self):
        """Stop audio recording."""
        try:
            self.is_recording = False

            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

            self.update_recording_status("⚪ Not Recording")
            self.log("Recording stopped")

            # Process recorded audio
            if self.audio_data:
                self.process_recorded_audio()

        except Exception as e:
            self.log(f"Error stopping recording: {str(e)}")

    def process_recorded_audio(self):
        """Process the recorded audio data."""
        try:
            # Convert audio data to numpy array
            audio_bytes = b"".join(self.audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

            self.log("Processing audio...")

            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_array)

            self.log(f"✅ Audio processed successfully: {len(processed_audio)} samples")

            # Handle based on current mode
            if self.current_mode == "enrollment":
                self.complete_enrollment_with_audio(processed_audio)
            elif self.current_mode == "verification":
                self.verify_with_audio(processed_audio)

        except Exception as e:
            self.log(f"❌ Error processing audio: {str(e)}")
            messagebox.showerror(
                "Processing Error", f"Failed to process audio: {str(e)}"
            )
            self.update_status("🔴 Failed - Try again")

    def complete_enrollment_with_audio(self, processed_audio):
        """Complete enrollment after recording - collect 3 samples."""
        try:
            user_id = self.user_id_var.get().strip()

            if not user_id:
                messagebox.showerror("Error", "User ID is missing!")
                return

            # Run anti-spoofing check on enrollment audio
            self.log("🔍 Checking audio authenticity...")
            
            # Extract spectrogram for anti-spoofing analysis
            mel_spec = librosa.feature.melspectrogram(
                y=processed_audio,
                sr=16000,
                n_mels=128,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Run anti-spoofing detection (lenient for enrollment since admin-supervised)
            spoof_result = self.anti_spoofing.detect_spoofing(
                audio=processed_audio,
                spectrogram=mel_spec_db,
                sr=16000,
                mode='enrollment'
            )
            
            # Log anti-spoofing score but don't block during enrollment (admin supervised)
            self.log(f"ℹ️ Anti-spoofing score: {spoof_result['confidence']:.1%} (logged only)")
            self.log(f"✅ Enrollment proceeds (admin supervised - anti-spoofing disabled)")

            # Add current sample
            self.registration_samples.append(processed_audio)
            self.log(
                f"✅ Sample {self.current_sample_number}/{self.total_samples_needed} recorded"
            )

            # Check if we need more samples
            if self.current_sample_number < self.total_samples_needed:
                self.current_sample_number += 1

                # Prompt for next sample
                self.update_status(
                    f"⏳ Sample {self.current_sample_number - 1} done. Recording next..."
                )
                self.log(
                    f"🎤 Please record sample {self.current_sample_number}/{self.total_samples_needed}"
                )

                # Wait 2 seconds then record next sample
                self.root.after(2000, lambda: self._record_next_sample())
                return

            # All samples collected - create profile by averaging embeddings
            self.log(f"Processing {len(self.registration_samples)} samples...")

            embeddings_list = []
            for sample in self.registration_samples:
                embedding = self.profile_manager.extract_embedding_from_audio(sample)
                embeddings_list.append(embedding)

            embeddings_array = np.vstack(embeddings_list)
            avg_embedding = np.mean(embeddings_array, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            self.log(
                f"   Averaged {len(embeddings_list)} embeddings (dim={avg_embedding.shape[0]})"
            )

            # Create voice profile
            try:
                profile = self.profile_manager.create_profile_from_embeddings(
                    user_id, avg_embedding
                )

                self.update_status(
                    f"✅ User {user_id} registered with {self.total_samples_needed} samples!"
                )
                self.log(f"✅ Registration complete for {user_id}")

                messagebox.showinfo(
                    "Success!",
                    f"✅ User {user_id} registered successfully!\\n\\n"
                    f"Voice profile saved to MongoDB Cloud\n"
                    f"Samples: {self.total_samples_needed}\n\n"
                    "You can now login using the LOGIN button.",
                )
            except Exception as profile_error:
                raise Exception(
                    f"Failed to save profile to MongoDB: {str(profile_error)}"
                )

            # Reset
            self.registration_samples = []
            self.current_sample_number = 0

        except Exception as e:
            self.log(f"❌ Enrollment failed: {str(e)}")
            messagebox.showerror("Enrollment Error", f"Failed to enroll user: {str(e)}")
            self.update_status("🔴 Registration failed")
            self.registration_samples = []
            self.current_sample_number = 0

    def _record_next_sample(self):
        """Record the next voice sample."""
        self.update_status(
            f"🔴 Recording Sample {self.current_sample_number}/{self.total_samples_needed}... Speak now!"
        )
        self.start_recording("enrollment")

    def verify_with_audio(self, processed_audio):
        """Verify user after recording - ONLY authenticate registered users."""
        try:
            user_id = self.login_user_id_var.get().strip()

            if not user_id:
                messagebox.showerror("Error", "User ID is missing!")
                return

            # CHECK 1: Verify user is registered in MongoDB
            if not self.profile_manager.user_exists(user_id):
                self.update_status(f"❌ User {user_id} not registered")
                self.log(f"❌ User {user_id} is not registered in MongoDB", user_id=user_id, level="WARNING")
                messagebox.showerror(
                    "User Not Found",
                    f"❌ User {user_id} is not registered!\n\n"
                    "Please register first using the REGISTER button.",
                )
                return

            self.log(f"Verifying registered user: {user_id}", user_id=user_id)

            # CHECK 2: Anti-Spoofing Detection
            self.log("🔍 Running anti-spoofing checks...")
            
            # Extract spectrogram for anti-spoofing analysis
            mel_spec = librosa.feature.melspectrogram(
                y=processed_audio,
                sr=16000,
                n_mels=128,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Run comprehensive anti-spoofing detection (strict for verification)
            spoof_result = self.anti_spoofing.detect_spoofing(
                audio=processed_audio,
                spectrogram=mel_spec_db,
                sr=16000,
                mode='verification'
            )
            
            # Anti-spoofing detection results (monitoring only - avoiding false positives)
            self.log(f"   Anti-spoofing confidence: {spoof_result['confidence']:.1%}", user_id=user_id)
            self.log(f"   CNN score: {spoof_result['cnn_score']:.1%}", user_id=user_id)
            self.log(f"   Replay score: {spoof_result['replay_score']:.1%}", user_id=user_id)
            self.log(f"ℹ️ Anti-spoofing monitoring only (needs more genuine training data)", user_id=user_id)

            # CHECK 3: Verify voice matches registered profile
            result = self.profile_manager.verify(user_id, processed_audio)

            if result.get("verified", False):
                similarity = result.get("similarity", 0)

                # CHECK 4: Threshold - require 90% or higher match (stricter to prevent false positives)
                if similarity >= 0.90:
                    self.update_status(
                        f"✅ Login successful! (Match: {similarity:.2%})"
                    )
                    self.log(f"✅ User {user_id} authenticated successfully!", user_id=user_id)
                    
                    # Log successful authentication to database
                    if self.db_instance:
                        try:
                            from datetime import datetime
                            self.db_instance.save_log(
                                level="INFO",
                                message=f"User {user_id} authenticated successfully",
                                module="authentication_login",
                                user_id=user_id,
                                extra_data={
                                    "similarity": float(similarity),
                                    "status": "SUCCESS",
                                    "login_time": datetime.utcnow().isoformat(),
                                    "threshold": 0.90
                                }
                            )
                        except Exception as log_err:
                            print(f"⚠️ Failed to log authentication: {log_err}")
                    
                    # Store user info for monitoring
                    self.current_user = user_id
                    self.current_user_embedding = result.get("stored_embedding")

                    # Show success message
                    messagebox.showinfo(
                        "Authentication Successful!",
                        f"✅ Welcome User {user_id}!\n\n"
                        f"Status: VERIFIED ✓\n"
                        f"Voice Match: {similarity:.2%}\n"
                        f"Confidence: {'HIGH' if similarity > 0.90 else 'GOOD'}\n\n"
                        f"Real-time voice monitoring has been started automatically.",
                    )
                    
                    # Automatically start voice monitoring
                    self.start_voice_monitoring_auto()
                        
                else:
                    self.update_status(
                        f"❌ Login failed (Match: {similarity:.2%} < 90%)"
                    )
                    self.log(f"❌ Voice match too low for {user_id}: {similarity:.2%}", user_id=user_id, level="WARNING")
                    
                    # Log failed authentication attempt
                    if self.db_instance:
                        try:
                            from datetime import datetime
                            self.db_instance.save_log(
                                level="WARNING",
                                message=f"Authentication failed for {user_id} - Voice match too low",
                                module="authentication_login",
                                user_id=user_id,
                                extra_data={
                                    "similarity": float(similarity),
                                    "status": "FAILED",
                                    "reason": "Low similarity score",
                                    "login_time": datetime.utcnow().isoformat(),
                                    "threshold": 0.90
                                }
                            )
                        except Exception as log_err:
                            print(f"⚠️ Failed to log authentication failure: {log_err}")

                    messagebox.showerror(
                        "Authentication Failed",
                        f"❌ NOT VERIFIED ✗\\n\\n"
                        f"Voice does not match registered profile!\\n\\n"
                        f"Match score: {similarity:.2%} (minimum 90% required)\\n\\n"
                        "This may not be the registered user.",
                    )
            else:
                similarity = result.get("similarity", 0)
                self.update_status(f"❌ Login failed")
                self.log(f"❌ Authentication failed for {user_id}", user_id=user_id, level="ERROR")

                messagebox.showerror(
                    "Authentication Failed",
                    f"❌ Voice verification failed!\\n\\nAccess denied.",
                )

        except Exception as e:
            self.log(f"❌ Verification failed: {str(e)}")
            messagebox.showerror(
                "Verification Error", f"Failed to verify user: {str(e)}"
            )
            self.update_status("🔴 Login failed")

    def load_audio_file(self):
        """Load an audio file for enrollment."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[
                    ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                    ("WAV files", "*.wav"),
                    ("All files", "*.*"),
                ],
            )

            if file_path:
                # Load and preprocess audio file
                audio_data, sr = self.audio_processor.load_audio(file_path)
                processed_audio = self.audio_processor.preprocess_audio(audio_data)

                self.last_processed_audio = processed_audio
                self.log(f"Loaded audio file: {os.path.basename(file_path)}")

        except Exception as e:
            self.log(f"Error loading audio file: {str(e)}")
            messagebox.showerror("File Error", f"Failed to load audio file: {str(e)}")

    def complete_enrollment(self):
        """Complete user enrollment with processed audio."""
        try:
            user_id = self.user_id_var.get().strip()
            if not user_id:
                messagebox.showerror("Error", "Please enter a User ID")
                return

            if not hasattr(self, "last_processed_audio"):
                messagebox.showerror("Error", "Please record or load audio first")
                return

            # Check if user already exists
            existing_embeddings = self.profile_manager.get_user_embeddings(user_id)
            if existing_embeddings is not None:
                if not messagebox.askyesno(
                    "User Exists", f"User {user_id} already exists. Overwrite?"
                ):
                    return

            self.update_status("🔄 Processing enrollment...")
            self.profile_manager.create_profile(user_id, self.last_processed_audio)

            self.log(f"✅ User {user_id} enrolled successfully")
            self.update_status("🟢 Ready")
            messagebox.showinfo("Success", f"User {user_id} enrolled successfully!")

        except Exception as e:
            self.log(f"Error during enrollment: {str(e)}")
            self.update_status("🔴 Error")
            messagebox.showerror("Enrollment Error", f"Failed to enroll user: {str(e)}")

    def quick_verify(self):
        """Perform quick verification."""
        try:
            user_id = self.user_id_var.get().strip()
            if not user_id:
                messagebox.showerror("Error", "Please enter a User ID")
                return

            if not hasattr(self, "last_processed_audio"):
                messagebox.showerror("Error", "Please record or load audio first")
                return

            # Check if user is enrolled
            enrolled_embeddings = self.profile_manager.get_user_embeddings(user_id)
            if enrolled_embeddings is None:
                messagebox.showerror("Error", f"User {user_id} is not enrolled")
                return

            # Perform verification
            self.update_status("🔄 Verifying...")
            start_time = time.time()

            result = self.profile_manager.verify(user_id, self.last_processed_audio)
            processing_time = time.time() - start_time

            similarity = result.get("similarity", 0.0)
            is_verified = result.get("verified", False)

            if is_verified:
                result_msg = (
                    f"✅ VERIFIED\nSimilarity: {similarity:.3f}\nTime: {processing_time:.2f}s"
                )
                self.log(f"✅ User {user_id} verified (similarity: {similarity:.3f})")
                self.update_status("🟢 Verified")
                messagebox.showinfo("Verification Result", result_msg)
            else:
                result_msg = (
                    f"❌ NOT VERIFIED\nSimilarity: {similarity:.3f}\nTime: {processing_time:.2f}s"
                )
                self.log(
                    f"❌ User {user_id} verification failed (similarity: {similarity:.3f})"
                )
                self.update_status("🔴 Not Verified")
                messagebox.showwarning("Verification Result", result_msg)

        except Exception as e:
            self.log(f"Error during verification: {str(e)}")
            self.update_status("🔴 Error")
            messagebox.showerror(
                "Verification Error", f"Failed to verify user: {str(e)}"
            )

    def start_voice_monitoring_auto(self):
        """Automatically start voice monitoring after successful login."""
        try:
            if not self.current_user or self.current_user_embedding is None:
                self.log("⚠️ Cannot start monitoring: Missing user data")
                return
                
            if self.current_monitor and self.current_monitor.is_monitoring:
                self.log(f"⚠️ Monitoring already active for {self.current_user}")
                return
                
            self.log(f"🎤 Starting automatic voice monitoring for {self.current_user}")
            
            # Create voice monitor
            self.current_monitor = VoiceMonitor(
                user_id=self.current_user,
                enrolled_embedding=self.current_user_embedding,
                check_interval=1.0,  # Check every 1 second (real-time)
                threshold=0.80,  # 80% similarity required
                alert_callback=self.monitoring_alert_handler,
            )
            
            # Start monitoring
            self.current_monitor.start_monitoring()
            
            # Register with admin panel
            AdminPanel.add_monitor(self.current_user, self.current_monitor)
            
            # Log monitoring session start to database
            if self.db_instance:
                try:
                    self.db_instance.save_log(
                        level="INFO",
                        message=f"Voice monitoring session started for {self.current_user}",
                        module="admin_panel_session",
                        user_id=self.current_user,
                        extra_data={
                            "action": "Monitoring Started",
                            "total_checks": 0,
                            "success_rate": 0.0,
                            "status": "Active",
                            "check_interval": 1.0,
                            "threshold": 0.80
                        }
                    )
                except Exception as log_err:
                    print(f"⚠️ Failed to log monitoring session start: {log_err}")
            
            self.log(f"✅ Monitoring started successfully for {self.current_user}")
            self.log(f"   Check interval: 1 second (real-time)")
            self.log(f"   Threshold: 80%")
            
        except Exception as e:
            self.log(f"❌ Error starting monitoring: {str(e)}")
            messagebox.showerror(
                "Monitoring Error",
                f"Failed to start voice monitoring:\n{str(e)}"
            )
            
    def monitoring_alert_handler(self, result):
        """Handle monitoring alerts when verification fails repeatedly."""
        user_id = result.get('user_id', 'Unknown')
        similarity = result.get('similarity', 0)
        failures = result.get('consecutive_failures', 0)
        
        # Log alert without showing middle popup dialog
        self.log(f"🚨 ALERT: Multiple monitoring failures for {user_id}")
        self.log(f"   Failures: {failures}, Last similarity: {similarity:.1%}")
        self.log(f"   ⚠️ Session may be compromised!")
        
        # Note: Red notification alert is shown automatically by VoiceMonitor
        # No middle popup dialog needed
    
    def open_admin_panel(self):
        """Open the admin panel in a new window."""
        try:
            self.log("Opening admin panel...")
            
            # Create admin panel in new thread to keep it responsive
            def run_admin():
                panel = AdminPanel()
                panel.run()
            
            admin_thread = threading.Thread(target=run_admin, daemon=True)
            admin_thread.start()
            
            self.log("✅ Admin panel opened")
            
        except Exception as e:
            self.log(f"❌ Error opening admin panel: {str(e)}")
            messagebox.showerror(
                "Admin Panel Error",
                f"Failed to open admin panel:\n{str(e)}"
            )

    def on_closing(self):
        """Handle application closing."""
        try:
            if self.is_recording:
                self.stop_recording()

            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()

            self.root.destroy()

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            self.root.destroy()


def main():
    """Main function to run the desktop application."""
    try:
        # Create the main window
        root = tk.Tk()

        # Create the application
        app = VoiceAuthApp(root)

        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)

        # Start the GUI event loop
        root.mainloop()

    except Exception as e:
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
