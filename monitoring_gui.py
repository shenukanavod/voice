"""
Desktop Voice Monitoring Window
Real-time voice monitoring interface with visual feedback
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from typing import Optional
import numpy as np

from voice_monitoring import VoiceMonitor
from db import VoiceDatabase
from app.config import settings


class MonitoringWindow:
    """GUI window for real-time voice monitoring."""

    def __init__(self, parent, user_id: str, enrolled_embedding: np.ndarray):
        """
        Initialize monitoring window.

        Args:
            parent: Parent tkinter window
            user_id: Authenticated user ID
            enrolled_embedding: User's enrolled voice embedding
        """
        self.parent = parent
        self.user_id = user_id
        self.window = None
        self.monitor = None

        # Initialize voice monitor
        self.monitor = VoiceMonitor(
            user_id=user_id,
            enrolled_embedding=enrolled_embedding,
            check_interval=30.0,  # Check every 30 seconds
            threshold=settings.VERIFICATION_THRESHOLD,
            alert_callback=self.on_alert,
        )

        self.create_window()

    def create_window(self):
        """Create the monitoring window."""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Voice Monitoring - {self.user_id}")
        self.window.geometry("600x500")
        self.window.configure(bg="#0a1628")

        # Prevent closing without stopping monitor
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        # Header
        header = tk.Frame(self.window, bg="#1a2d4d", height=80)
        header.pack(fill="x", padx=20, pady=20)

        title = tk.Label(
            header,
            text="🎤 Live Voice Monitoring",
            font=("Segoe UI", 18, "bold"),
            bg="#1a2d4d",
            fg="#00d4ff",
        )
        title.pack(pady=10)

        subtitle = tk.Label(
            header,
            text=f"User: {self.user_id}",
            font=("Segoe UI", 11),
            bg="#1a2d4d",
            fg="#a0b0c0",
        )
        subtitle.pack()

        # Status Frame
        status_frame = tk.Frame(self.window, bg="#1a2d4d", relief=tk.SOLID, bd=1)
        status_frame.pack(fill="x", padx=20, pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="⏸️  Monitoring: Not Started",
            font=("Segoe UI", 12, "bold"),
            bg="#1a2d4d",
            fg="#ffaa00",
            pady=15,
        )
        self.status_label.pack()

        # Statistics Frame
        stats_frame = tk.LabelFrame(
            self.window,
            text="Statistics",
            font=("Segoe UI", 11, "bold"),
            bg="#0a1628",
            fg="#00d4ff",
            relief=tk.SOLID,
            bd=1,
        )
        stats_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Stats grid
        self.total_checks_var = tk.StringVar(value="0")
        self.success_checks_var = tk.StringVar(value="0")
        self.failed_checks_var = tk.StringVar(value="0")
        self.success_rate_var = tk.StringVar(value="0%")
        self.last_check_var = tk.StringVar(value="Never")
        self.last_similarity_var = tk.StringVar(value="N/A")

        stats = [
            ("Total Checks:", self.total_checks_var),
            ("Successful:", self.success_checks_var),
            ("Failed:", self.failed_checks_var),
            ("Success Rate:", self.success_rate_var),
            ("Last Check:", self.last_check_var),
            ("Last Similarity:", self.last_similarity_var),
        ]

        for i, (label, var) in enumerate(stats):
            row = i // 2
            col = i % 2

            label_widget = tk.Label(
                stats_frame,
                text=label,
                font=("Segoe UI", 10),
                bg="#0a1628",
                fg="#a0b0c0",
                anchor="w",
            )
            label_widget.grid(row=row, column=col * 2, padx=20, pady=8, sticky="w")

            value_widget = tk.Label(
                stats_frame,
                textvariable=var,
                font=("Segoe UI", 10, "bold"),
                bg="#0a1628",
                fg="#ffffff",
                anchor="w",
            )
            value_widget.grid(row=row, column=col * 2 + 1, padx=10, pady=8, sticky="w")

        # Activity Log
        log_frame = tk.LabelFrame(
            self.window,
            text="Activity Log",
            font=("Segoe UI", 11, "bold"),
            bg="#0a1628",
            fg="#00d4ff",
            relief=tk.SOLID,
            bd=1,
        )
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.log_text = tk.Text(
            log_frame,
            height=8,
            font=("Consolas", 9),
            bg="#0d1929",
            fg="#ffffff",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Control Buttons
        button_frame = tk.Frame(self.window, bg="#0a1628")
        button_frame.pack(fill="x", padx=20, pady=15)

        self.start_btn = tk.Button(
            button_frame,
            text="▶️  Start Monitoring",
            font=("Segoe UI", 11, "bold"),
            bg="#00cc66",
            fg="white",
            activebackground="#00ff88",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_monitoring,
            padx=20,
            pady=8,
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️  Stop Monitoring",
            font=("Segoe UI", 11, "bold"),
            bg="#ff4444",
            fg="white",
            activebackground="#ff6666",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.stop_monitoring,
            padx=20,
            pady=8,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=5)

        self.close_btn = tk.Button(
            button_frame,
            text="✖️  Close",
            font=("Segoe UI", 11, "bold"),
            bg="#555555",
            fg="white",
            activebackground="#777777",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.on_close,
            padx=20,
            pady=8,
        )
        self.close_btn.pack(side="right", padx=5)

        # Start update loop
        self.update_display()
        self.log("Monitoring window ready")

    def start_monitoring(self):
        """Start voice monitoring."""
        self.monitor.start_monitoring()
        self.status_label.config(text="🎤 Monitoring: ACTIVE", fg="#00ff00")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log("✅ Monitoring started")

    def stop_monitoring(self):
        """Stop voice monitoring."""
        self.monitor.stop_monitoring()
        self.status_label.config(text="⏸️  Monitoring: STOPPED", fg="#ffaa00")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("⏹️  Monitoring stopped")

    def update_display(self):
        """Update display with latest statistics."""
        if self.monitor:
            status = self.monitor.get_monitoring_status()

            # Update stats
            self.total_checks_var.set(str(status["total_checks"]))
            self.success_checks_var.set(str(status["successful_checks"]))
            self.failed_checks_var.set(str(status["failed_checks"]))
            self.success_rate_var.set(f"{status['success_rate']:.1%}")

            if status["last_check"]:
                last_time = datetime.fromisoformat(status["last_check"]["timestamp"])
                self.last_check_var.set(last_time.strftime("%H:%M:%S"))
                self.last_similarity_var.set(
                    f"{status['last_check']['similarity']:.1%}"
                )

        # Schedule next update
        if self.window and self.window.winfo_exists():
            self.window.after(1000, self.update_display)

    def on_alert(self, result):
        """Handle security alert."""
        alert_msg = (
            f"⚠️  SECURITY ALERT!\n\n"
            f"Voice verification failed multiple times.\n\n"
            f"Similarity: {result['similarity']:.1%}\n"
            f"Required: {result['threshold']:.1%}\n\n"
            f"The session may be compromised."
        )

        self.log(f"🚨 ALERT: Verification failed!")
        
        # Show alert dialog
        self.window.after(0, lambda: messagebox.showwarning(
            "Security Alert",
            alert_msg,
            parent=self.window
        ))

    def log(self, message: str):
        """Add message to activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        if self.log_text:
            self.log_text.insert("end", log_entry)
            self.log_text.see("end")

    def on_close(self):
        """Handle window close."""
        if self.monitor and self.monitor.is_monitoring:
            response = messagebox.askyesno(
                "Monitoring Active",
                "Monitoring is still active. Stop monitoring and close?",
                parent=self.window,
            )
            if not response:
                return

            self.stop_monitoring()

        if self.window:
            self.window.destroy()


# Example standalone usage
if __name__ == "__main__":
    from embedding import EmbeddingExtractor
    from audio_recording import VoiceRecorder
    from preprocessing import AudioPreprocessor

    print("Testing Monitoring Window...")

    # Create test enrollment
    recorder = VoiceRecorder()
    preprocessor = AudioPreprocessor()
    extractor = EmbeddingExtractor("models/speaker_embedding_model.pth")

    print("🎤 Speak for 3 seconds to create test enrollment...")
    audio = recorder.record()
    mfcc = preprocessor.extract_mfcc(audio)
    enrolled_embedding = extractor.extract_embedding(mfcc)

    # Create GUI
    root = tk.Tk()
    root.withdraw()  # Hide main window

    window = MonitoringWindow(root, "test_user", enrolled_embedding)

    root.mainloop()
