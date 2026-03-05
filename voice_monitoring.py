"""
Continuous Voice Monitoring Module
Monitors user's voice during active session to detect if a different person takes over
"""

import time
import threading
import numpy as np
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from typing import Callable, Optional, Dict, Any
from pathlib import Path

from audio_recording import VoiceRecorder
from preprocessing import AudioPreprocessor
from embedding import EmbeddingExtractor
from scipy.spatial.distance import cosine
from app.config import settings
import librosa

# Import database logging
try:
    from db import VoiceDatabase
    from db_logger import setup_database_logging
    
    # Initialize database logging
    db_instance = VoiceDatabase(settings.MONGODB_URL)
    db_logger = setup_database_logging(db_instance, "voice_monitoring", console_output=True)
except Exception as e:
    print(f"⚠️  Database logging initialization failed: {e}")
    db_instance = None
    db_logger = None


class SmallAlertPopup:
    """Small non-blocking popup notification for unauthorized voice detection."""
    
    def __init__(self, title, message, alert_type="warning"):
        self.root = tk.Toplevel()
        self.root.title(title)
        
        # Small window size - doesn't block user's work
        self.root.geometry("400x140")
        self.root.resizable(False, False)
        
        # Position at top-middle of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 400) // 2  # Center horizontally
        y = 50  # Top of screen with some padding
        self.root.geometry(f"+{x}+{y}")
        
        # Always on top but not modal
        self.root.attributes('-topmost', True)
        
        # Color scheme based on alert type
        if alert_type == "error":
            bg_color = "#ff4444"
            fg_color = "white"
        else:
            bg_color = "#ffaa00"
            fg_color = "black"
        
        self.root.configure(bg=bg_color)
        
        # Message label
        msg_label = tk.Label(
            self.root,
            text=message,
            font=("Segoe UI", 10, "bold"),
            bg=bg_color,
            fg=fg_color,
            wraplength=320,
            justify="center"
        )
        msg_label.pack(pady=15)
        
        # OK button
        ok_btn = tk.Button(
            self.root,
            text="OK",
            command=self.close,
            font=("Segoe UI", 9, "bold"),
            bg="white" if alert_type == "error" else "#333333",
            fg="black" if alert_type == "error" else "white",
            relief=tk.FLAT,
            padx=20,
            pady=5,
            cursor="hand2"
        )
        ok_btn.pack(pady=5)
        
        # Auto-close after 10 seconds
        self.root.after(10000, self.close)
    
    def close(self):
        try:
            self.root.destroy()
        except:
            pass


class SmallRedFlag:
    """Small red flag indicator for different speaker detection."""
    
    def __init__(self):
        self.root = tk.Toplevel()
        self.root.title("Alert")
        
        # Tiny flag window
        self.root.geometry("80x80")
        self.root.resizable(False, False)
        
        # Position at top-right corner
        screen_width = self.root.winfo_screenwidth()
        x = screen_width - 100
        y = 20
        self.root.geometry(f"+{x}+{y}")
        
        # Always on top
        self.root.attributes('-topmost', True)
        self.root.configure(bg="#ff0000")
        
        # Remove window decorations for clean look
        self.root.overrideredirect(True)
        
        # Red flag emoji
        flag_label = tk.Label(
            self.root,
            text="🚩",
            font=("Segoe UI", 40),
            bg="#ff0000",
            fg="white"
        )
        flag_label.pack(expand=True)
        
        # Make it clickable to close
        flag_label.bind("<Button-1>", lambda e: self.close())
        
    def close(self):
        try:
            self.root.destroy()
        except:
            pass


class VoiceMonitor:
    """
    Continuous voice monitoring for active sessions.
    Periodically captures voice samples and compares with enrolled profile.
    """

    def __init__(
        self,
        user_id: str,
        enrolled_embedding: np.ndarray,
        model_path: str = "models/speaker_embedding_model.pth",
        check_interval: float = 30.0,  # Check every 30 seconds
        threshold: float = 0.80,  # Same threshold as authentication
        alert_callback: Optional[Callable] = None,
    ):
        """
        Initialize voice monitor.

        Args:
            user_id: ID of the authenticated user
            enrolled_embedding: The user's enrolled voice embedding
            model_path: Path to trained model
            check_interval: Seconds between voice checks
            threshold: Similarity threshold (0.80 = 80% match required)
            alert_callback: Function to call when verification fails
        """
        self.user_id = user_id
        self.enrolled_embedding = enrolled_embedding
        self.check_interval = check_interval
        self.threshold = threshold
        self.alert_callback = alert_callback

        # Initialize components
        self.recorder = VoiceRecorder(
            sample_rate=settings.SAMPLE_RATE,
            duration=settings.AUDIO_DURATION,
        )
        self.preprocessor = AudioPreprocessor(sample_rate=settings.SAMPLE_RATE)
        self.embedding_extractor = EmbeddingExtractor(
            model_path=model_path, device="cpu"
        )

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.verification_history = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # Alert after 3 consecutive failures
        
        # Unauthorized voice tracking
        self.different_speaker_start_time = None
        self.red_flag_popup = None
        self.unauthorized_alert_shown = False

        self._log("Voice Monitor initialized", "INFO")
        print(f"✅ Voice Monitor initialized for user: {user_id}")
        print(f"   Check interval: {check_interval}s")
        print(f"   Threshold: {threshold * 100}%")

    def _log(self, message: str, level: str = "INFO", **extra_data):
        """Log message to database and console."""
        # Log to database if available
        if db_instance:
            try:
                # Convert numpy types to Python native types
                clean_extra_data = {}
                for key, value in extra_data.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        clean_extra_data[key] = value.item()
                    elif isinstance(value, (np.ndarray, np.generic)):
                        clean_extra_data[key] = value.tolist() if hasattr(value, 'tolist') else value.item()
                    else:
                        clean_extra_data[key] = value
                
                db_instance.save_log(
                    level=level,
                    message=message,
                    module="voice_monitoring",
                    user_id=self.user_id,
                    extra_data=clean_extra_data
                )
            except Exception as e:
                print(f"⚠️  Database logging error: {e}")

    def start_monitoring(self):
        """Start continuous voice monitoring in background thread."""
        if self.is_monitoring:
            self._log("Monitoring already active", "WARNING")
            print("⚠️  Monitoring already active")
            return

        self.is_monitoring = True
        self.consecutive_failures = 0
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self._log("Voice monitoring started", "INFO")
        print(f"🎤 Voice monitoring started for {self.user_id}")

    def stop_monitoring(self):
        """Stop continuous voice monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self._log("Voice monitoring stopped", "INFO")
        print(f"⏹️  Voice monitoring stopped for {self.user_id}")

    def _monitoring_loop(self):
        """Main monitoring loop - runs in background thread."""
        self._log("Monitoring loop started", "INFO")
        print(f"🔄 Monitoring loop started")

        while self.is_monitoring:
            try:
                # Perform voice check
                result = self._perform_check()

                # Skip if no speech detected (silent/no one speaking)
                if result is None:
                    # No speech detected - don't log anything
                    if self.is_monitoring:
                        time.sleep(self.check_interval)
                    continue

                # Log result
                self.verification_history.append(result)

                # Handle verification result
                if result["verified"]:
                    self.consecutive_failures = 0
                    
                    # Clear unauthorized tracking if voice is verified
                    if self.different_speaker_start_time is not None:
                        self.different_speaker_start_time = None
                        self.unauthorized_alert_shown = False
                        # Close red flag if open
                        if self.red_flag_popup:
                            try:
                                self.red_flag_popup.close()
                                self.red_flag_popup = None
                            except:
                                pass
                    
                    speaker_type = "SAME SPEAKER" if not result.get("multiple_speakers", False) else "MULTIPLE SPEAKERS"
                    self._log(
                        f"Voice check passed - {speaker_type} - Similarity: {result['similarity']:.2%}",
                        "INFO",
                        similarity=result['similarity'],
                        multiple_speakers=result.get('multiple_speakers', False),
                        speaker_status=speaker_type
                    )
                    print(f"✅ Voice check passed - {speaker_type} - Similarity: {result['similarity']:.2%} at {result['timestamp']}")
                else:
                    self.consecutive_failures += 1
                    speaker_type = "DIFFERENT SPEAKER" if not result.get("multiple_speakers", False) else "MULTIPLE SPEAKERS"
                    
                    # Track different speaker detection time
                    if self.different_speaker_start_time is None:
                        self.different_speaker_start_time = time.time()
                        # Show small red flag immediately
                        try:
                            if self.red_flag_popup is None:
                                self.red_flag_popup = SmallRedFlag()
                        except Exception as popup_err:
                            print(f"⚠️ Could not show red flag: {popup_err}")
                    else:
                        # Check if different speaker for 5 seconds
                        duration = time.time() - self.different_speaker_start_time
                        if duration >= 5.0 and not self.unauthorized_alert_shown:
                            # Show small popup message
                            try:
                                SmallAlertPopup(
                                    "🚨 Security Alert",
                                    f"⚠️ Unauthorized person's voice detected!\n\nDifferent speaker for {int(duration)} seconds.",
                                    alert_type="error"
                                )
                                self.unauthorized_alert_shown = True
                            except Exception as popup_err:
                                print(f"⚠️ Could not show alert popup: {popup_err}")
                    
                    self._log(
                        f"Voice check FAILED - {speaker_type} - Similarity: {result['similarity']:.2%} (Failure #{self.consecutive_failures})",
                        "WARNING",
                        similarity=result['similarity'],
                        consecutive_failures=self.consecutive_failures,
                        multiple_speakers=result.get('multiple_speakers', False),
                        speaker_status=speaker_type
                    )
                    print(f"❌ Voice check FAILED - {speaker_type} - Similarity: {result['similarity']:.2%} (Failure #{self.consecutive_failures})")

                    # Alert if too many consecutive failures
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self._trigger_alert(result)

                # Wait before next check
                if self.is_monitoring:
                    time.sleep(self.check_interval)

            except Exception as e:
                print(f"⚠️  Monitoring error: {str(e)}")
                time.sleep(self.check_interval)  # Continue monitoring despite errors

    def _detect_multiple_speakers(self, audio_data: np.ndarray) -> bool:
        """
        Detect if multiple speakers are present in the audio.
        Uses energy variance and spectral flux analysis.

        Args:
            audio_data: Audio signal

        Returns:
            True if multiple speakers suspected, False otherwise
        """
        try:
            # Calculate frame-wise energy
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # Energy variance (multiple speakers = higher variance)
            if len(energy) > 1:
                energy_variance = np.var(energy)
                energy_mean = np.mean(energy)
                normalized_variance = energy_variance / (energy_mean + 1e-10)
                
                # Calculate spectral flux (frequency content changes)
                spectral_flux = np.sqrt(np.mean(np.diff(energy) ** 2))
                
                # Thresholds calibrated for multiple speaker detection
                high_variance = normalized_variance > 15.0
                high_flux = spectral_flux > 0.5
                
                # If both indicators are high, suspect multiple speakers
                if high_variance and high_flux:
                    return True
                    
            return False
            
        except Exception as e:
            # If detection fails, return False (don't assume multiple speakers)
            return False

    def _detect_speech_activity(self, audio_data: np.ndarray) -> bool:
        """
        Detect if there is actual human speech in the audio.
        Returns False if audio is silent or just noise.
        """
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate zero crossing rate (voice has lower ZCR than noise)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
            
            # Voice activity thresholds
            min_energy = 0.01  # Minimum RMS for speech
            max_zcr = 0.3      # Maximum zero crossing rate for speech
            
            has_energy = rms > min_energy
            is_voice_like = zero_crossings < max_zcr
            
            return has_energy and is_voice_like
            
        except Exception as e:
            return False

    def _perform_check(self) -> Dict[str, Any]:
        """
        Perform a single voice verification check.

        Returns:
            Dictionary with check results, or None if no speech detected
        """
        try:
            # Record audio
            audio_data = self.recorder.record()

            # First check if there's actual speech
            has_speech = self._detect_speech_activity(audio_data)
            
            if not has_speech:
                # No speech detected - return None to skip logging
                return None

            # Detect multiple speakers
            multiple_speakers = self._detect_multiple_speakers(audio_data)

            # Preprocess
            mfcc = self.preprocessor.extract_mfcc(audio_data)

            # Extract embedding using correct method name
            current_embedding = self.embedding_extractor.extract(audio_data)

            # Calculate similarity
            similarity = 1 - cosine(self.enrolled_embedding, current_embedding)
            verified = similarity >= self.threshold

            result = {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id,
                "similarity": similarity,
                "threshold": self.threshold,
                "verified": verified,
                "multiple_speakers": multiple_speakers,
                "consecutive_failures": self.consecutive_failures,
            }
            
            # Save detailed check result to database
            if db_instance:
                try:
                    level = "INFO" if verified else "WARNING"
                    message = f"Voice check {'passed' if verified else 'FAILED'} - Similarity: {similarity:.2%}"
                    if multiple_speakers:
                        message += " [MULTIPLE VOICES DETECTED]"
                    
                    speaker_status = "SAME SPEAKER" if (verified and not multiple_speakers) else ("DIFFERENT SPEAKER" if not verified else "MULTIPLE SPEAKERS")
                    
                    db_instance.save_log(
                        level=level,
                        message=message,
                        module="voice_monitoring_check",
                        user_id=self.user_id,
                        extra_data={
                            "similarity": float(similarity),
                            "threshold": float(self.threshold),
                            "verified": bool(verified),  # Convert numpy bool to Python bool
                            "multiple_speakers": bool(multiple_speakers),  # Convert numpy bool to Python bool
                            "consecutive_failures": int(self.consecutive_failures),  # Ensure int
                            "audio_samples": int(len(audio_data)),  # Ensure int
                            "speaker_status": str(speaker_status)  # Ensure string
                        }
                    )
                except Exception as log_error:
                    print(f"⚠️  Failed to log check to database: {log_error}")

            return result

        except Exception as e:
            error_msg = f"Check failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id,
                "similarity": 0.0,
                "threshold": self.threshold,
                "verified": False,
                "multiple_speakers": False,
                "error": str(e),
                "consecutive_failures": self.consecutive_failures,
            }
            
            # Log error to database
            if db_instance:
                try:
                    db_instance.save_log(
                        level="ERROR",
                        message=error_msg,
                        module="voice_monitoring_check",
                        user_id=self.user_id,
                        extra_data={"error": str(e)}
                    )
                except:
                    pass
            
            return result

    def _trigger_alert(self, result: Dict[str, Any]):
        """
        Trigger alert when verification fails multiple times.

        Args:
            result: Latest verification result
        """
        multiple_speakers_warning = ""
        if result.get("multiple_speakers", False):
            multiple_speakers_warning = "\n⚠️  MULTIPLE SPEAKERS DETECTED! ⚠️\n"
        
        alert_message = (
            f"⚠️  SECURITY ALERT: Voice verification failed {self.consecutive_failures} times!\n"
            f"User: {self.user_id}\n"
            f"Last similarity: {result['similarity']:.2%}\n"
            f"Required: {self.threshold:.2%}\n"
            f"Time: {result['timestamp']}\n"
            f"{multiple_speakers_warning}"
            f"Possible reasons:\n"
            f"  - Different person speaking\n"
            f"  - Multiple people speaking simultaneously\n"
            f"  - Poor audio quality\n"
            f"  - Background noise\n"
        )

        print("=" * 70)
        print(alert_message)
        print("=" * 70)
        
        # Save security alert to database
        if db_instance:
            try:
                db_instance.save_log(
                    level="CRITICAL",
                    message=f"SECURITY ALERT: Voice verification failed {self.consecutive_failures} consecutive times",
                    module="voice_monitoring_alert",
                    user_id=self.user_id,
                    extra_data={
                        "consecutive_failures": int(self.consecutive_failures),
                        "similarity": float(result.get('similarity', 0)),
                        "threshold": float(self.threshold),
                        "multiple_speakers": bool(result.get("multiple_speakers", False)),
                        "timestamp": str(result.get("timestamp")),
                        "alert_message": str(alert_message),
                        "alert_type": "Voice Verification Failed"
                    }
                )
            except Exception as e:
                print(f"⚠️  Failed to log alert to database: {e}")

        # Call custom alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(result)
            except Exception as e:
                print(f"❌ Alert callback error: {str(e)}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring statistics
        """
        total_checks = len(self.verification_history)
        if total_checks == 0:
            success_rate = 0.0
        else:
            successful_checks = sum(1 for r in self.verification_history if r["verified"])
            success_rate = successful_checks / total_checks

        return {
            "is_monitoring": self.is_monitoring,
            "user_id": self.user_id,
            "total_checks": total_checks,
            "successful_checks": sum(1 for r in self.verification_history if r["verified"]),
            "failed_checks": sum(1 for r in self.verification_history if not r["verified"]),
            "success_rate": success_rate,
            "consecutive_failures": self.consecutive_failures,
            "last_check": self.verification_history[-1] if self.verification_history else None,
        }

    def get_verification_history(self) -> list:
        """Get full verification history."""
        return self.verification_history.copy()


# Example usage
def on_alert(result):
    """Custom alert handler - called when verification fails repeatedly."""
    print(f"\n🚨 CUSTOM ALERT HANDLER TRIGGERED!")
    print(f"   Action: Lock session/request re-authentication")
    # Add your custom actions here:
    # - Lock the application
    # - Send notification
    # - Log security event
    # - Request re-authentication
    # - etc.


if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "VOICE MONITORING - TEST MODE")
    print("=" * 70)

    # For testing, create a dummy enrollment
    print("\n1️⃣  Loading model and creating test enrollment...")
    extractor = EmbeddingExtractor("models/speaker_embedding_model.pth")
    recorder = VoiceRecorder()
    preprocessor = AudioPreprocessor()

    print("\n🎤 Speak for 3 seconds to create enrollment...")
    audio = recorder.record()
    mfcc = preprocessor.extract_mfcc(audio)
    enrolled_embedding = extractor.extract_embedding(mfcc)
    print("✅ Enrollment created")

    # Start monitoring
    print("\n2️⃣  Starting voice monitor...")
    monitor = VoiceMonitor(
        user_id="test_user",
        enrolled_embedding=enrolled_embedding,
        check_interval=10.0,  # Check every 10 seconds for testing
        threshold=0.80,
        alert_callback=on_alert,
    )

    monitor.start_monitoring()

    print("\n3️⃣  Monitoring active! Speak periodically to verify...")
    print("   (Monitor will check your voice every 10 seconds)")
    print("   Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(5)
            status = monitor.get_monitoring_status()
            print(f"\r📊 Status: {status['total_checks']} checks | "
                  f"Success rate: {status['success_rate']:.1%} | "
                  f"Failures: {status['consecutive_failures']}", end="")
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping monitor...")
        monitor.stop_monitoring()
        
        # Show final statistics
        print("\n" + "=" * 70)
        print(" " * 20 + "MONITORING SUMMARY")
        print("=" * 70)
        status = monitor.get_monitoring_status()
        print(f"Total checks: {status['total_checks']}")
        print(f"Successful: {status['successful_checks']}")
        print(f"Failed: {status['failed_checks']}")
        print(f"Success rate: {status['success_rate']:.1%}")
        print("=" * 70)
