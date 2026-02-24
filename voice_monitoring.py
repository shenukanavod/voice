"""
Continuous Voice Monitoring Module
Monitors user's voice during active session to detect if a different person takes over
"""

import time
import threading
import numpy as np
from datetime import datetime
from typing import Callable, Optional, Dict, Any
from pathlib import Path

from audio_recording import VoiceRecorder
from preprocessing import AudioPreprocessor
from embedding import EmbeddingExtractor
from scipy.spatial.distance import cosine
from app.config import settings
import librosa


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

        print(f"✅ Voice Monitor initialized for user: {user_id}")
        print(f"   Check interval: {check_interval}s")
        print(f"   Threshold: {threshold * 100}%")

    def start_monitoring(self):
        """Start continuous voice monitoring in background thread."""
        if self.is_monitoring:
            print("⚠️  Monitoring already active")
            return

        self.is_monitoring = True
        self.consecutive_failures = 0
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"🎤 Voice monitoring started for {self.user_id}")

    def stop_monitoring(self):
        """Stop continuous voice monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print(f"⏹️  Voice monitoring stopped for {self.user_id}")

    def _monitoring_loop(self):
        """Main monitoring loop - runs in background thread."""
        print(f"🔄 Monitoring loop started")

        while self.is_monitoring:
            try:
                # Perform voice check
                result = self._perform_check()

                # Log result
                self.verification_history.append(result)

                # Handle verification result
                if result["verified"]:
                    self.consecutive_failures = 0
                    multi_voice_indicator = " [🚨 MULTIPLE VOICES DETECTED]" if result.get("multiple_speakers", False) else ""
                    print(f"✅ Voice check passed - Similarity: {result['similarity']:.2%} at {result['timestamp']}{multi_voice_indicator}")
                else:
                    self.consecutive_failures += 1
                    multi_voice_indicator = " [🚨 MULTIPLE VOICES DETECTED]" if result.get("multiple_speakers", False) else ""
                    print(f"❌ Voice check FAILED - Similarity: {result['similarity']:.2%} (Failure #{self.consecutive_failures}){multi_voice_indicator}")

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

    def _perform_check(self) -> Dict[str, Any]:
        """
        Perform a single voice verification check.

        Returns:
            Dictionary with check results
        """
        try:
            # Record audio
            audio_data = self.recorder.record()

            # Detect multiple speakers
            multiple_speakers = self._detect_multiple_speakers(audio_data)

            # Preprocess
            mfcc = self.preprocessor.extract_mfcc(audio_data)

            # Extract embedding using correct method name
            current_embedding = self.embedding_extractor.extract(audio_data)

            # Calculate similarity
            similarity = 1 - cosine(self.enrolled_embedding, current_embedding)
            verified = similarity >= self.threshold

            return {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id,
                "similarity": similarity,
                "threshold": self.threshold,
                "verified": verified,
                "multiple_speakers": multiple_speakers,
                "consecutive_failures": self.consecutive_failures,
            }

        except Exception as e:
            print(f"❌ Check failed: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id,
                "similarity": 0.0,
                "threshold": self.threshold,
                "verified": False,
                "multiple_speakers": False,
                "error": str(e),
                "consecutive_failures": self.consecutive_failures,
            }

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
