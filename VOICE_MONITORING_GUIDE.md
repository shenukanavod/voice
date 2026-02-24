# 🎤 Voice Monitoring Feature

## Overview
Continuous voice monitoring allows you to verify that the authenticated user remains the same person throughout a session. This is useful for:
- Online exams/proctoring
- Secure sessions
- Continuous authentication
- Detecting session takeover

## How It Works

### 1. Authentication Phase
- User logs in with voice authentication
- System extracts and verifies voice embedding
- Upon successful login, user is prompted to enable monitoring

### 2. Monitoring Phase
- System periodically records short audio samples (every 30 seconds by default)
- Compares new voice samples with enrolled profile
- Tracks verification success rate
- Alerts if voice verification fails repeatedly

### 3. Alert System
- After 3 consecutive verification failures, the system triggers an alert
- Possible reasons for failures:
  - Different person speaking
  - Poor audio quality
  - Excessive background noise
  - Microphone issues

## Usage

### Option 1: Through Desktop App
```bash
python desktop_app.py
```

1. Register a new user (if not already registered)
2. Login with voice authentication
3. When prompted "Would you like to enable continuous voice monitoring?" select **Yes**
4. The monitoring window will open automatically

### Option 2: Standalone Monitoring Test
```bash
python voice_monitoring.py
```
This will:
- Create a test enrollment
- Start monitoring
- Check voice every 10 seconds (for testing)
- Display real-time statistics

### Option 3: Custom Integration
```python
from voice_monitoring import VoiceMonitor
from embedding import EmbeddingExtractor
import numpy as np

# Create user enrollment first
extractor = EmbeddingExtractor("models/speaker_embedding_model.pth")
enrolled_embedding = extractor.extract_embedding(mfcc_data)

# Start monitoring
monitor = VoiceMonitor(
    user_id="user123",
    enrolled_embedding=enrolled_embedding,
    check_interval=30.0,  # Check every 30 seconds
    threshold=0.80,  # 80% similarity required
    alert_callback=my_custom_alert_handler
)

monitor.start_monitoring()

# Later... stop monitoring
monitor.stop_monitoring()

# Get statistics
status = monitor.get_monitoring_status()
print(f"Success rate: {status['success_rate']:.1%}")
```

## Features

### Monitoring Window
- **Real-time Statistics**: Shows total checks, success rate, failures
- **Activity Log**: Displays all verification events with timestamps
- **Visual Status**: Color-coded status indicators
- **Manual Controls**: Start/stop monitoring, view history

### Configuration Options

```python
VoiceMonitor(
    user_id="john_doe",           # User identifier
    enrolled_embedding=embedding,  # User's voice embedding
    check_interval=30.0,          # Seconds between checks (default: 30)
    threshold=0.80,               # Match threshold (default: 0.80)
    alert_callback=my_handler,    # Custom alert function (optional)
)
```

### Custom Alert Handler
```python
def my_alert_handler(result):
    """Called when verification fails multiple times."""
    print(f"Alert for user: {result['user_id']}")
    print(f"Similarity: {result['similarity']:.2%}")
    
    # Your custom actions:
    # - Lock the session
    # - Send notification
    # - Log security event
    # - Request re-authentication
    # - etc.
```

## Monitoring Statistics

```python
status = monitor.get_monitoring_status()
```

Returns:
- `is_monitoring`: Whether monitoring is active
- `user_id`: User being monitored
- `total_checks`: Total verification checks performed
- `successful_checks`: Number of successful verifications
- `failed_checks`: Number of failed verifications
- `success_rate`: Percentage of successful checks
- `consecutive_failures`: Current consecutive failure count
- `last_check`: Most recent check result with timestamp

## API

### VoiceMonitor Class

#### Methods
- `start_monitoring()` - Start continuous monitoring
- `stop_monitoring()` - Stop monitoring
- `get_monitoring_status()` - Get current statistics
- `get_verification_history()` - Get full history of all checks

#### Configuration
- `check_interval`: Time between voice checks (seconds)
- `threshold`: Minimum similarity for verification (0.0 - 1.0)
- `max_consecutive_failures`: Failures before alert (default: 3)

## Use Cases

### 1. Online Exam Proctoring
```python
# Start monitoring when exam begins
monitor.start_monitoring()

# Exam duration...

# Stop when exam ends
monitor.stop_monitoring()

# Generate exam report
history = monitor.get_verification_history()
suspicious_events = [h for h in history if not h['verified']]
```

### 2. Secure Work Session
```python
# Monitor during sensitive work
monitor = VoiceMonitor(
    user_id=employee_id,
    enrolled_embedding=employee_voice,
    check_interval=60.0,  # Check every minute
    threshold=0.85,       # Higher security
)
```

### 3. Continuous Authentication
```python
# Combine with other biometrics
def multi_factor_alert(result):
    # Trigger additional authentication
    request_fingerprint()
    # or face recognition
    # or password re-entry
```

## Requirements

- Trained voice model: `models/speaker_embedding_model.pth`
- Microphone access
- User enrollment completed
- Python packages: see `requirements.txt`

## Troubleshooting

### High False Rejection Rate
- **Issue**: Legitimate user frequently fails verification
- **Solutions**:
  - Lower threshold (e.g., 0.75 instead of 0.80)
  - Increase check interval (less frequent checks)
  - Improve microphone quality
  - Reduce background noise
  - Re-enroll user with better quality samples

### Monitoring Not Starting
- **Issue**: Monitoring window doesn't open
- **Solutions**:
  - Ensure user is authenticated first
  - Check that embedding is available
  - Verify model file exists
  - Check console for error messages

### Performance Issues
- **Issue**: System lag during monitoring
- **Solutions**:
  - Increase check interval (e.g., 60s instead of 30s)
  - Run on faster hardware
  - Close other applications

## Security Notes

- Monitoring runs in background thread (non-blocking)
- Each check is independent (failure doesn't affect next check)
- Alert threshold prevents false alarms from single failures
- All checks are logged for audit purposes
- Monitoring stops automatically when window is closed

## Advanced

### Integration with Session Management
```python
class SecureSession:
    def __init__(self, user_id, enrolled_embedding):
        self.monitor = VoiceMonitor(
            user_id=user_id,
            enrolled_embedding=enrolled_embedding,
            alert_callback=self.handle_breach
        )
        
    def start(self):
        self.monitor.start_monitoring()
        
    def handle_breach(self, result):
        # Lock session
        self.lock_session()
        # Notify admin
        self.send_security_alert(result)
        # Require re-authentication
        self.request_reauth()
```

## Files

- `voice_monitoring.py` - Core monitoring logic
- `monitoring_gui.py` - GUI window for desktop app
- `desktop_app.py` - Integrated desktop application (updated)

---

**Created with ❤️ for secure voice authentication**
