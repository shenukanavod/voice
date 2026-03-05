# Voice Monitoring System - Usage Guide

## Overview
After successful login, voice monitoring starts **automatically** in the background. The admin panel allows viewing all active monitoring sessions.

## How It Works

### 1. User Login Flow
1. User clicks "Verification" button
2. Enters User ID
3. Records voice sample (minimum 3 seconds)
4. System verifies voice (requires ≥90% match)
5. ✅ **Monitoring starts automatically** upon successful login
6. User sees confirmation message

### 2. Background Monitoring
- **Check Interval**: Every 1 second (real-time)
- **Threshold**: 80% similarity required
- **Alert Trigger**: 3 consecutive failures
- User doesn't need to do anything - runs in background

### 3. Admin Panel
- Click "👤 Admin Panel" button on landing page
- View all active monitoring sessions
- See real-time statistics and status

## Admin Panel Features

### Statistics Dashboard
- Total Sessions (all time)
- Total Checks Performed
- Overall Success Rate
- Total Alerts Triggered

### Session Table Columns
- **User ID**: The authenticated user
- **Status**: Active (✓ Running, ⏸ Stopped)
- **Checks**: Number of voice checks performed
- **Success**: Successful verifications
- **Failed**: Failed verifications  
- **Rate**: Success percentage
- **Last Check**: Timestamp of last verification

### Actions
- **Right-click on session** for menu:
  - View Details: See current status
  - Stop Monitoring: End session
  - View History: Check logs
- **Auto-refresh**: Updates every 2 seconds

## Status Indicators

### Monitoring Status
- ✅ **Verified**: Voice match ≥80%
- ⚠️ **Warning**: Voice match <80%
- 🚨 **Alert**: 3+ consecutive failures

### What Triggers Alerts?
- Different person speaking
- Poor audio quality
- Excessive background noise
- Microphone issues

## Security Features

### Automatic Protection
- Continuous identity verification
- No user intervention required
- Real-time anomaly detection
- Immediate alert on suspicious activity

### Alert Response
When alert triggers:
1. System shows warning popup
2. Logs security incident
3. Admin can review in admin panel
4. Consider session compromise

## Technical Details

### Monitoring Parameters
```python
Check Interval: 1 second (real-time)
Similarity Threshold: 0.80 (80%)
Max Consecutive Failures: 3
Alert Callback: Yes (popup + log)
```

### Audio Requirements
- Sample Rate: 16kHz
- Duration: 3 seconds minimum
- Quality: Clear speech, minimal noise

## Troubleshooting

### Monitoring Not Starting
- Ensure successful login (≥90% match)
- Check console logs for errors
- Verify microphone access

### False Alerts
- Reduce background noise
- Speak clearly into microphone
- Check microphone positioning
- Ensure same environment as enrollment

### Admin Panel Not Opening
- Check console for error messages
- Make sure no firewall blocking
- Try restarting application

## Architecture

```
User Login (≥90% match)
    ↓
Auto-Start Monitoring
    ↓
VoiceMonitor Instance Created
    ↓
Background Thread (checks every 30s)
    ↓
AdminPanel.add_monitor() → Tracks Session
    ↓
Admin Can View in Real-Time
```

## Files Involved
- `desktop_app.py`: Main application with auto-start integration
- `voice_monitoring.py`: VoiceMonitor class (background checks)
- `admin_panel.py`: AdminPanel class (UI for viewing sessions)
- `model.py`: Voice model for embeddings
- `embedding.py`: Similarity comparison

## Support
For issues or questions:
1. Check console logs in desktop app
2. Review monitoring status in admin panel
3. Verify audio quality and environment
4. Check model is loaded correctly (speaker_embedding_model.pth)
