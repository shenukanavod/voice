"""
Authentication Attempt Limiter
Prevents brute force attacks by limiting failed authentication attempts.

Features:
- Max 3 failed attempts per user
- 30-second lockout after reaching limit
- Counter resets only after successful live verification
- Persistent storage across sessions
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path


class AttemptLimiter:
    """
    Manages authentication attempt tracking and lockouts.
    
    Security Features:
    - Tracks failed attempts per user
    - Enforces lockout period after max attempts
    - Resets counter only on successful verification
    - Persists data across restarts
    """
    
    MAX_ATTEMPTS = 3
    LOCKOUT_DURATION = 30  # seconds
    
    def __init__(self, storage_path: str = "data/attempt_limits.json"):
        """
        Initialize the attempt limiter.
        
        Args:
            storage_path: Path to JSON file for persistent storage
        """
        self.storage_path = storage_path
        self.data: Dict[str, Dict] = self._load_data()
        
    def _load_data(self) -> Dict[str, Dict]:
        """Load attempt data from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load attempt data: {e}")
                return {}
        return {}
    
    def _save_data(self):
        """Save attempt data to disk."""
        try:
            # Ensure directory exists
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save attempt data: {e}")
    
    def _get_user_data(self, user_id: str) -> Dict:
        """Get or initialize user attempt data."""
        if user_id not in self.data:
            self.data[user_id] = {
                'failed_attempts': 0,
                'locked_until': None,
                'last_failure': None,
                'last_success': None
            }
        return self.data[user_id]
    
    def is_locked(self, user_id: str) -> Tuple[bool, Optional[float]]:
        """
        Check if user is currently locked out.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_locked, remaining_seconds)
            remaining_seconds is None if not locked
        """
        user_data = self._get_user_data(user_id)
        locked_until = user_data.get('locked_until')
        
        if locked_until is None:
            return False, None
        
        current_time = time.time()
        
        # Check if lockout period has expired
        if current_time >= locked_until:
            # Lockout expired, clear the lock
            user_data['locked_until'] = None
            self._save_data()
            return False, None
        
        # Still locked
        remaining = locked_until - current_time
        return True, remaining
    
    def record_failure(self, user_id: str) -> Dict:
        """
        Record a failed authentication attempt.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with:
                - failed_attempts: Current count
                - is_locked: Whether user is now locked
                - locked_until: Timestamp when lock expires (if locked)
                - remaining_attempts: How many attempts left before lock
        """
        user_data = self._get_user_data(user_id)
        current_time = time.time()
        
        # Increment failure count
        user_data['failed_attempts'] += 1
        user_data['last_failure'] = current_time
        
        # Check if we should lock the user
        if user_data['failed_attempts'] >= self.MAX_ATTEMPTS:
            locked_until = current_time + self.LOCKOUT_DURATION
            user_data['locked_until'] = locked_until
            self._save_data()
            
            return {
                'failed_attempts': user_data['failed_attempts'],
                'is_locked': True,
                'locked_until': locked_until,
                'lockout_duration': self.LOCKOUT_DURATION,
                'remaining_attempts': 0
            }
        
        self._save_data()
        remaining = self.MAX_ATTEMPTS - user_data['failed_attempts']
        
        return {
            'failed_attempts': user_data['failed_attempts'],
            'is_locked': False,
            'locked_until': None,
            'remaining_attempts': remaining
        }
    
    def record_success(self, user_id: str):
        """
        Record a successful authentication.
        Resets failed attempt counter and clears any lockout.
        
        Args:
            user_id: User identifier
        """
        user_data = self._get_user_data(user_id)
        current_time = time.time()
        
        # Reset counters
        user_data['failed_attempts'] = 0
        user_data['locked_until'] = None
        user_data['last_success'] = current_time
        
        self._save_data()
    
    def get_status(self, user_id: str) -> Dict:
        """
        Get current attempt status for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with current status information
        """
        user_data = self._get_user_data(user_id)
        is_locked, remaining_time = self.is_locked(user_id)
        
        return {
            'user_id': user_id,
            'failed_attempts': user_data['failed_attempts'],
            'is_locked': is_locked,
            'remaining_lockout_seconds': remaining_time,
            'remaining_attempts': max(0, self.MAX_ATTEMPTS - user_data['failed_attempts']),
            'last_failure': user_data.get('last_failure'),
            'last_success': user_data.get('last_success')
        }
    
    def reset_user(self, user_id: str):
        """
        Manually reset a user's attempt data.
        Use for administrative purposes only.
        
        Args:
            user_id: User identifier
        """
        if user_id in self.data:
            del self.data[user_id]
            self._save_data()
    
    def cleanup_expired(self):
        """
        Remove lockout data for users whose lockout has expired.
        Call periodically for maintenance.
        """
        current_time = time.time()
        
        for user_id in list(self.data.keys()):
            user_data = self.data[user_id]
            locked_until = user_data.get('locked_until')
            
            if locked_until and current_time >= locked_until:
                user_data['locked_until'] = None
        
        self._save_data()


def format_lockout_message(remaining_seconds: float) -> str:
    """
    Format a user-friendly lockout message.
    
    Args:
        remaining_seconds: Seconds until lockout expires
        
    Returns:
        Formatted message string
    """
    if remaining_seconds < 1:
        return "Account temporarily locked. Please try again in a moment."
    
    return f"Account locked. Please try again in {int(remaining_seconds)} seconds."


# Example usage and testing
if __name__ == "__main__":
    print("=== Attempt Limiter Test ===\n")
    
    # Create limiter instance
    limiter = AttemptLimiter(storage_path="data/test_attempt_limits.json")
    test_user = "test_user_001"
    
    print(f"Testing attempt limiting for user: {test_user}\n")
    
    # Test 1: Check initial status
    print("Test 1: Initial status")
    status = limiter.get_status(test_user)
    print(f"Failed attempts: {status['failed_attempts']}")
    print(f"Is locked: {status['is_locked']}")
    print(f"Remaining attempts: {status['remaining_attempts']}")
    print()
    
    # Test 2: Record failures
    print("Test 2: Recording failed attempts")
    for i in range(1, 4):
        result = limiter.record_failure(test_user)
        print(f"Attempt {i}: {result['failed_attempts']} failures, "
              f"{result['remaining_attempts']} remaining, "
              f"Locked: {result['is_locked']}")
    print()
    
    # Test 3: Check if locked
    print("Test 3: Check lockout status")
    is_locked, remaining = limiter.is_locked(test_user)
    if is_locked:
        print(f"✓ User is locked for {remaining:.1f} more seconds")
        print(f"Message: {format_lockout_message(remaining)}")
    print()
    
    # Test 4: Try to record another failure while locked
    print("Test 4: Attempt while locked")
    result = limiter.record_failure(test_user)
    print(f"Failed attempts: {result['failed_attempts']}")
    print(f"Still locked: {result['is_locked']}")
    print()
    
    # Test 5: Wait for lockout to expire
    print("Test 5: Waiting for lockout to expire...")
    print("(Sleeping 5 seconds for demo - real lockout is 30 seconds)")
    
    # Temporarily shorten lockout for testing
    limiter.LOCKOUT_DURATION = 5
    limiter.data[test_user]['locked_until'] = time.time() + 5
    limiter._save_data()
    
    time.sleep(5.5)
    
    is_locked, remaining = limiter.is_locked(test_user)
    print(f"After waiting: Locked = {is_locked}")
    print()
    
    # Test 6: Successful authentication resets counter
    print("Test 6: Successful authentication")
    limiter.record_success(test_user)
    status = limiter.get_status(test_user)
    print(f"After success:")
    print(f"  Failed attempts: {status['failed_attempts']}")
    print(f"  Is locked: {status['is_locked']}")
    print(f"  Remaining attempts: {status['remaining_attempts']}")
    print()
    
    # Test 7: Reset user
    print("Test 7: Manual reset")
    limiter.reset_user(test_user)
    status = limiter.get_status(test_user)
    print(f"After reset: {status['failed_attempts']} failed attempts")
    print()
    
    print("=== All Tests Complete ===")
    print("\nIntegration Instructions:")
    print("1. Import: from attempt_limiter import AttemptLimiter, format_lockout_message")
    print("2. Initialize: limiter = AttemptLimiter()")
    print("3. Before verification: is_locked, remaining = limiter.is_locked(user_id)")
    print("4. On failure: limiter.record_failure(user_id)")
    print("5. On success: limiter.record_success(user_id)")
