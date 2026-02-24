import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

class AntiSpoofingCNN(nn.Module):
    """
    CNN-based anti-spoofing model for detecting replay and synthetic attacks.
    
    This model analyzes spectral and temporal patterns to distinguish
    between genuine and spoofed audio samples.
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 2):
        super(AntiSpoofingCNN, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input spectrogram tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

class SpectralAnomaly:
    """
    Spectral anomaly detection for anti-spoofing.
    
    Uses statistical analysis of spectral features to detect anomalies
    that may indicate spoofing attacks.
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        self.is_trained = False
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract spectral features for anomaly detection.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Feature vector
        """
        features = []
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.min(spectral_centroids),
            np.max(spectral_centroids)
        ])
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.min(spectral_rolloff),
            np.max(spectral_rolloff)
        ])
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features.extend([
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.min(spectral_bandwidth),
            np.max(spectral_bandwidth)
        ])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr),
            np.min(zcr),
            np.max(zcr)
        ])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([
            np.mean(chroma),
            np.std(chroma),
            np.min(chroma),
            np.max(chroma)
        ])
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend([
            np.mean(spectral_contrast),
            np.std(spectral_contrast),
            np.min(spectral_contrast),
            np.max(spectral_contrast)
        ])
        
        return np.array(features)
    
    def train(self, genuine_audio_samples: list, sr: int = 16000):
        """
        Train anomaly detection models on genuine audio samples.
        
        Args:
            genuine_audio_samples: List of genuine audio arrays
            sr: Sample rate
        """
        # Extract features from genuine samples
        features = []
        for audio in genuine_audio_samples:
            feature_vector = self.extract_spectral_features(audio, sr)
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Train models
        self.isolation_forest.fit(features)
        self.one_class_svm.fit(features)
        
        self.is_trained = True
    
    def predict(self, audio: np.ndarray, sr: int = 16000) -> Tuple[bool, float]:
        """
        Predict if audio is genuine or spoofed.
        
        Args:
            audio: Audio signal to analyze
            sr: Sample rate
            
        Returns:
            Tuple of (is_genuine, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        features = self.extract_spectral_features(audio, sr).reshape(1, -1)
        
        # Get predictions from both models
        iso_pred = self.isolation_forest.predict(features)[0]
        svm_pred = self.one_class_svm.predict(features)[0]
        
        # Get decision scores
        iso_score = self.isolation_forest.decision_function(features)[0]
        svm_score = self.one_class_svm.decision_function(features)[0]
        
        # Combine predictions (both models should agree for high confidence)
        is_genuine = (iso_pred == 1) and (svm_pred == 1)
        
        # Combine confidence scores
        confidence = (iso_score + svm_score) / 2
        confidence = max(0, min(1, (confidence + 1) / 2))  # Normalize to [0, 1]
        
        return is_genuine, confidence

class ReplayDetector:
    """
    Replay attack detection using acoustic environment analysis.
    """
    
    def __init__(self):
        self.reference_noise_profile = None
        self.reference_reverb_profile = None
    
    def extract_acoustic_features(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Extract acoustic environment features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of acoustic features
        """
        features = {}
        
        # Noise floor estimation
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        noise_floor = np.percentile(magnitude, 10)
        features['noise_floor'] = float(noise_floor)
        
        # Reverberation estimation using spectral decay
        # Compute energy decay in different frequency bands
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        
        # Analyze decay in each mel band
        decay_rates = []
        for i in range(mel_spec.shape[0]):
            band_energy = mel_spec[i, :]
            if np.max(band_energy) > 0:
                # Fit exponential decay
                try:
                    log_energy = np.log(band_energy + 1e-10)
                    time_axis = np.arange(len(log_energy))
                    decay_rate = np.polyfit(time_axis, log_energy, 1)[0]
                    decay_rates.append(abs(decay_rate))
                except:
                    decay_rates.append(0)
        
        features['mean_decay_rate'] = float(np.mean(decay_rates))
        features['std_decay_rate'] = float(np.std(decay_rates))
        
        # High-frequency content analysis (often affected by recording quality)
        high_freq_energy = np.sum(magnitude[magnitude.shape[0]//2:, :])
        total_energy = np.sum(magnitude)
        features['high_freq_ratio'] = float(high_freq_energy / (total_energy + 1e-10))
        
        return features
    
    def set_reference_profile(self, reference_audio: np.ndarray, sr: int = 16000):
        """
        Set reference acoustic profile from enrollment audio.
        
        Args:
            reference_audio: Reference audio from enrollment
            sr: Sample rate
        """
        self.reference_noise_profile = self.extract_acoustic_features(reference_audio, sr)
    
    def detect_replay(self, test_audio: np.ndarray, sr: int = 16000) -> Tuple[bool, float]:
        """
        Detect if audio is a replay attack.
        
        Args:
            test_audio: Audio to test
            sr: Sample rate
            
        Returns:
            Tuple of (is_replay, confidence)
        """
        if self.reference_noise_profile is None:
            return False, 0.5  # Cannot determine without reference
        
        test_features = self.extract_acoustic_features(test_audio, sr)
        
        # Compare acoustic features
        differences = []
        for key in self.reference_noise_profile:
            if key in test_features:
                ref_val = self.reference_noise_profile[key]
                test_val = test_features[key]
                if ref_val != 0:
                    diff = abs(test_val - ref_val) / abs(ref_val)
                    differences.append(diff)
        
        if not differences:
            return False, 0.5
        
        # Calculate overall difference
        mean_diff = np.mean(differences)
        
        # Threshold for replay detection (tunable)
        replay_threshold = 0.3
        is_replay = mean_diff > replay_threshold
        
        # Confidence based on how far from threshold
        confidence = min(1.0, mean_diff / replay_threshold)
        
        return is_replay, confidence

class AntiSpoofingSystem:
    """
    Complete anti-spoofing system combining multiple detection methods.
    """
    
    def __init__(self):
        self.cnn_model = AntiSpoofingCNN()
        self.spectral_anomaly = SpectralAnomaly()
        self.replay_detector = ReplayDetector()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model.to(self.device)
    
    def load_cnn_model(self, model_path: str):
        """Load pre-trained CNN model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.eval()
        except Exception as e:
            print(f"Warning: Could not load CNN model: {e}")
    
    def train_spectral_anomaly(self, genuine_samples: list, sr: int = 16000):
        """Train spectral anomaly detector."""
        self.spectral_anomaly.train(genuine_samples, sr)
    
    def set_reference_profile(self, reference_audio: np.ndarray, sr: int = 16000):
        """Set reference profile for replay detection."""
        self.replay_detector.set_reference_profile(reference_audio, sr)
    
    def detect_spoofing(self, audio: np.ndarray, spectrogram: np.ndarray, sr: int = 16000) -> Dict[str, any]:
        """
        Comprehensive spoofing detection.
        
        Args:
            audio: Audio signal
            spectrogram: Spectrogram for CNN analysis
            sr: Sample rate
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'is_genuine': True,
            'confidence': 1.0,
            'cnn_score': 0.5,
            'spectral_anomaly_score': 0.5,
            'replay_score': 0.5,
            'details': {}
        }
        
        try:
            # CNN-based detection
            if spectrogram is not None:
                with torch.no_grad():
                    # Prepare input tensor
                    if len(spectrogram.shape) == 2:
                        spectrogram = spectrogram[np.newaxis, np.newaxis, :, :]
                    elif len(spectrogram.shape) == 3:
                        spectrogram = spectrogram[np.newaxis, :, :, :]
                    
                    input_tensor = torch.FloatTensor(spectrogram).to(self.device)
                    
                    # Get prediction
                    logits = self.cnn_model(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    
                    # Assuming class 0 is genuine, class 1 is spoofed
                    genuine_prob = probabilities[0, 0].item()
                    results['cnn_score'] = genuine_prob
                    results['details']['cnn_genuine_prob'] = genuine_prob
            
            # Spectral anomaly detection
            if self.spectral_anomaly.is_trained:
                is_genuine_spectral, spectral_confidence = self.spectral_anomaly.predict(audio, sr)
                results['spectral_anomaly_score'] = spectral_confidence if is_genuine_spectral else 1 - spectral_confidence
                results['details']['spectral_anomaly_genuine'] = is_genuine_spectral
                results['details']['spectral_anomaly_confidence'] = spectral_confidence
            
            # Replay detection
            is_replay, replay_confidence = self.replay_detector.detect_replay(audio, sr)
            results['replay_score'] = 1 - replay_confidence if is_replay else replay_confidence
            results['details']['is_replay'] = is_replay
            results['details']['replay_confidence'] = replay_confidence
            
            # Combine scores
            scores = [
                results['cnn_score'],
                results['spectral_anomaly_score'],
                results['replay_score']
            ]
            
            # Weighted average (can be tuned)
            weights = [0.5, 0.3, 0.2]
            combined_score = sum(w * s for w, s in zip(weights, scores))
            
            results['confidence'] = combined_score
            results['is_genuine'] = combined_score > 0.5
            
        except Exception as e:
            print(f"Error in spoofing detection: {e}")
            results['error'] = str(e)
        
        return results

