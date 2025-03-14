import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class AudioFeatureExtractor(nn.Module):
    """Extract features from raw audio using STFT and convolutional layers"""
    def __init__(self, 
                 n_fft=1024, 
                 hop_length=256, 
                 win_length=1024,
                 output_dim=512,
                 n_mels=128,
                 sample_rate=16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        # STFT parameters
        self.register_buffer('window', torch.hann_window(win_length))
        
        # Mel spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, kernel_size=3, padding=1),
        )
        
    def calculate_output_frames(self, input_length):
        """Calculate the number of time frames after STFT"""
        return math.ceil((input_length - self.win_length) / self.hop_length) + 1
        
    def calculate_required_input_length(self, desired_output_frames):
        """Calculate input length needed to produce desired output frames"""
        return (desired_output_frames - 1) * self.hop_length + self.win_length
    
    def forward(self, audio):
        """
        Args:
            audio: Raw audio waveform (batch_size, time_steps, 1)
        Returns:
            features: Extracted features (batch_size, new_time_steps, output_dim)
        """
        # Remove channel dimension and ensure correct shape
        batch_size = audio.shape[0]
        original_length = audio.shape[1]
        audio = audio.squeeze(-1)  # (batch_size, time_steps)
        
        # Store original length for reconstruction
        self.original_length = original_length
        
        # Calculate expected number of frames after STFT
        expected_frames = self.calculate_output_frames(original_length)
        
        # Apply STFT to get spectrogram
        spec = self.mel_spec(audio)  # (batch_size, n_mels, time)
        actual_frames = spec.shape[2]
        
        # Log mel spectrogram for better feature representation
        log_spec = torch.log(spec + 1e-5)
        
        # Apply convolutional feature extractor
        features = self.conv_layers(log_spec)  # (batch_size, output_dim, time)
        
        # Transpose to get (batch_size, time, output_dim)
        features = features.transpose(1, 2)
        
        return features


class AudioReconstructor(nn.Module):
    """Reconstruct audio from features"""
    def __init__(self, 
                 input_dim=512,
                 n_fft=1024, 
                 hop_length=256, 
                 win_length=1024,
                 n_mels=128,
                 sample_rate=16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        # Convolutional decoder to go from features to mel spectrogram
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, n_mels, kernel_size=3, padding=1),
        )
        
        # MelScale for conversion from mel to linear spectrogram
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft//2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        # Griffin-Lim for phase reconstruction
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0
        )
        
    def calculate_expected_audio_length(self, num_frames):
        """Calculate expected audio length after inverse STFT"""
        return (num_frames - 1) * self.hop_length + self.win_length
    
    def forward(self, features, original_length=None):
        """
        Args:
            features: Model output features (batch_size, time_steps, input_dim)
            original_length: Original audio length to match (optional)
        Returns:
            audio: Reconstructed audio (batch_size, time_steps, 1)
        """
        # Transpose to channel-first format for convolution
        features = features.transpose(1, 2)  # (batch_size, input_dim, time)
        
        # Convert features back to mel spectrogram
        mel_spec = self.conv_layers(features)  # (batch_size, n_mels, time)
        
        # Store the number of time frames
        num_frames = mel_spec.shape[2]
        
        # Convert from log to linear scale
        mel_spec = torch.exp(mel_spec) - 1e-5
        mel_spec = torch.clamp(mel_spec, min=0)
        
        # Process each item in batch
        batch_size = features.shape[0]
        audio_batch = []
        
        for i in range(batch_size):
            # Convert mel spectrogram to linear spectrogram
            linear_spec = self.inverse_mel(mel_spec[i])
            
            # Reconstruct audio using Griffin-Lim
            audio = self.griffin_lim(linear_spec)
            
            # Add to batch
            audio_batch.append(audio.unsqueeze(0))
        
        # Combine batch and add channel dimension
        audio = torch.cat(audio_batch, dim=0).unsqueeze(-1)
        
        # If original length is provided and different from reconstruction length,
        # perform smart trimming or padding
        if original_length is not None and audio.shape[1] != original_length:
            # Calculate the expected audio length based on STFT parameters
            expected_length = self.calculate_expected_audio_length(num_frames)
            
            if audio.shape[1] < original_length:
                # Pad if too short (typically at the end due to STFT boundary effects)
                padding = original_length - audio.shape[1]
                # Use reflection padding to avoid discontinuities
                audio = F.pad(audio.squeeze(-1), (0, padding), mode='reflect').unsqueeze(-1)
            else:
                # Trim if too long, taking center portion to preserve important content
                excess = audio.shape[1] - original_length
                start = excess // 2
                audio = audio[:, start:start+original_length, :]
        
        return audio


class AudioProcessingTokenizer(nn.Module):
    """Full audio processing pipeline with tokenization"""
    def __init__(self, 
                 feature_dim=512,
                 num_tokens=1024,
                 n_fft=1024, 
                 hop_length=256,
                 n_mels=128,
                 sample_rate=16000):
        super().__init__()
        
        # Feature extraction from raw audio
        self.feature_extractor = AudioFeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            output_dim=feature_dim,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        # Discrete tokenization
        self.tokenizer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_tokens)
        )
        self.embedding = nn.Embedding(num_tokens, feature_dim)
        
        # Audio reconstruction
        self.reconstructor = AudioReconstructor(
            input_dim=feature_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        self.original_audio_length = None
    
    def forward(self, audio, return_tokens=False):
        """
        Args:
            audio: Raw audio (batch_size, time_steps, 1)
            return_tokens: Whether to return discrete tokens
        Returns:
            embeddings: Tokenized features
            tokens: Discrete tokens (if return_tokens=True)
        """
        # Store original audio length for reconstruction
        self.original_audio_length = audio.shape[1]
        
        # Extract features
        features = self.feature_extractor(audio)
        
        # Tokenize features
        logits = self.tokenizer(features)
        tokens = torch.argmax(logits, dim=-1)
        embeddings = self.embedding(tokens)
        
        if return_tokens:
            return embeddings, tokens
        return embeddings
    
    def reconstruct(self, embeddings):
        """
        Reconstruct audio from embeddings
        Args:
            embeddings: Tokenized features from model output
        Returns:
            audio: Reconstructed audio with original length
        """
        return self.reconstructor(embeddings, original_length=self.original_audio_length)


if __name__ == "__main__":
    # Create examples to demonstrate the audio tokenizer
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    from sklearn.metrics import mean_squared_error
    
    # Generate test audio for demonstration (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)
    
    def generate_test_audio(frequency, duration=1.0, sample_rate=16000):
        """Generate a test tone with the given frequency"""
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * frequency * t)
        # Add harmonics for a more complex signal
        audio += 0.3 * torch.sin(2 * torch.pi * frequency * 2 * t)
        audio += 0.1 * torch.sin(2 * torch.pi * frequency * 3 * t)
        # Normalize
        audio = audio / torch.max(torch.abs(audio))
        return audio.unsqueeze(0).unsqueeze(-1)  # [1, samples, 1]
    
    # Create two test signals
    audio1 = generate_test_audio(440)  # A4 note
    audio2 = generate_test_audio(523.25)  # C5 note
    
    # Initialize the audio tokenizer with careful parameter selection
    # Choose parameters that minimize artifacts in the reconstruction
    n_fft = 1024
    hop_length = n_fft // 4  # 75% overlap for better reconstruction
    tokenizer = AudioProcessingTokenizer(
        feature_dim=512,
        num_tokens=1024,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=sample_rate
    )
    
    def process_and_evaluate(audio, name="tone"):
        """Process audio through tokenizer and evaluate reconstruction quality"""
        print(f"\nProcessing {name} (shape: {audio.shape})...")
        
        with torch.no_grad():
            # Get embeddings and tokens
            embeddings, tokens = tokenizer(audio, return_tokens=True)
            print(f"Generated tokens shape: {tokens.shape}")
            
            # Get token statistics
            unique_tokens = torch.unique(tokens)
            print(f"Unique tokens used: {len(unique_tokens)} out of 1024")
            
            # Reconstruct audio from embeddings
            reconstructed_audio = tokenizer.reconstruct(embeddings)
            print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
            
            # Ensure same length for comparison (handle STFT frame boundary effects)
            min_length = min(audio.shape[1], reconstructed_audio.shape[1])
            
            # Use consistent shapes for comparison
            audio_trim = audio[:, :min_length, :]
            recon_trim = reconstructed_audio[:, :min_length, :]
            
            # Calculate error metrics
            mse = mean_squared_error(
                audio_trim.squeeze().numpy(), 
                recon_trim.squeeze().numpy()
            )
            print(f"MSE between original and reconstruction: {mse:.6f}")
            
            # Calculate SNR (Signal-to-Noise Ratio)
            signal_power = np.mean(np.square(audio_trim.squeeze().numpy()))
            noise_power = mse
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            print(f"SNR: {snr:.2f} dB")
            
        # Plot time domain comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.title(f"Original Audio - {name}")
        plt.plot(audio_trim.squeeze().numpy())
        
        plt.subplot(3, 1, 2)
        plt.title(f"Reconstructed Audio - {name}")
        plt.plot(recon_trim.squeeze().numpy())
        
        plt.subplot(3, 1, 3)
        plt.title(f"Error Signal (Original - Reconstructed)")
        plt.plot(audio_trim.squeeze().numpy() - recon_trim.squeeze().numpy())
        
        plt.tight_layout()
        plt.savefig(f"audio_tokenization_{name}_time.png")
        plt.close()
        
        # Plot frequency domain comparison
        plt.figure(figsize=(15, 10))
        
        # Original audio spectrogram
        plt.subplot(2, 1, 1)
        D_orig = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_trim.squeeze().numpy(), n_fft=2048)), 
            ref=np.max
        )
        plt.title(f"Original Spectrogram - {name}")
        librosa.display.specshow(D_orig, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        # Reconstructed audio spectrogram
        plt.subplot(2, 1, 2)
        D_recon = librosa.amplitude_to_db(
            np.abs(librosa.stft(recon_trim.squeeze().numpy(), n_fft=2048)), 
            ref=np.max
        )
        plt.title(f"Reconstructed Spectrogram - {name}")
        librosa.display.specshow(D_recon, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(f"audio_tokenization_{name}_freq.png")
        plt.close()
        
        return embeddings, tokens, recon_trim
    
    # Process both test signals
    _, tokens1, recon1 = process_and_evaluate(audio1, "A4_note")
    _, tokens2, recon2 = process_and_evaluate(audio2, "C5_note")
    
    # Compare token distributions
    print("\nAnalyzing token distributions...")
    token_set1 = set(tokens1.flatten().tolist())
    token_set2 = set(tokens2.flatten().tolist())
    common_tokens = token_set1.intersection(token_set2)
    print(f"Common tokens between A4 and C5: {len(common_tokens)}")
    print(f"Unique tokens for A4: {len(token_set1 - token_set2)}")
    print(f"Unique tokens for C5: {len(token_set2 - token_set1)}")
    
    print("\nDemo complete! Saved comparison plots in current directory.")
