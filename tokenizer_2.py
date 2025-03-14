import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AudioTokenizer(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=16, debug=False):
        super().__init__()
        self.encoder = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size
        )
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug
        
    def forward(self, x):
        # Permute to [batch, channels, length] for Conv1d
        x_permuted = x.permute(0, 2, 1).contiguous()
        
        # Only print debug info when debug flag is True
        if self.debug:
            batch_size, seq_len, channels = x.shape
            print(f"Tokenizer input: [batch={batch_size}, len={seq_len}, ch={channels}]")
            print(f"Permuted input shape: {x_permuted.shape}")
            print(f"Conv1d params: in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}")
            print("shape before Conv1d:", x_permuted.shape)
        
        # Apply encoding
        encoded = self.encoder(x_permuted)
        
        # Permute back to [batch, length, channels]
        encoded = encoded.permute(0, 2, 1)
        
        # Ensure output tensor is contiguous
        encoded = encoded.contiguous()
        
        if self.debug:
            print(f"Tokenizer output: {encoded.shape}")
            
        return encoded


class InverseAudioTokenizer(nn.Module):
    def __init__(self, in_channels=64, out_channels=1, original_kernel_size=16, debug=False):
        super().__init__()
        self.decoder = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels*original_kernel_size,
            kernel_size=1,
            stride=1
        )
        self.original_kernel_size = original_kernel_size
        self.debug = debug
        
    def forward(self, x, original_length=None):
        # Ensure input tensor is contiguous
        x = x.contiguous()
        
        # Permute to [batch, channels, length] for Conv1d
        x_permuted = x.permute(0, 2, 1)
        
        # Decode to intermediate representation
        decoded = self.decoder(x_permuted)
        
        # Reshape to get back original audio dimensions
        batch_size, channels, time_steps = decoded.shape
        
        # Reshape: [batch, channels, time] -> [batch, time*16, 1]
        reshaped = decoded.permute(0, 2, 1).reshape(batch_size, time_steps * self.original_kernel_size, 1)
        
        # Ensure output tensor is contiguous
        reshaped = reshaped.contiguous()
        
        # If original_length provided, trim to match
        if original_length is not None:
            reshaped = reshaped[:, :original_length, :]
            
        return reshaped


if __name__ == "__main__":
    # When running as a standalone script, enable debug mode for testing
    # Generate a sample audio signal (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a test signal with two frequencies
    frequency1, frequency2 = 440, 880
    audio = 0.5 * np.sin(2 * np.pi * frequency1 * t) + 0.5 * np.sin(2 * np.pi * frequency2 * t)
    
    # Convert to torch tensor with shape [batch_size, length, channels]
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 16000, 1]
    
    # Initialize tokenizer and inverse tokenizer with debug=True for testing
    tokenizer = AudioTokenizer(in_channels=1, out_channels=512, kernel_size=16, debug=True)
    inverse_tokenizer = InverseAudioTokenizer(in_channels=512, out_channels=1, original_kernel_size=16, debug=True)
    
    # Tokenize and reconstruct
    encoded = tokenizer(audio_tensor)
    print(f"Encoded shape: {encoded.shape}")  # Should be [1, length, channels]
    
    reconstructed = inverse_tokenizer(encoded, original_length=audio_tensor.shape[1])
    print(f"Reconstructed shape: {reconstructed.shape}")  # Should be [1, length, 1]
    
    # Get as numpy arrays for plotting
    original_np = audio_tensor.numpy()[0, :, 0]
    reconstructed_np = reconstructed.numpy()[0, :, 0]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.title("Original Audio")
    plt.plot(original_np)
    
    plt.subplot(3, 1, 2)
    plt.title("Encoded Representation (first 3 channels)")
    for i in range(min(3, encoded.shape[2])):
        plt.plot(encoded[0, :, i].numpy(), label=f"Channel {i}")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title("Reconstructed Audio")
    plt.plot(reconstructed_np)
    
    plt.tight_layout()
    
    print(f"Original audio shape: {audio_tensor.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate reconstruction error
    error = torch.mean((audio_tensor - reconstructed) ** 2).item()
    print(f"Mean squared reconstruction error: {error:.6f}")
    
    plt.savefig('audio_tokenization_results.png')
    plt.show()
