import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizer_2 import AudioTokenizer, InverseAudioTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Using GELU for better performance
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        # Using batch_first=True, so no need to transpose
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward with residual connection and layer norm
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class AudioDenoiserTransformer(nn.Module):
    def __init__(self, 
                 input_dim=128,
                 d_model=512, 
                 num_heads=8, 
                 num_layers=12, 
                 d_ff=2048, 
                 num_tokens=1024,
                 dropout=0.1,
                 max_len=5000,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=128,
                 sample_rate=16000):
        super().__init__()
        
        # Model dimensions for easy scaling
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sample_rate = sample_rate
        
        # Audio processing and tokenization front-end using the new tokenizer
        self.audio_tokenizer = AudioTokenizer(
            in_channels=1,
            out_channels=d_model,
            kernel_size=16,
            debug=False,  # Disable debug prints during training
        )
        
        # Add inverse tokenizer for audio reconstruction
        self.inverse_tokenizer = InverseAudioTokenizer(
            in_channels=d_model,
            out_channels=1,
            original_kernel_size=16,
            debug=False,  # Disable debug prints during training
        )
        
        # Input processing
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Causal mask for autoregressive generation
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        
    def forward(self, x, generate=False, max_new_tokens=None, return_embeddings=False):
        # print(f"Input shape: {x.shape}")
        # x shape should be [batch_size, length, channels]
        batch_size, seq_len, channels = x.shape
        
        # Ensure the input tensor is contiguous before tokenization
        x = x.contiguous()
        
        # First tokenize the raw audio input - tokenizer now handles permutation internally
        x_tokenized = self.audio_tokenizer(x)
        # print(f"After tokenization shape: {x_tokenized.shape}")
        
        # Ensure tensor is contiguous to avoid stride issues
        x_tokenized = x_tokenized.contiguous()
        
        # x_tokenized is now [batch, seq_len/stride, d_model]
        
        if not generate:
            # Standard training/inference pass
            x = self.input_proj(x_tokenized)
            x = self.pos_encoder(x)
            x = self.dropout(x)
            
            # Create causal mask for the current sequence length
            curr_seq_len = x.size(1)
            mask = self.causal_mask[:curr_seq_len, :curr_seq_len]
            
            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, mask=mask)
                
            # Project back to embedding dimension
            embeddings = self.output_proj(x)
            
            # Return embeddings if requested (useful for training specific components)
            if return_embeddings:
                return embeddings
                
            # Otherwise, detokenize back to audio using the inverse tokenizer
            denoised_audio = self.inverse_tokenizer(
                embeddings, 
                original_length=seq_len
            )
            
            return denoised_audio
        else:
            # Autoregressive generation
            generated = x_tokenized
            
            for _ in range(max_new_tokens if max_new_tokens else seq_len):
                # Process the sequence so far
                current_len = generated.size(1)
                x = self.input_proj(generated)
                x = self.pos_encoder(x)
                
                # Apply transformer blocks with appropriate masking
                mask = self.causal_mask[:current_len, :current_len]
                for block in self.transformer_blocks:
                    x = block(x, mask=mask)
                
                # Get next token prediction (only the last position)
                next_token_logits = self.output_proj(x[:, -1:, :])
                
                # Append the predicted token to the sequence
                generated = torch.cat([generated, next_token_logits], dim=1)
            
            # Detokenize the generated sequence
            if return_embeddings:
                return generated
                
            denoised_audio = self.inverse_tokenizer(
                generated,
                original_length=seq_len
            )
            
            return denoised_audio
    
    def denoise_audio(self, noisy_audio):
        """
        Denoise audio in one pass or autoregressively - now simplified to use forward
        """
        # Process the entire noisy audio through the model
        with torch.no_grad():
            # Ensure input has shape [batch_size, length, channels]
            if len(noisy_audio.shape) == 1:  # [length]
                noisy_audio = noisy_audio.unsqueeze(0).unsqueeze(-1)  # [1, length, 1]
            elif len(noisy_audio.shape) == 2:
                if noisy_audio.shape[1] == 1:  # [length, channels]
                    noisy_audio = noisy_audio.unsqueeze(0)  # [1, length, channels]
                else:  # [batch, length]
                    noisy_audio = noisy_audio.unsqueeze(-1)  # [batch, length, 1]
            elif len(noisy_audio.shape) == 3 and noisy_audio.shape[2] != 1:
                # If shape is [batch, channels, length], permute to [batch, length, channels]
                if noisy_audio.shape[1] != noisy_audio.shape[2]:  # Only if clearly [batch, channels, length]
                    noisy_audio = noisy_audio.permute(0, 2, 1)
            
            # Simply call forward - it now handles the complete pipeline
            return self(noisy_audio)
            
    @staticmethod
    def get_model_size(d_model, num_layers, num_heads, d_ff, input_dim, num_tokens):
        """
        Calculate approximate number of parameters in the model
        """
        # Input projection parameters
        params = input_dim * d_model + d_model
        
        # Embedding table if using tokens
        params += num_tokens * d_model
        
        # Each transformer block
        params_per_block = (
            # Self-attention
            3 * d_model * d_model + d_model +  # Q, K, V projections and bias
            d_model * d_model + d_model +      # Output projection and bias
            
            # Layer norms
            2 * (2 * d_model) +               # Gain and bias for each LN
            
            # Feed-forward
            d_model * d_ff + d_ff +           # First linear layer
            d_ff * d_model + d_model          # Second linear layer
        )
        
        params += num_layers * params_per_block
        
        # Output projection
        params += d_model * input_dim + input_dim
        
        return params


def create_audio_denoiser(
    model_size='base',  # 'small', 'base', 'large', 'xl'
    input_dim=128,      # Input feature dimension (e.g., mel spectrogram bins)
    sample_rate=16000,  # Add default sample_rate parameter
    custom_config=None  # For custom configurations
):
    """
    Create an audio denoiser model with different size configurations
    """
    if custom_config:
        # If custom config is provided, also include sample_rate unless it's already there
        if 'sample_rate' not in custom_config:
            custom_config['sample_rate'] = sample_rate
        return AudioDenoiserTransformer(**custom_config)
    
    model_configs = {
        'small': {
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 6,
            'd_ff': 1024,
            'num_tokens': 512,
            'dropout': 0.1
        },
        'base': {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 12,
            'd_ff': 2048,
            'num_tokens': 1024,
            'dropout': 0.1
        },
        'large': {
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 24,
            'd_ff': 3072,
            'num_tokens': 2048,
            'dropout': 0.1
        },
        'xl': {
            'd_model': 1024,
            'num_heads': 16,
            'num_layers': 36,
            'd_ff': 4096,
            'num_tokens': 4096,
            'dropout': 0.1
        }
    }
    
    config = model_configs[model_size]
    config['input_dim'] = input_dim
    config['sample_rate'] = sample_rate  # Add sample_rate to config
    
    model = AudioDenoiserTransformer(**config)
    
    # Print approximate model size
    num_params = AudioDenoiserTransformer.get_model_size(
        config['d_model'], config['num_layers'], 
        config['num_heads'], config['d_ff'],
        input_dim, config['num_tokens']
    )
    print(f"Created {model_size} model with approximately {num_params/1e6:.2f}M parameters")
    
    return model

# Example usage:
if __name__ == "__main__":
    # Create a small model for testing
    model = create_audio_denoiser('small', input_dim=80, sample_rate=16000)  # Explicitly pass sample_rate
    
    # Generate random noisy audio [batch_size, length, channels]
    noisy_audio = torch.randn(1, 4000, 1)  # 1 second of audio at 16kHz
    
    # Denoise the audio
    denoised_audio = model.denoise_audio(noisy_audio)
    
    print(f"Input shape: {noisy_audio.shape}")
    print(f"Output shape: {denoised_audio.shape}")
