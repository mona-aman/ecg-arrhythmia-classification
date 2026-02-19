"""
Multi-Representation ECG to 2D Converter
Supports various time-frequency transforms for ECG signals
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pywt
from scipy import signal
from typing import Optional, Tuple, Dict, Any
import warnings


class ECGTo2DConverter(nn.Module):
    """
    Convert ECG signals to 2D representations using various methods
    Supports: STFT, Mel, CWT, S-Transform, and more
    """
    
    def __init__(self,
                 method: str = 'mel',
                 seq_length: int = 2500,
                 sample_rate: int = 250,
                 normalize: bool = True,
                 **kwargs):
        super().__init__()
        
        self.method = method.lower()
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Initialize the specific transform
        if self.method == 'stft':
            self.transform = STFTTransform(sample_rate=sample_rate, **kwargs)
        elif self.method == 'mel':
            self.transform = MelSpectrogramTransform(sample_rate=sample_rate, **kwargs)
        elif self.method == 'cwt':
            self.transform = CWTTransform(sample_rate=sample_rate, **kwargs)
        elif self.method == 'stransform':
            self.transform = STransformTransform(sample_rate=sample_rate, **kwargs)
        elif self.method == 'mfcc':
            self.transform = MFCCTransform(sample_rate=sample_rate, **kwargs)
        elif self.method == 'chroma':
            self.transform = ChromaTransform(sample_rate=sample_rate, **kwargs)
        elif self.method == 'spectral_centroid':
            self.transform = SpectralCentroidTransform(sample_rate=sample_rate, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def forward(self, x):
        """
        Convert ECG signals to 2D representations
        
        Args:
            x: (batch_size, n_leads, seq_length)
        Returns:
            (batch_size, n_leads, height, width)
        """
        batch_size, n_leads, seq_length = x.shape
        
        # Process each lead separately
        representations = []
        for lead_idx in range(n_leads):
            lead_signal = x[:, lead_idx, :]  # (batch_size, seq_length)
            
            # Apply transform
            lead_repr = self.transform(lead_signal)  # (batch_size, height, width)
            representations.append(lead_repr)
        
        # Stack all leads: (batch_size, n_leads, height, width)
        representations = torch.stack(representations, dim=1)
        
        # Normalize if requested
        if self.normalize:
            representations = self._normalize_representations(representations)
        
        return representations
    
    def _normalize_representations(self, representations):
        """Normalize representations to [0, 1] range"""
        batch_size, n_leads, height, width = representations.shape
        
        # Reshape for normalization
        repr_flat = representations.view(batch_size, n_leads, -1)
        
        # Min-max normalization per lead per sample
        repr_min = repr_flat.min(dim=2, keepdim=True)[0]  # (batch_size, n_leads, 1)
        repr_max = repr_flat.max(dim=2, keepdim=True)[0]  # (batch_size, n_leads, 1)
        
        # Avoid division by zero
        repr_range = repr_max - repr_min
        repr_range = torch.where(repr_range > 1e-8, repr_range, torch.ones_like(repr_range))
        
        repr_normalized = (repr_flat - repr_min) / repr_range
        
        # Reshape back
        return repr_normalized.view(batch_size, n_leads, height, width)
    
    def get_output_shape(self):
        """Get output shape for a single sample"""
        dummy_input = torch.randn(1, 1, self.seq_length)
        with torch.no_grad():
            output = self.transform(dummy_input)
        return output.shape[1:]  # (height, width)


class STFTTransform(nn.Module):
    """Short-Time Fourier Transform"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 window: str = 'hann',
                 **kwargs):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        
        # STFT transform
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            window_fn=torch.hann_window if window == 'hann' else torch.hamming_window
        )
        
        # Convert to dB scale
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, freq_bins, time_frames)
        """
        # Use PyTorch's native STFT implementation on the input device (GPU when
        # available). This avoids torchaudio's internal backend kernels that can
        # trigger CUDA driver errors on some cluster builds and is much faster
        # than moving large batches to CPU.
        device = x.device

        # Prepare window on the same device
        if self.window == 'hann':
            window = torch.hann_window(self.n_fft, device=device)
        else:
            window = torch.hamming_window(self.n_fft, device=device)

        # torch.stft expects (batch, seq) or (seq,) input; our x is (batch, seq)
        # Return complex tensor and compute power spectrogram
        complex_spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )

        # complex_spec: (batch, freq_bins, time_frames) complex dtype
        # Convert to power (magnitude squared)
        spec_power = complex_spec.abs().pow(2.0)

        # Convert to dB: 10 * log10(power + eps)
        eps = 1e-10
        spec_db = 10.0 * torch.log10(spec_power + eps)

        return spec_db


class MelSpectrogramTransform(nn.Module):
    """Mel Spectrogram Transform"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 n_mels: int = 128,
                 f_min: float = 0.5,
                 f_max: float = 40.0,
                 **kwargs):
        super().__init__()
        
        # Mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0
        )
        
        # Convert to dB scale with proper handling of zeros
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, n_mels, time_frames)
        """
        # Run mel spectrogram on CPU to avoid potential CUDA/driver issues
        device = x.device
        x_cpu = x.detach().to('cpu').float()

        mel_spec = self.mel_spectrogram(x_cpu)
        mel_spec = mel_spec + 1e-9
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = torch.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)

        return mel_spec_db.to(device)


class CWTTransform(nn.Module):
    """Continuous Wavelet Transform (Scalogram)"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 wavelet: str = 'morl',
                 scales_min: float = 1,
                 scales_max: float = 128,
                 n_scales: int = 128,
                 **kwargs):
        super().__init__()
        
        self.wavelet = wavelet
        self.sample_rate = sample_rate
        
        # Generate scales - logarithmically spaced
        self.scales = np.logspace(np.log10(scales_min), np.log10(scales_max), n_scales)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, n_scales, seq_length)
        """
        batch_size, seq_length = x.shape
        device = x.device
        
        # Convert to numpy for pywt
        x_np = x.cpu().numpy()
        
        scalograms = []
        
        for i in range(batch_size):
            signal_i = x_np[i]
            
            # Compute CWT
            coefficients, _ = pywt.cwt(signal_i, self.scales, self.wavelet)
            
            # Take absolute value (magnitude)
            scalogram = np.abs(coefficients)
            
            scalograms.append(scalogram)
        
        # Convert back to torch tensor
        scalograms = np.stack(scalograms, axis=0)
        scalograms = torch.from_numpy(scalograms).float().to(device)
        
        return scalograms


class STransformTransform(nn.Module):
    """S-Transform (Stockwell Transform)"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 f_min: float = 0.5,
                 f_max: float = 40.0,
                 n_freq: int = 128,
                 **kwargs):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.n_freq = n_freq
        
        # Generate frequency array
        self.freqs = np.linspace(f_min, f_max, n_freq)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, n_freq, seq_length)
        """
        batch_size, seq_length = x.shape
        device = x.device
        
        # Convert to numpy
        x_np = x.cpu().numpy()
        
        stransforms = []
        
        for i in range(batch_size):
            signal_i = x_np[i]
            
            # Compute S-transform (simplified version using STFT with varying window)
            st_result = self._compute_s_transform(signal_i)
            
            stransforms.append(st_result)
        
        # Convert back to torch tensor
        stransforms = np.stack(stransforms, axis=0)
        stransforms = torch.from_numpy(stransforms).float().to(device)
        
        return stransforms
    
    def _compute_s_transform(self, signal):
        """Simplified S-transform implementation"""
        n_samples = len(signal)
        st_result = np.zeros((self.n_freq, n_samples))
        
        for i, freq in enumerate(self.freqs):
            if freq == 0:
                continue
            
            # Window width inversely proportional to frequency
            window_width = int(self.sample_rate / freq)
            window_width = max(window_width, 3)  # Minimum window size
            
            # Apply windowed FFT
            for t in range(n_samples):
                start = max(0, t - window_width // 2)
                end = min(n_samples, t + window_width // 2)
                
                if end - start < 3:
                    continue
                
                windowed_signal = signal[start:end]
                
                # Apply Gaussian window
                window = signal.windows.gaussian(len(windowed_signal), std=window_width/6)
                windowed_signal = windowed_signal * window
                
                # FFT
                fft_result = np.fft.fft(windowed_signal)
                freq_bin = int(freq * len(windowed_signal) / self.sample_rate)
                
                if freq_bin < len(fft_result):
                    st_result[i, t] = np.abs(fft_result[freq_bin])
        
        return st_result


class MFCCTransform(nn.Module):
    """Mel-Frequency Cepstral Coefficients"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 n_mfcc: int = 40,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 n_mels: int = 128,
                 **kwargs):
        super().__init__()
        
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, n_mfcc, time_frames)
        """
        # Run MFCC on CPU to avoid CUDA/driver issues in some environments
        device = x.device
        x_cpu = x.detach().to('cpu').float()
        mfcc_cpu = self.mfcc(x_cpu)
        return mfcc_cpu.to(device)


class ChromaTransform(nn.Module):
    """Chroma Features (adapted for ECG frequency range)"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 n_chroma: int = 12,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 **kwargs):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Use STFT as base
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, n_chroma, time_frames)
        """
        # Get spectrogram (compute on CPU to avoid CUDA driver issues)
        device = x.device
        x_cpu = x.detach().to('cpu').float()
        spec = self.stft(x_cpu)  # (batch_size, freq_bins, time_frames)

        batch_size, freq_bins, time_frames = spec.shape

        # Simple chroma mapping (divide frequency bins into chroma bins)
        bins_per_chroma = freq_bins // self.n_chroma

        chroma_features = []
        for i in range(self.n_chroma):
            start_bin = i * bins_per_chroma
            end_bin = min((i + 1) * bins_per_chroma, freq_bins)

            chroma_bin = spec[:, start_bin:end_bin, :].sum(dim=1, keepdim=True)
            chroma_features.append(chroma_bin)

        chroma = torch.cat(chroma_features, dim=1)

        return chroma.to(device)


class SpectralCentroidTransform(nn.Module):
    """Spectral Centroid and other spectral features"""
    
    def __init__(self,
                 sample_rate: int = 250,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 n_features: int = 32,
                 **kwargs):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_features = n_features
        
        # Base spectrogram
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length)
        Returns:
            (batch_size, n_features, time_frames)
        """
        # Get spectrogram (compute on CPU to avoid CUDA driver issues)
        device = x.device
        x_cpu = x.detach().to('cpu').float()
        spec = self.stft(x_cpu)  # (batch_size, freq_bins, time_frames)

        batch_size, freq_bins, time_frames = spec.shape

        # Frequency array on the same device as spec (CPU)
        freqs = torch.linspace(0, self.sample_rate / 2, freq_bins, device=spec.device)
        freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, freq_bins, 1)

        # Spectral centroid
        spectral_centroid = (spec * freqs).sum(dim=1, keepdim=True) / (spec.sum(dim=1, keepdim=True) + 1e-8)

        # Spectral rolloff (simplified)
        cumsum_spec = torch.cumsum(spec, dim=1)
        total_energy = spec.sum(dim=1, keepdim=True)
        rolloff_threshold = 0.85 * total_energy

        # Find rolloff frequency
        rolloff_indices = (cumsum_spec > rolloff_threshold).float().argmax(dim=1, keepdim=True)
        spectral_rolloff = freqs.squeeze()[rolloff_indices.squeeze()].unsqueeze(1)

        # Spectral bandwidth (simplified)
        mean_freq = spectral_centroid
        bandwidth = torch.sqrt(((freqs - mean_freq) ** 2 * spec).sum(dim=1, keepdim=True) / (spec.sum(dim=1, keepdim=True) + 1e-8))

        # Combine features and repeat to reach n_features
        features = torch.cat([spectral_centroid, spectral_rolloff, bandwidth], dim=1)

        # Repeat or pad to reach desired number of features
        while features.shape[1] < self.n_features:
            features = torch.cat([features, features], dim=1)

        # Truncate if necessary
        features = features[:, :self.n_features, :]

        return features.to(device)


def get_available_methods():
    """Get list of available 2D representation methods"""
    return [
        'stft',           # Short-Time Fourier Transform
        'mel',            # Mel Spectrogram
        'cwt',            # Continuous Wavelet Transform
        'stransform',     # S-Transform
        'mfcc',           # Mel-Frequency Cepstral Coefficients
        'chroma',         # Chroma Features
        'spectral_centroid'  # Spectral Features
    ]


def get_method_info():
    """Get information about each method"""
    return {
        'stft': {
            'name': 'Short-Time Fourier Transform',
            'description': 'Classic time-frequency decomposition',
            'output_type': 'Magnitude spectrogram',
            'computational_cost': 'Low'
        },
        'mel': {
            'name': 'Mel Spectrogram',
            'description': 'Perceptually motivated frequency scale',
            'output_type': 'Mel-scaled spectrogram',
            'computational_cost': 'Low'
        },
        'cwt': {
            'name': 'Continuous Wavelet Transform',
            'description': 'Multi-resolution time-frequency analysis',
            'output_type': 'Scalogram',
            'computational_cost': 'High'
        },
        'stransform': {
            'name': 'S-Transform',
            'description': 'Frequency-dependent resolution',
            'output_type': 'S-transform magnitude',
            'computational_cost': 'High'
        },
        'mfcc': {
            'name': 'Mel-Frequency Cepstral Coefficients',
            'description': 'Compact spectral representation',
            'output_type': 'Cepstral coefficients',
            'computational_cost': 'Medium'
        },
        'chroma': {
            'name': 'Chroma Features',
            'description': 'Pitch class profiles',
            'output_type': 'Chroma vectors',
            'computational_cost': 'Low'
        },
        'spectral_centroid': {
            'name': 'Spectral Features',
            'description': 'Statistical spectral descriptors',
            'output_type': 'Feature vectors',
            'computational_cost': 'Low'
        }
    }


if __name__ == "__main__":
    # Test all methods
    print("Testing ECG to 2D conversion methods")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    n_leads = 12
    seq_length = 2500
    sample_rate = 250
    
    x = torch.randn(batch_size, n_leads, seq_length)
    print(f"Input shape: {x.shape}")
    print()
    
    methods = get_available_methods()
    method_info = get_method_info()
    
    for method in methods:
        print(f"Testing {method.upper()}:")
        print(f"  Name: {method_info[method]['name']}")
        print(f"  Description: {method_info[method]['description']}")
        
        try:
            converter = ECGTo2DConverter(method=method, sample_rate=sample_rate)
            output = converter(x)
            
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ 2D representation: {output.shape[2]} x {output.shape[3]}")
            print(f"  ✓ Computational cost: {method_info[method]['computational_cost']}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()