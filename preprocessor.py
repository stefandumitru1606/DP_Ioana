import numpy as np
import matplotlib.pyplot as plt
import librosa

sample_location = 'dataset/Folk/140.mp3'
orig_sr = 44100
audio_file, sr = librosa.load(sample_location,sr=orig_sr)

target_sr = 22050
resampled_audio_file = librosa.resample(audio_file,orig_sr=sr,target_sr=target_sr)

block_size = 10
block_length = target_sr * block_size

audio_blocks = []

for i in range(0,len(resampled_audio_file),block_length):
    block = resampled_audio_file[i:i+block_length]
    audio_blocks.append(block)

n_fft = 512
hop_length = n_fft // 2

spectrograms = []

for block in audio_blocks:
    stft_complex = librosa.stft(block, n_fft=n_fft, hop_length=hop_length)
    eps = 0.001
    log_stft = np.log(np.abs(stft_complex)+eps)
    spectrogram = np.flipud(log_stft)
    spectrograms.append(spectrogram)

plt.figure(figsize=(8,10))

for i in range(3):  # Plot the first three components
    plt.subplot(3, 1, i + 1)
    img = plt.imshow(spectrograms[i])
    plt.colorbar(img)
    plt.title(f"Spectrogram {i + 1}")

plt.tight_layout()
plt.show()
