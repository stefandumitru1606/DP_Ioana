import librosa
import os
import numpy as np
import matplotlib.pyplot as plt


audio_path = 'dataset/Folk/194.mp3'
image_path = 'test_stft'


orig_sr = 44100
target_sr = 22050

audio_file, _ = librosa.load(audio_path, sr=orig_sr)
resampled_audio_file = librosa.resample(audio_file, orig_sr=orig_sr, target_sr=target_sr)

n_fft = 512
hop_length = n_fft // 2
stft_complex = librosa.stft(resampled_audio_file, n_fft=n_fft, hop_length=hop_length)   # complex representation
eps = 0.001
log_stft = np.log(np.abs(stft_complex) + eps)   # logarithm representation
spectrogram = np.flipud(log_stft)   # flipped log - should be used as input

print(spectrogram.shape)
# plt.figure()
# plt.imshow(spectrogram)
# plt.axis('off')
# plt.show()

spectrogram_resized = spectrogram[1:, :]    # crop to be 128 pixels tall

width = 256
dpi_val = 69.4  # 34.8 for 128x128

for i in range(10):     # split the main spectrogram into 10 spectrogram representations (128x128) for CNN input
    start_col = i * width
    end_col = start_col + width
    segment = spectrogram_resized[:, start_col:end_col]

    plt.figure()
    plt.imshow(segment)
    plt.axis('off')
    plt.savefig(f"{image_path}/{os.path.splitext(os.path.basename(audio_path))[0]}_{i}.png", bbox_inches='tight', pad_inches=0,dpi=dpi_val)
    plt.close()

