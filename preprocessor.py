import os
import librosa
import numpy as np
import matplotlib.pyplot as plt


def process_spectrograms(audio_path, image_path):

    orig_sr = 44100
    target_sr = 22050

    audio_file, _ = librosa.load(audio_path, sr=orig_sr)
    resampled_audio_file = librosa.resample(audio_file, orig_sr=orig_sr, target_sr=target_sr)

    n_fft = 256  # 512 for 256x256
    hop_length = n_fft * 2  # // 2 for 256x256
    stft_complex = librosa.stft(resampled_audio_file, n_fft=n_fft, hop_length=hop_length)  # complex representation
    eps = 0.001
    log_stft = np.log(np.abs(stft_complex) + eps)  # logarithm representation
    spectrogram = np.flipud(log_stft)  # flipped log - should be used as input

    # print(spectrogram.shape)
    # plt.figure()
    # plt.imshow(spectrogram)
    # plt.axis('off')
    # plt.show()

    spectrogram_resized = spectrogram[1:, :]  # crop to be 2^M pixels tall (M=7,8)

    width = 128
    dpi_val = 34.8  # 69.4 for 256x256

    for i in range(10):  # split the main spectrogram into 10 spectrogram representations (256x256) for CNN input
        start_col = i * width
        end_col = start_col + width
        segment = spectrogram_resized[:, start_col:end_col]

        plt.figure()
        plt.imshow(segment)
        plt.axis('off')
        plt.savefig(f"{image_path}/{os.path.splitext(os.path.basename(audio_path))[0]}_{i}.png", bbox_inches='tight',
                    pad_inches=0, dpi=dpi_val)
        plt.close()


dataset_dir = 'dataset'
image_dir_base = 'dataset_images_03s_128x128'

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.mp3'):
            genre_label = os.path.basename(root)
            image_dir = os.path.join(image_dir_base, genre_label)
            os.makedirs(image_dir, exist_ok=True)

            audio_path = os.path.join(root, file)
            process_spectrograms(audio_path, image_dir)
