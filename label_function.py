import os
import pandas as pd
from pydub import AudioSegment

AUDIO_DIR = 'fma_small/'
tracks = pd.read_csv('tracks.csv', index_col=0, header=[0, 1])

os.makedirs('Samples', exist_ok=True)

small = tracks['set', 'subset'] <= 'small'

y_small = tracks.loc[small, ('track', 'genre_top')]

for subdir in os.listdir(AUDIO_DIR):
    subdir_path = os.path.join(AUDIO_DIR, subdir)
    if os.path.isdir(subdir_path):
        for track_id, genre in y_small.items():
            genre_folder = os.path.join('Samples', str(genre).replace('/', '_'))  # Use genre as folder name

            # Check if genre_folder is a string, otherwise skip iteration
            if not isinstance(genre_folder, str):
                continue

            os.makedirs(genre_folder, exist_ok=True)

            mp3_filename = os.path.join(subdir_path, f'{track_id:06}.mp3')
            out_mp3_filename = os.path.join(genre_folder, f'{track_id}.mp3')

            if os.path.exists(mp3_filename):
                print(f"Copying {mp3_filename} to {out_mp3_filename}")
                AudioSegment.from_file(mp3_filename, format="mp3").export(out_mp3_filename, format="mp3")
