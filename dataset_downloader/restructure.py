import pandas as pd, os, glob, shutil

df = pd.read_csv('selected_tracks.csv')
track_to_genre = dict(zip(df['track_id'].astype(str), df['genre']))

dataset_dir = 'dataset'
for f in glob.glob(os.path.join(dataset_dir, '*.mp3')):
    fname = os.path.basename(f)
    track_id = fname.split('_')[0]
    genre = track_to_genre.get(track_id, 'Unknown')
    genre_dir = os.path.join(dataset_dir, genre.replace('/', '_'))
    os.makedirs(genre_dir, exist_ok=True)
    dest = os.path.join(genre_dir, fname)
    shutil.move(f, dest)
    print(f'{fname} -> {genre}/')

print('Done restructuring')
