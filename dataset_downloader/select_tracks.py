import pandas as pd
import numpy as np

tracks = pd.read_csv('fma/fma_metadata/tracks.csv', header=[0, 1], low_memory=False)
tracks = tracks.iloc[2:].copy()
tracks.columns = tracks.columns.map('_'.join).str.strip('_')
tracks = tracks.rename(columns={tracks.columns[0]: 'track_id'})

np.random.seed(42)

available = tracks[
    tracks['track_genre_top'].notna() &
    tracks['track_title'].notna() &
    tracks['artist_name'].notna() &
    tracks['track_duration'].notna()
].copy()

genre_counts = available['track_genre_top'].value_counts()
total_genres = len(genre_counts)
target_total = 1000
base_per_genre = target_total // total_genres

selected = pd.DataFrame()
allocation = {}

for genre in genre_counts.index:
    pool = available[available['track_genre_top'] == genre]
    n = min(base_per_genre, len(pool))
    allocation[genre] = n

shortfall = target_total - sum(allocation.values())
if shortfall > 0:
    deficit_genres = [g for g in genre_counts.index if genre_counts[g] > allocation[g]]
    while shortfall > 0:
        for g in deficit_genres:
            if shortfall <= 0:
                break
            if genre_counts[g] > allocation[g]:
                allocation[g] += 1
                shortfall -= 1

for genre, n in allocation.items():
    pool = available[available['track_genre_top'] == genre]
    sampled = pool.sample(n=n, random_state=42)
    selected = pd.concat([selected, sampled])

selected = selected.reset_index(drop=True)

output = selected[['track_id', 'track_title', 'artist_name', 'track_genre_top',
                    'track_duration', 'album_title', 'track_license']].copy()
output.columns = ['track_id', 'title', 'artist', 'genre', 'duration', 'album', 'license']
output.to_csv('selected_tracks.csv', index=False)

print("=== SELECTION SUMMARY ===")
print(f"Total tracks selected: {len(output)}")
print("Per-genre breakdown:")
for g, count in output['genre'].value_counts().sort_index().items():
    print(f"  {g}: {count}")
print("Selected tracks saved to selected_tracks.csv")
print("Reproducible: run this script again with seed=42 to get identical output")
