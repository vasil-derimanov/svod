# === analyze_detailed_votes.py версия v1.1-max-seconds-aware ===
import csv
from pathlib import Path
from collections import Counter

root_dir = Path(__file__).parent

# Намираме последния detailed_votes CSV
detailed_files = sorted(root_dir.glob("detailed_votes_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not detailed_files:
    print("⚠ Няма намерени detailed_votes_*.csv файлове.")
    exit()

detailed_csv_path = detailed_files[0]
print(f"=== analyze_detailed_votes.py версия v1.1-max-seconds-aware ===")
print(f"Чета от файл: {detailed_csv_path.name}")

video_votes = {}

with open(detailed_csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["Video"]
        if video not in video_votes:
            video_votes[video] = []
        video_votes[video].append((
            row["Face"] if row["Face"] else None,
            row["YOLO"] if row["YOLO"] else None,
            row["MobileNet"] if row["MobileNet"] else None,
            row["Hough"] if row["Hough"] else None,
            row["Aspect"] if row["Aspect"] else None
        ))

total_frames = sum(len(votes) for votes in video_votes.values())
print(f"Общо анализирани кадри: {total_frames}")

for video, votes in video_votes.items():
    print(f"\n🎥 {video}")
    for idx, model_name in enumerate(["Face", "YOLO", "MobileNet", "Hough", "Aspect"]):
        model_votes = [v[idx] for v in votes if v[idx] is not None]
        if model_votes:
            most_common, count = Counter(model_votes).most_common(1)[0]
            percent = count / len(model_votes) * 100
            print(f"  {model_name}: {percent:.0f}% {most_common} ({count}/{len(model_votes)})")
        else:
            print(f"  {model_name}: няма данни")