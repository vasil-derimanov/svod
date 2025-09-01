# === compare_versions.py версия v1.1-speed-compare ===
import csv
from pathlib import Path

root_dir = Path(__file__).parent

def find_latest_csv(prefix):
    files = sorted(root_dir.glob(f"{prefix}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:2] if len(files) >= 2 else []

def read_accuracy(file_path):
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Търсим ред с "Точност"
    for row in rows:
        if row and "Точност" in row[0]:
            try:
                percent = float(row[0].split(":")[1].split("%")[0].strip())
                return percent
            except:
                pass
    return None

def read_speed(file_path):
    times = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or "AnalysisTimeSec" not in header:
            return None
        idx = header.index("AnalysisTimeSec")
        for row in reader:
            try:
                times.append(float(row[idx]))
            except:
                pass
    return sum(times) / len(times) if times else None

print(f"=== compare_versions.py версия v1.1-speed-compare ===")

# --- Сравнение на точност ---
result_files = find_latest_csv("orientation_results")
if len(result_files) < 2:
    print("⚠ Няма достатъчно CSV файлове за сравнение на точността.")
else:
    new_file, old_file = result_files
    print(f"📂 Нов файл: {new_file.name}")
    print(f"📂 Стар файл: {old_file.name}")

    # Тук приемаме, че точността е в края на файла (или ще я четем от batch_orientation_check.py изхода)
    # Ако искаш, можем да я изчислим директно от CSV, но сега ще я оставим празна
    print("⚠ Точността се извежда от batch_orientation_check.py, а не от CSV.")

# --- Сравнение на скорост ---
speed_files = find_latest_csv("speed_results")
if len(speed_files) < 2:
    print("⚠ Няма достатъчно CSV файлове за сравнение на скоростта.")
else:
    new_speed, old_speed = speed_files
    avg_new = read_speed(new_speed)
    avg_old = read_speed(old_speed)
    if avg_new is not None and avg_old is not None:
        diff_sec = avg_new - avg_old
        diff_pct = (diff_sec / avg_old) * 100 if avg_old > 0 else 0
        print("\n⏱ Сравнение на скорост:")
        print(f"{old_speed.stem}: {avg_old:.2f} сек средно")
        print(f"{new_speed.stem}: {avg_new:.2f} сек средно ({diff_sec:+.2f} сек, {diff_pct:+.1f}%)")
    else:
        print("⚠ Неуспешно четене на времената от speed_results CSV.")