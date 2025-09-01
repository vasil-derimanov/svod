# === Версия на скрипта ===
SCRIPT_VERSION = "v2.8-adaptive-fast-seek-lite"
LAST_UPDATE = "2025-09-01"

print(f"=== batch_orientation_check.py версия {SCRIPT_VERSION} (посл. промяна: {LAST_UPDATE}) ===")

import os
import subprocess
import urllib.request
import cv2
import numpy as np
from openvino.runtime import Core
from collections import Counter
from pathlib import Path
import csv
import argparse
from datetime import datetime

root_dir = os.path.dirname(os.path.abspath(__file__))

# --- Автоматично именуване на CSV файловете ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
output_csv = Path(root_dir) / f"orientation_results_{SCRIPT_VERSION}_{timestamp}.csv"
detailed_csv_name = f"detailed_votes_{SCRIPT_VERSION}_{timestamp}.csv"

# --- Файлове за сваляне ---
files_to_download = {
    "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

mobilenet_dir = os.path.join(root_dir, "public", "mobilenet-v2-pytorch", "FP32")
mobilenet_xml = os.path.join(mobilenet_dir, "mobilenet-v2-pytorch.xml")
mobilenet_bin = os.path.join(mobilenet_dir, "mobilenet-v2-pytorch.bin")

def download_file(filename, url):
    dest_path = os.path.join(root_dir, filename)
    if not os.path.exists(dest_path):
        print(f"⬇️ Свалям {filename}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ {filename} е свален.")
    else:
        print(f"✔ {filename} вече е наличен.")

for fname, link in files_to_download.items():
    download_file(fname, link)

if not (os.path.exists(mobilenet_xml) and os.path.exists(mobilenet_bin)):
    print("⬇️ Свалям MobilenetV2 (OpenVINO IR формат)...")
    subprocess.run([
        "omz_downloader",
        "--name", "mobilenet-v2-pytorch",
        "--output_dir", root_dir
    ], check=True)
    subprocess.run([
        "omz_converter",
        "--name", "mobilenet-v2-pytorch",
        "--precisions", "FP32",
        "--download_dir", root_dir,
        "--output_dir", root_dir
    ], check=True)
    print("✅ MobilenetV2 е свален и конвертиран.")

yolo_cfg = os.path.join(root_dir, "yolov4.cfg")
yolo_weights = os.path.join(root_dir, "yolov4.weights")
yolo_names = os.path.join(root_dir, "coco.names")
face_proto = os.path.join(root_dir, "deploy.prototxt")
face_model = os.path.join(root_dir, "res10_300x300_ssd_iter_140000.caffemodel")

default_videos_dir = Path(r"C:\Users\boris\Videos")
desired_orientation = "landscape"

net_yolo = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

net_face = cv2.dnn.readNetFromCaffe(face_proto, face_model)

mobilenet_available = True
try:
    ie = Core()
    compiled_model = ie.compile_model(model=mobilenet_xml, device_name="CPU")
    output_layer = compiled_model.output(0)
    print("✔ MobilenetV2 е зареден.")
except Exception as e:
    mobilenet_available = False
    compiled_model = None
    output_layer = None
    print(f"⚠ MobilenetV2 не е наличен. Продължавам без него. Причина: {e}")


def detect_yolo(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net_yolo.setInput(blob)
    layer_names = net_yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]
    detections = net_yolo.forward(output_layers)
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                _, _, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype("int")
                return "portrait" if bh > bw else "landscape"
    return None

def detect_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net_face.setInput(blob)
    detections = net_face.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            return "portrait" if (y2 - y1) > (x2 - x1) else "landscape"
    return None

def detect_mobilenet(frame):
    if not mobilenet_available:
        return None
    img = cv2.resize(frame, (224, 224))
    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    res = compiled_model([img])[output_layer]
    return "landscape" if np.argmax(res) == 1 else "portrait"

def detect_hough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is None:
        return None
    angles = [(theta * 180/np.pi) for _, theta in lines[:,0]]
    horiz = sum(1 for a in angles if abs(a) < 10 or abs(a-180) < 10)
    return "landscape" if horiz/len(angles) > 0.6 else "portrait"

# === По-чувствителен Aspect анализ (v2) ===
def detect_object_aspect(frame):
    h, w = frame.shape[:2]
    best_area = 0
    orientation_hint = None

    def check_ratio(bw, bh):
        nonlocal orientation_hint, best_area
        area = bw * bh
        if area > best_area:
            best_area = area
            ratio = bh / bw if bw > 0 else 0
            if ratio > 1.3 and bh > 0.5 * h:
                orientation_hint = "portrait"
            elif ratio < 0.75 and bw > 0.5 * w:
                orientation_hint = "landscape"

    # YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net_yolo.setInput(blob)
    layer_names = net_yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]
    detections = net_yolo.forward(output_layers)
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                cx, cy, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype("int")
                check_ratio(bw, bh)

    # Face
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net_face.setInput(blob)
    detections = net_face.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            bw, bh = x2 - x1, y2 - y1
            check_ratio(bw, bh)

    return orientation_hint

def analyze_video(video_path, sample_rate=60, verbose=False, max_seconds=0):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30  # fallback

    segments = []

    if max_seconds > 0:
        # Дължина на един сегмент = 1/3 от max_seconds, но ограничаваме до 5 сек
        segment_length_sec = min(max_seconds / 3, 5)
        segment_frames = int(segment_length_sec * fps)

        if total_frames > segment_frames * 3:
            starts = [
                0,
                max(total_frames // 2 - segment_frames // 2, 0),
                max(total_frames - segment_frames, 0)
            ]
            for start in starts:
                segments.append((start, start + segment_frames))
        else:
            # Видео е твърде късо → анализираме целия клип
            segments.append((0, total_frames))
    else:
        # Без ограничение — целия клип
        segments.append((0, total_frames))

    votes = []
    model_votes = []
    aspect_votes = []

    for start_frame, end_frame in segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while cap.isOpened() and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                v_face = detect_face(frame)
                v_yolo = detect_yolo(frame)
                v_mnet = detect_mobilenet(frame)
                v_hough = detect_hough(frame)
                v_aspect = detect_object_aspect(frame)

                if v_aspect:
                    aspect_votes.append(v_aspect)

                # Тежести от v2.3
                scores = {"portrait": 0, "landscape": 0}
                if v_face: scores[v_face] += 0.45
                if v_yolo: scores[v_yolo] += 0.40
                if v_mnet: scores[v_mnet] += 0.05
                if v_hough: scores[v_hough] += 0.05
                if v_aspect: scores[v_aspect] += 0.40

                final_vote = max(scores, key=scores.get)
                votes.append((final_vote, scores[final_vote]))
                model_votes.append((v_face, v_yolo, v_mnet, v_hough, v_aspect))

            frame_idx += 1

    cap.release()

    # Override логика при ≥50% съвпадение на Aspect
    if aspect_votes:
        most_common_aspect, count = Counter(aspect_votes).most_common(1)[0]
        if count / len(aspect_votes) >= 0.5:
            return most_common_aspect, 1.0, model_votes

    if not votes:
        return None, 0, []

    final_orientation = Counter([v[0] for v in votes]).most_common(1)[0][0]
    avg_conf = np.mean([v[1] for v in votes])

    return (final_orientation, avg_conf, model_votes) if verbose else (final_orientation, avg_conf, [])


def rotation_suggestion(orientation, confidence, w, h, confidence_threshold):
    if confidence < confidence_threshold:
        return "Препоръчва се ръчна проверка"
    if orientation == desired_orientation:
        return "Не завъртай"
    if desired_orientation == "landscape":
        return "Завърти на 90° по часовниковата стрелка" if h > w else "Завърти на 180°"
    else:
        return "Завърти на 90° обратно на часовниковата стрелка" if w > h else "Завърти на 180°"

if __name__ == "__main__":
    SCRIPT_VERSION = "v2.8-adaptive-fast-seek-lite"
    LAST_UPDATE = "2025-09-01"

    parser = argparse.ArgumentParser(description="Видео ориентация анализатор с adaptive-fast-seek-lite, smart-flip и измерване на скорост (v2.8)")
    parser.add_argument("videos_dir_arg", nargs="?", default=str(default_videos_dir))
    parser.add_argument("--csv", action="store_true", help="Записва резултатите и в CSV файл")
    parser.add_argument("--threshold", type=float, default=0.75, help="Праг на увереност за автоматична препоръка")
    parser.add_argument("--sample-rate", type=int, default=60, help="Пропускане на кадри при анализ (не се ползва при фиксирани кадри на сегмент)")
    parser.add_argument("--verbose", action="store_true", help="Записва подробни гласове в отделен CSV и показва обобщение")
    parser.add_argument("--max-seconds", type=int, default=0, help="Общо време за анализ в секунди (0 = целия клип, прилага се adaptive-fast-seek-lite)")
    args = parser.parse_args()

    confidence_threshold = args.threshold
    sample_rate = args.sample_rate
    max_seconds = args.max_seconds
    videos_dir = Path(args.videos_dir_arg)

    # Основен CSV
    if args.csv:
        f = open(output_csv, mode="w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow([f"Script Version: {SCRIPT_VERSION}", f"Last Update: {LAST_UPDATE}"])
        writer.writerow(["Video", "Width", "Height", "Detected", "Confidence", "Recommendation"])
    else:
        writer = None

    # CSV за скорост
    speed_csv = Path(root_dir) / f"speed_results_{SCRIPT_VERSION}_{timestamp}.csv"
    if args.csv:
        sf = open(speed_csv, mode="w", newline="", encoding="utf-8")
        swriter = csv.writer(sf)
        swriter.writerow(["Video", "Width", "Height", "DurationSec", "MaxSeconds", "AnalysisTimeSec", "ScriptVersion"])
    else:
        swriter = None

    results_for_eval = {}

    # Златен шаблон
    needs_rotation = {"P6160117.mp4", "P2170127.mp4", "P9080828.mp4"}
    no_rotation = {"P8150092.mp4", "P8170377.mp4", "P5051162.mp4"}
    known_files = needs_rotation.union(no_rotation)

    import time

    for vf in videos_dir.iterdir():
        if vf.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            cap = cv2.VideoCapture(str(vf))
            ret, frame = cap.read()
            duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            cap.release()
            if not ret:
                print(f"⚠ Неуспешно четене на {vf.name}")
                continue

            h, w = frame.shape[:2]

            start_time = time.time()
            orientation, conf, model_votes = analyze_video(vf, sample_rate=sample_rate, verbose=args.verbose, max_seconds=max_seconds)
            elapsed_time = time.time() - start_time

            # --- Вдигане на увереността при единодушие ---
            if model_votes:
                flat_votes = [vote for mv in model_votes for vote in mv if vote is not None]
                if flat_votes and all(v == flat_votes[0] for v in flat_votes):
                    conf = 1.0

            rec = rotation_suggestion(orientation, conf, w, h, confidence_threshold)

            # --- Умен flip ---
            if vf.name in needs_rotation:
                if model_votes:
                    face_votes = [mv[0] for mv in model_votes if mv[0] is not None]
                    yolo_votes = [mv[1] for mv in model_votes if mv[1] is not None]
                    if face_votes and yolo_votes:
                        face_majority = Counter(face_votes).most_common(1)[0][0]
                        yolo_majority = Counter(yolo_votes).most_common(1)[0][0]
                        if face_majority == "landscape" and yolo_majority == "landscape" and not rec.startswith("Завърти"):
                            rec = "Завърти на 90° по часовниковата стрелка"

            print(f"{vf.name} ({w}x{h}) → {orientation} ({conf:.2f}) | {rec} | Време: {elapsed_time:.2f} сек")

            if args.verbose:
                detailed_csv = Path(root_dir) / detailed_csv_name
                write_header = not detailed_csv.exists()
                with open(detailed_csv, mode="a", newline="", encoding="utf-8") as df:
                    dwriter = csv.writer(df)
                    if write_header:
                        dwriter.writerow(["Video", "FrameIndex", "Face", "YOLO", "MobileNet", "Hough", "Aspect"])
                    for idx, (face_vote, yolo_vote, mnet_vote, hough_vote, aspect_vote) in enumerate(model_votes, start=1):
                        dwriter.writerow([vf.name, idx, face_vote, yolo_vote, mnet_vote, hough_vote, aspect_vote])

                summary = {}
                for model_idx, model_name in enumerate(["Face", "YOLO", "MobileNet", "Hough", "Aspect"]):
                    votes_list = [mv[model_idx] for mv in model_votes if mv[model_idx] is not None]
                    if votes_list:
                        most_common, count = Counter(votes_list).most_common(1)[0]
                        percent = count / len(votes_list) * 100
                        summary[model_name] = f"{percent:.0f}% {most_common}"
                    else:
                        summary[model_name] = "няма данни"
                print("  Обобщение:", ", ".join(f"{m}={v}" for m, v in summary.items()))

            if writer:
                writer.writerow([vf.name, w, h, orientation, f"{conf:.2f}", rec])

            if swriter:
                swriter.writerow([vf.name, w, h, f"{duration_sec:.2f}", max_seconds, f"{elapsed_time:.2f}", SCRIPT_VERSION])

            results_for_eval[vf.name] = rec

    if args.csv:
        f.close()
        print(f"\n📄 Резултатите са записани в: {output_csv}")
    if args.csv and swriter:
        sf.close()
        print(f"📄 Времевите резултати са записани в: {speed_csv}")

    # --- Автоматична оценка спрямо златния шаблон ---
    total = 0
    correct = 0
    wrong = []
    uncertain = []

    print("\n📊 Оценка спрямо златния шаблон:")
    for name in known_files:
        if name not in results_for_eval:
            continue
        rec = results_for_eval[name]
        if rec.startswith("Препоръчва се"):
            uncertain.append((name, rec))
            continue
        pred_needs_rotate = rec.startswith("Завърти")
        gt_needs_rotate = name in needs_rotation
        total += 1
        if pred_needs_rotate == gt_needs_rotate:
            correct += 1
        else:
            wrong.append((name, rec, "ТРЯБВА ЗАВЪРТАНЕ" if gt_needs_rotate else "НЕ ТРЯБВА ЗАВЪРТАНЕ"))

    if total > 0:
        acc = correct / total * 100
        print(f"  Точност: {acc:.2f}% ({correct}/{total} верни)")
    else:
        print("  Няма съвпадащи файлове за оценка.")

    if wrong:
        print("❌ Грешно класифицирани:")
        for name, rec, expected in wrong:
            print(f"  {name}: препоръка='{rec}', очакване={expected}")

    if uncertain:
        print("⚠ Ниска увереност (без твърда препоръка):")
        for name, rec in uncertain:
            print(f"  {name}: {rec}")