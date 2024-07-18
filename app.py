import sys
import cv2
import csv
import torch
from collections import defaultdict


# Function to load the YOLO model
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True)
    return model


# Function to detect and define regions dynamically
def detect_regions(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the video")
        sys.exit(1)

    height, width, _ = frame.shape
    region_size = 100  # Adjust this size as needed

    regions = {
        'A': ((0, 0), (region_size, region_size)),
        'B': ((width - region_size, 0), (width, region_size)),
        'C': ((0, height - region_size), (region_size, height)),
        'D': ((width - region_size, height - region_size), (width, height)),
        'E': ((width // 2 - region_size // 2, height // 2 - region_size // 2),
              (width // 2 + region_size // 2, height // 2 + region_size // 2)),
        'F': ((width // 2 - region_size // 2, height // 2 + region_size // 2),
              (width // 2 + region_size // 2, height // 2 + region_size // 2 + region_size))
    }
    cap.release()
    return regions


# Function to determine which region a point is in
def get_region(point, regions):
    for region, ((x1, y1), (x2, y2)) in regions.items():
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
            return region
    return None


# Function to process the video and count vehicle movements
def process_video(model, video_path, regions):
    vehicle_counts = {f"{start}{end}": 0 for start in regions.keys() for end in regions.keys() if start != end}
    tracked_vehicles = defaultdict(list)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            vehicle_id = row['class']
            center_x = int((row['xmin'] + row['xmax']) / 2)
            center_y = int((row['ymin'] + row['ymax']) / 2)
            current_region = get_region((center_x, center_y), regions)

            if current_region:
                tracked_vehicles[vehicle_id].append(current_region)
                if len(tracked_vehicles[vehicle_id]) > 1:
                    start = tracked_vehicles[vehicle_id][-2]
                    end = tracked_vehicles[vehicle_id][-1]
                    if start != end:
                        vehicle_counts[f"{start}{end}"] += 1

    cap.release()

    return vehicle_counts


# Function to write counts to CSV
def write_counts_to_csv(counts, output_csv_path):
    vehicle_types = ["Car", "Bike", "Bus", "Truck"]  # Example vehicle types
    turning_patterns = list(counts.keys())

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Vehicle Type"] + turning_patterns)

        for vehicle_type in vehicle_types:
            row = [vehicle_type] + [counts[pattern] for pattern in turning_patterns]
            writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python app.py <weights_path> <video_path> <output_csv_path>")
        sys.exit(1)

    weights_path = sys.argv[1]
    video_path = sys.argv[2]
    output_csv_path = sys.argv[3]

    model = load_model(weights_path)
    regions = detect_regions(video_path)
    counts = process_video(model, video_path, regions)
    write_counts_to_csv(counts, output_csv_path)
