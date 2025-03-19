import os
import re
import cv2
import numpy as np

class DataLoader:
    def __init__(self, base_path, batch_size=32):
        self.base_path = base_path
        self.camera_dirs = ["CAMERA_1"]  # Only using CAMERA_1
        self.ego_info_path = os.path.join(base_path, "EGO_INFO")
        self.all_camera_files = {}
        self.ego_frames = {}
        self.batch_size = batch_size  # Batch size for processing
        self.matched_frames = []

        self._load_data()
        self.match_data()

    def _load_data(self):
        """Load data from the specified directories."""
        self._load_camera_files()
        self._load_ego_info_files()

    def _load_camera_files(self):
        """Load camera files from CAMERA_1."""
        for camera_dir in self.camera_dirs:
            camera_path = os.path.join(self.base_path, camera_dir)
            if os.path.exists(camera_path):
                camera_files = sorted(
                    [f for f in os.listdir(camera_path) if f.endswith(".jpeg")],
                    key=lambda f: int(''.join(filter(str.isdigit, f)))  # 파일명에서 숫자만 추출하여 정렬
                )
                self.all_camera_files[camera_dir] = {
                    os.path.join(camera_path, f): f for f in camera_files
                }

    def _load_ego_info_files(self):
        """Load EGO_INFO files."""
        if os.path.exists(self.ego_info_path):
            ego_files = sorted([f for f in os.listdir(self.ego_info_path) if f.endswith(".txt")])
            self.ego_frames = {
                int(''.join(filter(str.isdigit, f.split("_")[0]))): os.path.join(self.ego_info_path, f)
                for f in ego_files
            }

    def _extract_frame_number(self, filename):
        """Extract frame number from file name."""
        return int(''.join(filter(str.isdigit, filename.split("_")[0])))

    def match_data(self):
        """Match camera and EGO_INFO files."""
        ego_frame_numbers = sorted(self.ego_frames.keys())

        for camera_dir, camera_files in self.all_camera_files.items():
            for camera_file_path, camera_file in camera_files.items():
                camera_frame = self._extract_frame_number(camera_file)

                # Find closest frame
                closest_ego_frame = min(ego_frame_numbers, key=lambda ego_frame: abs(ego_frame - camera_frame))

                # Read the EGO_INFO file and extract GT and input data
                with open(self.ego_frames[closest_ego_frame], "r") as f:
                    ego_data = f.readlines()
                    data = self._parse_ego_info_data(ego_data)

                # Load the camera image
                camera_image = self._load_camera_image(camera_file_path)

                # Store the matched frame
                self.matched_frames.append({
                    "camera_data": camera_image,
                    "ego_data": data,
                })

    def _load_camera_image(self, camera_file_path):
        """Load camera image and preprocess it."""
        image = cv2.imread(camera_file_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def _parse_ego_info_data(self, ego_data):
        """Parse EGO_INFO data into input and GT."""
        try:
            input_data = {
                "position": [float(val) for val in ego_data[0].split(":")[1].strip().split()],
                "orientation": [float(val) for val in ego_data[1].split(":")[1].strip().split()],
                "enu_velocity": [float(val) for val in ego_data[2].split(":")[1].strip().split()],
                "velocity": [float(val) for val in ego_data[3].split(":")[1].strip().split()],
                "angularVelocity": [float(val) for val in ego_data[4].split(":")[1].strip().split()],
                "acceleration": [float(val) for val in ego_data[5].split(":")[1].strip().split()],
                # `linkid`는 문자열로 저장
                "linkid": ego_data[9].split(":")[1].strip(),
                "trafficlightid": ego_data[10].split(":")[1].strip(),
                "turn_signal_lamp": float(ego_data[11].split(":")[1].strip()),
            }

            gt_data = {
                "accel": float(ego_data[6].split(":")[1].strip()),
                "brake": float(ego_data[7].split(":")[1].strip()),
                "steer": float(ego_data[8].split(":")[1].strip()),
            }
        except ValueError as e:
            print(f"Error parsing EGO_INFO data: {e}")
            raise

        return {"input": input_data, "gt": gt_data}

    # Link ID x
    def __iter__(self):
        """Yield batches of data."""
        for i in range(0, len(self.matched_frames), self.batch_size):
            batch = self.matched_frames[i:i + self.batch_size]
            camera_images = np.array([item["camera_data"] for item in batch])
            
            # EGO 입력 데이터를 처리 (숫자형 데이터만 포함)
            ego_inputs = np.array([
                [
                    float(value)  # 모든 값을 float로 변환
                    for sublist in item["ego_data"]["input"].values()
                    for value in (sublist if isinstance(sublist, list) else [sublist])
                    if isinstance(value, (int, float)) or value.replace('.', '', 1).isdigit()  # 숫자형 데이터만 포함
                ]
                for item in batch
            ])

            # Ground Truth 데이터를 처리
            gt_data = np.array([list(item["ego_data"]["gt"].values()) for item in batch])
            yield camera_images, ego_inputs, gt_data




