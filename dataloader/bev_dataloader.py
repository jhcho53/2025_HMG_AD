import os
import cv2
import numpy as np
import pandas as pd
import torch

class MultiCamDataLoader:
    def __init__(self, base_root, map_name, camera_dirs, batch_size=32, img_size=(224, 480), time_steps=4):
        """
        Args:
            base_root: 데이터셋 루트 경로
            map_name: 맵 이름
            camera_dirs: 카메라 디렉토리 리스트 (예: ["CAMERA_1", ..., "CAMERA_5"])
            batch_size: 배치 크기
            img_size: 이미지 크기 (H, W)
            time_steps: 시퀀스 길이 (T)
        """
        self.base_root = base_root
        self.map_name = map_name
        self.camera_dirs = camera_dirs
        self.batch_size = batch_size
        self.img_size = img_size
        self.time_steps = time_steps
        self.scenario_paths = []
        self.all_camera_files = {}
        self.ego_frames = {}
        self.matched_frames = []
        self.calibration_data = {}  # Calibration 데이터를 저장할 딕셔너리
        self._load_calibration_data()
        self.current_scenario = None
        self.global_path = None
        self._load_scenarios()
        self.device = "cuda"

    def _load_calibration_data(self):
        """Load calibration data (intrinsics and extrinsics) for all cameras."""
        calibration_path = os.path.join(self.base_root, "Calibration")
        for camera_dir in self.camera_dirs:
            intrinsic_file = os.path.join(calibration_path, f"{camera_dir}__intrinsic.npy")
            extrinsic_file = os.path.join(calibration_path, f"{camera_dir}__extrinsic.npy")
            if os.path.exists(intrinsic_file) and os.path.exists(extrinsic_file):
                self.calibration_data[camera_dir] = {
                    "intrinsic": np.load(intrinsic_file),
                    "extrinsic": np.load(extrinsic_file),
                }
            else:
                print(f"Warning: Missing calibration files for {camera_dir}")

    def _load_scenarios(self):
        """Load all scenarios for the specified map."""
        self.scenario_paths = sorted(
            os.path.join(self.base_root, d)
            for d in os.listdir(self.base_root)
            if d.startswith(self.map_name) and os.path.isdir(os.path.join(self.base_root, d))
        )

    def set_scenario(self, scenario_path):
        """Set the current scenario and prepare data."""
        self.current_scenario = os.path.basename(scenario_path)  # Scenario 이름
        self.base_path = scenario_path
        self.ego_info_path = os.path.join(self.base_path, "EGO_INFO")
        self.all_camera_files = {}
        self.ego_frames = {}
        self.matched_frames = []
        self._load_data()
        self.match_data()

        # Load Global Path
        global_path_file = os.path.join(scenario_path, "global_path.csv")
        if os.path.exists(global_path_file):
            self.global_path = self._load_global_path(global_path_file)
        else:
            print(f"Warning: No global_path.csv found for scenario {self.current_scenario}")
            self.global_path = None

    def _load_global_path(self, file_path):
        """Load Global Path from a CSV file."""
        global_path = pd.read_csv(file_path)
        return global_path[['PositionX (m)', 'PositionY (m)']].to_numpy()

    def _load_data(self):
        """Load data from the specified directories."""
        self._load_camera_files()
        self._load_ego_info_files()

    def _load_camera_files(self):
        """Load camera files from all camera directories."""
        for camera_dir in self.camera_dirs:
            camera_path = os.path.join(self.base_path, camera_dir)
            if os.path.exists(camera_path):
                camera_files = sorted(
                    [f for f in os.listdir(camera_path) if f.endswith(".jpeg")],
                    key=lambda f: int(''.join(filter(str.isdigit, f)))
                )
                self.all_camera_files[camera_dir] = [
                    os.path.join(camera_path, f) for f in camera_files
                ]

    def _load_ego_info_files(self):
        """Load EGO_INFO files."""
        if os.path.exists(self.ego_info_path):
            ego_files = sorted([f for f in os.listdir(self.ego_info_path) if f.endswith(".txt")])
            self.ego_frames = {
                int(''.join(filter(str.isdigit, f.split("_")[0]))): os.path.join(self.ego_info_path, f)
                for f in ego_files
            }

    def match_data(self):
        """Match camera and EGO_INFO files into time-step sequences."""
        ego_frame_numbers = sorted(self.ego_frames.keys())
        min_length = min(len(ego_frame_numbers), len(next(iter(self.all_camera_files.values()))))

        self.matched_frames = []
        for start_idx in range(0, min_length - self.time_steps + 1):
            sequence_data = {"camera_data": [], "ego_data": []}
            for t in range(self.time_steps):
                current_idx = start_idx + t
                ego_frame = ego_frame_numbers[current_idx]

                # Load camera data for all cameras
                time_step_camera_data = []
                for camera_dir, camera_files in self.all_camera_files.items():
                    if current_idx < len(camera_files):  # 인덱스 확인
                        camera_frame_path = camera_files[current_idx]
                        camera_image = self._load_camera_image(camera_frame_path)
                        time_step_camera_data.append(camera_image)
                sequence_data["camera_data"].append(np.stack(time_step_camera_data))

                # Load corresponding EGO_INFO
                ego_file = self.ego_frames[ego_frame]
                with open(ego_file, "r") as f:
                    ego_data = self._parse_ego_info_data(f.readlines())
                sequence_data["ego_data"].append(ego_data)

            self.matched_frames.append(sequence_data)


    def _load_camera_image(self, camera_file_path):
        """
        Load camera image and preprocess it.
        Original image shape: (720, 1280, 3)
        Resized image shape: (224, 480, 3)
        """
        image = cv2.imread(camera_file_path)
        image = cv2.resize(image, (480, 224))
        image = image / 255.0  # Normalize to [0, 1]
        return image.transpose(2, 0, 1)  # Convert to (C, H, W)

    def _parse_ego_info_data(self, ego_data):
        """Parse EGO_INFO data into input and GT."""
        input_data = {
            "position": [float(val) for val in ego_data[0].split(":")[1].strip().split()],
            "orientation": [float(val) for val in ego_data[1].split(":")[1].strip().split()],
            "enu_velocity": [float(val) for val in ego_data[2].split(":")[1].strip().split()],
            "velocity": [float(val) for val in ego_data[3].split(":")[1].strip().split()],
            "angularVelocity": [float(val) for val in ego_data[4].split(":")[1].strip().split()],
            "acceleration": [float(val) for val in ego_data[5].split(":")[1].strip().split()],
            # 문자열 필드는 그대로 저장
            "linkid": ego_data[9].split(":")[1].strip(),
            "trafficlightid": ego_data[10].split(":")[1].strip(),
            "turn_signal_lamp": float(ego_data[11].split(":")[1].strip()),
        }

        gt_data = {
            "accel": float(ego_data[6].split(":")[1].strip()),
            "brake": float(ego_data[7].split(":")[1].strip()),
            "steer": float(ego_data[8].split(":")[1].strip()),
        }
        return {"input": input_data, "gt": gt_data}


    def __iter__(self):
        """Yield batches of data."""
        for i in range(0, len(self.matched_frames), self.batch_size):
            batch = self.matched_frames[i:i + self.batch_size]

            # (B, T, N, C, H, W)
            camera_images = torch.tensor(
                [np.stack(item["camera_data"]) for item in batch],
                dtype=torch.float32,
                device=self.device
            )

            # (B, T, N, 3, 3) - Intrinsics
            intrinsics = torch.tensor(
                [[self.calibration_data[camera_dir]["intrinsic"] for camera_dir in self.camera_dirs]
                for _ in range(self.time_steps)
                for _ in range(self.batch_size)],
                dtype=torch.float32,
                device=self.device
            ).reshape(self.batch_size, self.time_steps, len(self.camera_dirs), 3, 3)

            # (B, T, N, 4, 4) - Extrinsics
            extrinsics = torch.tensor(
                [[self.calibration_data[camera_dir]["extrinsic"] for camera_dir in self.camera_dirs]
                for _ in range(self.time_steps)
                for _ in range(self.batch_size)],
                dtype=torch.float32,
                device=self.device
            ).reshape(self.batch_size, self.time_steps, len(self.camera_dirs), 4, 4)

            # (B, T, input-dim)
            ego_inputs = []
            for item in batch:
                sequence_inputs = []
                for step in item["ego_data"]:
                    current_state = []
                    for sublist in step["input"].values():
                        if isinstance(sublist, list):  # 리스트 데이터 처리
                            current_state.extend(
                                [float(v) for v in sublist if self._is_float(v)]  # 숫자만 추가
                            )
                        elif self._is_float(sublist):  # 단일 값 처리
                            current_state.append(float(sublist))
                    sequence_inputs.append(current_state)
                ego_inputs.append(sequence_inputs)

            # (B, T, output-dim)
            gt_data = torch.tensor(
                [[list(step["gt"].values()) for step in item["ego_data"]] for item in batch],
                dtype=torch.float32,
                device=self.device
            )

            yield camera_images, intrinsics, extrinsics, torch.tensor(ego_inputs, dtype=torch.float32, device=self.device), gt_data


    def _is_float(self, value):
        """Check if a value can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False



    def _get_next_path_points(self, global_path, current_position, n_points=5):
        """Get the next n path points relative to the current position."""
        distances = np.linalg.norm(global_path - np.array(current_position), axis=1)
        nearest_idx = np.argmin(distances)
        next_points = global_path[nearest_idx:nearest_idx + n_points]
        if len(next_points) < n_points:
            padding = np.zeros((n_points - len(next_points), global_path.shape[1]))
            next_points = np.vstack((next_points, padding))
        return next_points

    def _compute_relative_path(self, next_points, current_position):
        """Compute relative path points to the current position."""
        return (next_points - np.array(current_position)).flatten()

# --- Batch 0 ---
# Camera Images Shape:   (2, 4, 5, 3, 224, 224)
# Intrinsics Shape:      (2, 4, 5, 3, 3)
# Extrinsics Shape:      (2, 4, 5, 4, 4)
# Ego Inputs Shape:      (2, 4, 19)
# Ground Truth Shape:    (2, 4, 3)

if __name__ == "__main__":
    # 루트 디렉토리 및 카메라 설정
    base_root = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E"
    map_name = "R_KR_PG_KATRI__HMG"
    camera_dirs = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "CAMERA_5"]

    # DataLoader 설정
    loader = MultiCamDataLoader(
        base_root=base_root,
        map_name=map_name,
        camera_dirs=camera_dirs,
        batch_size=2,  # B=2
        img_size=(224, 224),  # 이미지 크기
        time_steps=4  # T=4
    )

    # 첫 번째 시나리오 로드
    if not loader.scenario_paths:
        print("No scenarios found!")
    else:
        scenario_path = loader.scenario_paths[0]
        loader.set_scenario(scenario_path)
        print(f"Loaded scenario: {scenario_path}")

        # 데이터 로드 및 shape 확인
        for batch_idx, (camera_images, intrinsics, extrinsics, ego_inputs, gt_data) in enumerate(loader):
            print(f"--- Batch {batch_idx} ---")
            print(f"Camera Images Shape:   {camera_images.shape}")  # (B, T, N, C, H, W)
            print(f"Intrinsics Shape:      {intrinsics.shape}")     # (B, T, N, 3, 3)
            print(f"Extrinsics Shape:      {extrinsics.shape}")     # (B, T, N, 4, 4)
            print(f"Ego Inputs Shape:      {ego_inputs.shape}")     # (B, T, input-dim)
            print(f"Ground Truth Shape:    {gt_data.shape}")       # (B, T, output-dim)

            # 첫 배치만 출력
            break
