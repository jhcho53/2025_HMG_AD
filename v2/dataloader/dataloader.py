import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re

class camDataLoader(Dataset):
    def __init__(self, root_dir, num_timesteps=3, image_size=(135, 240), map_size=(144, 144), 
                 hd_map_dir="HD_MAP", gt_hd_map_dir="GT_HD_MAP", ego_info_dir="EGO_INFO", 
                 traffic_info_dir="TRAFFIC_INFO", num_traffic_classes=10):
        """
        Args:
            root_dir (str): CALIBRATION 및 시나리오 폴더들이 있는 루트 디렉토리.
            num_timesteps (int): 입력 프레임 개수 (이후 2프레임은 미래 GT로 사용).
            image_size (tuple): 출력 이미지 크기 (height, width).
            map_size (tuple): HD Map 이미지 크기 (height, width).
            hd_map_dir (str): 각 시나리오 내 HD_MAP 폴더 이름.
            gt_hd_map_dir (str): 각 시나리오 내 GT_HD_MAP 폴더 이름 (BEV segmentation GT용).
            ego_info_dir (str): 각 시나리오 내 EGO_INFO 폴더 이름.
            traffic_info_dir (str): 각 시나리오 내 TRAFFIC_INFO 폴더 이름.
            num_traffic_classes (int): traffic GT의 클래스 수 (one-hot 인코딩에 사용).
        """
        self.root_dir = root_dir
        self.num_timesteps = num_timesteps          # 입력 프레임 개수
        self.future_steps = 2                       # 미래 GT 프레임 개수 (고정)
        self.total_steps = self.num_timesteps + self.future_steps  # 총 사용 프레임 수 (예: 5)
        self.map_size = map_size
        self.hd_map_dir = hd_map_dir
        self.gt_hd_map_dir = gt_hd_map_dir           # GT_HD_MAP 폴더 이름 (BEV segmentation GT)
        self.ego_info_dir = ego_info_dir
        self.traffic_info_dir = traffic_info_dir      # TRAFFIC_INFO 폴더 이름
        self.num_traffic_classes = num_traffic_classes  # traffic GT 클래스 개수

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),   # 이미지 크기 고정
            transforms.ToTensor()              # 텐서 변환
        ])

        # Calibration 파일 로드
        calibration_dir = os.path.join(root_dir, "Calibration")
        assert os.path.isdir(calibration_dir), f"Calibration directory not found: {calibration_dir}"

        camera_files = [f for f in os.listdir(calibration_dir) if f.endswith(".npy")]
        camera_ids = sorted(set(f.split("__")[0] for f in camera_files))
        self.num_cameras = len(camera_ids)

        self.intrinsic_data = []
        self.extrinsic_data = []
        for camera_id in camera_ids:
            intrinsic_file = os.path.join(calibration_dir, f"{camera_id}__intrinsic.npy")
            extrinsic_file = os.path.join(calibration_dir, f"{camera_id}__extrinsic.npy")

            assert os.path.isfile(intrinsic_file), f"Intrinsic file not found: {intrinsic_file}"
            assert os.path.isfile(extrinsic_file), f"Extrinsic file not found: {extrinsic_file}"

            self.intrinsic_data.append(torch.tensor(np.load(intrinsic_file), dtype=torch.float32))
            self.extrinsic_data.append(torch.tensor(np.load(extrinsic_file), dtype=torch.float32))

        # 시나리오 폴더 로드
        self.scenario_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("R_KR_")]
        )

        # 각 시나리오별로 CAMERA, EGO_INFO, TRAFFIC_INFO 데이터 수집
        self.camera_data = []  # (scenario_dir, [카메라별 이미지 파일 리스트])
        self.ego_data = []     # 각 시나리오의 EGO_INFO 파일 경로 리스트
        self.traffic_data = [] # 각 시나리오의 TRAFFIC_INFO 파일 경로 리스트 (EGO_INFO와 순서 일치)

        for scenario_dir in self.scenario_dirs:
            # CAMERA 데이터 처리
            camera_dirs = sorted(
                [os.path.join(scenario_dir, d) for d in os.listdir(scenario_dir)
                 if d.upper().startswith("CAMERA_")]
            )
            assert camera_dirs, f"No CAMERA_* folders found in {scenario_dir}"

            camera_files = []
            for camera_dir in camera_dirs:
                assert os.path.isdir(camera_dir), f"{camera_dir} is not a directory"
                files = sorted(
                    [os.path.join(camera_dir, f) for f in os.listdir(camera_dir) if f.endswith(".jpeg")]
                )
                assert files, f"No .jpeg files found in {camera_dir}"
                camera_files.append(files)
            # 모든 카메라가 동일한 프레임 수를 가지고 있는지 확인
            num_frames = len(camera_files[0])
            assert all(len(files) == num_frames for files in camera_files), (
                f"Mismatch in number of frames across cameras in {scenario_dir}"
            )
            self.camera_data.append((scenario_dir, camera_files))

            # EGO_INFO 파일 처리
            ego_info_path = os.path.join(scenario_dir, self.ego_info_dir)
            assert os.path.isdir(ego_info_path), f"EGO_INFO directory not found: {ego_info_path}"
            ego_files = sorted(
                [os.path.join(ego_info_path, f) for f in os.listdir(ego_info_path) if f.endswith(".txt")]
            )
            assert ego_files, f"No EGO_INFO files found in {ego_info_path}"
            self.ego_data.append(ego_files)

            # TRAFFIC_INFO 파일 처리 (EGO_INFO 시퀀스와 동일한 순서로 정렬)
            traffic_info_path = os.path.join(scenario_dir, self.traffic_info_dir)
            if os.path.isdir(traffic_info_path):
                traffic_files = sorted(
                    [os.path.join(traffic_info_path, f) for f in os.listdir(traffic_info_path) if f.endswith(".txt")]
                )
                if len(traffic_files) != len(ego_files):
                    print(f"Warning: Number of TRAFFIC_INFO files ({len(traffic_files)}) does not match EGO_INFO files ({len(ego_files)}) in {scenario_dir}")
                self.traffic_data.append(traffic_files)
            else:
                print(f"Warning: TRAFFIC_INFO directory not found: {traffic_info_path}")
                self.traffic_data.append(None)

        # 총 샘플 수 계산 (각 샘플은 total_steps 프레임을 필요로 함)
        self.num_frames = sum(len(camera_files[0]) - self.total_steps + 1 for _, camera_files in self.camera_data)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # idx가 속하는 시나리오와 해당 프레임 인덱스 결정
        cumulative_frames = 0
        for scenario_idx, (scenario_dir, camera_files) in enumerate(self.camera_data):
            num_scenario_frames = len(camera_files[0]) - self.total_steps + 1
            if idx < cumulative_frames + num_scenario_frames:
                frame_idx = idx - cumulative_frames
                break
            cumulative_frames += num_scenario_frames

        temporal_images = []
        ego_info_data = []
        control_info_data = []
        traffic_class_seq = []   # 입력 시퀀스 내 프레임에 해당하는 traffic classification 값을 저장
        camera_indices = []

        # total_steps 만큼 반복 (입력 프레임 + 미래 GT 프레임은 EGO_INFO 및 이미지 처리에 사용)
        for t in range(self.total_steps):
            # --- 이미지 로드 (CAMERA) ---
            images_per_camera = []
            for cam_idx in range(len(camera_files)):
                image_path = camera_files[cam_idx][frame_idx + t]
                image = Image.open(image_path).convert("RGB")
                image = self.image_transform(image)
                images_per_camera.append(image)
            camera_indices.append(frame_idx + t)
            temporal_images.append(torch.stack(images_per_camera, dim=0))  # (num_cameras, C, H, W)

            # --- EGO_INFO 파일 로드 ---
            ego_file_path = self.ego_data[scenario_idx][frame_idx + t]
            ego_info_dict = {}
            with open(ego_file_path, 'r') as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key in {"position", "orientation", "enu_velocity", "velocity", "angularVelocity", "acceleration"}:
                            ego_info_dict[key] = list(map(float, value.split()))
                        elif key in {"accel", "brake", "steer"}:
                            if "control" not in ego_info_dict:
                                ego_info_dict["control"] = []
                            ego_info_dict["control"].append(float(value))
                        elif key == "trafficlightid":
                            # trafficlightid는 문자열 그대로 저장 (예: "C119BS010046")
                            ego_info_dict["trafficlightid"] = value

            # flatten: 기존 키들에 대해 (없으면 길이 3의 0.0 벡터로 채움)
            ego_info_values = []
            for k in ["position", "orientation", "enu_velocity", "velocity", "angularVelocity", "acceleration", "control"]:
                default_length = 3
                ego_info_values.extend(ego_info_dict.get(k, [0.0] * default_length))
            ego_info_data.append(ego_info_values)
            
            # control 부분만 따로 저장
            control_data = ego_info_dict.get("control", [0.0] * 3)
            control_info_data.append(control_data)
            
            # --- TRAFFIC_INFO 파일 로드 및 traffic classification 결정 ---
            # 입력 시퀀스(frames 0 ~ num_timesteps-1)에 대해서만 처리
            if t < self.num_timesteps:
                if "trafficlightid" not in ego_info_dict or ego_info_dict["trafficlightid"] in [None, "", "null"]:
                    traffic_class = 0
                else:
                    ego_traffic_id = ego_info_dict["trafficlightid"]
                    if self.traffic_data[scenario_idx] is not None:
                        traffic_file_path = self.traffic_data[scenario_idx][frame_idx + t]
                        with open(traffic_file_path, 'r') as f_traffic:
                            traffic_dict = {}
                            for line in f_traffic:
                                if ":" in line:
                                    k2, v2 = line.split(":", 1)
                                    k2 = k2.strip()
                                    tokens = v2.strip().split()
                                    if tokens:
                                        # 마지막 토큰을 분류값(정수형)으로 사용
                                        traffic_dict[k2] = int(tokens[-1])
                        traffic_class = traffic_dict.get(ego_traffic_id, 0)
                    else:
                        traffic_class = 0
                traffic_class_seq.append(traffic_class)

        # 최종 traffic classification 값은 입력 시퀀스의 마지막 프레임(현재 프레임)의 값 사용
        current_traffic_class = traffic_class_seq[-1] if traffic_class_seq else 0
        # one-hot 인코딩: traffic GT를 (클래스 수,) 형태의 벡터로 변환
        traffic_gt = torch.zeros(self.num_traffic_classes, dtype=torch.float32)
        if current_traffic_class < self.num_traffic_classes:
            traffic_gt[current_traffic_class] = 1.0

        # EGO_INFO 텐서 생성 및 입력/미래 분리
        ego_info_tensor = torch.tensor(ego_info_data, dtype=torch.float32)  # (total_steps, feature_dim)
        ego_info_input = ego_info_tensor[:self.num_timesteps]      # (num_timesteps, feature_dim)
        ego_info_future = ego_info_tensor[self.num_timesteps:]       # (future_steps, feature_dim)
        control_info_tensor = torch.tensor(control_info_data, dtype=torch.float32)
        control_info_future = control_info_tensor[self.num_timesteps+1:]
        
        # --- 이미지 데이터 (CAMERA) ---
        # temporal_images: 리스트 길이 total_steps, 각 원소 (num_cameras, C, H, W)
        # 스택 후 (total_steps, num_cameras, C, H, W)로 변환
        temporal_images = torch.stack(temporal_images, dim=1)  # (num_cameras, total_steps, C, H, W)
        temporal_images = temporal_images.permute(1, 0, 2, 3, 4)  # (total_steps, num_cameras, C, H, W)
        images_input = temporal_images[:self.num_timesteps]       # (num_timesteps, num_cameras, C, H, W)
        images_future = temporal_images[self.num_timesteps:]        # (future_steps, num_cameras, C, H, W)

        # --- Calibration (intrinsic & extrinsic) ---
        intrinsic = torch.stack(self.intrinsic_data, dim=0).unsqueeze(1).repeat(1, self.total_steps, 1, 1)
        extrinsic = torch.stack(self.extrinsic_data, dim=0).unsqueeze(1).repeat(1, self.total_steps, 1, 1)

        intrinsic = intrinsic.permute(1, 0, 2, 3)  # (total_steps, num_cameras, 3, 3)
        extrinsic = extrinsic.permute(1, 0, 2, 3)  # (total_steps, num_cameras, 4, 4)

        intrinsic_input = intrinsic[:self.num_timesteps]
        intrinsic_future = intrinsic[self.num_timesteps:]
        extrinsic_input = extrinsic[:self.num_timesteps]
        extrinsic_future = extrinsic[self.num_timesteps:]

        # --- HD Map 데이터 로드 및 전처리 ---
        hd_map_images = self._load_hd_map(scenario_dir, camera_indices)
        hd_map_input = None
        hd_map_future = None
        if hd_map_images is not None:
            hd_map_images = np.stack([self._process_hd_map(frame) for frame in hd_map_images])
            hd_map_images = torch.tensor(hd_map_images, dtype=torch.float32)  # (total_steps, channels, H, W)
            hd_map_input = hd_map_images[:self.num_timesteps]
            hd_map_future = hd_map_images[self.num_timesteps:]

        # --- GT HD Map 데이터 로드 및 전처리 (BEV segmentation GT) ---
        gt_hd_map_images = self._load_gt_hd_map(scenario_dir, camera_indices)
        gt_hd_map_input = None
        gt_hd_map_future = None
        if gt_hd_map_images is not None:
            gt_hd_map_images = np.stack([self._process_gt_hd_map(frame) for frame in gt_hd_map_images])
            gt_hd_map_images = torch.tensor(gt_hd_map_images, dtype=torch.float32)  # (total_steps, channels, H, W)
            gt_hd_map_input = gt_hd_map_images[:self.num_timesteps]
            gt_hd_map_future = gt_hd_map_images[self.num_timesteps:]

        return {
            "images_input": images_input,         # (num_timesteps, num_cameras, C, H, W)
            "images_future": images_future,         # (future_steps, num_cameras, C, H, W)
            "intrinsic_input": intrinsic_input,     # (num_timesteps, num_cameras, 3, 3)
            "intrinsic_future": intrinsic_future,   # (future_steps, num_cameras, 3, 3)
            "extrinsic_input": extrinsic_input,     # (num_timesteps, num_cameras, 4, 4)
            "extrinsic_future": extrinsic_future,   # (future_steps, num_cameras, 4, 4)
            "hd_map_input": hd_map_input,           # (num_timesteps, channels, H, W) 또는 None
            "hd_map_future": hd_map_future,         # (future_steps, channels, H, W) 또는 None
            "gt_hd_map_input": gt_hd_map_input,       # (num_timesteps, channels, H, W) 또는 None (BEV segmentation GT)
            "gt_hd_map_future": gt_hd_map_future,     # (future_steps, channels, H, W) 또는 None (BEV segmentation GT)
            "ego_info": ego_info_input,             # (num_timesteps, feature_dim)
            "ego_info_future": ego_info_future,     # (future_steps, feature_dim)
            "traffic": traffic_gt,                  # (num_traffic_classes,) -> DataLoader에서 (배치, num_traffic_classes)로 조합됨.
            "scenario": scenario_dir,
            "control": control_info_future
        }

    def _process_hd_map(self, hd_map_frame):
        """HD Map 프레임을 다중 채널 텐서로 전처리."""
        exterior = (hd_map_frame[:, :, 0] > 200).astype(np.float32)  # Red channel
        interior = (hd_map_frame[:, :, 1] > 200).astype(np.float32)  # Green channel
        lane = (hd_map_frame[:, :, 2] > 200).astype(np.float32)      # Blue channel
        crosswalk = ((hd_map_frame[:, :, 0] > 200) & (hd_map_frame[:, :, 1] > 200)).astype(np.float32)  # Yellow
        traffic_light = ((hd_map_frame[:, :, 1] > 200) & (hd_map_frame[:, :, 2] > 200)).astype(np.float32)  # Cyan
        ego_vehicle = ((hd_map_frame[:, :, 0] > 200) & (hd_map_frame[:, :, 2] > 200)).astype(np.float32)  # Magenta

        return np.stack([exterior, interior, lane, crosswalk, traffic_light, ego_vehicle], axis=0)

    def _load_hd_map(self, scenario_path, camera_indices):
        """HD Map 이미지를 불러오고, 카메라 인덱스와 동기화."""
        hd_map_path = os.path.join(scenario_path, self.hd_map_dir)

        if not os.path.exists(hd_map_path):
            print(f"Warning: HD Map directory not found: {hd_map_path}")
            return None

        def extract_number(file_name):
            match = re.search(r'(\d+)', file_name)
            return int(match.group(1)) if match else float('inf')

        hd_map_files = sorted(
            [f for f in os.listdir(hd_map_path) if f.endswith(".png")],
            key=extract_number
        )

        if not hd_map_files:
            print(f"Warning: No HD Map files found in directory: {hd_map_path}")
            return None

        hd_map_images = []
        for idx in camera_indices:
            if idx < len(hd_map_files):
                file_path = os.path.join(hd_map_path, hd_map_files[idx])
                hd_map_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if hd_map_image is None:
                    print(f"Warning: Failed to load HD Map image: {file_path}")
                    continue
                hd_map_image = cv2.resize(hd_map_image, self.map_size)
                hd_map_images.append(hd_map_image)

        return np.array(hd_map_images)

    def _load_gt_hd_map(self, scenario_path, camera_indices):
        """GT_HD_MAP 폴더에서 BEV segmentation GT용 HD Map 이미지를 불러오고, 카메라 인덱스와 동기화."""
        gt_hd_map_path = os.path.join(scenario_path, self.gt_hd_map_dir)

        if not os.path.exists(gt_hd_map_path):
            print(f"Warning: GT HD Map directory not found: {gt_hd_map_path}")
            return None

        def extract_number(file_name):
            match = re.search(r'(\d+)', file_name)
            return int(match.group(1)) if match else float('inf')

        gt_hd_map_files = sorted(
            [f for f in os.listdir(gt_hd_map_path) if f.endswith(".png")],
            key=extract_number
        )

        if not gt_hd_map_files:
            print(f"Warning: No GT HD Map files found in directory: {gt_hd_map_path}")
            return None

        gt_hd_map_images = []
        for idx in camera_indices:
            if idx < len(gt_hd_map_files):
                file_path = os.path.join(gt_hd_map_path, gt_hd_map_files[idx])
                gt_hd_map_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if gt_hd_map_image is None:
                    print(f"Warning: Failed to load GT HD Map image: {file_path}")
                    continue
                gt_hd_map_image = cv2.resize(gt_hd_map_image, self.map_size)
                gt_hd_map_images.append(gt_hd_map_image)

        return np.array(gt_hd_map_images)

    def _process_gt_hd_map(self, gt_hd_map_frame):
        """
        GT_HD_MAP 프레임을 BEV segmentation GT용으로 전처리합니다.
        여기서는 HD Map과 동일한 방식으로 다중 채널 바이너리 마스크를 생성하지만,
        실제 segmentation label 형식(예: single-channel class index)으로 변환하려면
        색상 매핑 또는 다른 후처리 작업이 필요할 수 있습니다.
        """
        exterior = (gt_hd_map_frame[:, :, 0] > 200).astype(np.float32)
        interior = (gt_hd_map_frame[:, :, 1] > 200).astype(np.float32)
        lane = (gt_hd_map_frame[:, :, 2] > 200).astype(np.float32)
        crosswalk = ((gt_hd_map_frame[:, :, 0] > 200) & (gt_hd_map_frame[:, :, 1] > 200)).astype(np.float32)
        traffic_light = ((gt_hd_map_frame[:, :, 1] > 200) & (gt_hd_map_frame[:, :, 2] > 200)).astype(np.float32)
        ego_vehicle = ((gt_hd_map_frame[:, :, 0] > 200) & (gt_hd_map_frame[:, :, 2] > 200)).astype(np.float32)
        return np.stack([exterior, interior, lane, crosswalk, traffic_light, ego_vehicle], axis=0)
