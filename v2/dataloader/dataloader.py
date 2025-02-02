import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re

class camDataLoader(Dataset):
    def __init__(self, root_dir, num_timesteps=4, image_size=(135, 240), map_size=(200, 200), hd_map_dir="HD_MAP", ego_info_dir="EGO_INFO"):
        """
        Args:
            root_dir (str): Root directory containing CALIBRATION and scenario folders.
            num_timesteps (int): Number of time steps for temporal data.
            image_size (tuple): Desired size of the output images (height, width).
            map_size (tuple): Desired size of HD Map images (height, width).
            hd_map_dir (str): Directory name for HD Maps inside each scenario folder.
            ego_info_dir (str): Directory name for EGO information inside each scenario folder.
        """
        self.root_dir = root_dir
        self.num_timesteps = num_timesteps
        self.map_size = map_size
        self.hd_map_dir = hd_map_dir
        self.ego_info_dir = ego_info_dir
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),   # Resize images to a fixed size
            transforms.ToTensor()           # Convert images to PyTorch tensors
        ])

        # Load calibration files
        calibration_dir = os.path.join(root_dir, "Calibration")
        assert os.path.isdir(calibration_dir), f"Calibration directory not found: {calibration_dir}"

        # Collect intrinsic and extrinsic calibration data
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

            # Load NumPy files
            self.intrinsic_data.append(torch.tensor(np.load(intrinsic_file), dtype=torch.float32))
            self.extrinsic_data.append(torch.tensor(np.load(extrinsic_file), dtype=torch.float32))

        # Load scenario folders
        self.scenario_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("R_KR_")]
        )

        # Validate folder structure and gather camera data
        self.camera_data = []  # List of (scenario_dir, [camera_1_files, camera_2_files, ...])
        self.ego_data = []
        for scenario_dir in self.scenario_dirs:
            camera_dirs = sorted(
                [os.path.join(scenario_dir, d) for d in os.listdir(scenario_dir) if d.upper().startswith("CAMERA_")]
            )
            assert camera_dirs, f"No CAMERA_* folders found in {scenario_dir}"

            camera_files = []
            for camera_dir in camera_dirs:
                # Ensure camera_dir is a directory
                assert os.path.isdir(camera_dir), f"{camera_dir} is not a directory"

                files = sorted(
                    [os.path.join(camera_dir, f) for f in os.listdir(camera_dir) if f.endswith(".jpeg")]
                )
                assert files, f"No .jpeg files found in {camera_dir}"
                camera_files.append(files)

            # Ensure all cameras have the same number of frames
            num_frames = len(camera_files[0])
            assert all(len(files) == num_frames for files in camera_files), (
                f"Mismatch in number of frames across cameras in {scenario_dir}"
            )
            self.camera_data.append((scenario_dir, camera_files))

            # Load EGO_INFO file paths
            ego_info_path = os.path.join(scenario_dir, self.ego_info_dir)
            assert os.path.isdir(ego_info_path), f"EGO_INFO directory not found: {ego_info_path}"
            ego_files = sorted(
                [os.path.join(ego_info_path, f) for f in os.listdir(ego_info_path) if f.endswith(".txt")]
            )
            assert ego_files, f"No EGO_INFO files found in {ego_info_path}"
            self.ego_data.append(ego_files)


        # Calculate total frames across scenarios
        self.num_frames = sum(len(camera_files[0]) - num_timesteps + 1 for _, camera_files in self.camera_data)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # Find which scenario the index belongs to
        cumulative_frames = 0
        for scenario_idx, (scenario_dir, camera_files) in enumerate(self.camera_data):
            num_scenario_frames = len(camera_files[0]) - self.num_timesteps + 1
            if idx < cumulative_frames + num_scenario_frames:
                frame_idx = idx - cumulative_frames
                break
            cumulative_frames += num_scenario_frames

        # Load temporal image data for the selected scenario and frame index
        temporal_images = []
        ego_info_data = []
        camera_indices = []

        for t in range(self.num_timesteps):
            images_per_camera = []
            for camera_idx in range(len(camera_files)):
                image_path = camera_files[camera_idx][frame_idx + t]
                image = Image.open(image_path).convert("RGB")  # Open image and ensure it's in RGB format
                image = self.image_transform(image)           # Apply transformations
                images_per_camera.append(image)
        
            camera_indices.append(frame_idx + t)

            # Stack along the camera axis
            temporal_images.append(torch.stack(images_per_camera, dim=0))  # (num_cameras, c, h, w)
            # Load corresponding EGO_INFO data
            ego_file_path = self.ego_data[scenario_idx][frame_idx + t]
            with open(ego_file_path, 'r') as f:
                ego_info_dict = {}
                for line in f:
                    key, value = line.split(":")
                    key = key.strip()
                    value = value.strip()
                    if key in {"position", "orientation", "enu_velocity", "velocity", "angularVelocity", "acceleration"}:
                        ego_info_dict[key] = list(map(float, value.split()))
                    elif key in {"accel", "brake", "steer"}:
                        # 그룹화된 accel, brake, steer 처리
                        if "control" not in ego_info_dict:
                            ego_info_dict["control"] = []  # 그룹 초기화
                        ego_info_dict["control"].append(float(value))
                
                # Flatten the dictionary into a single list of values
                ego_info_values = []
                for key in ["position", "orientation", "enu_velocity", "velocity", "angularVelocity", "acceleration", "control"]:
                    ego_info_values.extend(ego_info_dict.get(key, [0.0] * (3 if key != "control" else 3)))
                ego_info_data.append(ego_info_values)
                
        ego_info_tensor = torch.tensor(ego_info_data, dtype=torch.float32)

        # Stack along the time axis and swap time and cam_num dimensions
        temporal_images = torch.stack(temporal_images, dim=1)  # (num_cameras, time_steps, c, h, w)

        # Expand intrinsic and extrinsic matrices along time dimension
        intrinsic = torch.stack(self.intrinsic_data, dim=0).unsqueeze(1).repeat(1, self.num_timesteps, 1, 1)  # (num_cameras, time_steps, 3, 3)
        extrinsic = torch.stack(self.extrinsic_data, dim=0).unsqueeze(1).repeat(1, self.num_timesteps, 1, 1)  # (num_cameras, time_steps, 4, 4)

        # Load and process HD Map data
        hd_map_images = self._load_hd_map(scenario_dir, camera_indices)
        if hd_map_images is not None:
            hd_map_images = np.stack([self._process_hd_map(frame) for frame in hd_map_images])
            hd_map_images = torch.tensor(hd_map_images, dtype=torch.float32)  # Convert to tensor

        return {
            "images": temporal_images.permute(1, 0, 2, 3, 4),  # (batch, time, cam_num, c, h, w)
            "intrinsic": intrinsic.permute(1, 0, 2, 3),       # (batch, time, cam_num, 3, 3)
            "extrinsic": extrinsic.permute(1, 0, 2, 3),       # (batch, time, cam_num, 4, 4)
            "scenario": self.camera_data[scenario_idx][0],
            "hd_map": hd_map_images if hd_map_images is not None else None,  # (time_steps, channels, h, w)
            "ego_info": ego_info_tensor
        }

    def _process_hd_map(self, hd_map_frame):
        """Process HD Map frame into a multi-channel tensor."""
        exterior = (hd_map_frame[:, :, 0] > 200).astype(np.float32)  # Red channel
        interior = (hd_map_frame[:, :, 1] > 200).astype(np.float32)  # Green channel
        lane = (hd_map_frame[:, :, 2] > 200).astype(np.float32)      # Blue channel
        crosswalk = ((hd_map_frame[:, :, 0] > 200) & (hd_map_frame[:, :, 1] > 200)).astype(np.float32)  # Yellow
        traffic_light = ((hd_map_frame[:, :, 1] > 200) & (hd_map_frame[:, :, 2] > 200)).astype(np.float32)  # Cyan
        ego_vehicle = ((hd_map_frame[:, :, 0] > 200) & (hd_map_frame[:, :, 2] > 200)).astype(np.float32)  # Magenta

        return np.stack([exterior, interior, lane, crosswalk, traffic_light, ego_vehicle], axis=0)

    def _load_hd_map(self, scenario_path, camera_indices):
        """Load pre-generated HD Map images and synchronize with camera indices."""
        hd_map_path = os.path.join(scenario_path, self.hd_map_dir)

        if not os.path.exists(hd_map_path):
            print(f"Warning: HD Map directory not found: {hd_map_path}")
            return None

        # Extract and sort HD Map files by numerical order
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

        # Synchronize HD Map files with camera indices
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





'''
Image Shape: torch.Size([batch, time_step, 5, 3, 270, 480])
Intrinsics Shape: torch.Size([batch, time_step, 5, 3, 3])
Extrinsics Shape: torch.Size([batch, time_step, 5, 4, 4])
HD Map Shape: torch.Size([batch, time_step, 6, 256, 256])
Ego_info Shape: torch.Size([batch, time_step, 21])
'''