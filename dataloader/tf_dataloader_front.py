import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, base_root, map_name, batch_size=32, seq_len=5):
        self.base_root = base_root
        self.map_name = map_name
        self.batch_size = batch_size
        self.seq_len = seq_len  # 시퀀스 길이 추가
        self.scenario_paths = []
        self.camera_dirs = ["CAMERA_1"]
        self.all_camera_files = {}
        self.ego_frames = {}
        self.matched_frames = []
        self.current_scenario = None
        self.global_path = None  # Global Path 저장
        self._load_scenarios()

    def _load_scenarios(self):
        """Load all scenarios for the specified map."""
        map_scenarios = [
            os.path.join(self.base_root, d) for d in os.listdir(self.base_root)
            if d.startswith(self.map_name) and os.path.isdir(os.path.join(self.base_root, d))
        ]
        self.scenario_paths = sorted(map_scenarios)

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
                    key=lambda f: int(''.join(filter(str.isdigit, f)))
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

    def match_data(self):
        """Match camera and EGO_INFO files."""
        ego_frame_numbers = sorted(self.ego_frames.keys())
        self.matched_frames = []

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
                

    def _extract_frame_number(self, filename):
        """Extract frame number from file name."""
        return int(''.join(filter(str.isdigit, filename.split("_")[0])))

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
    def _prepare_ego_inputs(self, ego_input):
        """
        ego_input: EGO 상태 정보를 포함한 딕셔너리
        Returns: numpy array 형태의 EGO 상태 정보
        """
        # 필요한 키를 정의 (정렬된 순서로)
        keys = [
            "position", "orientation", "enu_velocity", "velocity",
            "angularVelocity", "acceleration", "linkid", "trafficlightid", "turn_signal_lamp"
        ]

        processed_input = []
        for key in keys:
            if isinstance(ego_input[key], list):  # 리스트인 경우 확장
                processed_input.extend(ego_input[key])
            elif isinstance(ego_input[key], (float, int)):  # 숫자인 경우 추가
                processed_input.append(ego_input[key])

        return np.array(processed_input, dtype=np.float32)

    def _compute_relative_path_for_item(self, ego_input):
        """
        ego_input: 현재 상태 정보를 포함한 딕셔너리 (position 포함)
        Returns: numpy array 형태의 상대 경로 정보
        """
        if self.global_path is None:
            # Global path가 없으면 0으로 채움
            return np.zeros(10, dtype=np.float32)

        # 현재 위치 추출
        current_position = np.array(ego_input["position"][:2])  # x, y 좌표

        # Global Path에서 가장 가까운 점 찾기
        distances = np.linalg.norm(self.global_path - current_position, axis=1)
        nearest_idx = np.argmin(distances)

        # 다음 n개의 경로 점 가져오기
        n_points = 5
        next_points = self.global_path[nearest_idx:nearest_idx + n_points]

        # 패딩 처리 (경로 점이 부족할 경우)
        if len(next_points) < n_points:
            padding = np.zeros((n_points - len(next_points), 2), dtype=np.float32)
            next_points = np.vstack((next_points, padding))

        # 현재 위치를 기준으로 상대 좌표 계산
        relative_path = next_points - current_position
        return relative_path.flatten()

    def __iter__(self):
        """Yield batches of data as sequences."""
        # 시퀀스를 구성하는 길이
        seq_len = 10  # 시퀀스 길이

        # 전체 데이터 시퀀스 구성
        sequences = [
            self.matched_frames[i:i + seq_len]
            for i in range(len(self.matched_frames) - seq_len + 1)
        ]

        # 배치로 나누기
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]  # 배치 추출

            # 시퀀스 데이터 준비
            try:
                camera_images = np.array([
                    [item["camera_data"] for item in seq] for seq in batch
                ])  # (batch_size, seq_len, H, W, C)

                ego_inputs = np.array([
                    [
                        np.concatenate(
                            [
                                self._prepare_ego_inputs(item["ego_data"]["input"]),
                                self._compute_relative_path_for_item(item["ego_data"]["input"])
                            ]
                        )
                        for item in seq
                    ]
                    for seq in batch
                ])  # (batch_size, seq_len, ego_dim)

                gt_data = np.array([
                    [list(item["ego_data"]["gt"].values()) for item in seq]
                    for seq in batch
                ])  # (batch_size, seq_len, 3)

            except KeyError as e:
                print("KeyError while preparing batch:", e)
                continue

            yield camera_images, ego_inputs, gt_data



        
