import os
import re

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.camera_dirs = [f"CAMERA_{i}" for i in range(1, 6)]
        self.ego_info_path = os.path.join(base_path, "EGO_INFO")
        self.object_info_path = os.path.join(base_path, "OBJECT_INFO")
        self.all_camera_files = {}
        self.ego_frames = {}
        self.object_frames = {}
        self.matched_data = {}

        self._load_data()

    def _load_data(self):
        """Load data from the specified directories."""
        self._load_camera_files()
        self._load_ego_info_files()
        self._load_object_info_files()

    def _load_camera_files(self):
        """Load camera files from CAMERA_1 ~ CAMERA_5."""
        for camera_dir in self.camera_dirs:
            camera_path = os.path.join(self.base_path, camera_dir)
            if os.path.exists(camera_path):
                camera_files = sorted([f for f in os.listdir(camera_path) if f.endswith(".jpeg")])
                camera_files_with_path = {os.path.join(camera_path, f): f for f in camera_files}
                self.all_camera_files[camera_dir] = camera_files_with_path

    def _load_ego_info_files(self):
        """Load EGO_INFO files."""
        if os.path.exists(self.ego_info_path):
            ego_files = sorted([f for f in os.listdir(self.ego_info_path) if f.endswith(".txt")])
            self.ego_frames = {
                int(''.join(filter(str.isdigit, f.split("_")[0]))): os.path.join(self.ego_info_path, f)
                for f in ego_files
            }

    def _load_object_info_files(self):
        """Load OBJECT_INFO files and extract frame numbers."""
        if os.path.exists(self.object_info_path):
            object_files = sorted([f for f in os.listdir(self.object_info_path) if f.endswith(".txt")])
            for f in object_files:
                match = re.search(r"\d+", f)
                if match:
                    frame_number = int(match.group())
                    self.object_frames[frame_number] = os.path.join(self.object_info_path, f)

    def _extract_frame_number(self, filename):
        """Extract frame number from file name."""
        return int(''.join(filter(str.isdigit, filename.split("_")[0])))

    def match_data(self):
        """Match camera, EGO_INFO, and OBJECT_INFO files."""
        self.matched_data = {}
        ego_frame_numbers = sorted(self.ego_frames.keys())
        object_frame_numbers = sorted(self.object_frames.keys())

        for camera_dir, camera_files in self.all_camera_files.items():
            matched_frames = []
            for camera_file_path, camera_file in camera_files.items():
                camera_frame = self._extract_frame_number(camera_file)

                # Find closest frames
                closest_ego_frame = min(ego_frame_numbers, key=lambda ego_frame: abs(ego_frame - camera_frame))
                closest_object_frame = min(object_frame_numbers, key=lambda obj_frame: abs(obj_frame - camera_frame))

                matched_frames.append({
                    "camera_frame": camera_frame,
                    "camera_file": camera_file_path,
                    "ego_frame": closest_ego_frame,
                    "ego_file": self.ego_frames[closest_ego_frame],
                    "object_frame": closest_object_frame,
                    "object_file": self.object_frames[closest_object_frame],
                })

            self.matched_data[camera_dir] = matched_frames

    def debug_output(self):
        """Print debug information about the matched data."""
        for camera_dir, matched_frames in self.matched_data.items():
            print(f"=== {camera_dir} 매칭된 프레임 ===")
            for frame in matched_frames:
                print(
                    f"Camera Frame {frame['camera_frame']}: {frame['camera_file']} -> "
                    f"Closest EGO Frame {frame['ego_frame']}: {frame['ego_file']} -> "
                    f"Closest Object Frame {frame['object_frame']}: {frame['object_file']}"
                )

            print("\n=== 디버깅 결과 ===")
            if len(matched_frames) == len(self.all_camera_files[camera_dir]):
                print(f"{camera_dir}: 모든 카메라 프레임이 EGO_INFO 및 OBJECT_INFO 데이터와 매칭되었습니다.")
            else:
                print(f"{camera_dir}: 일부 카메라 프레임이 매칭되지 않았습니다.")

# 사용 예제
if __name__ == "__main__":
    base_path = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_0"
    data_loader = DataLoader(base_path)
    data_loader.match_data()
    data_loader.debug_output()
