import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -------------------------
# (1) 고정된 맵 관련 데이터 로드
# -------------------------
road_mesh_file = '/home/vip/hd/Dataset_sample/Map_Data/R_KR_PR_Sangam_DP/road_mesh_out_line_polygon_set.json'
lane_boundary_file = '/home/vip/hd/Dataset_sample/Map_Data/R_KR_PR_Sangam_DP/lane_boundary_set.json'
crosswalk_file = '/home/vip/hd/Dataset_sample/Map_Data/R_KR_PR_Sangam_DP/singlecrosswalk_set.json'
traffic_light_file = '/home/vip/hd/Dataset_sample/Map_Data/R_KR_PR_Sangam_DP/traffic_light_set.json'

with open(road_mesh_file, 'r') as file:
    road_mesh_data = json.load(file)

with open(lane_boundary_file, 'r') as file:
    lane_data = json.load(file)

lane_boundary_points_by_type = {}
for entry in lane_data:
    lane_type = str(entry.get('lane_type', 'unknown'))
    if lane_type not in lane_boundary_points_by_type:
        lane_boundary_points_by_type[lane_type] = []
    lane_boundary_points_by_type[lane_type].append(entry['points'])

with open(crosswalk_file, 'r') as file:
    crosswalk_data = json.load(file)
crosswalk_points = [entry['points'] for entry in crosswalk_data]

with open(traffic_light_file, 'r') as file:
    traffic_light_data = json.load(file)

traffic_light_points_by_subtype = {}
for entry in traffic_light_data:
    sub_type = str(entry.get('sub_type', 'unknown'))
    if sub_type not in traffic_light_points_by_subtype:
        traffic_light_points_by_subtype[sub_type] = []
    traffic_light_points_by_subtype[sub_type].append(entry['point'][:2])

# -------------------------
# (2) 도로 mesh(Exterior/Interior) 분류 함수
# -------------------------
def classify_and_extract_points(data):
    classified_points = {}
    for polygon in data:
        polygon_class = polygon.get('class', 'unknown')
        if polygon_class not in classified_points:
            classified_points[polygon_class] = []
        if 'points' in polygon:
            classified_points[polygon_class].append(polygon['points'])
    return classified_points

classified_exterior_points = classify_and_extract_points(road_mesh_data)
classified_interior_points = classify_and_extract_points(road_mesh_data)

# -------------------------
# (3) EGO_INFO 파싱 함수
# -------------------------
def parse_ego_info(file_path):
    ego_info = {}
    if not os.path.isfile(file_path):
        return ego_info
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            ego_info[key.strip()] = value.strip()
    return ego_info

# -------------------------
# (4) 객체 정보 파싱 함수
# -------------------------
def parse_object_info(file_path):
    objects = []
    if not os.path.isfile(file_path):
        return objects
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 10:
                continue
            numeric_vals = list(map(float, tokens[1:]))

            x     = numeric_vals[0]
            y     = numeric_vals[1]
            z     = numeric_vals[2]
            roll  = numeric_vals[3]
            pitch = numeric_vals[4]
            yaw   = numeric_vals[5]  # degree
            length= numeric_vals[6]
            width = numeric_vals[7]
            height= numeric_vals[8]
            
            obj_info = {
                'x': x,
                'y': y,
                'z': z,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'length': length,
                'width': width,
                'height': height
            }
            objects.append(obj_info)
    return objects

# -------------------------
# (5) 유틸 함수
# -------------------------
def calculate_distance(point, ego_pos):
    return np.sqrt((point[0] - ego_pos[0])**2 + (point[1] - ego_pos[1])**2)

def filter_points_within_range(points, ego_pos, max_distance):
    filtered = []
    for polygon in points:
        filtered_polygon = [p for p in polygon if calculate_distance(p, ego_pos) <= max_distance]
        if filtered_polygon:
            filtered.append(filtered_polygon)
    return filtered

def compute_relative_path(global_path, ego_position, n_points=5):
    distances = np.linalg.norm(global_path - ego_position[:2], axis=1)
    nearest_idx = np.argmin(distances)
    next_points = global_path[nearest_idx:nearest_idx + n_points]
    if len(next_points) < n_points:
        padding = np.zeros((n_points - len(next_points), 2), dtype=np.float32)
        next_points = np.vstack((next_points, padding))
    relative_path = next_points - ego_position[:2]
    return relative_path

def assign_colors(categories):
    cmap = cm.get_cmap('tab20', len(categories))
    return {category: cmap(idx) for idx, category in enumerate(categories)}

# -------------------------
# (5-1) 로컬 변환 함수
# -------------------------
def rotate_point(point, theta):
    c, s = np.cos(theta), np.sin(theta)
    x, y = point
    x_rot = x*c + y*s
    y_rot = -x*s + y*c
    return (x_rot, y_rot)

def global_to_local_north_up(point_global, ego_pos, ego_yaw_deg):
    ego_yaw_rad = np.deg2rad(ego_yaw_deg)
    shifted = (point_global[0] - ego_pos[0],
               point_global[1] - ego_pos[1])
    theta = ego_yaw_rad - (np.pi / 2)
    return rotate_point(shifted, theta)

def polygon_global_to_local_north_up(polygons, ego_pos, ego_yaw_deg):
    local_polygons = []
    for polygon in polygons:
        local_polygon = []
        for pt in polygon:
            local_pt = global_to_local_north_up(pt, ego_pos, ego_yaw_deg)
            local_polygon.append(local_pt)
        local_polygons.append(local_polygon)
    return local_polygons

# -------------------------
# (객체용) 바운딩 박스 corner 계산
# -------------------------
def get_bbox_corners_global(x, y, yaw_deg, length, width):
    """
    중심(x, y)에서 yaw_deg만큼 회전된 사각형 corners (4개)
    순서: [왼아래, 왼위, 오른위, 오른아래] 처럼 시계/반시계
    """
    half_l = length * 0.5
    half_w = width  * 0.5
    
    # (로컬 좌표) -> yaw=0
    # 여기서 앞쪽을 +X, 왼쪽을 +Y 라고 가정
    corners_local = [
        (-half_l, -half_w),  # 좌하
        (-half_l,  half_w),  # 좌상
        ( half_l,  half_w),  # 우상
        ( half_l, -half_w)   # 우하
    ]
    
    yaw_rad = np.deg2rad(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    
    corners_global = []
    for lx, ly in corners_local:
        gx = x + (lx*cos_y - ly*sin_y)
        gy = y + (lx*sin_y + ly*cos_y)
        corners_global.append((gx, gy))
    
    return corners_global

# -------------------------
# (6) 시각화용 함수
# -------------------------
def plot_polygon_local(polygons, color='blue'):
    for polygon in polygons:
        x = [pt[0] for pt in polygon]
        y = [pt[1] for pt in polygon]
        plt.plot(x, y, color=color)

# 맵 데이터 색상
lane_boundary_colors = assign_colors([])
exterior_colors      = assign_colors([])
interior_colors      = assign_colors([])
traffic_light_colors = assign_colors([])

lane_boundary_colors = assign_colors(lane_boundary_points_by_type.keys())
exterior_colors      = assign_colors(classified_exterior_points.keys())
interior_colors      = assign_colors(classified_interior_points.keys())
traffic_light_colors = assign_colors(traffic_light_points_by_subtype.keys())

# -------------------------
# (7) 시나리오별 처리
# -------------------------
for scenario_idx in range(254, 373):
    scenario_dir = f'/home/vip/hd/Dataset/R_KR_PR_Sangam_DP__HMG_Scenario_{scenario_idx}'
    
    if not os.path.isdir(scenario_dir):
        continue
    
    global_path_file = os.path.join(scenario_dir, 'global_path.csv')
    ego_info_dir     = os.path.join(scenario_dir, 'EGO_INFO')
    object_info_dir  = os.path.join(scenario_dir, 'OBJECT_INFO')  # <- 폴더 (가정)
    output_dir       = os.path.join(scenario_dir, 'GT_HD_MAP')
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isfile(global_path_file):
        print(f"[WARNING] {global_path_file} 파일이 존재하지 않습니다. 스킵.")
        continue
    
    global_path_data = pd.read_csv(global_path_file)
    global_path_points = global_path_data[['PositionX (m)', 'PositionY (m)']].values
    
    if not os.path.isdir(ego_info_dir):
        print(f"[WARNING] {ego_info_dir} 폴더가 존재하지 않습니다. 스킵.")
        continue
    
    txt_files = sorted(glob.glob(os.path.join(ego_info_dir, '*.txt')))
    
    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        try:
            file_number = int(base_name)
        except ValueError:
            continue
        
        if file_number % 10 != 0:
            continue
        
        ego_info = parse_ego_info(txt_file)
        if ('position' not in ego_info) or ('orientation' not in ego_info):
            continue
        
        ego_position = np.array(list(map(float, ego_info['position'].split())))
        orientation_list = list(map(float, ego_info['orientation'].split()))
        if len(orientation_list) < 3:
            continue
        ego_yaw_deg = orientation_list[-1]
        
        # 객체 파일
        object_file = os.path.join(object_info_dir, f'object_info_{file_number}.txt')
        object_list = parse_object_info(object_file)
        
        # 맵 필터링(50m)
        filtered_classified_exterior_points = {
            cls: filter_points_within_range(points, ego_position[:2], 50)
            for cls, points in classified_exterior_points.items()
        }
        filtered_classified_interior_points = {
            cls: filter_points_within_range(points, ego_position[:2], 50)
            for cls, points in classified_interior_points.items()
        }
        filtered_lane_boundary_points_by_type = {
            lt: filter_points_within_range(points, ego_position[:2], 50)
            for lt, points in lane_boundary_points_by_type.items()
        }
        filtered_traffic_light_points_by_subtype = {
            st: [
                pt for pt in pts if calculate_distance(pt, ego_position[:2]) <= 50
            ]
            for st, pts in traffic_light_points_by_subtype.items()
        }
        filtered_crosswalk_points = filter_points_within_range(crosswalk_points, ego_position[:2], 50)
        
        relative_path = compute_relative_path(global_path_points, ego_position, n_points=5)
        
        # 시각화
        plt.figure(figsize=(10, 10))
        
        # Exterior
        for cls, polygons in filtered_classified_exterior_points.items():
            local_polygons = polygon_global_to_local_north_up(polygons, ego_position[:2], ego_yaw_deg)
            plot_polygon_local(local_polygons, color=exterior_colors[cls])
        
        # Interior
        for cls, polygons in filtered_classified_interior_points.items():
            local_polygons = polygon_global_to_local_north_up(polygons, ego_position[:2], ego_yaw_deg)
            plot_polygon_local(local_polygons, color=interior_colors[cls])
        
        # Lane boundary
        for lt, polygons in filtered_lane_boundary_points_by_type.items():
            local_polygons = polygon_global_to_local_north_up(polygons, ego_position[:2], ego_yaw_deg)
            plot_polygon_local(local_polygons, color=lane_boundary_colors[lt])
        
        # Traffic lights
        for st, pts in filtered_traffic_light_points_by_subtype.items():
            for pt in pts:
                local_pt = global_to_local_north_up(pt, ego_position[:2], ego_yaw_deg)
                plt.scatter(local_pt[0], local_pt[1],
                            color=traffic_light_colors[st],
                            s=80, zorder=5
                            )
        
        # Crosswalk
        local_crosswalks = polygon_global_to_local_north_up(filtered_crosswalk_points, ego_position[:2], ego_yaw_deg)
        plot_polygon_local(local_crosswalks, color='orange')
        
        # Global path
        for r_pt in relative_path:
            global_pt = (r_pt[0] + ego_position[0], r_pt[1] + ego_position[1])
            local_pt = global_to_local_north_up(global_pt, ego_position[:2], ego_yaw_deg)
            plt.scatter(local_pt[0], local_pt[1], color='cyan', s=50, zorder=5)
        
        # Ego 차량
        plt.scatter(0, 0, color='green', s=100, zorder=5)
        
        # 객체 (박스 + fill)
        fill_color = 'gray'
        for obj in object_list:
            ox, oy = obj['x'], obj['y']
            if calculate_distance((ox, oy), ego_position[:2]) > 50:
                continue
            
            yaw_deg_obj = obj['yaw']
            length_obj  = obj['length']
            width_obj   = obj['width']
            
            # 4개 코너(전역)
            corners_global = get_bbox_corners_global(
                x=ox, y=oy, yaw_deg=yaw_deg_obj,
                length=length_obj, width=width_obj
            )
            
            # 로컬 변환
            corners_local = [
                global_to_local_north_up(pt, ego_position[:2], ego_yaw_deg)
                for pt in corners_global
            ]
            # 폴리곤 폐합(첫 점 다시 추가)
            corners_local.append(corners_local[0])
            
            # x,y 분리
            x_vals = [c[0] for c in corners_local]
            y_vals = [c[1] for c in corners_local]
            
            # 내부 채우기
            plt.fill(x_vals, y_vals, color=fill_color, alpha=0.5)
            # # 테두리는 검은색(선택)
            # plt.plot(x_vals, y_vals, color='black', linewidth=1)
            
            # # 중앙점 찍기(선택)
            # center_local = global_to_local_north_up((ox, oy), ego_position[:2], ego_yaw_deg)
            # plt.scatter(center_local[0], center_local[1], color='black', s=20)
        
        # Plot config
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.legend(loc='upper right', fontsize=8)
        
        save_name = f'Scenario_{scenario_idx}_EGO_{file_number}.png'
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"[Scenario {scenario_idx}] {file_number}.txt -> {save_path} 저장 완료.")

print("모든 시나리오의 10단위 EGO_INFO 파일 시각화 + 박스 fill 완료.")
