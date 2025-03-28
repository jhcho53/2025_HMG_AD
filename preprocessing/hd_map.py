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
road_mesh_file = '/home/jaehyeon/Desktop/VIPLAB/HD_E2E/Map_Data/R_KR_PG_KATRI/road_mesh_out_line_polygon_set.json'
lane_boundary_file = '/home/jaehyeon/Desktop/VIPLAB/HD_E2E/Map_Data/R_KR_PG_KATRI/lane_boundary_set.json'
crosswalk_file = '/home/jaehyeon/Desktop/VIPLAB/HD_E2E/Map_Data/R_KR_PG_KATRI/singlecrosswalk_set.json'
traffic_light_file = '/home/jaehyeon/Desktop/VIPLAB/HD_E2E/Map_Data/R_KR_PG_KATRI/traffic_light_set.json'

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
    # traffic_light_set.json 예시에 따라 point는 [x, y, z] 형태라고 가정
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
    with open(file_path, 'r') as f:
        for line in f:
            # 'key: value' 형태라 가정
            key, value = line.split(':', 1)
            ego_info[key.strip()] = value.strip()
    return ego_info

# -------------------------
# (4) 유틸 함수
# -------------------------
def calculate_distance(point, ego_pos):
    return np.sqrt((point[0] - ego_pos[0])**2 + (point[1] - ego_pos[1])**2)

def filter_points_within_range(points, ego_pos, max_distance):
    """
    points: 
        [
          [ (x1, y1), (x2, y2), ... ],  # 여러 폴리곤 중 하나
          [ (x1, y1), (x2, y2), ... ],  # 또 다른 폴리곤
          ...
        ]
    """
    filtered = []
    for polygon in points:
        filtered_polygon = [
            p for p in polygon
            if calculate_distance(p, ego_pos) <= max_distance
        ]
        if filtered_polygon:
            filtered.append(filtered_polygon)
    return filtered

def compute_relative_path(global_path, ego_position, n_points=5):
    """
    global_path: shape (N, 2) (전역 x, y)
    ego_position: (x, y, z)
    """
    distances = np.linalg.norm(global_path - ego_position[:2], axis=1)
    nearest_idx = np.argmin(distances)
    next_points = global_path[nearest_idx:nearest_idx + n_points]
    # 패딩 (만약 데이터가 부족할 경우)
    if len(next_points) < n_points:
        padding = np.zeros((n_points - len(next_points), 2), dtype=np.float32)
        next_points = np.vstack((next_points, padding))
    # ego_position을 기준으로 상대 좌표를 단순 계산 (원점 고려 없이 빼기만 하는 예시)
    # 실제로는 바로 쓰기보다는 global->local 회전 변환이 필요할 수 있음
    relative_path = next_points - ego_position[:2]
    return relative_path

def assign_colors(categories):
    cmap = cm.get_cmap('tab20', len(categories))
    return {category: cmap(idx) for idx, category in enumerate(categories)}

# -------------------------
# (4-1) BEV 변환(전역 -> 로컬) 유틸 함수
# -------------------------
def rotate_point(point, theta):
    """
    point: (x, y)
    theta: 이 함수는 'R(-θ) = [[cosθ, sinθ],
                             [-sinθ, cosθ]]' 역할(즉, -θ 회전)
    """
    c, s = np.cos(theta), np.sin(theta)
    x, y = point
    x_rot = x*c + y*s
    y_rot = -x*s + y*c
    return (x_rot, y_rot)

def global_to_local_north_up(point_global, ego_pos, ego_yaw_deg):
    """
    point_global: (x, y) in global frame
    ego_pos: (x_ego, y_ego)
    ego_yaw_deg: 차량 yaw (도, degree)
    
    - yaw_deg=0 -> 전역 상 +x를 본다고 가정
    - 로컬 좌표에서 '차량 진행 방향'을 +y로 맞추려면
      실제 + (pi/2 - yaw_rad) 회전이 필요함.
    - 하지만 rotate_point가 -θ 회전 행렬이므로, 인자= (yaw_rad - pi/2)를 넣으면 됨.
    """
    # 1) degree -> rad 변환
    ego_yaw_rad = np.deg2rad(ego_yaw_deg)
    
    # 2) 평행이동
    shifted = (point_global[0] - ego_pos[0],
               point_global[1] - ego_pos[1])
    
    # 3) 최종 회전 (rotate_point는 -θ 회전)
    theta = ego_yaw_rad - (np.pi / 2)
    return rotate_point(shifted, theta)

def polygon_global_to_local_north_up(polygons, ego_pos, ego_yaw_deg):
    """
    polygons: [ [ (x1,y1), (x2,y2), ... ],  [ ... ] ]
    return: 동일 구조로 변환된 로컬 폴리곤들
    (차량 방향이 항상 +y 축)
    """
    local_polygons = []
    for polygon in polygons:
        local_polygon = []
        for pt in polygon:
            local_pt = global_to_local_north_up(pt, ego_pos, ego_yaw_deg)
            local_polygon.append(local_pt)
        local_polygons.append(local_polygon)
    return local_polygons

# -------------------------
# (5) 시각화용 함수
# -------------------------
def plot_polygon(points, color='blue'):
    """
    points: [ [ (x1,y1), (x2,y2), ... ],  [ ... ], ... ]
    """
    for polygon in points:
        x = [pt[0] for pt in polygon]
        y = [pt[1] for pt in polygon]
        plt.plot(x, y, color=color)

def plot_polygon_local(polygons, color='blue'):
    """로컬 좌표계(polygons) 시각화"""
    for polygon in polygons:
        x = [pt[0] for pt in polygon]
        y = [pt[1] for pt in polygon]
        plt.plot(x, y, color=color)

# 사전 색상 할당(맵 데이터용)
lane_boundary_colors = assign_colors(lane_boundary_points_by_type.keys())
traffic_light_colors = assign_colors(traffic_light_points_by_subtype.keys())
exterior_colors = assign_colors(classified_exterior_points.keys())
interior_colors = assign_colors(classified_interior_points.keys())

# -------------------------
# (6) 시나리오를 순회하며 작업
# -------------------------
# 예: R_KR_PG_KATRI__HMG_Scenario_0, R_KR_PG_KATRI__HMG_Scenario_1, ...
for scenario_idx in range(0, 50):
    scenario_dir = f'/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_{scenario_idx}'
    
    # 해당 시나리오 폴더가 존재하지 않으면 스킵
    if not os.path.isdir(scenario_dir):
        continue
    
    # 시나리오별 global_path.csv 경로
    global_path_file = os.path.join(scenario_dir, 'global_path.csv')
    # EGO_INFO 폴더
    ego_info_dir = os.path.join(scenario_dir, 'EGO_INFO')
    # 출력 이미지 폴더
    output_dir = os.path.join(scenario_dir, 'HD_MAP')
    os.makedirs(output_dir, exist_ok=True)
    
    # global_path.csv 존재 여부 확인
    if not os.path.isfile(global_path_file):
        print(f"[WARNING] {global_path_file} 파일이 존재하지 않습니다. 스킵.")
        continue
    
    # global path 로드
    global_path_data = pd.read_csv(global_path_file)
    global_path_points = global_path_data[['PositionX (m)', 'PositionY (m)']].values
    
    # EGO_INFO 폴더가 없으면 스킵
    if not os.path.isdir(ego_info_dir):
        print(f"[WARNING] {ego_info_dir} 폴더가 존재하지 않습니다. 스킵.")
        continue
    
    # EGO_INFO 내 모든 txt 파일 순회
    txt_files = sorted(glob.glob(os.path.join(ego_info_dir, '*.txt')))
    
    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        
        # 파일 이름이 숫자가 아닐 경우 스킵
        try:
            file_number = int(base_name)
        except ValueError:
            continue
        
        # 10단위 파일만 처리
        if file_number % 10 != 0:
            continue
        
        # EGO_INFO 파싱
        ego_info = parse_ego_info(txt_file)
        
        # 예) ego_info['position'] -> "x y z"
        #    ego_info['orientation'] -> "roll pitch yaw(도)"  <- 여기서 yaw는 degree로 가정
        ego_position = np.array(list(map(float, ego_info['position'].split())))  # (x, y, z)
        orientation_list = list(map(float, ego_info['orientation'].split()))
        ego_yaw_deg = orientation_list[-1]  # roll, pitch, yaw(도)
        
        # 맵 데이터 필터링 (30m 이내)
        filtered_classified_exterior_points = {
            cls: filter_points_within_range(points, ego_position[:2], 30)
            for cls, points in classified_exterior_points.items()
        }
        filtered_classified_interior_points = {
            cls: filter_points_within_range(points, ego_position[:2], 30)
            for cls, points in classified_interior_points.items()
        }
        filtered_lane_boundary_points_by_type = {
            lt: filter_points_within_range(points, ego_position[:2], 30)
            for lt, points in lane_boundary_points_by_type.items()
        }
        filtered_traffic_light_points_by_subtype = {
            st: [
                pt for pt in pts
                if calculate_distance(pt, ego_position[:2]) <= 30
            ]
            for st, pts in traffic_light_points_by_subtype.items()
        }
        filtered_crosswalk_points = filter_points_within_range(crosswalk_points, ego_position[:2], 30)
        
        # Global Path에서 ego_position 주변 n개 점 선택 (상대 좌표)
        relative_path = compute_relative_path(global_path_points, ego_position, n_points=5)
        
        # -------------------------
        # (7) 시각화: "차량 방향을 북쪽(+y 축)"으로 고정한 BEV(Ego 중심) 시점
        # -------------------------
        plt.figure(figsize=(10, 10))
        
        # (7-1) Exterior (Global -> Local 변환 후 플롯)
        for cls, polygons in filtered_classified_exterior_points.items():
            local_polygons = polygon_global_to_local_north_up(polygons, ego_position[:2], ego_yaw_deg)
            plot_polygon_local(local_polygons, color=exterior_colors[cls])
        
        # (7-2) Interior
        for cls, polygons in filtered_classified_interior_points.items():
            local_polygons = polygon_global_to_local_north_up(polygons, ego_position[:2], ego_yaw_deg)
            plot_polygon_local(local_polygons, color=interior_colors[cls])
        
        # (7-3) Lane Boundaries
        for lt, polygons in filtered_lane_boundary_points_by_type.items():
            local_polygons = polygon_global_to_local_north_up(polygons, ego_position[:2], ego_yaw_deg)
            plot_polygon_local(local_polygons, color=lane_boundary_colors[lt])
        
        # (7-4) Traffic Lights (점)
        for st, pts in filtered_traffic_light_points_by_subtype.items():
            for pt in pts:
                local_pt = global_to_local_north_up(pt, ego_position[:2], ego_yaw_deg)
                plt.scatter(local_pt[0], local_pt[1], color=traffic_light_colors[st], s=80, zorder=5)
        
        # (7-5) Crosswalk
        local_crosswalks = polygon_global_to_local_north_up(filtered_crosswalk_points, ego_position[:2], ego_yaw_deg)
        plot_polygon_local(local_crosswalks, color='orange')
        
        # (7-6) Global Path 일부 점 (Ego 기준 표시)
        #       relative_path 자체는 단순히 (global_path - ego_pos)지만,
        #       실제 회전을 적용해야 정확히 '차량이 북쪽을 보게' 표현됨
        for r_pt in relative_path:
            global_pt = (r_pt[0] + ego_position[0], r_pt[1] + ego_position[1])
            local_pt = global_to_local_north_up(global_pt, ego_position[:2], ego_yaw_deg)
            plt.scatter(local_pt[0], local_pt[1], color='cyan', s=50, zorder=5)
        
        # (7-7) Ego Vehicle 위치 (로컬 좌표계에선 항상 (0, 0))
        plt.scatter(0, 0, color='green', s=100, zorder=5)
        
        # 축 그리드, 축 범위 등
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')   # 필요 시 on/off
        plt.legend()
        
        # 이미지를 파일로 저장
        save_name = f'Scenario_{scenario_idx}_EGO_{file_number}.png'
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"[Scenario {scenario_idx}] {file_number}.txt -> {save_path} 저장 완료.")

print("모든 시나리오의 10단위 EGO_INFO 파일 시각화(차량 방향 북쪽 고정, yaw=degree) 완료되었습니다.")
