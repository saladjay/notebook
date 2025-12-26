import random
from copy import deepcopy
region_values = [500000000,5000000000000,0,0]
def calculate_iou(box1, box2):
    """
    计算两个矩形的IOU (交并比)
    矩形格式: [x1, y1, x2, y2] (左上角坐标和右下角坐标)
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 避免除零错误
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def generate_partial_overlap_boxes(iou_target=0.3, box_sizes:list[int, int]=[100, 100], distance_ratio=0.5):
    """
    生成部分重叠的两个矩形，通过控制距离来调整IOU值。
    
    参数:
        iou_target: 目标IOU值
        box_size: 矩形尺寸
        distance_ratio: 两矩形中心点距离与边长的比例 (0-1)
    """
    # 第一个矩形位置固定
    box1 = [50, 50, 50 + box_sizes[0], 50 + box_sizes[1]]
    
    # 根据距离比例计算第二个矩形的位置
    distance = int(box_sizes[0] * distance_ratio)
    box2 = [50 + distance, 50 + distance, 
            50 + distance + box_sizes[0], 50 + distance + box_sizes[1]]
    
    actual_iou = calculate_iou(box1, box2)
    
    # 调整距离以达到目标IOU
    max_iterations = 100
    for i in range(max_iterations):
        current_iou = calculate_iou(box1, box2)
        
        if abs(current_iou - iou_target) < 0.01:  # 容差
            break
        
        # 根据当前IOU调整距离
        if current_iou > iou_target:
            distance += 2  # 增加距离降低IOU
        else:
            distance -= 2  # 减少距离增加IOU
            
        # 更新第二个矩形位置
        box2 = [50 + distance, 50 + distance, 
                50 + distance + box_sizes[0], 50 + distance + box_sizes[1]]
    
    return box1, box2, calculate_iou(box1, box2)

def _iou(region1, region2):
    x1 = max(region1["x"], region2["x"])
    y1 = max(region1["y"], region2["y"])
    x2 = min(region1["x"] + region1["w"], region2["x"] + region2["w"])
    y2 = min(region1["y"] + region1["h"], region2["y"] + region2["h"])
    intersection = max(0, abs(x2 - x1)) * max(0, abs(y2 - y1))
    area1 = region1["w"] * region1["h"]
    area2 = region2["w"] * region2["h"]
    return intersection / (area1 + area2 - intersection)

def get_pseudo_label_regions(nb_regions = 1, iou = None, special_wh_ratio = None, width_range = None, height_range = None):
    regions = []
    global region_values
    if iou is None:
        for i in range(nb_regions):
            regions.append({
                "x": random.randint(0, 100),
                "y": random.randint(0, 100),
                "w": random.randint(0, 100),
                "h": random.randint(0, 100),
                "severe": random.random() * 10,
            })
            region_values[0] = min(regions[-1]['x'], region_values[0])
            region_values[1] = max(regions[-1]['y'], region_values[1])
            region_values[2] = max(regions[-1]['w'], region_values[2])
            region_values[3] = max(regions[-1]['h'], region_values[3])
        return regions
    else:
        for i in range(nb_regions // 2 + 1):
            box1, box2, _iou = generate_partial_overlap_boxes(iou_target=iou, box_size=100, distance_ratio=0.5)
            for box in [box1, box2]:
                regions.append({
                    "x": box[0],
                    "y": box[1],
                    "w": box[2] - box[0],
                    "h": box[3] - box[1],
                    "severe": random.random() * 10,
                })
                region_values[0] = min(regions[-1]['x'], region_values[0])
                region_values[1] = max(regions[-1]['y'], region_values[1])
                region_values[2] = max(regions[-1]['w'], region_values[2])
                region_values[3] = max(regions[-1]['h'], region_values[3])
        return regions
    
def get_pseudo_region_ranges():
    global region_values
    return deepcopy(region_values)