import json
import os
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLConfig
import torch
import torch.nn as nn
import re
from PIL import ImageDraw
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import copy

# 系统提示词
SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''

SYSTEM_PROMPT = SYSTEM_PROMPT.strip()

class ScreenSpotDataset(Dataset):
    def __init__(self, json_data_list, base_image_dir):
        self.json_data_list = json_data_list
        self.base_image_dir = base_image_dir
        
    def __len__(self):
        return len(self.json_data_list)
    
    def __getitem__(self, idx):
        return self.json_data_list[idx]

def custom_collate_fn(batch):
    """自定义collate函数，直接返回列表"""
    return batch

class EvaluationSampler:
    """
    评估用的采样器，确保不重复不丢弃样本
    """
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(dataset)
        
        # 计算每个rank的样本数
        self.samples_per_rank = self.total_size // self.num_replicas
        self.remainder = self.total_size % self.num_replicas
        
        # 计算该rank的样本范围
        self.start_idx = self.rank * self.samples_per_rank + min(self.rank, self.remainder)
        self.end_idx = self.start_idx + self.samples_per_rank + (1 if self.rank < self.remainder else 0)
        
        self.num_samples = self.end_idx - self.start_idx
        self.indices = list(range(self.start_idx, self.end_idx))
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return self.num_samples

def _is_point_in_rectangle(point, bbox):
    """检查点是否在矩形边界框内"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def _is_point_in_polygon(point, polygon):
    """检查点是否在多边形内"""
    x, y = point
    n = len(polygon) // 2
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i

    return inside

def is_point_in_target(point, box_type, box_coordinates, image_size):
    """
    根据box_type判断点是否在目标区域内
    """
    if box_type == "bbox":
        # bbox格式: [x, y, width, height]
        x, y, width, height = box_coordinates
        bbox_rect = [x, y, x + width, y + height]
        return _is_point_in_rectangle(point, bbox_rect)
    
    elif box_type == "polygon":
        # polygon格式: [x1, y1, x2, y2, ...]
        return _is_point_in_polygon(point, box_coordinates)
    
    elif box_type == "refusal":
        # refusal格式: 所有点坐标应为负值
        return all(coord < 0 for coord in point)
    
    else:
        print(f"Warning: Unknown box_type: {box_type}")
        return False

def get_top_attention_regions(image, instruction, processor, model, device, top_k=100, patch_size=28, max_regions=10, subimage_width=1288, subimage_height=728):
    """
    基于attention scores选择覆盖最多top attention token的子图区域，并按覆盖点数排序
    """
    if hasattr(model, 'module'):
        original_model = model.module
    else:
        original_model = model
    
    width, height = image.size
    
    # 调整图片大小用于获取attention scores
    resized_height, resized_width = smart_resize(
        height, width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height))
    
    # 准备单图消息
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)
    }
    
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": resized_image},
            {"type": "text", "text": instruction}
        ]
    }
    
    # 处理输入
    image_inputs, video_inputs = process_vision_info([system_message, user_message])
    text = processor.apply_chat_template([system_message, user_message], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    
    # 前向传播获取attention scores
    with torch.no_grad():
        outputs = original_model.generate(**inputs, max_new_tokens=32, do_sample=False, temperature=0.0, use_cache=True, pad_token_id=151645)
        captured_states = original_model.get_captured_states()
    
    # 获取attention scores
    query_states = captured_states["query_states"]
    key_states = captured_states["key_states"]
    
    batch_size, num_heads, q_len, head_dim = query_states.shape
    
    # 计算attention scores
    attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
    attention_scores = attention_scores.squeeze(2)
    
    # 应用softmax得到attention weights
    attention_weights = nn.functional.softmax(attention_scores, dim=-1)
    attention_weights = attention_weights.max(dim=1)[0]  # (1, L_v)
    
    # 获取视觉token的attention scores
    input_ids = inputs["input_ids"]
    img_id_starts = torch.where(input_ids[0].eq(151652))[0]  # 图像开始token
    img_id_ends = torch.where(input_ids[0].eq(151653))[0]    # 图像结束token
    
    if len(img_id_starts) > 0:
        img_start = img_id_starts[0]
        img_end = img_id_ends[0]
        visual_attention = attention_weights[0, img_start + 1:img_end]  # 视觉token的attention scores
        
        # 获取top-k attention的token索引
        top_k = min(top_k, len(visual_attention))
        top_indices = torch.topk(visual_attention, top_k).indices.cpu().numpy()
        
        # 将token索引转换为原图坐标
        num_patches_h = resized_height // patch_size
        num_patches_w = resized_width // patch_size
        
        top_positions = []
        for token_idx in top_indices:
            # 将token索引转换为patch坐标
            patch_y = token_idx // num_patches_w
            patch_x = token_idx % num_patches_w
            
            # 将patch坐标转换为原图坐标（中心点）
            center_x = (patch_x + 0.5) * patch_size * (width / resized_width)
            center_y = (patch_y + 0.5) * patch_size * (height / resized_height)
            top_positions.append((center_x, center_y))
        
        # 为每个top position创建子图区域，并按覆盖点数排序
        ranked_regions = rank_regions_by_coverage(
            top_positions, width, height, max_regions, 
            subimage_width, subimage_height
        )
        
        return ranked_regions, (resized_width, resized_height)
    else:
        # 如果没有找到视觉token，使用默认的划分
        return get_default_subimage_regions(width, height, max_regions), (resized_width, resized_height)

def rank_regions_by_coverage(positions, width, height, max_regions=10, subimage_width=1288, subimage_height=728):
    """
    为每个position创建子图区域，并按覆盖的position数量排序
    """
    if len(positions) == 0:
        return get_default_subimage_regions(width, height, max_regions)
    
    positions_array = np.array(positions)
    
    # 限制子图大小不超过原图
    sub_w = min(subimage_width, width)
    sub_h = min(subimage_height, height)
    
    # 为每个position创建子图区域
    candidate_regions = []
    
    for center in positions_array:
        center_x, center_y = center
        
        # 计算区域边界
        left = max(0, center_x - sub_w / 2)
        top = max(0, center_y - sub_h / 2)
        right = min(width, left + sub_w)
        bottom = min(height, top + sub_h)
        
        # 调整边界确保大小为sub_w x sub_h
        if right - left < sub_w:
            if left == 0:
                right = sub_w
            else:
                left = right - sub_w
        if bottom - top < sub_h:
            if top == 0:
                bottom = sub_h
            else:
                top = bottom - sub_h
        
        region = (int(left), int(top), int(right), int(bottom))
        
        # 计算该区域覆盖的position数量
        covered_count = 0
        for pos in positions_array:
            x, y = pos
            if left <= x <= right and top <= y <= bottom:
                covered_count += 1
        
        candidate_regions.append((region, covered_count, (center_x, center_y)))
    
    # 按覆盖点数降序排序
    candidate_regions.sort(key=lambda x: x[1], reverse=True)
    
    # 去重并选择前max_regions个区域
    selected_regions = []
    seen_regions = set()
    
    for region, count, center in candidate_regions:
        if region not in seen_regions and len(selected_regions) < max_regions:
            selected_regions.append({
                'region': region,
                'coverage': count,
                'center': center
            })
            seen_regions.add(region)
    
    print(f"Selected {len(selected_regions)} regions ranked by coverage:")
    for i, reg_info in enumerate(selected_regions):
        print(f"Region {i+1}: {reg_info['region']}, coverage: {reg_info['coverage']}, center: {reg_info['center']}")
    
    return selected_regions

def get_default_subimage_regions(width, height, num_regions=4):
    """获取默认的子图区域"""
    regions = []
    if num_regions == 4:
        # 2x2网格
        sub_w = width // 2
        sub_h = height // 2
        regions = [
            {'region': (0, 0, sub_w, sub_h), 'coverage': 0, 'center': (sub_w/2, sub_h/2)},  # 左上
            {'region': (sub_w, 0, width, sub_h), 'coverage': 0, 'center': (sub_w + sub_w/2, sub_h/2)},  # 右上
            {'region': (0, sub_h, sub_w, height), 'coverage': 0, 'center': (sub_w/2, sub_h + sub_h/2)},  # 左下
            {'region': (sub_w, sub_h, width, height), 'coverage': 0, 'center': (sub_w + sub_w/2, sub_h + sub_h/2)}  # 右下
        ]
    elif num_regions == 1:
        regions = [{'region': (0, 0, width, height), 'coverage': 0, 'center': (width/2, height/2)}]
    else:
        # 其他数量的默认划分
        cols = int(np.ceil(np.sqrt(num_regions)))
        rows = int(np.ceil(num_regions / cols))
        sub_w = width // cols
        sub_h = height // rows
        
        for i in range(rows):
            for j in range(cols):
                if len(regions) < num_regions:
                    left = j * sub_w
                    top = i * sub_h
                    right = (j + 1) * sub_w if j < cols - 1 else width
                    bottom = (i + 1) * sub_h if i < rows - 1 else height
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    regions.append({
                        'region': (left, top, right, bottom),
                        'coverage': 0,
                        'center': (center_x, center_y)
                    })
    
    return regions

def extract_coordinates(raw_string):
    """从模型输出中提取坐标"""
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except:
        return (0, 0)

def process_single_subimage(subimage, instruction, processor, model, device, offset_x=0, offset_y=0, resize=False):
    """处理单个子图的推理，返回坐标和输出文本"""
    if hasattr(model, 'module'):
        original_model = model.module
    else:
        original_model = model
    
    # 调整子图大小
    if resize:
        subimage = subimage.resize((subimage.width * 2, subimage.height * 2))
    resized_height, resized_width = smart_resize(
        subimage.height,
        subimage.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized_subimage = subimage.resize((resized_width, resized_height))
    
    scale_x = subimage.width / resized_width
    scale_y = subimage.height / resized_height
    
    # 准备消息
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)
    }
    
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": resized_subimage},
            {"type": "text", "text": instruction}
        ]
    }
    
    # 处理输入
    image_inputs, video_inputs = process_vision_info([system_message, user_message])
    text = processor.apply_chat_template([system_message, user_message], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    
    # 生成预测
    with torch.no_grad():
        output_ids = original_model.generate(**inputs, max_new_tokens=32, do_sample=False, temperature=0.0, use_cache=True, pad_token_id=151645)
    
    # 解码输出
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    # 提取坐标
    pred_x, pred_y = extract_coordinates(output_text)
    if resize:
        pred_x = pred_x // 2
        pred_y = pred_y // 2
    # 应用缩放和偏移
    final_x = int(pred_x * scale_x + offset_x)
    final_y = int(pred_y * scale_y + offset_y)
    
    return (final_x, final_y), output_text

def are_coordinates_consistent(coord1, coord2, threshold=5):
    """检查两个坐标是否一致（横纵坐标差都小于阈值）"""
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold

def find_closest_pair(coordinates):
    """找到所有坐标点中距离最近的一对"""
    if len(coordinates) < 2:
        return None, float('inf')
    
    min_distance = float('inf')
    closest_pair = None
    
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            coord1 = coordinates[i]['point']
            coord2 = coordinates[j]['point']
            
            # 计算欧几里得距离
            distance = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_pair = (coordinates[i], coordinates[j])
    
    return closest_pair, min_distance

def process_single_image(json_data, model, processor, base_image_dir, device, max_inferences=10, consistency_threshold=14):
    """处理单个图片的推理流程，统计所有预测点，按相近点分组，选择最大的group"""
    if hasattr(model, 'module'):
        original_model = model.module
    else:
        original_model = model
    
    try:
        # 读取图片
        img_path = os.path.join(base_image_dir, json_data["image_path"])
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found, skipping...")
            return None
        
        image = Image.open(img_path)
        width, height = image.size
        
        # 获取边界框信息
        box_type = json_data["box_type"]
        box_coordinates = json_data["box_coordinates"]
        image_size = json_data["image_size"]
        instruction = json_data["instruction"]
        
        print(f"Processing: {json_data['image_path']}")
        print(f"Original size: {width}x{height}, Box type: {box_type}, Instruction: {instruction}")
        
        # 初始化结果字典
        result = {
            "id": json_data["id"],
            "filename": json_data["image_path"],
            "instruction": instruction,
            "box_type": box_type,
            "box_coordinates": box_coordinates,
            "image_size": image_size,
            "full_image_prediction": None,
            "subimage_predictions": [],
            "all_predictions": [],
            "groups": [],
            "largest_group": None,
            "final_prediction": None,
            "inference_count": 0,
            "GUI_types": json_data.get("GUI_types", [])
        }
        
        all_coordinates = []  # 存储所有预测坐标
        
        # 第一步：全图预测
        print("=" * 30)
        print("Step 1: Full Image Prediction")
        print("=" * 30)
        
        full_coord, full_output = process_single_subimage(image, instruction, processor, model, device)
        full_x, full_y = full_coord
        full_in_target = is_point_in_target((full_x, full_y), box_type, box_coordinates, image_size)
        
        full_pred = {
            "point": (full_x, full_y),
            "in_target": full_in_target,
            "output": full_output,
            "region": (0, 0, width, height),
            "coverage": 0,
            "stage": "full_image"
        }
        
        result["full_image_prediction"] = full_pred
        all_coordinates.append(full_pred)
        result["inference_count"] += 1
        
        print(f"Full image prediction: ({full_x}, {full_y}), In target: {full_in_target}")
        
        # 第二步：获取按覆盖率排序的区域
        print("=" * 30)
        print("Step 2: Getting ranked regions by coverage...")
        ranked_regions, resized_size = get_top_attention_regions(
            image, instruction, processor, model, device, 
            max_regions=max_inferences
        )
        
        print(f"Found {len(ranked_regions)} ranked regions")
        
        # 第三步：推理所有子图（最多max_inferences次）
        print("=" * 30)
        print(f"Step 3: Subimage inference (up to {max_inferences} times)")
        print("=" * 30)
        
        subimage_predictions = []
        
        for i, region_info in enumerate(ranked_regions):
            if i >= max_inferences:  # 最多推理max_inferences次
                print(f"Reached maximum subimage inference count ({max_inferences}), stopping...")
                break
                
            region = region_info['region']
            coverage = region_info['coverage']
            center = region_info['center']
            
            left, top, right, bottom = region
            subimage = image.crop(region)
            
            print(f"Subimage inference {i + 1}: Region {i+1} (coverage: {coverage})")
            print(f"Region: {region}, Center: {center}")
            
            # 推理当前子图
            coord, output_text = process_single_subimage(
                subimage, instruction, processor, model, device, left, top, resize=True
            )
            
            pred_x, pred_y = coord
            in_target = is_point_in_target((pred_x, pred_y), box_type, box_coordinates, image_size)
            
            prediction = {
                "point": (pred_x, pred_y),
                "in_target": in_target,
                "output": output_text,
                "region": region,
                "coverage": coverage,
                "stage": f"subimage_{i+1}"
            }
            
            subimage_predictions.append(prediction)
            all_coordinates.append(prediction)
            result["inference_count"] += 1
            
            print(f"Prediction: ({pred_x}, {pred_y}), In target: {in_target}")
        
        result["subimage_predictions"] = subimage_predictions
        result["all_predictions"] = all_coordinates
        
        # 第四步：将相近的点分组
        print("=" * 30)
        print("Step 4: Grouping similar coordinates...")
        print("=" * 30)
        
        def find_coordinate_groups(predictions, threshold=5):
            """将相近的坐标点分组"""
            groups = []
            assigned = set()
            
            for i, pred1 in enumerate(predictions):
                if i in assigned:
                    continue
                    
                # 创建新group
                group = [pred1]
                assigned.add(i)
                
                # 查找所有相近的点
                for j, pred2 in enumerate(predictions):
                    if j not in assigned and are_coordinates_consistent(pred1["point"], pred2["point"], threshold):
                        group.append(pred2)
                        assigned.add(j)
                
                groups.append(group)
            
            return groups
        
        groups = find_coordinate_groups(all_coordinates, consistency_threshold)
        result["groups"] = groups
        
        print(f"Found {len(groups)} coordinate groups:")
        for i, group in enumerate(groups):
            group_points = [pred["point"] for pred in group]
            group_sizes = [pred["stage"] for pred in group]
            print(f"Group {i+1}: {len(group)} points - {group_points} (stages: {group_sizes})")
        
        # 第五步：选择最大的group
        print("=" * 30)
        print("Step 5: Selecting largest group...")
        print("=" * 30)
        
        def get_coverage_value(pred):
            """安全获取覆盖率数值"""
            coverage = pred.get('coverage', 0)
            if isinstance(coverage, str):
                return 0
            return coverage
        
        def calculate_group_score(group):
            """计算group的得分（点数 + 平均覆盖率/1000）"""
            size = len(group)
            avg_coverage = sum(get_coverage_value(pred) for pred in group) / len(group) if group else 0
            # 主要按点数排序，点数相同时按平均覆盖率排序
            return size + avg_coverage / 1000
        
        if groups:
            # 计算每个group的得分
            group_scores = []
            for i, group in enumerate(groups):
                score = calculate_group_score(group)
                group_scores.append((i, score, len(group), 
                                   sum(get_coverage_value(pred) for pred in group) / len(group)))
            
            # 按得分排序（降序）
            group_scores.sort(key=lambda x: x[1], reverse=True)
            
            best_group_idx = group_scores[0][0]
            best_group = groups[best_group_idx]
            result["largest_group"] = best_group
            
            print(f"Group scores:")
            for i, score, size, avg_coverage in group_scores:
                print(f"  Group {i+1}: size={size}, avg_coverage={avg_coverage:.1f}, score={score:.4f}")
            
            # 在最佳group中选择覆盖率最高的预测作为最终结果
            best_pred = max(best_group, key=lambda pred: get_coverage_value(pred))
            
            result["final_prediction"] = best_pred.copy()
            result["final_prediction"]["stage"] = f"group_{best_group_idx+1}_best_coverage"
            
            print(f"Selected group {best_group_idx + 1} with {len(best_group)} points")
            print(f"Selected prediction: {best_pred['point']} (coverage: {get_coverage_value(best_pred)})")
        else:
            # 如果没有group，使用全图预测
            result["final_prediction"] = full_pred.copy()
            result["final_prediction"]["stage"] = "default_full_image"
            print("No groups found, using full image prediction")
        
        # 统计准确率
        predictions_in_target = [pred["in_target"] for pred in all_coordinates]
        result["any_correct"] = any(predictions_in_target) if predictions_in_target else False
        
        # 统计group准确率
        group_correct = any(any(pred["in_target"] for pred in group) for group in groups) if groups else False
        result["group_correct"] = group_correct
        
        print("=" * 50)
        print("Summary:")
        print(f"Total inferences: {result['inference_count']} (1 full + {len(subimage_predictions)} subimages)")
        print(f"Total groups: {len(groups)}")
        print(f"Full image prediction: {full_pred['point']} in target: {full_pred['in_target']}")
        
        for i, pred in enumerate(subimage_predictions):
            print(f"Subimage {i+1}: {pred['point']} in target: {pred['in_target']} (coverage: {pred['coverage']})")
        
        if groups:
            print(f"Largest group: {len(result['largest_group'])} points")
            group_points = [pred["point"] for pred in result["largest_group"]]
            print(f"Group points: {group_points}")
        
        print(f"Final prediction: {result['final_prediction']['point']} in target: {result['final_prediction']['in_target']}")
        print(f"Any prediction correct: {result['any_correct']}")
        print(f"Group prediction correct: {result['group_correct']}")
        print("=" * 50)
        
        return result
        
    except Exception as e:
        print(f"Error processing {json_data.get('image_path', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        return rank, world_size, gpu
    else:
        print("Not using distributed mode")
        return 0, 1, 0

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_data_by_category(json_file):
    """按类别加载数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)["classified"]
    
    # 按类别组织数据
    category_data = {}
    for category_name, samples in data.items():
        category_data[category_name] = samples
    
    return category_data

def main():
    # 设置分布式
    rank, world_size, gpu = setup_distributed()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--max_inferences', type=int, default=4, help='Maximum number of subimage inferences per image')
    parser.add_argument('--attn_layer', type=int, default=20, help='layer index for attention-based region selection')
    parser.add_argument('--target_token_id', type=str, default=",", help='target token id for attention-based region selection')
    parser.add_argument('--base_image_dir', type=str, default="/run/determined/NAS1/public/yunzhu/OSWorld-G/benchmark/images", help='Base directory for images')
    parser.add_argument('--json_file', type=str, default="/run/determined/NAS1/public/yunzhu/OSWorld-G/benchmark/OSWorld-G_refined_classified.json", help='Path to the JSON file with annotations')
    parser.add_argument('--model_path', type=str, default="/run/determined/NAS1/UI-TARS-1.5-7B", help='Path to the pre-trained model')
    args = parser.parse_args()
    
    # 只在主进程打印信息
    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Attention layer for region selection: {args.attn_layer}")
        print(f"Target token id for region selection: {args.target_token_id}")
        print(f"Max subimage inferences per image: {args.max_inferences}")
        print("Running: Full image + subimage predictions with group selection")
    
    # 加载模型
    # model_path = "/run/determined/NAS1/HelloKKMe/grounding-r1-7B" 
    model_path = args.model_path
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    
    token_selector_config = {"target_token_id": args.target_token_id, "target_layer_idx": args.attn_layer}
    for k, v in token_selector_config.items():
        setattr(config, k, v)
    
    # 每个进程加载自己的模型实例
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu}"
    )
    
    # 将模型移动到当前GPU并包装为DDP
    model.eval()
    
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_path,
        min_pixels=3136,
        max_pixels=4096 * 2160
    )
    
    # 读取JSON文件（按类别）
    json_file = args.json_file
    base_image_dir = args.base_image_dir
    
    # 按类别加载数据
    category_data = load_data_by_category(json_file)
    
    # 只在主进程打印类别信息
    if rank == 0:
        print(f"Loaded {len(category_data)} categories:")
        for category_name, samples in category_data.items():
            print(f"  {category_name}: {len(samples)} samples")
    
    # 按类别处理数据
    all_results = {}
    
    for category_name, samples in category_data.items():
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Processing category: {category_name}")
            print(f"{'='*60}")
        
        # 创建数据集
        dataset = ScreenSpotDataset(samples, base_image_dir)
        
        # 使用自定义的评估采样器，确保不重复不丢弃样本
        if world_size > 1:
            sampler = EvaluationSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # 处理每个图片
        local_results = []
        
        # 只在主进程创建进度条
        if rank == 0:
            pbar = tqdm(total=len(dataloader), desc=f"Processing {category_name}")

        for batch in dataloader:
            # 处理批次中的每个样本
            for json_data in batch:
                result = process_single_image(json_data, model, processor, base_image_dir, gpu, args.max_inferences)
                if result is not None:
                    local_results.append(result)
            
            # 更新进度条（只在主进程）
            if rank == 0:
                pbar.update(1)

        if rank == 0:
            pbar.close()
        
        # 收集所有进程的结果
        if world_size > 1:
            # 将所有结果收集到主进程
            all_category_results = [None] * world_size
            dist.gather_object(local_results, all_category_results if rank == 0 else None, dst=0)
            
            if rank == 0:
                # 合并所有结果
                category_results = []
                for res_list in all_category_results:
                    if res_list is not None:
                        category_results.extend(res_list)
            else:
                category_results = []
        else:
            category_results = local_results
        
        # 只在主进程保存类别结果
        if rank == 0:
            all_results[category_name] = category_results
    
    # 只在主进程进行最终统计和保存
    if rank == 0:
        # 计算每个类别的准确率和总体准确率
        category_stats = {}
        overall_stats = {
            "total": 0,
            "full_correct": 0,
            "final_correct": 0,
            "any_correct": 0,
            "group_correct": 0
        }
        
        # 统计每个类别
        for category_name, results in all_results.items():
            total = len(results)
            
            # 计算各种准确率
            full_correct = sum(1 for r in results if r["full_image_prediction"] and r["full_image_prediction"]["in_target"])
            final_correct = sum(1 for r in results if r["final_prediction"] and r["final_prediction"]["in_target"])
            any_correct = sum(1 for r in results if r["any_correct"])
            group_correct = sum(1 for r in results if r.get("group_correct", False))
            
            # 计算group相关统计
            total_groups = sum(len(r.get("groups", [])) for r in results)
            avg_groups_per_image = total_groups / total if total > 0 else 0
            
            # 计算每个推理次数的平均预测数
            avg_inferences = sum(r["inference_count"] for r in results) / total if total > 0 else 0
            avg_subimages = avg_inferences - 1  # 减去全图预测
            
            full_accuracy = full_correct / total if total > 0 else 0
            final_accuracy = final_correct / total if total > 0 else 0
            any_accuracy = any_correct / total if total > 0 else 0
            group_accuracy = group_correct / total if total > 0 else 0
            
            # 保存类别统计
            category_stats[category_name] = {
                "total": int(total),
                "full_correct": int(full_correct),
                "final_correct": int(final_correct),
                "any_correct": int(any_correct),
                "group_correct": int(group_correct),
                "full_accuracy": float(full_accuracy),
                "final_accuracy": float(final_accuracy),
                "any_accuracy": float(any_accuracy),
                "group_accuracy": float(group_accuracy),
                "avg_inferences": float(avg_inferences),
                "avg_subimages": float(avg_subimages),
                "total_groups": int(total_groups),
                "avg_groups_per_image": float(avg_groups_per_image)
            }
            
            # 累加到总体统计
            overall_stats["total"] += total
            overall_stats["full_correct"] += full_correct
            overall_stats["final_correct"] += final_correct
            overall_stats["any_correct"] += any_correct
            overall_stats["group_correct"] += group_correct
        
        # 计算总体准确率
        overall_full_accuracy = overall_stats["full_correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
        overall_final_accuracy = overall_stats["final_correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
        overall_any_accuracy = overall_stats["any_correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
        overall_group_accuracy = overall_stats["group_correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
        
        overall_stats["full_accuracy"] = float(overall_full_accuracy)
        overall_stats["final_accuracy"] = float(overall_final_accuracy)
        overall_stats["any_accuracy"] = float(overall_any_accuracy)
        overall_stats["group_accuracy"] = float(overall_group_accuracy)
        
        # 打印详细结果
        print("\n" + "="*80)
        print("CATEGORY-WISE RESULTS:")
        print("="*80)
        
        for category_name, stats in category_stats.items():
            print(f"\n{category_name}:")
            print(f"  Total: {stats['total']}")
            print(f"  Full image accuracy: {stats['full_accuracy']:.4f} ({stats['full_correct']}/{stats['total']})")
            print(f"  Final prediction accuracy: {stats['final_accuracy']:.4f} ({stats['final_correct']}/{stats['total']})")
            print(f"  Any prediction correct: {stats['any_accuracy']:.4f} ({stats['any_correct']}/{stats['total']})")
            print(f"  Group prediction correct: {stats['group_accuracy']:.4f} ({stats['group_correct']}/{stats['total']})")
        
        print("\n" + "="*80)
        print("OVERALL RESULTS:")
        print("="*80)
        print(f"Total images processed: {overall_stats['total']}")
        print(f"Overall Full image accuracy: {overall_stats['full_accuracy']:.4f} ({overall_stats['full_correct']}/{overall_stats['total']})")
        print(f"Overall Final prediction accuracy: {overall_stats['final_accuracy']:.4f} ({overall_stats['final_correct']}/{overall_stats['total']})")
        print(f"Overall Any prediction correct: {overall_stats['any_accuracy']:.4f} ({overall_stats['any_correct']}/{overall_stats['total']})")
        print(f"Overall Group prediction correct: {overall_stats['group_accuracy']:.4f} ({overall_stats['group_correct']}/{overall_stats['total']})")
        print("="*80)
        
        # 准备保存的数据
        output_data = {
            "summary": {
                "overall": overall_stats,
                "categories": category_stats,
                "max_subimage_inferences": args.max_inferences
            },
            "detailed_results": all_results
        }
        
        # 生成输出文件名
        model_name = "_".join(args.model_path.split('/'))
        output_file = f"osworldg_rst/group_selection_k{args.max_inferences}_{model_name}_category_refine_results.json"
        if not os.path.exists("osworldg_rst"):
            os.makedirs("osworldg_rst", exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()