import json
import os
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor
from qwen3_vl import Qwen3VLForConditionalGeneration
import torch
import torch.nn as nn
import re
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
from qwen3_vl import Qwen3VLConfig
import itertools

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

def get_top_attention_regions(image, instruction, processor, model, device, top_k=100, patch_size=32, max_regions=10, subimage_width=1288, subimage_height=728):
    """
    基于attention scores选择覆盖最多top attention token的子图区域，并按覆盖点数排序
    """
    if max_regions == 0:
        return [], []
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
        # 匹配多种格式的坐标
        patterns = [
            r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)",  # (x, y)
            r"\[(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\]",  # [x, y]
            r"(-?\d*\.?\d+),\s*(-?\d*\.?\d+)",      # x, y
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_string)
            if matches:
                # 尝试解析为float
                try:
                    x = float(matches[0][0])
                    y = float(matches[0][1])
                    
                    # 如果坐标看起来是像素坐标（大于10），转换为相对坐标
                    if x > 10 or y > 10:
                        # 假设这是1000x1000坐标系
                        if x <= 1000 and y <= 1000:
                            return x / 1000, y / 1000
                        # 否则返回None，表示格式错误
                        else:
                            return None
                    else:
                        # 已经是相对坐标
                        return x, y
                except:
                    return None
        
        # 如果没有匹配到任何坐标
        return None
    except:
        return None

def process_single_subimage(subimage, instruction, processor, model, device, offset_x=0, offset_y=0, resize=False):
    """处理单个子图的推理，返回坐标和输出文本"""
    if hasattr(model, 'module'):
        original_model = model.module
    else:
        original_model = model

    if resize:
        subimage = subimage.resize((subimage.width * 2, subimage.height * 2))   
    width, height = subimage.size
 

    
    # 调整子图大小用于推理
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized_subimage = subimage.resize((resized_width, resized_height))
    
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
    coords = extract_coordinates(output_text)
    
    if coords is None:
        return None, output_text
    
    pred_x_rel, pred_y_rel = coords
    
    # 确保坐标在0-1之间
    pred_x_rel = max(0.0, min(1.0, pred_x_rel))
    pred_y_rel = max(0.0, min(1.0, pred_y_rel))
    
    # 将子图相对坐标转换为原图相对坐标
    if resize:
        final_x = pred_x_rel * subimage.width // 2 + offset_x
        final_y = pred_y_rel * subimage.height // 2 + offset_y
    else:
        final_x = pred_x_rel * subimage.width + offset_x
        final_y = pred_y_rel * subimage.height + offset_y

    
    
    return (final_x, final_y), output_text

def are_coordinates_consistent(coord1, coord2, threshold=14):
    """检查两个坐标是否一致（使用相对坐标阈值）"""
    if coord1 is None or coord2 is None:
        return False
    
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold

def find_coordinate_groups_knn(predictions, threshold=14):
    """使用简单KNN算法进行聚类：所有点互相之间都满足差值条件才在一个组"""
    if not predictions:
        return []

    groups = []
    assigned = set()

    for i in range(len(predictions)):
        if i in assigned:
            continue

        # 找到所有与当前点相互之间都满足条件的点
        current_group = [predictions[i]]
        assigned.add(i)

        # 候选点索引列表
        candidate_indices = [i]

        # 检查所有未分配的点
        for j in range(len(predictions)):
            if j in assigned:
                continue

            # 检查点j与组内所有点是否都满足条件
            all_close = True
            for k in candidate_indices:
                if not are_coordinates_consistent(predictions[k]["point"], predictions[j]["point"], threshold):
                    all_close = False
                    break

            if all_close:
                current_group.append(predictions[j])
                assigned.add(j)
                candidate_indices.append(j)

        groups.append(current_group)

    # 按组大小排序（大的在前）
    groups.sort(key=lambda x: len(x), reverse=True)

    return groups

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

def process_single_image(json_data, model, processor, base_image_dir, device, max_inferences=10, consistency_threshold=14):
    """处理单个图片的推理流程，使用多图预测+group选择，输出官方格式"""
    try:
        # 读取图片
        img_path = os.path.join(base_image_dir, json_data["img_filename"])
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found, skipping...")
            return None
        
        image = Image.open(img_path)
        width, height = image.size
        img_size = [width, height]
        
        # 获取任务相关信息
        instruction = json_data.get("instruction", "")
        instruction_cn = json_data.get("instruction_cn", "")
        gt_type = json_data.get("gt_type", "positive")
        instruction_style = json_data.get("instruction_style", "instruction")
        language = json_data.get("language", "en")
        
        # 根据语言选择指令
        if language == "cn" and instruction_cn:
            prompt_to_evaluate = instruction_cn
        else:
            prompt_to_evaluate = instruction
        
        print(f"Processing: {json_data['img_filename']}")
        
        # 存储所有预测
        all_predictions = []
        
        # 1. 全图预测
        full_coord, full_output = process_single_subimage(image, prompt_to_evaluate, processor, model, device, resize=False)
        if full_coord is not None:
            all_predictions.append({
                "point": full_coord,
                "output": full_output,
                "coverage": 0,
                "stage": "full_image",
                "region": (0, 0, width, height)
            })
        
        # 2. 获取按覆盖率排序的区域并进行子图预测
        ranked_regions, _ = get_top_attention_regions(
            image, prompt_to_evaluate, processor, model, device, 
            max_regions=max_inferences
        )
        
        for i, region_info in enumerate(ranked_regions):
            if i >= max_inferences:
                break
                
            region = region_info['region']
            coverage = region_info['coverage']
            left, top, right, bottom = region
            subimage = image.crop(region)
            
            # 推理当前子图
            coord, output_text = process_single_subimage(
                subimage, prompt_to_evaluate, processor, model, device, left, top, resize=True
            )
            
            if coord is not None:
                all_predictions.append({
                    "point": coord,
                    "output": output_text,
                    "coverage": coverage,
                    "stage": f"subimage_{i+1}",
                    "region": region
                })
        
        # 3. 将相近的点分组并选择最佳预测
        final_prediction = None
        raw_response = ""
        
        if all_predictions:
            # 分组
            groups = find_coordinate_groups_knn(all_predictions, consistency_threshold)
            
            if groups:
                # 计算每个group的得分
                group_scores = []
                for i, group in enumerate(groups):
                    score = calculate_group_score(group)
                    group_scores.append((i, score, len(group)))
                
                # 按得分排序（降序）
                group_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 选择最佳group
                best_group_idx = group_scores[0][0]
                best_group = groups[best_group_idx]
                
                # 在最佳group中选择覆盖率最高的预测作为最终结果
                best_pred = max(best_group, key=lambda pred: get_coverage_value(pred))
                final_point = best_pred["point"]
                raw_response = best_pred["output"]
                
                print(f"Selected group {best_group_idx + 1} with {len(best_group)} points")
            else:
                # 如果没有group，选择第一个预测
                final_point = all_predictions[0]["point"]
                raw_response = all_predictions[0]["output"]
        else:
            # 如果没有预测结果
            final_point = None
        
        # 4. 构建官方格式的结果
        result = {
            "img_path": img_path,
            "platform": json_data.get("platform", "unknown"),
            "application": json_data.get("application", "unknown"),
            "lang": language,
            "instruction_style": instruction_style,
            "prompt_to_evaluate": prompt_to_evaluate,
            "gt_type": gt_type,
            "ui_type": json_data.get("ui_type", "unknown"),
            "task_filename": json_data.get("id", "unknown").split('_')[0] if 'id' in json_data else "unknown",
            "img_size": img_size,
            "raw_response": raw_response,
            # 额外的调试信息（不会影响官方评估）
            "_debug": {
                "num_predictions": len(all_predictions),
                "num_groups": len(groups) if 'groups' in locals() else 0,
                "predictions": all_predictions
            }
        }
        
        if "group" in json_data:
            result["group"] = json_data["group"]
        
        # 根据gt_type处理结果
        if gt_type == "positive":
            if final_point is not None:
                point_in_pixel = list(final_point)
                # point_in_pixel = [point[0] * width, point[1] * height]
                result["pred"] = point_in_pixel
                result["point"] = point_in_pixel
                result["bbox"] = json_data.get("bbox", [0, 0, 0, 0])
                
                # 评估正确性
                bbox = json_data["bbox"]
                # bbox是[x, y, w, h]格式，转换为[x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                
                # 转换为相对坐标
                # x1_rel = x1 / width
                # y1_rel = y1 / height
                # x2_rel = x2 / width
                # y2_rel = y2 / height
                
                if x1 <= point_in_pixel[0] <= x2 and y1 <= point_in_pixel[1] <= y2:
                    result["correctness"] = "correct"
                else:
                    result["correctness"] = "wrong"
            else:
                result["pred"] = None
                result["point"] = None
                result["bbox"] = json_data.get("bbox", [0, 0, 0, 0])
                result["correctness"] = "wrong_format"
        
        elif gt_type == "negative":
            if final_point is not None:
                # 有坐标输出表示positive
                result["result"] = "positive"
                point = list(final_point)
                point_in_pixel = [point[0] * width, point[1] * height]
                result["pred"] = point_in_pixel
                result["point"] = point
                result["correctness"] = "wrong"
            else:
                # 无坐标输出表示negative
                result["result"] = "negative"
                result["pred"] = None
                result["point"] = None
                result["correctness"] = "correct"
        
        print(f"Final prediction: {result.get('pred', 'None')}")
        print(f"Correctness: {result.get('correctness', 'N/A')}")
        print("=" * 50)
        
        return result
        
    except Exception as e:
        print(f"Error processing {json_data.get('img_filename', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============ 官方评估函数（保持不变） ============

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("lang") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
    return filtered_results

def make_combinations(results, platform=False, group=False, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("lang"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []
    attribute_combinations = list(itertools.product(*filtered_values.values()))
    combinations = [dict(zip(filtered_values.keys(), combination)) for combination in attribute_combinations]
    return combinations

def calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res.get("correctness") == "correct")
    wrong_format_num = sum(1 for res in results if res.get("correctness") == "wrong_format")
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")
    text_correct = sum(1 for res in text_results if res.get("correctness") == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res.get("correctness") == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics

def evaluate_fine_grained(results):
    combinations = make_combinations(results, platform=True, application=True, instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        filtered_results = collect_results_to_eval(results, platform=platform, application=application, instruction_style=inst_style, gt_type=gt_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_seeclick_paper_style(results):
    combinations = make_combinations(results, platform=True, instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        filtered_results = collect_results_to_eval(results, platform=platform, instruction_style=inst_style, gt_type=gt_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    combinations = make_combinations(results, application=True)
    evaluation_result = {}
    for combo in combinations:
        application = combo.get("application")
        filtered_results = collect_results_to_eval(results, application=application)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"app:{application}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    combinations = make_combinations(results, group=True)
    evaluation_result = {}
    for combo in combinations:
        group = combo.get("group")
        filtered_results = collect_results_to_eval(results, group=group)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"group:{group}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_overall(results):
    metrics = calc_metric_for_result_list(results)
    return metrics

def evaluate(results):
    result_report = {"details": [], "metrics": {}}
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)
    result_report["details"] = results
    return result_report

# ============ 主函数 ============

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
        return 0, 1, 0

class SequentialDistributedSampler(DistributedSampler):
    """顺序分布式采样器，不重复样本，确保不丢弃样本"""
    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.drop_last = False

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # 计算每个rank应该处理的样本数（不重复样本）
        num_samples = len(indices)
        samples_per_rank = num_samples // self.num_replicas
        remainder = num_samples % self.num_replicas

        # 为当前rank分配样本范围
        start_idx = self.rank * samples_per_rank + min(self.rank, remainder)
        end_idx = start_idx + samples_per_rank + (1 if self.rank < remainder else 0)

        assigned_indices = indices[start_idx:end_idx]

        return iter(assigned_indices)

def main():
    # 设置分布式
    rank, world_size, gpu = setup_distributed()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--max_inferences', type=int, default=5, help='Maximum number of subimage inferences per image')
    parser.add_argument('--attn_layer', type=int, default=20, help='layer index for attention-based region selection')
    parser.add_argument('--target_token_id', type=str, default=",", help='target token id for attention-based region selection')
    parser.add_argument('--model_path', type=str, default="/run/determined/NAS1/Qwen/Qwen3-VL-8B-Instruct", help='Path to the pre-trained model')
    parser.add_argument('--json_file_dir', type=str, default="/run/determined/NAS1/ScreenSpot-Pro/annotations", help='Path to the JSON file with annotations')
    parser.add_argument('--base_image_dir', type=str, default="/run/determined/NAS1/ScreenSpot-Pro/images", help='Base directory for images')
    parser.add_argument('--inst_style', type=str, default="instruction", help='Instruction style to use')
    parser.add_argument('--language', type=str, default="en", help='Language to use')
    parser.add_argument('--gt_type', type=str, default="positive", help='Ground truth type')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON')
    args = parser.parse_args()
    
    # 只在主进程打印信息
    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Max subimage inferences per image: {args.max_inferences}")
        print(f"Instruction style: {args.inst_style}")
        print(f"Language: {args.language}")
        print(f"GT type: {args.gt_type}")
        print("Running: Multi-image prediction with group selection")
    
    # 加载模型
    model_path = args.model_path 
    config = Qwen3VLConfig.from_pretrained(model_path)
    
    token_selector_config = {"target_token_id": args.target_token_id, "target_layer_idx": args.attn_layer}
    for k, v in token_selector_config.items():
        setattr(config, k, v)
    
    # 每个进程加载自己的模型实例
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu}"
    )
    
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=3136,
        max_pixels=4096 * 2160
    )
    
    # 读取JSON文件
    json_file_dir = args.json_file_dir
    base_image_dir = args.base_image_dir
    
    json_data_list = []
    json_file_paths = [f for f in os.listdir(json_file_dir) if f.endswith(('.json', '.jsonl'))]
    
    for json_file_path in json_file_paths:
        full_path = os.path.join(json_file_dir, json_file_path)
        with open(full_path, 'r') as f:
            if json_file_path.endswith('.jsonl'):
                for line in f:
                    data = json.loads(line)
                    data["instruction_style"] = args.inst_style
                    data["language"] = args.language
                    data["gt_type"] = args.gt_type
                    json_data_list.append(data)
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        item["instruction_style"] = args.inst_style
                        item["language"] = args.language
                        item["gt_type"] = args.gt_type
                        json_data_list.append(item)
                else:
                    data["instruction_style"] = args.inst_style
                    data["language"] = args.language
                    data["gt_type"] = args.gt_type
                    json_data_list.append(data)
    
    # 创建数据集和数据加载器
    dataset = ScreenSpotDataset(json_data_list, base_image_dir)
    
    if world_size > 1:
        sampler = SequentialDistributedSampler(dataset, num_replicas=world_size, rank=rank)
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
    
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Processing images")

    for batch in dataloader:
        for json_data in batch:
            result = process_single_image(json_data, model, processor, base_image_dir, gpu, args.max_inferences)
            if result is not None:
                local_results.append(result)
        
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()
    
    # 收集所有进程的结果
    if world_size > 1:
        all_results = [None] * world_size
        dist.gather_object(local_results, all_results if rank == 0 else None, dst=0)
        
        if rank == 0:
            results = []
            for res_list in all_results:
                if res_list is not None:
                    results.extend(res_list)
            results.sort(key=lambda x: (x.get("task_filename", ""), x.get("img_path", "")))
        else:
            results = []
    else:
        results = local_results
    
    # 只在主进程进行评估和保存
    if rank == 0:
        # 进行官方格式的评估
        result_report = evaluate(results)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        # 保存结果
        with open(args.output_path, 'w') as f:
            json.dump(result_report, f, indent=4)
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS:")
        print("="*80)
        
        # 打印总体结果
        overall = result_report["metrics"]["overall"]
        print(f"Total samples: {overall['num_total']}")
        print(f"Correct actions: {overall['num_correct_action']}")
        print(f"Action accuracy: {overall['action_acc']:.4f}")
        print(f"Text accuracy: {overall['text_acc']:.4f}")
        print(f"Icon accuracy: {overall['icon_acc']:.4f}")
        print(f"Wrong format: {overall['wrong_format_num']}")
        
        print("\nResults saved to:", args.output_path)
        print("="*80)

if __name__ == "__main__":
    main()