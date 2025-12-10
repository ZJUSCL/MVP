import datetime
import json
import os
from PIL import Image
from qwen3_vl import Qwen3VLConfig
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor
from qwen3_vl import Qwen3VLForConditionalGeneration
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

def get_border_regions(image, max_inferences, border_width=32):
    """
    根据max_inferences获取边框扩展区域
    max_inferences=2: 返回左侧和上侧边框扩展
    max_inferences=4: 返回左侧、上侧、右侧、下侧边框扩展
    """
    width, height = image.size
    regions = []
    
    if max_inferences == 2:
        # 左侧边框扩展
        left_region = {
            'region': (0, 0, width + border_width, height),
            'coverage': 1,
            'center': (width/2 + border_width/2, height/2),
            'border_type': 'left',
            'offset_x': border_width,
            'offset_y': 0
        }
        regions.append(left_region)
        
        # 上侧边框扩展
        top_region = {
            'region': (0, 0, width, height + border_width),
            'coverage': 1,
            'center': (width/2, height/2 + border_width/2),
            'border_type': 'top',
            'offset_x': 0,
            'offset_y': border_width
        }
        regions.append(top_region)
        
    elif max_inferences == 4:
        # 左侧边框扩展
        left_region = {
            'region': (0, 0, width + border_width, height),
            'coverage': 1,
            'center': (width/2 + border_width/2, height/2),
            'border_type': 'left',
            'offset_x': border_width,
            'offset_y': 0
        }
        regions.append(left_region)
        
        # 上侧边框扩展
        top_region = {
            'region': (0, 0, width, height + border_width),
            'coverage': 1,
            'center': (width/2, height/2 + border_width/2),
            'border_type': 'top',
            'offset_x': 0,
            'offset_y': border_width
        }
        regions.append(top_region)
        
        # 右侧边框扩展
        right_region = {
            'region': (-border_width, 0, width, height),
            'coverage': 1,
            'center': (width/2 - border_width/2, height/2),
            'border_type': 'right',
            'offset_x': -border_width,
            'offset_y': 0
        }
        regions.append(right_region)
        
        # 下侧边框扩展
        bottom_region = {
            'region': (0, -border_width, width, height),
            'coverage': 1,
            'center': (width/2, height/2 - border_width/2),
            'border_type': 'bottom',
            'offset_x': 0,
            'offset_y': -border_width
        }
        regions.append(bottom_region)
    
    print(f"Generated {len(regions)} border regions for max_inferences={max_inferences}")
    for i, region in enumerate(regions):
        print(f"Region {i+1}: {region['border_type']} border, region: {region['region']}")
    
    return regions

def extract_coordinates(raw_string):
    """从模型输出中提取坐标"""
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except:
        return (0, 0)

def create_image_with_border(image, border_type, border_width=32, border_color=(255, 255, 255)):
    """
    创建带边框的图像
    border_type: 'left', 'top', 'right', 'bottom'
    """
    width, height = image.size
    
    if border_type == 'left':
        # 左侧增加边框
        new_width = width + border_width
        new_image = Image.new('RGB', (new_width, height), border_color)
        new_image.paste(image, (border_width, 0))
        return new_image, border_width, 0
    
    elif border_type == 'top':
        # 上侧增加边框
        new_height = height + border_width
        new_image = Image.new('RGB', (width, new_height), border_color)
        new_image.paste(image, (0, border_width))
        return new_image, 0, border_width
    
    elif border_type == 'right':
        # 右侧增加边框
        new_width = width + border_width
        new_image = Image.new('RGB', (new_width, height), border_color)
        new_image.paste(image, (0, 0))
        return new_image, 0, 0
    
    elif border_type == 'bottom':
        # 下侧增加边框
        new_height = height + border_width
        new_image = Image.new('RGB', (width, new_height), border_color)
        new_image.paste(image, (0, 0))
        return new_image, 0, 0
    
    else:
        return image, 0, 0

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
    final_x = int(pred_x / 1000 * subimage.width + offset_x)
    final_y = int(pred_y / 1000 * subimage.height + offset_y)
    
    return (final_x, final_y), output_text

def are_coordinates_consistent(coord1, coord2, threshold=5):
    """检查两个坐标是否一致（横纵坐标差都小于阈值）"""
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold

def process_single_image(json_data, model, processor, base_image_dir, device, max_inferences=2, consistency_threshold=14):
    """处理单个图片的推理流程，使用边框扩展方法"""
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
            "border_predictions": [],
            "all_predictions": [],
            "groups": [],
            "largest_group": None,
            "final_prediction": None,
            "inference_count": 0,
            "GUI_types": json_data.get("GUI_types", [])
        }
        
        all_coordinates = []  # 存储所有预测坐标
        
        # 第一步：原图预测
        print("=" * 30)
        print("Step 1: Original Image Prediction")
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
            "stage": "original_image"
        }
        
        result["full_image_prediction"] = full_pred
        all_coordinates.append(full_pred)
        result["inference_count"] += 1
        
        print(f"Original image prediction: ({full_x}, {full_y}), In target: {full_in_target}")
        
        # 第二步：获取边框扩展区域
        print("=" * 30)
        print("Step 2: Getting border regions...")
        border_regions = get_border_regions(image, max_inferences)
        
        print(f"Found {len(border_regions)} border regions")
        
        # 第三步：推理所有边框扩展图像
        print("=" * 30)
        print(f"Step 3: Border image inference ({len(border_regions)} times)")
        print("=" * 30)
        
        border_predictions = []
        
        for i, region_info in enumerate(border_regions):
            border_type = region_info['border_type']
            offset_x = region_info['offset_x']
            offset_y = region_info['offset_y']
            
            print(f"Border inference {i + 1}: {border_type} border")
            
            # 创建带边框的图像
            border_image, actual_offset_x, actual_offset_y = create_image_with_border(
                image, border_type, border_width=32
            )
            
            # 推理当前边框图像
            coord, output_text = process_single_subimage(
                border_image, instruction, processor, model, device, 
                -actual_offset_x, -actual_offset_y, resize=False
            )
            
            pred_x, pred_y = coord
            in_target = is_point_in_target((pred_x, pred_y), box_type, box_coordinates, image_size)
            
            prediction = {
                "point": (pred_x, pred_y),
                "in_target": in_target,
                "output": output_text,
                "border_type": border_type,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "stage": f"border_{border_type}"
            }
            
            border_predictions.append(prediction)
            all_coordinates.append(prediction)
            result["inference_count"] += 1
            
            print(f"Prediction: ({pred_x}, {pred_y}), In target: {in_target}")
        
        result["border_predictions"] = border_predictions
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
            group_stages = [pred["stage"] for pred in group]
            print(f"Group {i+1}: {len(group)} points - {group_points} (stages: {group_stages})")
        
        # 第五步：选择最大的group
        print("=" * 30)
        print("Step 5: Selecting largest group...")
        print("=" * 30)
        
        if groups and len(groups) != max_inferences + 1:
            # 找到最大的group
            largest_group = max(groups, key=len)
            result["largest_group"] = largest_group
            
            # 在最大的group中选择原图预测（如果存在），否则选择第一个
            original_in_group = [pred for pred in largest_group if pred["stage"] == "original_image"]
            if original_in_group:
                best_pred = original_in_group[0]
            else:
                best_pred = largest_group[0]
            
            result["final_prediction"] = best_pred.copy()
            result["final_prediction"]["stage"] = f"group_largest"
            
            print(f"Selected largest group with {len(largest_group)} points")
            print(f"Selected prediction: {best_pred['point']} (stage: {best_pred['stage']})")
        else:
            # 如果没有group，使用原图预测
            result["final_prediction"] = full_pred.copy()
            result["final_prediction"]["stage"] = "default_original"
            print("No groups found, using original image prediction")
        
        # 统计准确率
        predictions_in_target = [pred["in_target"] for pred in all_coordinates]
        result["any_correct"] = any(predictions_in_target) if predictions_in_target else False
        
        # 统计group准确率
        group_correct = any(any(pred["in_target"] for pred in group) for group in groups) if groups else False
        result["group_correct"] = group_correct
        
        print("=" * 50)
        print("Summary:")
        print(f"Total inferences: {result['inference_count']} (1 original + {len(border_predictions)} borders)")
        print(f"Total groups: {len(groups)}")
        print(f"Original image prediction: {full_pred['point']} in target: {full_pred['in_target']}")
        
        for i, pred in enumerate(border_predictions):
            print(f"Border {i+1} ({pred['border_type']}): {pred['point']} in target: {pred['in_target']}")
        
        if groups and len(groups) != max_inferences + 1:
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
        # dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        dist.init_process_group(
            backend='nccl', 
            world_size=world_size, 
            rank=rank,
            timeout=datetime.timedelta(minutes=30)  # 添加这行
        )
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
    parser.add_argument('--max_inferences', type=int, default=4, choices=[2, 4], help='Maximum number of border inferences per image (2 or 4)')
    parser.add_argument('--attn_layer', type=int, default=20, help='layer index for attention-based region selection')
    parser.add_argument('--target_token_id', type=str, default=",", help='target token id for attention-based region selection')
    parser.add_argument('--json_file', type=str, default="/run/determined/NAS1/public/yunzhu/OSWorld-G/benchmark/OSWorld-G_refined_classified.json", help='Path to the JSON file with annotations')
    parser.add_argument('--base_image_dir', type=str, default="/run/determined/NAS1/public/yunzhu/OSWorld-G/benchmark/images", help='Base directory for images')
    parser.add_argument('--model_path', type=str, default="/run/determined/NAS1/Qwen/Qwen3-VL-8B-Instruct", help='Path to the pre-trained model')
    
    args = parser.parse_args()
    
    # 只在主进程打印信息
    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Attention layer for region selection: {args.attn_layer}")
        print(f"Target token id for region selection: {args.target_token_id}")
        print(f"Max border inferences per image: {args.max_inferences}")
        print("Running: Original image + border predictions with group selection")
    
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
    
    # 将模型移动到当前GPU并包装为DDP
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
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
            avg_borders = avg_inferences - 1  # 减去原图预测
            
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
                "avg_borders": float(avg_borders),
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
            print(f"  Original image accuracy: {stats['full_accuracy']:.4f} ({stats['full_correct']}/{stats['total']})")
            print(f"  Final prediction accuracy: {stats['final_accuracy']:.4f} ({stats['final_correct']}/{stats['total']})")
            print(f"  Any prediction correct: {stats['any_accuracy']:.4f} ({stats['any_correct']}/{stats['total']})")
            print(f"  Group prediction correct: {stats['group_accuracy']:.4f} ({stats['group_correct']}/{stats['total']})")
        
        print("\n" + "="*80)
        print("OVERALL RESULTS:")
        print("="*80)
        print(f"Total images processed: {overall_stats['total']}")
        print(f"Overall Original image accuracy: {overall_stats['full_accuracy']:.4f} ({overall_stats['full_correct']}/{overall_stats['total']})")
        print(f"Overall Final prediction accuracy: {overall_stats['final_accuracy']:.4f} ({overall_stats['final_correct']}/{overall_stats['total']})")
        print(f"Overall Any prediction correct: {overall_stats['any_accuracy']:.4f} ({overall_stats['any_correct']}/{overall_stats['total']})")
        print(f"Overall Group prediction correct: {overall_stats['group_accuracy']:.4f} ({overall_stats['group_correct']}/{overall_stats['total']})")
        print("="*80)
        
        # 准备保存的数据
        output_data = {
            "summary": {
                "overall": overall_stats,
                "categories": category_stats,
                "max_border_inferences": args.max_inferences
            },
            "detailed_results": all_results
        }
        
        # 生成输出文件名
        model_name = "_".join(args.model_path.split('/'))
        output_file = f"osworldg_rst/border_method_k{args.max_inferences}_{model_name}_refined_category_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()