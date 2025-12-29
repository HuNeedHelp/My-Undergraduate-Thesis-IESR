from io import BytesIO
import json
import os
from pathlib import Path
import argparse
import logging
import re
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from huggingface_hub import snapshot_download

import sys
sys.path.append('./code/model_training/perception/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 图像预处理
img_transform = T.Compose([
    T.Resize((384,384)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# 设置路径以及日志
ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=str(ROOT / 'get_street_score.log'))
logger = logging.getLogger(__name__)

model_dict = {
            'Safe':'/safety.pth', \
            'Lively': '/lively.pth', \
            'Wealthy': '/wealthy.pth',\
            'Beautiful':'/beautiful.pth',\
            'Boring': '/boring.pth',\
            'Depressing': '/depressing.pth',\
            }

def read_street_data(input_csv, output_csv, location_col):
    if input_csv.endswith('.csv'):
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv, encoding='utf-8')
        else:
            df = pd.read_csv(input_csv, encoding='utf-8')
    elif input_csv.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_csv)
    elif input_csv.endswith('dta'):
        df = pd.read_stata(input_csv)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    if location_col not in df.columns:
        raise ValueError(f"Column '{location_col}' not found in the input data.")
    df = df.dropna(subset=["offaddress"]).drop_duplicates("offaddress").reset_index(drop=True)
    return df

def openUrl(_url):
    # 设置请求头 request header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    response = requests.get(_url, headers=headers)
    if response.status_code == 200:  # 如果状态码为200，寿命服务器已成功处理了请求，则继续处理数据
        return response.content
    else:
        return None

def wgs2bd09mc(wgs_x, wgs_y):
    url = f'http://api.map.baidu.com/geoconv/v2/?coords={wgs_x},{wgs_y}&model=3&ak=5dM0jLSkXB8E695HdlbIB73MJSOiigjS'
    res = openUrl(url).decode()
    temp = json.loads(res)
    bd09mc_x = 0
    bd09mc_y = 0
    if temp['status'] == 0:
        bd09mc_x = temp['result'][0]['x']
        bd09mc_y = temp['result'][0]['y']

    return bd09mc_x, bd09mc_y

def getPanoId(_lng, _lat):
    # 获取百度街景中的svid get svid of baidu streetview
    url = "https://mapsv0.bdimg.com/?&qt=qsdata&x=%s&y=%s&l=17.031000000000002&action=0&mode=day" % (
        str(_lng), str(_lat))
    response = openUrl(url).decode("utf8")
    # print(response)
    if (response == None):
        return None
    reg = r'"id":"(.+?)",'
    pat = re.compile(reg)
    try:
        svid = re.findall(pat, response)[0]
        return svid
    except:
        return None

def grab_img_baidu(_url, _headers=None):
    if _headers == None:
        # 设置请求头 request header
        headers = {
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
            "Referer": "https://map.baidu.com/",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
    else:
        headers = _headers
    response = requests.get(_url, headers=headers)

    if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
        return response.content
    else:
        return None

def get_street_view_batch(street_scores, current_idx, batch_size=16):
    """
    每个坐标位置获取 东南西北 4 张街景图像
    返回:
        loc_idx:  当前 batch 中包含的坐标索引列表
        img_batch: Tensor [N, 3, 384, 384]
        new_idx:   下一个待处理的坐标索引
    """
    headings = ['0', '90', '180', '270']  # 北 东 南 西
    loc_idx = []
    images = []

    while len(images) < batch_size and current_idx < len(street_scores):
        # ===== 1. 取当前坐标（WGS84）=====
        wgs_x = street_scores.iloc[current_idx]['offlng']
        wgs_y = street_scores.iloc[current_idx]['offlat']

        # ===== 2. WGS84 → 百度墨卡托 =====
        try:
            bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
        except Exception as e:
            current_idx += 1
            logger.info(f"Coordinate conversion failed for index {current_idx}: {e}")
            continue

        # ===== 3. 获取 panoid =====
        svid = getPanoId(bd09mc_x, bd09mc_y)
        if svid is None:
            current_idx += 1
            continue

        # ===== 4. 拉取 4 个方向的街景 =====
        for h in headings:
            if len(images) >= batch_size:
                break

            url = (
                f'https://mapsv0.bdimg.com/?qt=pr3d'
                f'&fovy=90&quality=100'
                f'&panoid={svid}'
                f'&heading={h}'
                f'&pitch=0'
                f'&width=480&height=320'
            )

            img_bytes = grab_img_baidu(url)
            if img_bytes is None:
                continue

            try:
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                img_tensor = img_transform(img)
                images.append(img_tensor)
            except Exception as e:
                continue

        # 不管这个点拿到几张图，都算“处理完一个坐标”
        loc_idx.append(current_idx)
        current_idx += 1

    # ===== 5. 如果一个 batch 都没拿到 =====
    if len(images) == 0:
        return None, current_idx

    # ===== 6. 拼 batch =====
    img_batch = torch.stack(images, dim=0).to(device)

    return loc_idx, img_batch, current_idx


def get_street_score(input_csv, output_csv, location_col, score_type):
    street_scores = read_street_data(input_csv, output_csv, location_col)
    # 初始化分数列, 如果列不存在则创建列
    for st in score_type:
        if st not in street_scores.columns:
            street_scores[st] = np.nan

    # ======== 预测街景图像的感知分数 ========
    for score_type in score_type:
        logger.info(f"Start processing {len(street_scores)} companies for scores: {score_type}")
        
        # 初始化进度条和索引
        current_idx = 2272   # ❗❗❗从上次中断的地方继续, 初始应设置为0❗❗❗
        pbar = tqdm(total=len(street_scores), desc=f'Processing {score_type} scores. {current_idx + 1}/{len(street_scores)} companies completed')
        pbar.update(current_idx)

        # 设置模型路径
        # directory that stores all the models
        model_load_path = "./global-streetscapes/code/model_training/perception/models" 
        if not os.path.exists(model_load_path):
            Path(model_load_path).mkdir(parents=True, exist_ok=True)
            # download model
            print('Downloading models...')
            snapshot_download(repo_id="Jiani11/human-perception-place-pulse", allow_patterns=["*.pth", "README.md"], local_dir=model_load_path)
        model_path = model_load_path + model_dict[score_type]

        # 加载模型
        model = torch.load(model_path, map_location=torch.device(device))  
        model = nn.DataParallel(model)
        model = model.to(device)
        model.eval()
        while current_idx <= len(street_scores):
            # 获取当前批次的街景图像
            loc_idx, img_batch, new_idx = get_street_view_batch(street_scores, current_idx, batch_size=16)
            
            # predict
            pred = model(img_batch)
            pred = pred.softmax(dim=-1)[:, 1].detach().cpu().numpy()
            
            # 更新分数到 DataFrame, 每个坐标位置有 4 张图，取平均值
            for i, idx in enumerate(loc_idx):
                imgs_per_location = 4
                start = i * imgs_per_location
                end = start + imgs_per_location
                location_score = pred[start:end].mean()
                street_scores.at[idx, score_type] = round(location_score * 10, 2)   # 参考原来开源代码

            # 更新一下pbar进度条
            pbar.update(new_idx - current_idx)
            current_idx = new_idx

            # 每处理24个企业保存一次
            if (current_idx % 24 < 5):
                street_scores.to_csv(output_csv, index=False)
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default=str(ROOT.parent / 'data' / '上市公司地址.csv'), help='Path to the input file containing street data.')
    parser.add_argument('--output_csv', type=str, default=str(ROOT.parent / 'data' / 'street_scores.csv'), help='Path to the output CSV file to save street scores.')
    parser.add_argument('--location_col', type=str, default='offaddress', help='Column name for street location in the input file.')
    parser.add_argument('--score_type', nargs='+', default=['Beautiful', 'Safe'], help='List of perception scores to compute (e.g., beautiful, safe).')
    args = parser.parse_args()
    
    get_street_score(args.input_csv, args.output_csv, args.location_col, args.score_type)


