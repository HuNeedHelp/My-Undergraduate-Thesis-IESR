# 街景数据处理代码使用说明

本文档主要介绍如何在本项目中使用 `get_street_score.py` 脚本来为街景位置生成感知评分（例如 Beautiful、Safe 等）。

**目录与文件位置**

- 脚本位置：`global-streetscapes/get_street_score.py`
- 示例输入（仓库示例）：`data/上市公司地址.csv`
- 默认输出：`data/street_scores.csv`

**功能简介**

- `get_street_score.py`：对给定坐标（经纬度）批量抓取百度街景图像（每个位置多方向）并使用预训练的感知模型预测评分（模型说明见  
  👉 [Human perception – Global Streetscapes Wiki](https://github.com/ualsg/global-streetscapes/wiki/4b-Human-perception)），最终将评分回写到 CSV 中。
- 主要流程：坐标（WGS84）→ 百度墨卡托转换 → 获取 panoid → 拉取街景图片 → 图像预处理 → 用模型预测 → 将每位置的多张图平均为单个分数。

依赖与环境

- 推荐创建并激活虚拟环境（Windows 示例）：

```powershell

# 进入项目目录
cd global-streetscapes

# 创建 conda 虚拟环境（示例使用 Python 3.10）
conda create -n global-streetscapes python=3.10 -y

# 激活环境
conda activate global-streetscapes

```

- 安装依赖：

```bash

pip install -r requirements-cv-linux.txt

```

脚本参数与示例运行

- 常用参数：

  -`--input_csv`：输入文件路径（支持 CSV / Excel / Stata）

  -`--output_csv`：输出 CSV 路径

  -`--location_col`：输入文件中表示街道/坐标的列名（脚本示例使用 `offaddress`）

  -`--score_type`：要计算的感知评分列表（如 `Beautiful Safe`）
- 运行示例（在 `global-streetscapes` 目录下运行）：

```bash

python get_street_score.py --input_csv ../data/上市公司地址.csv --output_csv ../data/street_scores.csv --location_col offaddress --score_type Beautiful Safe

```

重要说明与注意事项

- 百度坐标转换：脚本中 `wgs2bd09mc` 使用了百度 API 的 `ak`（access key），当前代码中为硬编码占位，请更换为您自己的百度地图 AK 以保证转换成功。
- 日志：脚本会在 `global-streetscapes/get_street_score.log` 中记录运行信息与错误。
- 模型文件：首次运行时脚本会尝试从 Hugging Face 仓库下载预训练模型并存到 `code/model_training/perception/models`。请确保有网络访问权限与足够存储空间。
- GPU：脚本会优先使用 GPU（若可用），否则在 CPU 上运行；请根据机器配置安装对应的 PyTorch 版本。
- 中断恢复：脚本中有 `current_idx` 用于记录上次中断位置（示例代码里写了一个非零初始值用于恢复），实际使用请将其初始化为 `0` 或按需要调整以从中断处继续。
- 抓图频率与反爬：脚本直接请求百度街景接口并抓取图片，请遵循服务条款，合理控制请求速率，必要时加入代理或节流策略。

输出说明

- 输出 CSV 会包含原始列并新增每种 `score_type` 的评分列（数值在 0-10 范围内，脚本中使用乘 10 的缩放并四舍五入到两位小数）。

引用

- 本项目部分实现参考并使用了开源仓库：ualsg/global-streetscapes — https://github.com/ualsg/global-streetscapes

