from ollama_model.ChatModel import call_model
import logging
import os
from .generate_query_config import prompt

def generate_query():
    subjects = ['统计查询', '类别分布', '标注密度分布', '基础过滤查询', '小目标检测', '标注异常值检测', '多目标检测', '标注完整性', '类别不平衡', '数据集划分', '数据集重新划分查询', '新数据集构建', '多目标关系查询']

    subjects_not_support_now = ['拥挤场景分析', '图像尺寸统计', '​​数据增强策略', '精度-召回分析支持', '困难负样本挖掘​​', '模型对比分析', '尺寸特定性能​​']

    log_file = os.path.join(os.path.dirname(__file__), "generate_query.json")

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    for subject in subjects:
        new_prompt = prompt.replace("{subject}", subject)
        # call_model(new_prompt, logger, chatModel="Bailian")
        call_model(new_prompt, logger, chatModel="Ollama")

