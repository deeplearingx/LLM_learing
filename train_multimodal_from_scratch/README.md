# 使用方法

## 下载模型及数据
### 下载qwen2.5-0.5b和siglip2（图片解码器）
qwen2.5-0.5b: \
https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct \
siglip: \
此处使用的是如下版本的siglip2的base模型（模型小，但是效果可能没那么好，训练更快，显存要求更低）：\
https://hf-mirror.com/google/siglip2-base-patch16-224


### 下载数据集
1、预训练数据：\
图片数据：\
https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K \
中文文本数据：\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json \

2、SFT数据:\
图片数据:\
https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/sft_images.zip \
中文文本数据:\
https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/sft_data.jsonl

## 数据集下载脚本
down.py
## 开始训练
### 直接运行
预训练:\
python train.py\
SFT:\
python sft_train.py
### torchrun
预训练:\
torchrun --nproc_per_node=2 train.py
SFT:\
torchrun --nproc_per_node=2 sft_train.py
### deepspeed
预训练:\
deepspeed --include 'localhost:0,1' train.py\
SFT:\
deepspeed --include 'localhost:0,1' sft_train.py

## 测试
python test.py

## 可视化测试
python gradio_vlm.py


