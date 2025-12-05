from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
from typing import List, Dict, Any
import json
import os

# 复用你 train.py 里的 VLMConfig / VLM
from train import VLMConfig, VLM

# --------------------
# 1. 辅助函数：找到所有 assistant 段在 token 序列里的区间
# --------------------
def find_assistant_spans(tokenizer, input_ids: List[int]):
    """
    在 token id 序列里，找到所有
    <|im_start|> assistant ... <|im_end|>
    之间的 [start, end) 区间（不含 <|im_start|> 和 role 这个词，但含内容和 <|im_end|>）

    返回: List[(start, end)]
    """
    im_start_id = tokenizer("<|im_start|>", add_special_tokens=False)["input_ids"][0]
    im_end_id = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
    assistant_id = tokenizer("assistant", add_special_tokens=False)["input_ids"][0]

    spans = []
    i = 0
    n = len(input_ids)
    while i < n - 2:
        # 匹配模式: <|im_start|>, "assistant", 换行
        if (
            input_ids[i] == im_start_id
            and i + 1 < n
            and input_ids[i + 1] == assistant_id
        ):
            # 内容从 i+2 开始，一直到遇到 im_end_id 为止（包含 im_end）
            j = i + 2
            while j < n and input_ids[j] != im_end_id:
                j += 1
            if j < n and input_ids[j] == im_end_id:
                # [start, end) = [i+2, j+1)
                spans.append((i + 2, j + 1))
                i = j + 1
                continue
        i += 1

    return spans


# --------------------
# 2. Dataset：适配你给的 data 格式
# --------------------
class SFTDataset(Dataset):
    """
    一行一个 JSON，例如：

    {
      "conversations": [
        {"role": "user", "content": "在这个场景中...<image>"},
        {"role": "assistant", "content": "在这幅画中，团队合作起到了..."},
        {"role": "user", "content": "..."}, ...
      ],
      "image": "train-00000-of-00001_image_22_0.jpg"
    }
    """

    def __init__(self, images_path, data_path, tokenizer, processor, config: VLMConfig):
        super().__init__()
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.datas = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.datas.append(json.loads(line))

        print(f"[SFT] 样本数: {len(self.datas)}")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        sample = self.datas[idx]

        try:
            image_name = sample["image"]
            conversations = sample["conversations"]

            # 1) 组装 messages：system + 多轮 user/assistant
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for turn in conversations:
                role = turn["role"]  # "user" or "assistant"
                content = turn["content"]
                messages.append({"role": role, "content": content})

            # 2) 用 chat_template 得到整段对话文本
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 不额外加 assistant 起始
            )

            # 把 <image> 替换成 49 个 <|image_pad|>
            text = text.replace("<image>", "<|image_pad|>" * self.config.image_pad_num)

            # 3) tokenize 成 id 序列
            input_ids = self.tokenizer(text)["input_ids"]

            # 4) 构造 labels：只监督 assistant 内容，其余全是 pad_token_id
            pad_id = self.tokenizer.pad_token_id
            labels = [pad_id] * len(input_ids)

            spans = find_assistant_spans(self.tokenizer, input_ids)
            for s, e in spans:
                labels[s:e] = input_ids[s:e]

            # 5) 做一次 shift，对齐你的 pretrain 逻辑
            input_ids = input_ids[:-1]
            labels = labels[1:]

            # 6) 读图
            image_path = os.path.join(self.images_path, image_name)
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(
                images=image, return_tensors="pt"
            )["pixel_values"].squeeze(0)

        except Exception as e:
            # 防止坏样本崩掉训练
            print(f"[WARN] 样本 {idx} 解析失败：{e}，使用默认白图样本代替。")
            default_image = Image.new("RGB", (224, 224), color="white")
            pixel_values = self.processor(
                images=default_image, return_tensors="pt"
            )["pixel_values"].squeeze(0)

            q_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "图片内容是什么\n<image>"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ).replace("<image>", "<|image_pad|>" * self.config.image_pad_num)

            a_text = "图片内容为空" + self.tokenizer.eos_token
            q_ids = self.tokenizer(q_text)["input_ids"]
            a_ids = self.tokenizer(a_text)["input_ids"]
            input_ids = q_ids + a_ids
            labels = [self.tokenizer.pad_token_id] * len(q_ids) + a_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


# --------------------
# 3. DataCollator：pad 到 batch 里同长，拼 pixel_values
# --------------------
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, labels, pixel_values = [], [], []

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(labs + [pad_id] * pad_len)
            pixel_values.append(f["pixel_values"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values, dim=0),  # [B, 3, H, W]
        }


# --------------------
# 4. 训练入口
# --------------------
if __name__ == "__main__":
    # 1) 先从预训练阶段的 checkpoint 加载多模态模型
    pretrained_path = (
        "/root/autodl-tmp/ai_learing/llm_related/train_multimodal_from_scratch/save/pretrain/checkpoint-11000"
    )

    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)

    model = AutoModelForCausalLM.from_pretrained(pretrained_path).cuda()
    config: VLMConfig = model.config

    # 2) 冻结 vision + 两层 MLP，只训练 llm_model
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []
    for name, param in model.named_parameters():
        if "llm_model" in name:
            param.requires_grad = True
            trainable_names.append(name)

    print("将要训练的参数：")
    for n in trainable_names[:10]:
        print("  ", n)
    if len(trainable_names) > 10:
        print("  ... 共", len(trainable_names), "个参数组")
    print("总参数量：", sum(p.numel() for p in model.parameters()))
    print("可训练参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 3) tokenizer / processor
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4) 路径按你自己的来改
    images_path = "/root/autodl-tmp/ai_learing/datasets/SFT_images/sft_images"  # 放 train-00000-of-00001_image_22_0.jpg 等
    data_path = "/root/autodl-tmp/ai_learing/datasets/SFT_images/sft_data.jsonl"       # 就是你贴的这种 JSONL 文件
    output_dir = "save/sft_multiturn"

    # 5) Trainer 配置
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=4,   # 看显存调
        learning_rate=5e-5,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to="tensorboard",
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
        remove_unused_columns=False,     # 多模态必须关，否则 pixel_values 会被删掉
    )

    train_dataset = SFTDataset(images_path, data_path, tokenizer, processor, config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=MyDataCollator(tokenizer),
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_dir)
    trainer.save_state()
