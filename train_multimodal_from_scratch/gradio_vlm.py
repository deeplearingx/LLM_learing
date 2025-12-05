import gradio as gr
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM
import torch
from torch.nn import functional as F

device = "cuda:0"

# 1. 加载处理器和分词器
processor = AutoProcessor.from_pretrained(
    "/root/autodl-tmp/ai_learing/models/google/siglip2-base-patch16-224"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/ai_learing/models/Qwen/Qwen2.5-0.5B-Instruct"
)

# 2. 注册自定义模型
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

# 3. 加载预训练阶段和 SFT 阶段的权重
pretrain_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/ai_learing/llm_related/train_multimodal_from_scratch/save/pretrain/checkpoint-11000"
).to(device)

sft_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/ai_learing/llm_related/train_multimodal_from_scratch/save/sft_multiturn/checkpoint-4000"
).to(device)

pretrain_model.eval()
sft_model.eval()


def generate(
    mode,
    vision_mode,
    image_input,
    text_input,
    max_new_tokens=100,
    temperature=0.0,
    top_k=None,
):
    """
    mode: 'pretrain' or 'sft'
    vision_mode: 'real' 使用真实图片; 'zero' 使用全 0 图像(断开视觉)
    """

    if image_input is None:
        return "请先上传图片。"
    if not text_input:
        return "请先输入文本问题。"

    # 1. 构造带 <image> 的对话模板
    q_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{text_input}\n<image>"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    # 把 <image> 替换为 49 个 <|image_pad|>
    q_text = q_text.replace("<image>", "<|image_pad|>" * 49)

    # 2. 文本 -> input_ids
    inputs = tokenizer(q_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # 3. 图像 -> pixel_values (tensor)
    # gradio Image(type="pil") 已经是 PIL.Image
    image = image_input.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    # pixel_values: [1, 3, H, W]

    # 视觉模式实验：真实图片 vs 零图像
    if vision_mode == "zero":
        # 用同形状的全 0 图像，达到“断开视觉”的效果
        pixel_values = torch.zeros_like(pixel_values)

    eos = tokenizer.eos_token_id
    s = input_ids.shape[1]

    # 选择模型一次就好，不用每步判断
    model = pretrain_model if mode == "pretrain" else sft_model

    max_len = s + max_new_tokens - 1

    with torch.no_grad():
        while input_ids.shape[1] < max_len:
            # 关键：用关键字参数，并显式传 labels=None，避免 forward 缺参
            inference_res = model(
                input_ids=input_ids,
                attention_mask=None,
                pixel_values=pixel_values,
                labels=None,  # 如果你已经把 VLM.forward 改成 labels 可选，这行可以去掉
            )
            logits = inference_res.logits
            # 只用最后一个 token 的 logits
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # idx_next 是 shape [1,1] 的 tensor，要用 .item() 比较
            if idx_next.item() == eos:
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)

    # 只 decode 新生成的部分，并去掉 special tokens
    return tokenizer.decode(input_ids[:, s:][0], skip_special_tokens=True)


# 4. Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="选择图片")
        with gr.Column(scale=1):
            mode = gr.Radio(["pretrain", "sft"], label="选择模型", value="sft")
            vision_mode = gr.Radio(
                ["real", "zero"],
                label="视觉模式",
                value="real",
                info="real=使用真实图片; zero=使用全零图像(断开视觉)",
            )
            text_input = gr.Textbox(label="输入文本（例如：描述图片内容）")
            text_output = gr.Textbox(label="输出文本")
            generate_button = gr.Button("生成")
            generate_button.click(
                generate,
                inputs=[mode, vision_mode, image_input, text_input],
                outputs=text_output,
            )

if __name__ == "__main__":

    demo.launch(share=False, server_name="0.0.0.0", server_port=7891)
