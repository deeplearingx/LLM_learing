import os
import time
import requests
from tqdm import tqdm
from requests.exceptions import RequestException

# ==========================
# 配置区域
# ==========================

# ✅ 正确的镜像前缀：用 resolve，而不是 tree
#放入下载的链接
BASE_URL = "https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main"

# 要下载的文件列表
FILES = [
    "sft_data.jsonl",
    "sft_images.zip",
]

# 本地保存目录
SAVE_DIR = "/root/autodl-tmp/ai_learing/datasets/SFT_images"

# 重试设置
MAX_RETRIES = 5
RETRY_BASE_WAIT = 5


def download_file(url: str, save_path: str) -> bool:
    """支持断点续传 + 重试"""

    temp_size = 0
    if os.path.exists(save_path):
        temp_size = os.path.getsize(save_path)

    print(f"\n开始下载: {url}")
    print(f"👉 已有 {temp_size} 字节，将尝试断点续传...")

    headers = {"Range": f"bytes={temp_size}-"} if temp_size > 0 else {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, headers=headers, timeout=60)

            if resp.status_code in (200, 206):
                total_size = resp.headers.get("content-length")
                total_size = int(total_size) if total_size is not None else 0
                total_to_download = temp_size + total_size
                mode = "ab" if temp_size > 0 else "wb"

                with open(save_path, mode) as f, tqdm(
                    total=total_to_download,
                    initial=temp_size,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(save_path),
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

                print(f"✅ 下载完成: {save_path}")
                return True

            elif 500 <= resp.status_code < 600:
                wait_time = RETRY_BASE_WAIT * attempt
                print(
                    f"⚠️ 服务器异常 HTTP {resp.status_code}, "
                    f"第 {attempt}/{MAX_RETRIES} 次重试，{wait_time} 秒后重试..."
                )
                time.sleep(wait_time)
                continue

            else:
                print(f"❌ 无法下载（HTTP {resp.status_code}），停止重试")
                return False

        except RequestException as e:
            wait_time = RETRY_BASE_WAIT * attempt
            print(
                f"⚠️ 网络异常: {e}, "
                f"第 {attempt}/{MAX_RETRIES} 次重试，{wait_time} 秒后重试..."
            )
            time.sleep(wait_time)

    print(f"❌ 多次重试失败: {url}")
    return False


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_ok = True
    for fname in FILES:
        # ✅ 这里一定要加 '/'，否则会变成 mainsft_xxx 这种错误
        url = f"{BASE_URL}/{fname}"
        save_path = os.path.join(SAVE_DIR, fname)
        ok = download_file(url, save_path)
        all_ok = all_ok and ok

    if all_ok:
        print("\n🎉 所有文件成功下载！")
    else:
        print("\n⚠️ 有文件下载失败，请查看上方日志。")
