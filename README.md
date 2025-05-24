# VietMedGPT-LoRA
Supervised fine-tuning Vietnamese medical QA with LoRA + 4-bit on OpenChat 3.5

# OpenChat ViMedQA SFT 🩺🦙

> Supervised fine-tuning (LoRA + 4-bit) mô hình [OpenChat 3.5](https://huggingface.co/openchat) cho tác vụ Hỏi–Đáp Y khoa tiếng Việt.

![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![made with 🤗 Transformers](https://img.shields.io/badge/🤗%20Transformers-%F0%9F%A4%97-ff69b4)

---

## Table of Contents
1. [Giới thiệu](#giới-thiệu)
2. [Tính năng](#tính-năng)
3. [Chuẩn bị môi trường](#chuẩn-bị-môi-trường)


---

## Giới thiệu
Notebook `LLM_Finetune_SFT.ipynb` minh hoạ quy trình:
* Dùng **OpenChat 3.5**;
* Giảm dụng VRAM với **4-bit quantization** (BitsAndBytes);
* Áp dụng **LoRA** (rank 8, α 32) vào các tầng Q/V-proj;
* Thêm prompt ép “Chain-of-Thought” tiếng Việt;
* Fine-tune trên **`hungnm/vietnamese-medical-qa`** (≈ 5 k cặp Q–A);
* Xuất checkpoint `sft-openchat-medqa/`.

Kết quả: mô hình trả lời kèm các bước giải thích trước khi đưa ra câu trả lời cuối cùng

---

## Tính năng
| | Mô tả |
|---|---|
| **🔧 LoRA** | Chỉ huấn luyện <1 % tham số → nhanh & rẻ |
| **🧮 4-bit** | Tiết kiệm VRAM gấp ~2,6× so với fp16 |
| **🩺 ViMedQA** | Domain chuyên biệt Y khoa tiếng Việt |
| **🗣️ CoT Prompt** | Thúc đẩy lời giải có bước lập luận rõ ràng |
| **⚡ Trainer API** | Sử dụng `transformers.Trainer` – dễ tái lập |

---

## Chuẩn bị môi trường

```bash
git clone https://github.com/<username>/openchat-vimedqa-sft.git
cd openchat-vimedqa-sft

# Python ≥ 3.10
python -m venv .venv
source .venv/bin/activate

pip install torch transformers datasets peft bitsandbytes accelerate
