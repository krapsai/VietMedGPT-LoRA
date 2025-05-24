# VietMedGPT-LoRA 🩺🦙
Supervised fine-tuning for Vietnamese medical QA using LoRA and 4-bit quantization on OpenChat 3.5.

![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![made with 🤗 Transformers](https://img.shields.io/badge/🤗%20Transformers-%F0%9F%A4%97-ff69b4)

---

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Environment Setup](#environment-setup)

---

## 🧠 Introduction

The notebook `LLM_Finetune_SFT.ipynb` demonstrates how to:

- Load **OpenChat 3.5** as the base model.
- Apply **4-bit quantization** via BitsAndBytes to reduce VRAM usage.
- Use **LoRA adapters** (rank 8, α = 32) on Q/V projection layers.
- Add **Chain-of-Thought prompting** in Vietnamese.
- Fine-tune the model on the **`hungnm/vietnamese-medical-qa`** dataset (~5,000 QA pairs).
- Export a final model checkpoint to `sft-openchat-medqa/`.

✅ The resulting model provides answers with step-by-step reasoning, suitable for Vietnamese medical question answering.

---

## 🚀 Features

| Feature           | Description |
|------------------|-------------|
| **🔧 LoRA**       | Fine-tunes less than 1% of parameters → fast & memory-efficient |
| **🧮 4-bit**      | Reduces VRAM usage by ~2.6× compared to fp16 |
| **🩺 ViMedQA**    | Domain-specific QA in Vietnamese medical language |
| **🗣️ CoT Prompt** | Produces step-by-step reasoning before answering |
| **⚡ Trainer API**| Easy training via Hugging Face `transformers.Trainer` |

---

## 🛠 Environment Setup

```bash
git clone https://github.com/krapsai/openchat-vimedqa-sft.git
cd openchat-vimedqa-sft

# Python ≥ 3.10
python -m venv .venv
source .venv/bin/activate

pip install torch transformers datasets peft bitsandbytes accelerate
