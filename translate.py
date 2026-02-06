# This file is modified from https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/visual_language_chat/visual_language_chat.py

#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor
from pathlib import Path
from transformers import AutoTokenizer
import json


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help="Path to the model directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help="Text file")
    group.add_argument('--image', help="Image file")
    parser.add_argument("--device", help="Device to run the model on (default: CPU)", choices=["CPU", "GPU", "NPU"], default="CPU")
    parser.add_argument('--source_lang_code', required=True, help="Source language code, e.g., en or en-GB")
    parser.add_argument('--target_lang_code', required=True, help="Target language code, e.g., zh or zh-TW")
    args = parser.parse_args()

    if args.text is not None:
        content = "text"
        with open(args.text, "r", encoding="utf-8") as f:
            txt = f.read()
    else:
        content = "image"
        rgb = read_image(args.image)

    # GPU and NPU can be used as well.
    # Note: If NPU is selected, only the language model will be run on the NPU.
    enable_compile_cache = dict()
    if args.device == "GPU" or args.device == "NPU":
        # Cache compiled models on disk for GPU/NPU to save time on the next run.
        # It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"

    pipe = openvino_genai.VLMPipeline(args.model_dir, args.device, **enable_compile_cache)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 2048
    
    if content == "text":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": args.source_lang_code,
                        "target_lang_code": args.target_lang_code,
                        "text": txt
                    }
                ],
            }
        ]
    else:  # conten is "image"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source_lang_code": args.source_lang_code,
                        "target_lang_code": args.target_lang_code,
                        "image": "sample_image.png"   # file name doesn't care
                    }
                ],
            }
        ]
    print(f"Messages:\n{messages}\n")

    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    prompt = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"Prompt:\n{prompt}\n")
 
    # [WORKAROUND BEGINS]
    # As OpenVINO GenAI pipeline hasn't yet supported TranslateGemma. OpenVINO GenAI pipeline somehow formats input text with Gemma3 chat_template (see formated messages below) but validates it using the TranslateGemma chat_template, which results in errors.
    # formated messages=[{'content': '<bos><start_of_turn>user\nYou are a professional English (en) to
    #                                 Japanese (ja) translator. Your goal is to accurately convey the
    #                                 meaning and nuances of the original English text while adhering
    #                                 to Japanese grammar, vocabulary, and cultural sensitivities.\n
    #                                 Produce only the Japanese translation, without any additional
    #                                 explanations or commentary. Please translate the following English
    #                                 text into Japanese:\n\n\nhow are you doing today?<end_of_turn>\n<start_of_turn>model\n',
    #                     'role': 'user'
    #                    }]
    # Here we workaround this by forcing OpenVINO GenAI pipeline to use Gemma3 chat_template for validation
    GEMMA3_CHAT_TEMPLATE_FILE = "chat_template-gemma3.json"   # Downloaded from https://huggingface.co/google/gemma-3-4b-it/blob/main/chat_template.json
    with open(GEMMA3_CHAT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    GEMMA3_CHAT_TEMPLATE = data["chat_template"]
    pipe.get_tokenizer().set_chat_template(GEMMA3_CHAT_TEMPLATE)
    # [WORKAROUND ENDS]

    if content == "text":
        output = pipe.generate(prompt, generation_config=config)
    else:  # conten is "image"
        output = pipe.generate(prompt, images=rgb, generation_config=config)

    print(f"Output:\n{output}\n")


if '__main__' == __name__:
    main()
