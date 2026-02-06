# About TranslateGemma-OpenVINO

This Python sample code is to run [TranslateGemma](https://huggingface.co/google/gemma-3-4b-it) on Intel devices thru OpenVINO.

Files in this repo:
 - [`translate.py`](./translate.py) the main pipeline. (modified from [`visual_language_chat.py`](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/samples/python/visual_language_chat/visual_language_chat.py))
 - [`export-requirements.txt`](./export-requirements.txt) required Python packages for model export. ([`download link`](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/samples/export-requirements.txt))
 - [`deployment-requirements.txt`](./deployment-requirements.txt) required Python packages for model deployment. ([`download link`](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/samples/deployment-requirements.txt))
 - [`chat_template-gemma3.json`](./chat_template-gemma3.json) Gemma3 chat_template used to workaround the validation error of OV GenAI VLM pipeline ([`download link`](https://huggingface.co/google/gemma-3-4b-it/blob/main/chat_template.json))
- [`text_en.txt`](./text_en.txt) an English text file used to test text translation. ([`source`](https://learning.cambridgeinternational.org/classroom/pluginfile.php/219010/mod_label/intro/Writing_a_speech.pdf))
- [`text_zh-TW.txt`](./text_zh-TW.txt) an Chinese(Traditional) text file used to test text translation. ([`source`](https://zh.wikipedia.org/wiki/%E7%99%BB%E9%B8%9B%E9%9B%80%E6%A8%93))
- [`image_cs.jpg`](./image_cs.jpg) an image that contains Czech characters used to test image translation. ([`source`](https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg))
- [`image_en.png`](./image_en.png) an image that contains English characters used to test image translation. ([`source`](https://raw.githubusercontent.com/esalesky/vistra-benchmark/refs/heads/main/images/f488c322.png))


## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
pip install --upgrade-strategy eager -r export-requirements.txt
```

### Hugging Face login
The script needs to access Hugging Face models. To get the access, please visit below link and hit "log in" https://huggingface.co/google/gemma-3-4b-it with you account

Make sure your [access token](https://huggingface.co/settings/tokens) has been prepared. Make sure [huggingface-cli](https://huggingface.co/docs/huggingface_hub/v0.30.2/guides/cli) has been installed. Open a Command Prompt, run ```huggingface-cli login``` with your access token

### Export model
Then, run the export with Optimum CLI:
```sh
optimum-cli export openvino --model google/gemma-3-4b-it --trust-remote-code gemma-3-4b-it
```

## Run

### Usage
```
translate.py --model_dir MODEL_DIR
             --text TEXT
             --image IMAGE
             --device {CPU,GPU,NPU}
             --source_lang_code SOURCE_LANG_CODE
             --target_lang_code TARGET_LANG_CODE

The following arguments are required: --model_dir, --source_lang_code, -target_lang_code
Either --text TEXT or --image IMAGE should be provided
```
The `--device` can be `CPU`, `GPU` or `NPU`

Language code examples: `en`, `en-GB`, `zh` or `zh-TW`. Full language code can be found [`here`](https://huggingface.co/google/translategemma-4b-it/blob/main/chat_template.jinja)

## Text translation:
Command:
```
python translate.py --model_dir translategemma-4b-it --device GPU --source_lang_code zh-TW --target_lang_code en --text text_zh-TW.txt
```
Result:
```
白日依山盡，黃河入海流；欲窮千里目，更上一層樓。

Output:
As the sun sets behind the mountains, the Yellow River flows into the sea. To gain a broader perspective, one must climb to a higher vantage point.
```

## Image translation:
Command:
```
python translate.py --model_dir translategemma-4b-it --device GPU --source_lang_code cs --target_lang_code en --image image_cs.jpg
```

Input:
![](https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg)
```
Output:
Pedestrian Zone

Child Supervision

IZS, CBS in Supervision
0 - 24 hours
```

## Log
### Tested devices
The pipeline is verified on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 130V GPU, driver 32.0.101.8331 (11/26/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4514 (12/17/2025)```

### Sample log
[`log.txt`](./log.txt) is provided for reference


