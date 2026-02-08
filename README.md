# About TranslateGemma-OpenVINO

Python sample code that runs [TranslateGemma](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) on Intel devices (`CPU`, `GPU`, `NPU`) by using [`OpenVINO GenAI`](
https://github.com/openvinotoolkit/openvino.genai) pipeline.


Files in this repo:
 - [`translate.py`](./translate.py) the main pipeline. (modified from [`visual_language_chat.py`](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/samples/python/visual_language_chat/visual_language_chat.py))
 - [`export-requirements.txt`](./export-requirements.txt) required Python packages for model export. ([`download link`](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/samples/export-requirements.txt))
 - [`deployment-requirements.txt`](./deployment-requirements.txt) required Python packages for model deployment. ([`download link`](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/samples/deployment-requirements.txt))
 - [`chat_template-gemma3.json`](./chat_template-gemma3.json) Gemma3 chat_template used to workaround the validation error of OV GenAI VLM pipeline ([`download link`](https://huggingface.co/google/gemma-3-4b-it/blob/main/chat_template.json))
- [`text_en.txt`](./text_en.txt) an English text file used to test text translation. ([`source`](https://learning.cambridgeinternational.org/classroom/pluginfile.php/219010/mod_label/intro/Writing_a_speech.pdf))
- [`text_zh-TW.txt`](./text_zh-TW.txt) a (Traditional)Chinese text file used to test text translation. ([`source`](https://zh.wikipedia.org/wiki/%E7%99%BB%E9%B8%9B%E9%9B%80%E6%A8%93))
- [`image_cs.jpg`](./image_cs.jpg) an image that contains Czech characters used to test image translation. ([`source`](https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg))
- [`image_en.png`](./image_en.png) an image that contains English characters used to test image translation. ([`source`](https://raw.githubusercontent.com/esalesky/vistra-benchmark/refs/heads/main/images/f488c322.png))

# Quick Start Guide
## Prepare Model
### Install required packages
Input the following command to install required packages for model export. The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
pip install --upgrade-strategy eager -r export-requirements.txt
```

### Hugging Face login
The script needs to access Hugging Face models. To get the access, please visit  https://huggingface.co/google/translategemma-4b-it and hit `log in` to login with you account

Make sure your [access token](https://huggingface.co/settings/tokens) has been prepared. Make sure [huggingface-cli](https://huggingface.co/docs/huggingface_hub/v0.30.2/guides/cli) has been installed. Open a Command Prompt, run ```huggingface-cli login``` with your access token
```sh
pip install "huggingface_hub[cli]<1.0,>=0.34.0"
huggingface-cli login
```
- Transformers 4.55.4 requires huggingface-hub<1.0,>=0.34.0
### Download and export model
Then, run the export with Optimum CLI:
```sh
optimum-cli export openvino --model google/translategemma-4b-it --trust-remote-code translategemma-4b-it
```
- Exported models will be under model_dir (`translategemma-4b-it` in this example)
- The argument `--weight-format` can be used to quantize the model. See [Quantization](#Quantization) for the detail


## Run

### Install required packages
Input the following command to install required packages for model deployment.
```
pip install -r deployment-requirements.txt
```
### Pipeline usage
```
translate.py --model_dir MODEL_DIR
             --text TEXT
             --image IMAGE
             --device {CPU,GPU,NPU}
             --source_lang_code SOURCE_LANG_CODE
             --target_lang_code TARGET_LANG_CODE
```
- The arguments `--model_dir`, `--source_lang_code` and `-target_lang_code` are required:
- Either `--text TEXT` or `--image IMAGE` should be provided
- The `--device` can be `CPU`, `GPU` or `NPU`
- Language code examples: `en`, `en-GB`, `zh` or `zh-TW`. Full language code can be found [`here`](https://huggingface.co/google/translategemma-4b-it/blob/main/chat_template.jinja) or locally check `chat_template.jinja` under model_dir

### Run Text Translation
Command:
```
python translate.py --model_dir translategemma-4b-it --device GPU --source_lang_code zh-TW --target_lang_code en --text text_zh-TW.txt
```
Result:
```
Input:
白日依山盡，黃河入海流；欲窮千里目，更上一層樓。

Output:
As the sun sets behind the mountains, the Yellow River flows into the sea. To gain a broader perspective, one must climb to a higher vantage point.
```

### Run Image Translation
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

## Quantization
When exporting a model, the argument `--weight-format` can be used to quantize the model. The supported weights are `int8`, `int4` and `nf4`. Please visit [`OpenVINO model preparation guide`](https://openvinotoolkit.github.io/openvino.genai/docs/guides/model-preparation/convert-to-openvino) for the detail.

```sh
optimum-cli export openvino --model google/translategemma-4b-it --trust-remote-code --weight-format int8 translategemma-4b-it_int8
```
### Tested device
The pipeline is verified on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system with 32GB memory. GPU/NPU driver info below
* ```GPU: Intel(R) Arc(TM) 130V GPU, driver 32.0.101.8425 (1/16/2026)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4514 (12/17/2025)```

### Result
```
| Model                      | CPU    | GPU    | NPU    |
|----------------------------|--------|--------|--------|
| translategemma-4b-it       | OK     | OK     | OK     |
| translategemma-4b-it(int8) | OK     | OK     | OK     |
| translategemma-4b-it(int4) | OK     | OK     | Fail*  |
| translategemma-4b-it(nf4)  | OK     | OK     | OK**   |
```
- The int4 model fails to run on NPU, check [`log.txt`](./log.txt) for the detail
- The nf4 model can run on NPU but very slow
### Log
[`log.txt`](./log.txt) is provided for reference

