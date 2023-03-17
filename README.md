# ChatGLM WebUI

基于 [ChatGPT WebUI](https://github.com/dotmet/chatgpt_webui) 的一个简单的 [ChatGLM](https://github.com/THUDM/ChatGLM-6B) WebUI。

## 环境安装

`pip install -r requirements.txt`

Gradio 版本必须大于 3.21.0！

## 运行

通过 `git clone https://huggingface.co/THUDM/chatglm-6b` 下载模型文件到根目录下然后 `python main.py` 即可，默认状态至少需要 13GB 显存。

### 实参

 - `--path`：指定模型所在文件夹
 - `--low_vram`：4-bit 量化，6GB 显存可用
 - `--med_vram`：8-bit 量化，10GB 显存可用
 - `--cpu`：CPU运行，32G 内存可用
 - `--low_ram`：CPU运行，16G 内存可用