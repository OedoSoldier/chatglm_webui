# ChatGLM WebUI

基于 [ChatGPT WebUI](https://github.com/dotmet/chatgpt_webui) 的一个简单的 [ChatGLM](https://github.com/THUDM/ChatGLM-6B) WebUI。

## 环境安装

`pip install -r requirements.txt`

Gradio 版本必须大于 3.21.0！

Windows 下 CPU 运行需要安装编译器，可参考 (https://www.freesion.com/article/4185669814/)。

## 运行

通过 `git clone https://huggingface.co/THUDM/chatglm-6b-int4` 下载模型文件到根目录下然后 `python main.py` 即可，默认状态至少需要 4 GB 显存（CPU 运行则需要 5.2 GB 内存）。

### 实参

 - `--path`：指定模型所在文件夹
