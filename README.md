# ChatGLM WebUI

基于 [ChatGPT WebUI](https://github.com/dotmet/chatgpt_webui) 的一个简单的 [ChatGLM](https://github.com/THUDM/ChatGLM-6B) WebUI。

## 环境安装

`pip install -r requirements.txt`

Gradio 版本必须大于 3.21.0！

## 运行

通过 `git clone https://huggingface.co/THUDM/chatglm-6b` 下载模型文件到根目录下然后 `python main.py` 即可，默认状态至少需要 13GB 显存。

### 实参

 - `--path`：指定模型所在文件夹
 - `--tokenize_path` / `--t_path`：指定模型的 `tokenize` 所在文件夹(方便加载微调后的模型)
 - `--low_vram`：4-bit 量化，6GB 显存可用
 - `--med_vram`：8-bit 量化，10GB 显存可用
 - `--cpu`：CPU运行，32G 内存可用
 - `--low_ram`：CPU运行，16G 内存可用
 - `--example`: 示例文件的文件名
 - `--log`: 保存对话记录的目录

## Promotes 和示例

提供可以在输入时使用的 example ，在 `example.txt` 中，每行一句，启动时加载。

提供在对话开头使用的 `Promotes` ,在 `promotes` 文件夹中,每个文件包含一组，可以点击 `加载promotes` 刷新。

`Promotes` 格式与对话记录格式一致，如果 `output` 为空数组，则由模型生成第一句回复。如果`output` 不为空，则将 `Promotes` 和对应的 `output` 注入历史（可以迫使模型接受某些设定）