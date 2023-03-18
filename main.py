import argparse
import os
import json
import datetime
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel


def get_args():
    parser = argparse.ArgumentParser(description="ChatGLM Arguments")

    parser.add_argument(
        "--path", default="chatglm-6b", help="The path of ChatGLM model"
    )
    parser.add_argument(
        "--example", default="examples.txt", help="The path of input example"
    )
    parser.add_argument("--log", default="log", help="The path of output log")
    parser.add_argument(
        "--tokenize_path",
        "--t_path",
        default=None,
        help="The path of ChatGLM model",
    )

    parser.add_argument(
        "--promote_path", "--p_path", default="promotes", help="The path of promotes"
    )

    quantize_group = parser.add_mutually_exclusive_group()
    quantize_group.add_argument(
        "--low_vram", action="store_true", help="Use 4-bit quantization"
    )
    quantize_group.add_argument(
        "--med_vram", action="store_true", help="Use 8-bit quantization"
    )

    parser.add_argument("--cpu", action="store_true", help="Use CPU")

    parser.add_argument("--low_ram", action="store_true", help="Use CPU (low RAM)")

    return parser.parse_args()


args = get_args()

if not os.path.isdir(args.path):
    raise FileNotFoundError("Model not found")

if args.tokenize_path is None:
    args.tokenize_path = args.path
elif not os.path.isdir(args.tokenize_path):
    raise FileNotFoundError("promotes not found")

possible_promotes: list[str] = os.listdir(args.promote_path)
possible_promotes.append("none")


def update_possible_promotes():
    possible_promotes = os.listdir(args.promote_path)
    possible_promotes.append("none")
    return gr.Dropdown.update(choices=possible_promotes)


tokenizer = AutoTokenizer.from_pretrained(args.tokenize_path, trust_remote_code=True)
model = AutoModel.from_pretrained(args.path, trust_remote_code=True)

if args.cpu:
    model = model.float()
elif args.low_ram:
    model = model.bfloat16()
else:
    if args.low_vram:
        model = model.half().quantize(4).cuda()
    elif args.med_vram:
        model = model.half().quantize(8).cuda()
    else:
        model = model.half().cuda()

model = model.eval()


def parse_text(text):
    # lines = text.split("\n")
    # incode = False
    # for i, line in enumerate(lines):
    #     if "```" in line:
    #         item = line.split("`")[-1]
    #         if incode:
    #             lines[i] = "</code></pre>"
    #         elif item:
    #             lines[i] = f'<pre><code class="{item}">'
    #         else:
    #             lines[i] = '<pre><code>'
    #         incode = not incode
    #     else:
    #         if i > 0 and not incode:
    #             line = line.replace("<", "&lt;").replace(">", "&gt;")
    #             lines[i] = f"<br/>{line}"
    # return "".join(lines)
    return text


def load_promote(file_name: str) -> tuple[str, str]:
    with open(
        file=os.path.join(args.promote_path, file_name), encoding="utf-8"
    ) as promote_file:
        dict_list = json.load(promote_file)
        history = [
            (restore_str(item["input"]), restore_str(item["output"]))
            for item in dict_list
        ]
        return history[0]


def chat_wrapper(
    promote: str,
    query: str,
    styled_history: list[tuple[str, str]],
    history: list[tuple[str, str]],
    max_length: int,
    top_p: float,
    temperature: float,
):
    if promote != "none":
        promot_val = load_promote(promote)
        if len(history) == 0 and len(promot_val[1]) != 0:
            history = [promot_val]
        elif len(history) == 0 and len(promot_val[1]) == 0:
            query = promot_val[0]
        elif len(promot_val[1]) != 0:
            history[0] = promot_val
    message, history = model.chat(
        tokenizer,
        query,
        history=history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    )
    styled_history = [(parse_text(h[0]), parse_text(h[1])) for h in history]
    return styled_history, history, "", get_current_vram()


def regenerate_wrapper(
    promote, query, styled_history, history, max_length, top_p, temperature
):
    if len(history) == 0:
        return [], [], "", get_current_vram()
    new_styled_history, new_history, query, vram = edit_wrapper(styled_history, history)
    return chat_wrapper(
        promote, query, new_styled_history, new_history, max_length, top_p, temperature
    )


def edit_wrapper(styled_history, history):
    if len(history) == 0:
        return [], [], "", get_current_vram()
    query = history[-1][0]
    history = history[:-1]
    styled_history = styled_history[:-1]
    return styled_history, history, query, get_current_vram()


def reset_history():
    return [], [], "", get_current_vram()


def cut_str(s: str):
    return s.split("\n")


def restore_str(s):
    if isinstance(s, list):
        return "\n".join(s)
    return s


def save_history(history, promote):
    os.makedirs(args.log, exist_ok=True)
    dict_list = [{"input": cut_str(q), "output": cut_str(a)} for q, a in history]
    file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    if promote != "none":
        file_name = f"{os.path.splitext(promote)[0]}-{file_name}"
    file_name = os.path.join(args.log, file_name)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(dict_list, f, ensure_ascii=False, indent=2)
    return file_name


def save_config(max_length, top_p, temperature):
    with open("config.json", "w") as f:
        json.dump(
            {"max_length": max_length, "top_p": top_p, "temperature": temperature},
            f,
            indent=2,
        )


def load_history(file, styled_history, history):
    current_styled_history, current_history = styled_history.copy(), history.copy()
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            dict_list = json.load(f)
        history = [
            (restore_str(item["input"]), restore_str(item["output"]))
            for item in dict_list
        ]
        styled_history = [(parse_text(h[0]), parse_text(h[1])) for h in history]
    except BaseException:
        return current_styled_history, current_history, "", get_current_vram()
    return styled_history, history, "", get_current_vram()


def get_current_vram():
    return torch.cuda.memory_allocated(0) / (1024**3)


def main():
    with gr.Blocks(title="ChatGLM") as app:
        if not os.path.isfile("config.json"):
            save_config(4096, 0.7, 0.95)

        with open("config.json", "r", encoding="utf-8") as f:
            configs = json.loads(f.read())

        gr.Markdown("""<h1><center>ChatGLM WebUI</center></h1>""")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    save = gr.Button("保存对话 （在 log 文件夹下）")
                    file = gr.File(file_types=["json"])
                    load = gr.UploadButton(
                        "读取对话", file_types=["file"], file_count="single"
                    )
                promot = gr.Dropdown(possible_promotes, value="none")
                gr.Button("加载promotes").click(update_possible_promotes, outputs=promot)

                with gr.Accordion(label="设置"):
                    gr.Markdown(
                        """`Max Length` 是生成文本时的长度限制，`Top P` 控制输出文本中概率最高前 p 个单词的总概率，`Temperature` 控制生成文本的多样性和随机性。<br/>`Top P` 变小会生成更多样和不相关的文本；变大会生成更保守和相关的文本。<br/>`Temperature` 变小会生成更保守和相关的文本；变大会生成更奇特和不相关的文本。"""
                    )
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            step=0.01,
                            label="Top P",
                            value=configs["top_p"],
                        )
                        temperature = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            step=0.01,
                            label="Temperature",
                            value=configs["temperature"],
                        )
                    max_length = gr.Slider(
                        minimum=4.0,
                        maximum=4096.0,
                        step=4.0,
                        label="Max Length",
                        value=configs["max_length"],
                    )
                    save_conf = gr.Button("保存设置")
            with gr.Column(scale=7):
                gr.Markdown("""<h2>聊天记录</h2>""")
                chatbot = gr.Chatbot(elem_id="chatbot", show_label=False)
                vram = gr.Slider(
                    maximum=torch.cuda.get_device_properties(0).total_memory
                    / (1024**3),
                    value=get_current_vram(),
                    interactive=False,
                    label="显存",
                )
                state = gr.State([])
                message = gr.Textbox(placeholder="输入内容", label="用户：")
                if os.path.isfile(args.example):
                    with gr.Accordion(label="示例"):
                        with open(args.example, encoding="utf-8") as e:
                            gr.Examples(
                                [i.strip() for i in e.readlines()],
                                inputs=[message],
                                examples_per_page=100,
                            )
                with gr.Row():
                    submit = gr.Button("提交", variant="primary")
                    edit = gr.Button("修改问题")
                    regen = gr.Button("重新生成")
                delete = gr.Button("清空聊天", variant="stop")

        submit_list = [promot, message, chatbot, state, max_length, top_p, temperature]
        state_list = [chatbot, state, message, vram]

        save_conf.click(save_config, inputs=[max_length, top_p, temperature])
        load.upload(load_history, inputs=[load, chatbot, state], outputs=state_list)
        save.click(save_history, inputs=[state, promot], outputs=[file])
        message.submit(
            chat_wrapper,
            inputs=submit_list,
            outputs=state_list,
        )
        submit.click(
            chat_wrapper,
            inputs=submit_list,
            outputs=state_list,
            api_name="chat",
        )
        edit.click(edit_wrapper, inputs=submit_list[2:4], outputs=state_list)
        regen.click(regenerate_wrapper, inputs=submit_list, outputs=state_list)
        delete.click(reset_history, outputs=state_list)
        app.launch(debug=True, server_name="0.0.0.0", share=True, show_api=False)


if __name__ == "__main__":
    main()
