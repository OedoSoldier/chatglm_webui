import argparse
import os
import json
import datetime
import gradio as gr
from transformers import AutoTokenizer, AutoModel


def get_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument(
        "--path",
        default="chatglm-6b",
        help="The path of ChatGLM model")
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Use 4-bit quantization")
    parser.add_argument(
        "--med_vram",
        action="store_true",
        help="Use 8-bit quantization")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument(
        "--low_ram",
        action="store_true",
        help="Use CPU (low ram)")
    return parser.parse_args()


args = get_args()

if not os.path.isdir(args.path):
    raise FileNotFoundError("Model not found")

tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
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
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            items = line.split('`')
            if items[-1]:
                lines[i] = f'<pre><code class="{items[-1]}">'
            else:
                lines[i] = f'</code></pre>'
        else:
            if i > 0:
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                lines[i] = '<br/>' + line  # .replace(" ", "&nbsp;")
    return "".join(lines)


def chat(query, styled_history, history, max_length, top_p, temperature):
    message, history = model.chat(tokenizer, query, history=history,
                                  max_length=max_length, top_p=top_p, temperature=temperature)
    styled_history.append((parse_text(query), parse_text(message)))
    return styled_history, history


def regenerate(styled_history, history, max_length, top_p, temperature):
    query = history[-1][0]
    history = history[:-1]
    styled_history = styled_history[-1]
    return chat(query, styled_history, history, max_length, top_p, temperature)


def reset_history(styled_history, history):
    history = []
    styled_history = []
    return styled_history, history


def save_history(history):
    if not os.path.exists("log"):
        os.mkdir("log")
    dict_list = [{"input": item[0], "output": item[1]} for item in history]

    json_data = json.dumps(dict_list, indent=2, ensure_ascii=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"log/{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json_data)


def save_config(max_length, top_p, temperature):
    configs = {
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature}
    json_data = json.dumps(configs, indent=2)
    with open("config.json", "w") as f:
        f.write(json_data)


def load_history(file, styled_history, history):
    current_styled_history, current_history = styled_history.copy(), history.copy()
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            dict_list = json.loads(f.read())
        history = [(item["input"], item["output"]) for item in dict_list]
        styled_history = [
            (parse_text(
                item["input"]), parse_text(
                item["output"])) for item in dict_list]
    except BaseException:
        return current_styled_history, current_history
    return styled_history, history


def main():
    with gr.Blocks() as app:
        if not os.path.isfile("config.json"):
            save_config(2048, 0.7, 0.95)

        with open("config.json", "r", encoding="utf-8") as f:
            configs = json.loads(f.read())

        gr.Markdown("""<h1><center>ChatGLM</center></h1>""")

        with gr.Row():
            max_length = gr.Slider(
                minimum=0.0,
                maximum=4096.0,
                step=1.0,
                label="Max Length",
                value=configs["max_length"])
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label="Top P",
                value=configs["top_p"])
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label="Temperature",
                value=configs["temperature"])
        save_conf = gr.Button("保存设置")

        gr.Markdown("""<h2>聊天记录</h2>""")

        chatbot1 = gr.Chatbot(elem_id="chatbot", show_label=False)
        state = gr.State([])
        message = gr.Textbox(placeholder="输入内容", label="你：")

        with gr.Row():
            submit = gr.Button("提交")
            regen = gr.Button("重新生成")

        delete = gr.Button("清空聊天")

        with gr.Row():
            save = gr.Button("保存对话 （在 log 文件夹下）")
            load = gr.UploadButton(
                "读取对话",
                file_types=["file"],
                file_count="single")

        save_conf.click(save_config, inputs=[max_length, top_p, temperature])
        load.upload(
            load_history, inputs=[
                load, chatbot1, state], outputs=[
                chatbot1, state])
        save.click(save_history, inputs=[state])
        message.submit(
            chat,
            inputs=[
                message,
                chatbot1,
                state,
                max_length,
                top_p,
                temperature],
            outputs=[
                chatbot1,
                state])
        message.submit(lambda: "", None, message)
        submit.click(
            chat,
            inputs=[
                message,
                chatbot1,
                state,
                max_length,
                top_p,
                temperature],
            outputs=[
                chatbot1,
                state])
        submit.click(lambda: "", None, message)
        regen.click(
            regenerate, inputs=[
                chatbot1, state, max_length, top_p, temperature], outputs=[
                chatbot1, state])
        regen.click(lambda: "", None, message)
        delete.click(
            reset_history, inputs=[
                chatbot1, state], outputs=[
                chatbot1, state])
        delete.click(lambda: "", None, message)

        app.launch(debug=True)


if __name__ == '__main__':
    main()
