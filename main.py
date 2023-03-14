import argparse
import os
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
    parser.add_argument("--cpu", action="store_true", help="Use cpu")
    parser.add_argument(
        "--low_ram",
        action="store_true",
        help="Use cpu (low ram)")
    return parser.parse_args()


args = get_args()

if os.path.isdir(args.path):
    tokenizer = AutoTokenizer.from_pretrained(
        "chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("chatglm-6b", trust_remote_code=True)
else:
    raise FileNotFoundError("Model not found")

if args.low_vram:
    model = model.half().quantize(4).cuda()
elif args.med_vram:
    model = model.half().quantize(8).cuda()
elif args.cpu:
    model = model.float()
elif args.low_ram:
    model = model.bfloat16()
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


history = []
styled_history = []


def chat(query, styled_history, history):
    message, history = model.chat(tokenizer, query, history=history)
    styled_history.append((parse_text(query), parse_text(message)))
    # history.append((parse_text(query), parse_text(message)))
    return styled_history, history


def regenerate(styled_history, history):
    query = history[-1][0]
    history = history[:-1]
    styled_history = styled_history[-1]
    return chat(query, styled_history, history)


def reset_history(styled_history, history):
    history = []
    styled_history = []
    return styled_history, history


css = None

with gr.Blocks(css=css) as demo:

    gr.Markdown("""<h1><center>ChatGLM</center></h1>""")
    gr.Markdown("""<h2>聊天记录</h2>""")

    chatbot1 = gr.Chatbot(elem_id="chatbot", show_label=False)
    state = gr.State([])
    message = gr.Textbox(placeholder="输入内容", label="用户：")

    with gr.Row():
        submit = gr.Button("提交")
        regen = gr.Button("重新生成")

    delete = gr.Button("清空聊天")

    message.submit(
        chat, inputs=[
            message, chatbot1, state], outputs=[
            chatbot1, state])
    message.submit(lambda: "", None, message)
    submit.click(
        chat, inputs=[
            message, chatbot1, state], outputs=[
            chatbot1, state])
    submit.click(lambda: "", None, message)
    regen.click(
        regenerate, inputs=[
            chatbot1, state], outputs=[
            chatbot1, state])
    regen.click(lambda: "", None, message)
    delete.click(
        reset_history, inputs=[
            chatbot1, state], outputs=[
            chatbot1, state])
    delete.click(lambda: "", None, message)

    demo.launch(debug=True)
