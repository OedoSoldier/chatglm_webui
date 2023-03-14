import argparse
import os
import json
import datetime
import gradio as gr
from transformers import AutoTokenizer, AutoModel


def get_args():
    parser = argparse.ArgumentParser(description='ChatGLM Arguments')

    parser.add_argument('--path', default='chatglm-6b', help='The path of ChatGLM model')

    quantize_group = parser.add_mutually_exclusive_group()
    quantize_group.add_argument('--low_vram', action='store_true', help='Use 4-bit quantization')
    quantize_group.add_argument('--med_vram', action='store_true', help='Use 8-bit quantization')

    parser.add_argument('--cpu', action='store_true', help='Use CPU')

    parser.add_argument('--low_ram', action='store_true', help='Use CPU (low RAM)')

    return parser.parse_args()


args = get_args()

if not os.path.isdir(args.path):
    raise FileNotFoundError('Model not found')

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
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if '```' in line:
            item = line.split('`')[-1]
            if item:
                lines[i] = f'<pre><code class="{item}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                line = line.replace('<', '&lt;').replace('>', '&gt;')
                lines[i] = f'<br/>{line}'
    return ''.join(lines)


def chat_wrapper(query, styled_history, history, max_length, top_p, temperature):
    message, history = model.chat(tokenizer, query, history=history,
                                  max_length=max_length, top_p=top_p, temperature=temperature)
    styled_history.append((parse_text(query), parse_text(message)))
    return styled_history, history, ''


def regenerate_wrapper(styled_history, history, max_length, top_p, temperature):
    if len(history) == 0:
        return [], [], ''
    styled_history, history, query = edit_wrapper(styled_history, history)
    return chat_wrapper(query, styled_history, history, max_length, top_p, temperature)


def edit_wrapper(styled_history, history):
    if len(history) == 0:
        return [], [], ''
    query = history[-1][0]
    history = history[:-1]
    styled_history = styled_history[:-1]
    return styled_history, history, query


def reset_history():
    return [], [], ''


def save_history(history):
    os.makedirs('log', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in history]

    with open(f'log/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(dict_list, f, ensure_ascii=False, indent=2)


def save_config(max_length, top_p, temperature):
    with open('config.json', 'w') as f:
        json.dump({'max_length': max_length, 'top_p': top_p, 'temperature': temperature}, f, indent=2)


def load_history(file, styled_history, history):
    current_styled_history, current_history = styled_history.copy(), history.copy()
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            dict_list = json.load(f)
        history = [(item['input'], item['output']) for item in dict_list]
        styled_history = [(parse_text(item['input']), parse_text(item['output'])) for item in dict_list]
    except BaseException:
        return current_styled_history, current_history
    return styled_history, history, ''


def main():
    with gr.Blocks() as app:
        if not os.path.isfile('config.json'):
            save_config(2048, 0.7, 0.95)

        with open('config.json', 'r', encoding='utf-8') as f:
            configs = json.loads(f.read())

        gr.Markdown('''<h1><center>ChatGLM WebUI</center></h1>''')
        gr.Markdown('''`Max Length` 是生成文本时的长度限制，`Top P` 控制输出文本中概率最高前 p 个单词的总概率，`Temperature` 控制生成文本的多样性和随机性。<br/>`Top P` 变小会生成更多样和不相关的文本；变大会生成更保守和相关的文本。<br/>`Temperature` 变小会生成更保守和相关的文本；变大会生成更奇特和不相关的文本。''')

        with gr.Row():
            max_length = gr.Slider(minimum=4.0, maximum=4096.0, step=4.0, label='Max Length', value=configs['max_length'])
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=configs['top_p'])
            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=configs['temperature'])
        save_conf = gr.Button('保存设置')

        gr.Markdown("""<h2>聊天记录</h2>""")

        chatbot = gr.Chatbot(elem_id='chatbot', show_label=False)
        state = gr.State([])

        message = gr.Textbox(placeholder='输入内容', label='你：')

        with gr.Row():
            submit = gr.Button('提交')
            edit = gr.Button('修改问题')
            regen = gr.Button('重新生成')

        delete = gr.Button('清空聊天')

        with gr.Row():
            save = gr.Button('保存对话 （在 log 文件夹下）')
            load = gr.UploadButton('读取对话', file_types=['file'], file_count='single')

        submit_list = [message, chatbot, state, max_length, top_p, temperature]
        state_list = [chatbot, state, message]

        save_conf.click(save_config, inputs=[max_length, top_p, temperature])
        load.upload(load_history, inputs=[load, chatbot, state], outputs=state_list)
        save.click(save_history, inputs=[state])
        message.submit(chat_wrapper, inputs=submit_list, outputs=state_list)
        submit.click(chat_wrapper, inputs=submit_list, outputs=state_list)
        edit.click(edit_wrapper, inputs=submit_list[1:3], outputs=state_list)
        regen.click(regenerate_wrapper, inputs=submit_list[1:], outputs=state_list)
        delete.click(reset_history, outputs=state_list)

        app.launch(debug=True)


if __name__ == '__main__':
    main()
