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


def chat_wrapper(query, styled_history, history, max_length, top_p, temperature, memory_limit):
    if query == '':
        return [], [], ''
    if memory_limit == 0:
        history = []
        styled_history = []
    elif memory_limit > 0:
        history = history[-memory_limit:]
        styled_history = styled_history[-memory_limit:]
    message, history = model.chat(tokenizer, query, history=history,
                                  max_length=max_length, top_p=top_p, temperature=temperature)
    styled_history.append((parse_text(query), parse_text(message)))
    return styled_history, history, ''


def regenerate_wrapper(styled_history, history, max_length, top_p, temperature, memory_limit):
    if not history:
        return [], [], ''

    styled_history, history, query = edit_wrapper(styled_history, history)
    return chat_wrapper(query, styled_history, history, max_length, top_p, temperature, memory_limit)


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


def save_config(max_length=2048, top_p=0.7, temperature=0.95, memory_limit=-1.0):
    with open('config.json', 'w') as f:
        json.dump({'max_length': max_length, 'top_p': top_p, 'temperature': temperature, 'memory_limit': memory_limit}, f, indent=2)


def load_history(file, styled_history, history):
    current_styled_history, current_history = styled_history.copy(), history.copy()
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            dict_list = json.load(f)
        history = [(item['input'], item['output']) for item in dict_list]
        styled_history = [(parse_text(item['input']), parse_text(item['output'])) for item in dict_list]
    except BaseException:
        return current_styled_history, current_history, ''
    return styled_history, history, ''


def gr_show_and_load(history, evt: gr.SelectData):
    if evt.index[1] == 0:
        label = f'修改提问{evt.index[0]}：'
    else:
        label = f'修改回答{evt.index[0]}：'
    return {'visible': True, '__type__': 'update'}, {'value': history[evt.index[0]][evt.index[1]], 'label': label, '__type__': 'update'}, evt.index


def update_history(styled_history, history, log, idx):
    if log == '':
        return styled_history, history, {'visible': True, '__type__': 'update'},  {'value': history[idx[0]][idx[1]], '__type__': 'update'}, idx

    def swap_value(lst, idx, value):
        lst[idx[0]] = tuple(value if j == idx[1] else elem for j, elem in enumerate(lst[idx[0]]))
        return lst
    styled_history = swap_value(styled_history, idx, parse_text(log))
    history = swap_value(history, idx, log)
    return styled_history, history, {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []


def gr_hide():
    return {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []


with gr.Blocks() as demo:
    if not os.path.isfile('config.json'):
        save_config()

    with open('config.json', 'r', encoding='utf-8') as f:
        configs = json.loads(f.read())

    gr.Markdown('''<h1><center>ChatGLM WebUI</center></h1>''')
    gr.Markdown('''`Max Length` 是生成文本时的长度限制，`Top P` 控制输出文本中概率最高前 p 个单词的总概率，`Temperature` 控制生成文本的多样性和随机性。<br/>`Top P` 变小会生成更多样和不相关的文本；变大会生成更保守和相关的文本。<br/>`Temperature` 变小会生成更保守和相关的文本；变大会生成更奇特和不相关的文本。<br/>`Memory Limit` 对话记忆轮数，`-1` 为无限长，限制记忆可减小显存占用。''')

    with gr.Row():
        max_length = gr.Slider(minimum=4.0, maximum=4096.0, step=4.0, label='Max Length', value=configs['max_length'])
        top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=configs['top_p'])
        temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=configs['temperature'])
        memory_limit = gr.Slider(minimum=-1.0, maximum=10.0, step=1.0, label='Memory Limit', value=configs['memory_limit'])
    save_conf = gr.Button('保存设置')

    gr.Markdown('''<h2>提示：点击对话可以进行修改</h2>''')

    state = gr.State([])
    chatbot = gr.Chatbot(elem_id='chatbot', show_label=False)
    with gr.Row(visible=False) as edit_log:
        with gr.Column():
            log = gr.Textbox()
            with gr.Row():
                submit_log = gr.Button('保存')
                cancel_log = gr.Button('取消')
    log_idx = gr.State([])

    message = gr.Textbox(placeholder='输入内容', label='问：')

    with gr.Row():
        submit = gr.Button('提交')
        edit = gr.Button('修改上一问题')
        regen = gr.Button('重新生成')

    delete = gr.Button('清空聊天')

    with gr.Row():
        save = gr.Button('保存对话（在 `log` 文件夹下）')
        load = gr.UploadButton('读取对话', file_types=['file'], file_count='single')

    input_list = [message, chatbot, state, max_length, top_p, temperature, memory_limit]
    output_list = [chatbot, state, message]

    save_conf.click(save_config, inputs=input_list[3:])
    load.upload(load_history, inputs=[load, chatbot, state], outputs=output_list)
    save.click(save_history, inputs=[state])
    message.submit(chat_wrapper, inputs=input_list, outputs=output_list)
    submit.click(chat_wrapper, inputs=input_list, outputs=output_list)
    edit.click(edit_wrapper, inputs=input_list[1:3], outputs=output_list)
    regen.click(regenerate_wrapper, inputs=input_list[1:], outputs=output_list)
    delete.click(reset_history, outputs=output_list)
    chatbot.select(gr_show_and_load, inputs=[state], outputs=[edit_log, log, log_idx])
    edit_kwargs = {'inputs': [chatbot, state, log, log_idx], 'outputs': [chatbot, state, edit_log, log, log_idx]} 
    log.submit(update_history, **edit_kwargs)
    submit_log.click(update_history, **edit_kwargs)
    cancel_log.click(gr_hide, outputs=[edit_log, log, log_idx])


if __name__ == '__main__':
    demo.launch(debug=True)
