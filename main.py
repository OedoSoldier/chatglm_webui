import argparse
import os
import json
import datetime
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from bilibili_utils import BilibiliTranscriptReader


def get_args():
    parser = argparse.ArgumentParser(description='ChatGLM Arguments')

    parser.add_argument('--path', default='../chatglm_webui_dev/chatglm-6b', help='The path of ChatGLM model')

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


def extract_bv_strings(input_data):
    """
    从输入数据中提取包含 BV 的字符串，返回一个列表
    """
    try:
        # 正则表达式匹配
        pattern = re.compile(r'(?<=BV)[0-9A-Za-z]+')
        match = re.findall(pattern, input_data)

        # 返回匹配到的字符串
        return match
    except TypeError:
        raise ValueError("输入数据必须是字符串类型")
    except:
        raise ValueError("输入数据格式不正确")


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


def chat_wrapper(query, cookies, max_length, top_p, temperature):
    # query = extract_bv_strings(extract_bv_strings)
    styled_history = []
    history = []
    BTR = BilibiliTranscriptReader()
    data, pic, bvid = BTR.load_data(video_urls=query, cookies=cookies)
    if pic:
        query = """<a href="https://www.bilibili.com/video/BV{bvid}"><img src="{pic}" alt="" width="226" height="144" /></a>"""
    if data != "":
        message, history = model.chat(tokenizer, data, history=history,
                                      max_length=max_length, top_p=top_p, temperature=temperature)
    else:
        message = "不支持该视频"
        # history = [(query, message)]
    styled_history.append((query, parse_text(message)))
    return styled_history, history, ''


def save_config(max_length, top_p, temperature):
    with open('config.json', 'w') as f:
        json.dump({'max_length': max_length, 'top_p': top_p, 'temperature': temperature}, f, indent=2)


def main():
    with gr.Blocks() as app:
        if not os.path.isfile('config.json'):
            save_config(2048, 0.7, 0.95)

        with open('config.json', 'r', encoding='utf-8') as f:
            configs = json.loads(f.read())

        gr.Markdown('''<h1><center>ChatGLM 哔哩哔哩量子速看</center></h1>''')
        # gr.Markdown('''`Max Length` 是生成文本时的长度限制，`Top P` 控制输出文本中概率最高前 p 个单词的总概率，`Temperature` 控制生成文本的多样性和随机性。<br/>`Top P` 变小会生成更多样和不相关的文本；变大会生成更保守和相关的文本。<br/>`Temperature` 变小会生成更保守和相关的文本；变大会生成更奇特和不相关的文本。''')

        with gr.Row(visible=False):
            max_length = gr.Slider(minimum=4.0, maximum=4096.0, step=4.0, label='Max Length', value=configs['max_length'])
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=configs['top_p'])
            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=configs['temperature'])
            save_conf = gr.Button('保存设置')

        gr.Markdown("""<h2>聊天记录</h2>""")

        chatbot = gr.Chatbot(elem_id='chatbot', show_label=False)
        state = gr.State([])

        message = gr.Textbox(label='请输入 BV 号（或 AV 号）：')
        submit = gr.Button('提交')
        with gr.Accordion("设置"):
            cookies = gr.Textbox(label='请输入 B 站 Cookie（在 F12 - Network 的记录中）：')

        submit_list = [message, cookies, max_length, top_p, temperature]
        state_list = [chatbot, state, message]

        save_conf.click(save_config, inputs=[max_length, top_p, temperature])
        message.submit(chat_wrapper, inputs=submit_list, outputs=state_list)
        submit.click(chat_wrapper, inputs=submit_list, outputs=state_list)

        app.launch(debug=True)


if __name__ == '__main__':
    main()
