import argparse
import datetime
import json
import os
import re
import time

import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib

import numpy as np
from PIL import Image, ImageDraw

from copy import deepcopy

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

prompt_template = '''
<image>
Hello! Could you please help me to place {N} foreground elements over the background image of resolution {resolution} to craft an aesthetically pleasing, harmonious, balanced, and visually appealing {domain_name}?
Finding semantic-meaningful objects or visual foci on the background image at first might help in designing, and you should avoid any unnecessary blocking of them. 
For each layout, there are 3 additional user requirements and you are expected to generate a layout corresponding to them. Here is the user requirements: {cons_data}
Please return the result by completing the following JSON file. Each element's location and size should be represented by a bounding box described as [left, top, right, bottom], and each number is a continuous digit from 0 to 1.
Here is the initial JSON file: {json_data}
'''

ELEM_CLASSES = {
    "QB-Poster": ["title", "decoration", "subtitle", "itemtitle", "itemlogo", "item", "text", "textbackground", "object", "frame"],
    "CGL": ["text", "underlay", "embellishment"],
    "Ad Banners": ["header", "preheader", "postheader", "body text", "disclaimer / footnote", "button", "callout", "logo"]
}

CLS2COLOR = {
    "QB-Poster": {
        "title": "red", "subtitle": "green", "itemlogo": "orange", "item": "blue", "itemtitle": "yellow",
        "object": "purple", "textbackground": "pink", "decoration": "brown", "frame": "gray", "text": "cyan",
        "false": "black"
    },
    "CGL": {
        "text": "red", "underlay": "green", "embellishment": "blue", "false": "black"
    },
    "Ad Banners": {
        "header": "red", "preheader": "green", "postheader": "blue", "body text": "orange", "disclaimer / footnote": "purple",
        "button": "pink", "callout": "brown", "logo": "gray", "false": "black"
    }
}

def get_json_response(response):
    for i in range(len(response)):
        if i < len(response) - 1 and response[i:i+2] == "[{":
            lo = i
        elif i > 1 and response[i-1:i+1] == "}]":
            hi = i
    try:
        string = response[lo:hi+1].replace("'", '"')
        json_response = json.loads(string)
    except:
        json_response = None
    return json_response

def draw_box(img, elems, elems2, cls2color):
    W, H = img.size
    drawn_outline = img.copy()
    drawn_fill = img.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)
    for cls, box in elems:
        color = cls2color[cls]
        left, top, right, bottom = box
        _box = int(left * W), int(top * H), int(right * W), int(bottom * H)
        draw_ol.rectangle(_box, fill=None, outline=color, width=max(10 * (W + H) // (1242 + 1660), 1))
        draw_f.rectangle(_box, fill=color)
    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.1))
    drawn = Image.alpha_composite(drawn_outline, drawn_fill)
    return drawn

def draw_boxmap(json_response, background_image, cls2color):
    pic = background_image.convert("RGB")
    cls_box = [(elem['label'], elem['box']) for elem in json_response]
    print(cls_box)
    drawn = draw_box(background_image, cls_box, cls_box, cls2color)
    return drawn.convert("RGB")

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update

def init_json(elem_list, dataset):
    json_data = []
    for i, label in enumerate(ELEM_CLASSES[dataset]):
        num = int(elem_list[i])
        json_data += [{"label": label, "box": []} for _ in range(num) if num > 0]
    return json_data

def init_conv(request: gr.Request):
    logger.info(f"init_conversation. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot()) + (enable_btn,) * 3

def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}.")
    if image is not None:
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot()) + (disable_btn,) * 3

def qb_add_text(state, title_num, decoration_num, subtitle_num, itemtitle_num, itemlogo_num, item_num, text_num, 
            textbackground_num, object_num, frame_num, image, user_cons, image_process_mode, request: gr.Request):
    elem_list = [title_num, decoration_num, subtitle_num, itemtitle_num, itemlogo_num, \
        item_num, text_num, textbackground_num, object_num, frame_num]
    json_data = init_json(elem_list, dataset='QB-Poster')
    if image is not None:
        resolution = list(image.size)
    else:
        try:
            resolution = list(state.get_images(return_pil=True)[-1].size)
        except:
            resolution = [1242, 1660]
    text = prompt_template.replace('<image>\n', '<image>').replace('\n', '\\n').format(
        N=len(json_data),
        resolution=resolution,
        domain_name="poster with xiaohonshu style",
        cons_data=user_cons,
        json_data=json.dumps(json_data)
    )
    return add_text(state, text, image, image_process_mode, request)

def cgl_add_text(state, text_num, underlay_num, embellishment_num, image, user_cons, image_process_mode, request: gr.Request):
    elem_list = [text_num, underlay_num, embellishment_num]
    json_data = init_json(elem_list, dataset='CGL')
    if image is not None:
        resolution = list(image.size)
    else:
        try:
            resolution = list(state.get_images(return_pil=True)[-1].size)
        except:
            resolution = [513, 750]
    text = prompt_template.replace('<image>\n', '<image>').replace('\n', '\\n').format(
        N=len(json_data),
        resolution=resolution,
        domain_name="commercial poster",
        cons_data=user_cons,
        json_data=json.dumps(json_data)
    )
    return add_text(state, text, image, image_process_mode, request)

def banners_add_text(state, header_num, preheader_num, postheader_num, body_text, disclaimer_num, button_num,
        callout_num, logo_num, image, user_cons, image_process_mode, request: gr.Request):
    elem_list = [header_num, preheader_num, postheader_num, body_text, disclaimer_num, button_num, callout_num, logo_num]
    json_data = init_json(elem_list, dataset='Ad Banners')
    if image is not None:
        resolution = list(image.size)
    else:
        try:
            resolution = list(state.get_images(return_pil=True)[-1].size)
        except:
            resolution = [1080, 1080]
    text = prompt_template.replace('<image>\n', '<image>').replace('\n', '\\n').format(
        N=len(json_data),
        resolution=resolution,
        domain_name="commercial banner",
        cons_data=user_cons,
        json_data=json.dumps(json_data)
    )
    return add_text(state, text, image, image_process_mode, request)

def http_bot(state, model_selector, temperature, top_p, max_new_tokens, repeat_times, request: gr.Request, progress = gr.Progress()):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    
    if state.skip_next:
        yield (state, state.to_gradio_chatbot(), None) + (no_change_btn,) * 3
        return

    if len(state.messages) == state.offset + 2:
        template_name = "llava_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), None, disable_btn, disable_btn, disable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()
    if "xiaohonshu" in prompt:
        current_dataset = "QB-Poster"
    elif "commercial poster" in prompt:
        current_dataset = "CGL"
    elif "commercial banner" in prompt:
        current_dataset = "Ad Banners"

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    boxmaps, all_responses = [], []
    initial_json = re.findall(r'\[\{.*?\}\]', prompt)[0]
    elems_num = initial_json.count("label")
    total_length = elems_num * len('0.0000, 0.0000, 0.0000, 0.0000') + len(initial_json)\
        + len('Sure! Here is the design results: ')
    for t in range(repeat_times):
        try:
            response = requests.post(worker_addr + "/worker_generate_stream",
                headers=headers, json=pload, stream=True, timeout=20)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(prompt):].strip()
                    else:
                        output = data["text"] + f" (error_code: {data['error_code']})"
                        state.messages[-1][-1] = output
                        yield (state, state.to_gradio_chatbot(), boxmaps) + (enable_btn,) * 3
                        return
                    time.sleep(0.01)
                p = (len(output) + len(''.join(all_responses))) / (total_length * repeat_times)
                progress(p, desc=f'Generating the {t + 1}th output...')
        except requests.exceptions.RequestException as e:
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot(), boxmaps) + (enable_btn,) * 3
            return
        
        all_responses.append(output)
        json_response = get_json_response(output)
        if json_response is not None:
            boxmaps.append(draw_boxmap(json_response, all_images[-1], CLS2COLOR[current_dataset]))
    
    state.messages[-1][-1] = "".join([f"Design Result {i}:\n" + all_responses[i] + "\n\n" for i in range(len(all_responses))])
    yield (state, state.to_gradio_chatbot(), boxmaps) + (enable_btn,) * 3
    
    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# PosterLLaVA: Constructing a Unified Multi-modal Layout Generator with LLM
""")

tos_markdown = ("""
""")


learn_more_markdown = ("""
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode, cur_dir):
    imagebox_boxmap = gr.Gallery(label='Result(结果)', show_label=True, preview=False, columns=2, allow_preview=True, height=550)
    with gr.Blocks(title="PosterLLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                with gr.Accordion("Parameters", open=True) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=1024, step=64, interactive=True, label="Max output tokens",)
                    repeat_times = gr.Slider(minimum=1, maximum=64, value=4, step=1, interactive=True, label="Repeat times",)

                # cur_dir = os.path.dirname(os.path.abspath(__file__))
                # gr.Examples(examples=[
                #     [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                #     [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                # ], inputs=[imagebox, textbox])
            
            with gr.Column(scale=8):
                imagebox_boxmap.render()
                with gr.Tabs() as tabs:
                    with gr.Tab("QB-Poster"):
                        with gr.Row(variant='compact'):
                            object_num = gr.Checkbox(label='object', value=True)
                            frame_num = gr.Checkbox(label='frame', value=True)
                            title_num = gr.Checkbox(label='title', value=True)
                            decoration_num = gr.Slider(minimum=0, maximum=10, value=2, step=1, label='decoration')
                            subtitle_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='subtitle')
                            itemtitle_num = gr.Slider(minimum=0, maximum=10, value=0, step=1, label='itemtitle')
                            itemlogo_num = gr.Slider(minimum=0, maximum=10, value=3, step=1, label='itemlogo')
                            item_num = gr.Slider(minimum=0, maximum=10, value=3, step=1, label='item')
                            text_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='text')
                            textbackground_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='textbackground')
                            qb_elem_list = [title_num, decoration_num, subtitle_num, itemtitle_num, itemlogo_num,
                                        item_num, text_num, textbackground_num, object_num, frame_num]
                        with gr.Row():
                            qb_textbox = gr.Textbox(label='User Constraint', show_label=True,
                                placeholder="Enter text and press ENTER", container=True)
                        with gr.Column(scale=1, min_width=50):
                            qb_submit_btn = gr.Button(value="Generate", variant="primary", interactive=False)
                    with gr.Tab("CGL / PosterLayout"):
                        with gr.Row(variant='compact'):
                            cgl_text_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='text')
                            underlay_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='underlay')
                            embellishment_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='embellishment')
                            cgl_elem_list = [cgl_text_num, underlay_num, embellishment_num]
                        with gr.Row():
                            cgl_textbox = gr.Textbox(label='User Constraint', show_label=True,
                                placeholder="Enter text and press ENTER", container=True)
                        with gr.Column(scale=1, min_width=50):
                            cgl_submit_btn = gr.Button(value="Generate", variant="primary", interactive=False)
                    with gr.Tab("Ad Banners"):
                        with gr.Row(variant='compact'):
                            header_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='header')
                            preheader_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='pre-header')
                            postheader_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='post-header')
                            body_text = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='body text')
                            disclaimer_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='disclaimer / footnote')
                            button_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='button')
                            callout_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='callout')
                            logo_num = gr.Slider(minimum=0, maximum=10, value=1, step=1, label='logo')
                            banners_elem_list = [header_num, preheader_num, postheader_num, body_text, disclaimer_num,
                                            button_num, callout_num, logo_num]
                        with gr.Row():
                            banners_textbox = gr.Textbox(label='User Constraint', show_label=True,
                                placeholder="Enter your customized design requirements separated by ';' to control the sizes and positions of elements", container=True)
                        with gr.Column(scale=1, min_width=50):
                            banners_submit_btn = gr.Button(value="Generate", variant="primary", interactive=False)
                with gr.Accordion("Intermediate results", open=False):
                    gr.Markdown("The layout generation process with LLM")
                    chatbot = gr.Chatbot(elem_id="chatbot", label="LLM Conversations", height=550)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [qb_submit_btn, cgl_submit_btn, banners_submit_btn]
        imagebox.change(
            init_conv,
            None,
            [state, chatbot] + btn_list,
            queue=False
        )
        qb_submit_btn.click(
            qb_add_text,
            [state, *qb_elem_list, imagebox, qb_textbox, image_process_mode],
            [state, chatbot] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, repeat_times],
            [state, chatbot, imagebox_boxmap] + btn_list
        )
        cgl_submit_btn.click(
            cgl_add_text,
            [state, *cgl_elem_list, imagebox, cgl_textbox, image_process_mode],
            [state, chatbot] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, repeat_times],
            [state, chatbot, imagebox_boxmap] + btn_list
        )
        banners_submit_btn.click(
            banners_add_text,
            [state, *banners_elem_list, imagebox, banners_textbox, image_process_mode],
            [state, chatbot] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, repeat_times],
            [state, chatbot, imagebox_boxmap] + btn_list
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )