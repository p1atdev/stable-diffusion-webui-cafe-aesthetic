import os
from pathlib import Path
from glob import glob

import gradio as gr
from PIL import Image 

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks, ui
from modules import generation_parameters_copypaste as parameters_copypaste
import launch

script_dir = Path(scripts.basedir())
aesthetics = {} # name: pipeline

def library_check():
    if not launch.is_installed("transformers"):
        launch.run_pip("install transformers", "requirements for Cafe Aesthetic")

def model_check(name):
    if name not in aesthetics:
        library_check()
        from transformers import pipeline
        if name == "aesthetic":
            aesthetics["aesthetic"] = pipeline("image-classification", model="cafeai/cafe_aesthetic")
        elif name == "style":
            aesthetics["style"] = pipeline("image-classification", model="cafeai/cafe_style")
        elif name == "waifu":
            aesthetics["waifu"] = pipeline("image-classification", model="cafeai/cafe_waifu")

def judge_aesthetic(image):
    model_check("aesthetic")
    data = aesthetics["aesthetic"](image, top_k=2)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    return result

def judge_style(image):
    model_check("style")
    data = aesthetics["style"](image, top_k=5)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    return result

def judge_waifu(image):
    model_check("waifu")
    data = aesthetics["waifu"](image, top_k=5)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    return result

def judge(image):
    if image is None:
        return None, None, None
    aesthetic = judge_aesthetic(image)
    style = judge_style(image)
    waifu = judge_waifu(image)
    return aesthetic, style, waifu



def on_ui_tabs():
    batch_progress = 0 # max 1
    def progress_str():
        return int(batch_progress * 1000) / 10

    with gr.Blocks(analytics_enabled=False) as ui:
        
        with gr.Tabs():
            with gr.TabItem(label='Single'):
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        # with gr.Tabs():
                        image = gr.Image(source="upload", label="Image", interactive=True, type="pil")

                        single_start_btn = gr.Button(value="Judge", variant="primary")
                    
                    with gr.Column():
                        single_aesthetic_result = gr.Label(label="Aesthetic")
                        single_style_result = gr.Label(label="Style")
                        single_waifu_result = gr.Label(label="Waifu")
            
            with gr.TabItem(label='Batch'):
                gr.Markdown("Classify images whether are they aesthetic or not, and what style they are.")

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        input_dir = gr.Textbox(label="Image Directory", placeholder="path/to/classify", type="text")
                        output_dir = gr.Textbox(label="Output Directory", placeholder="path/of/output", type="text")

                        batch_start_btn = gr.Button(value="Start", variant="primary")

                    with gr.Column():
                        with gr.Column(variant="panel"):
                            gr.Markdown(f"#### Progress: {progress_str()}%")

                            # progress = gr.Slider(label="Progress", minimum=0, maximum=100, step=0.1, interactive=False, elem_id="progress_bar")
                            gr.HTML(f'<div class="h-1 mb-1 rounded bg-gradient-to-r group-hover:from-orange-500 from-orange-400 to-orange-200 dark:from-orange-400 dark:to-orange-600" style="width: {max(progress_str(), 1)}%;"></div>')

                        progress_aesthetic_result = gr.Label(label="Aesthetic")
                        progress_style_result = gr.Label(label="Style")
                        progress_waifu_result = gr.Label(label="Waifu")

                        progress_img = gr.Image(label="Current", interactive=False, type="pil")

        image.change(fn=judge, inputs=image, outputs=[single_aesthetic_result, single_style_result, single_waifu_result])
        single_start_btn.click(fn=judge, inputs=image, outputs=[single_aesthetic_result, single_style_result, single_waifu_result])


    return [(ui, "Aesthetic", "aesthetic")]

script_callbacks.on_ui_tabs(on_ui_tabs)