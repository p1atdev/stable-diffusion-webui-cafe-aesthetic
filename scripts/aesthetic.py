import os
from pathlib import Path
from glob import glob
import shutil

import gradio as gr
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks, ui
from modules import generation_parameters_copypaste as parameters_copypaste
import launch

script_dir = Path(scripts.basedir())
aesthetics = {}  # name: pipeline


def library_check():
    if not launch.is_installed("transformers"):
        launch.run_pip("install transformers", "requirements for Cafe Aesthetic")


def model_check(name):
    if name not in aesthetics:
        library_check()
        from transformers import pipeline

        if name == "aesthetic":
            aesthetics["aesthetic"] = pipeline(
                "image-classification", model="cafeai/cafe_aesthetic"
            )
        elif name == "style":
            aesthetics["style"] = pipeline(
                "image-classification", model="cafeai/cafe_style"
            )
        elif name == "waifu":
            aesthetics["waifu"] = pipeline(
                "image-classification", model="cafeai/cafe_waifu"
            )


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


def classify_outputs_folders(type):
    if type == "Aesthetic":
        return ["aesthetic", "not_aesthetic"]
    elif type == "Style":
        return ["anime", "other", "real_life", "3d", "manga_like"]
    elif type == "Waifu":
        return ["waifu", "not_waifu"]


def output_dir_previews_update(value, classify_type):
    if value == "":
        return
    folders = classify_outputs_folders(classify_type)
    output_dir_previews = "\n".join([f"- {Path(value)/f}" for f in folders])

    return f"Output dirs will be created like: \n{output_dir_previews}"


def progress_str(progress):
    return int(progress * 1000) / 10


def copy_or_move_files(img_path: Path, to: Path, copy, together):
    img_name = img_path.stem  # hoge.jpg
    if together:
        for p in img_path.parent.glob(f"{img_name}.*"):
            if copy:
                shutil.copy2(p, to / p.name)
            else:
                if os.path.exists(p):
                    p.rename(to / p.name)
                else:
                    print(f"Not found: {p}")
    else:
        if copy:
            shutil.copy2(img_path, to / img_path.name)
        else:
            img_path.rename(to / img_path.name)


def batch_classify(
    input_dir, output_dir, classify_type, output_style, together, basis, threshold
):
    print("Batch classifying started")
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        image_paths = [
            p
            for p in input_dir.iterdir()
            if (p.is_file and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"])
        ]

        print(f"Found {len(image_paths)} images")

        classifyer = None
        if classify_type == "Aesthetic":
            classifyer = judge_aesthetic
        elif classify_type == "Style":
            classifyer = judge_style
        elif classify_type == "Waifu":
            classifyer = judge_waifu

        folders = classify_outputs_folders(classify_type)

        for f in folders:
            os.makedirs(output_dir / f, exist_ok=True)

        for i, f in enumerate(image_paths):
            if f.is_dir():
                continue

            img = Image.open(f)
            result = classifyer(img)

            max_score = 0
            max_label = None

            for label, score in result.items():
                if basis == "Relative":
                    if score > max_score:
                        max_score = score
                        max_label = label
                elif basis == "Absolute":
                    if score > threshold and score > max_score:
                        max_score = score
                        max_label = label

            if max_label is None:
                continue

            copy_or_move_files(
                f, output_dir / max_label, output_style == "Copy", together
            )

            print(
                f"Classified {f.name} as {max_label} with {progress_str(max_score)}% confidence"
            )

        print("All done!")
        return "Done!"
    except Exception as e:
        return f"Error: {e}"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem(label="Single"):
                    with gr.Row().style(equal_height=False):
                        with gr.Column():
                            # with gr.Tabs():
                            image = gr.Image(
                                source="upload",
                                label="Image",
                                interactive=True,
                                type="pil",
                            )

                            single_start_btn = gr.Button(
                                value="Judge", variant="primary"
                            )

                        with gr.Column():
                            single_aesthetic_result = gr.Label(label="Aesthetic")
                            single_style_result = gr.Label(label="Style")
                            single_waifu_result = gr.Label(label="Waifu")

                with gr.TabItem(label="Batch"):

                    with gr.Row().style(equal_height=False):
                        with gr.Column():
                            input_dir_input = gr.Textbox(
                                label="Image Directory",
                                placeholder="path/to/classify",
                                type="text",
                            )
                            output_dir_input = gr.Textbox(
                                label="Output Directory",
                                placeholder="path/to/output",
                                type="text",
                            )

                            output_dir_previews_md = gr.Markdown("")

                            classify_type_radio = gr.Radio(
                                label="Classify type",
                                choices=["Aesthetic", "Style", "Waifu"],
                                value="Aesthetic",
                                interactive=True,
                            )

                            output_style_radio = gr.Radio(
                                label="Output style",
                                choices=["Copy", "Move"],
                                value="Copy",
                                interactive=True,
                            )
                            copy_or_move_captions_together = gr.Checkbox(
                                label="Copy or move captions together",
                                value=True,
                                interactive=True,
                            )

                            gr.Markdown("")

                            basis_radio = gr.Radio(
                                label="Basis",
                                choices=["Relative", "Absolute"],
                                value="Relative",
                                interactive=True,
                            )

                            absolute_slider = gr.Slider(
                                label="Threshold (Use only when basis is absolute)",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.5,
                            )

                            batch_start_btn = gr.Button(
                                value="Start", variant="primary"
                            )

                        with gr.Column():
                            status_block = gr.Label(label="Status", value="Idle")

                            ## Sadly I don't have a capable to implement progress bar...

                            # with gr.Column(variant="panel"):
                            #     progress_md = gr.Markdown("#### Progress: 0 %")

                            #     # progress = gr.Slider(label="Progress", minimum=0, maximum=100, step=0.1, interactive=False, elem_id="progress_bar")
                            #     progress_html = gr.HTML(f'<div class="h-1 mb-1 rounded bg-gradient-to-r group-hover:from-orange-500 from-orange-400 to-orange-200 dark:from-orange-400 dark:to-orange-600" style="width: {1}%;"></div>')

                            # progress_aesthetic_result = gr.Label(label="Aesthetic")
                            # progress_style_result = gr.Label(label="Style")
                            # progress_waifu_result = gr.Label(label="Waifu")

                            # progress_img = gr.Image(label="Current", interactive=False, type="pil")

        image.change(
            fn=judge,
            inputs=image,
            outputs=[single_aesthetic_result, single_style_result, single_waifu_result],
        )
        single_start_btn.click(
            fn=judge,
            inputs=image,
            outputs=[single_aesthetic_result, single_style_result, single_waifu_result],
        )

        output_dir_input.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio],
            outputs=[output_dir_previews_md],
        )
        classify_type_radio.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio],
            outputs=[output_dir_previews_md],
        )

        batch_start_btn.click(
            fn=batch_classify,
            inputs=[
                input_dir_input,
                output_dir_input,
                classify_type_radio,
                output_style_radio,
                copy_or_move_captions_together,
                basis_radio,
                absolute_slider,
            ],
            outputs=[status_block],
        )

    return [(ui, "Aesthetic", "aesthetic")]


script_callbacks.on_ui_tabs(on_ui_tabs)
