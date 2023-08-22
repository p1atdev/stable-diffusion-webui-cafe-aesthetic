import os
from pathlib import Path
from glob import escape
import shutil
import os
from modules import scripts, script_callbacks
from modules.shared import opts

import gradio as gr
from pathlib import Path
import torch
import torch.nn as nn
import clip

import gradio as gr
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from modules import scripts, script_callbacks
import launch

script_dir = Path(scripts.basedir())
aesthetics = {}  # name: pipeline

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


try:
    force_cpu = opts.ais_force_cpu
except:
    force_cpu = False

if force_cpu:
    print(f"{extension_name}: Forcing prediction model to run on CPU")
device = "cuda" if not force_cpu and torch.cuda.is_available() else "cpu"
# load the model you trained previously or the model available in this repo
pt_state = torch.load(state_name, map_location=torch.device(device=device))

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features


def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return (score.item() / 10) # make it match the other percentile scores


def library_check():
    if not launch.is_installed("transformers"):
        launch.run_pip("install transformers", "requirements for Cafe Aesthetic")


def model_check(name):
    if name not in aesthetics:
        library_check()
        from transformers import pipeline
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if name == "aesthetic":
            aesthetics["aesthetic"] = pipeline(
                "image-classification", model="cafeai/cafe_aesthetic", device=device
            )
        elif name == "style":
            aesthetics["style"] = pipeline(
                "image-classification", model="cafeai/cafe_style", device=device
            )
        elif name == "waifu":
            aesthetics["waifu"] = pipeline(
                "image-classification", model="cafeai/cafe_waifu", device=device
            )
        elif name == "chad":
            aesthetics["chad"] = get_score


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

def judge_chad(image):
    model_check("chad")
    result = {}
    score = aesthetics["chad"](image)
    result["Chad"] = score 
    return result


def judge(image):
    if image is None:
        return None, None, None
    aesthetic = judge_aesthetic(image)
    style = judge_style(image)
    waifu = judge_waifu(image)
    chad = judge_chad(image)
    return aesthetic, style, waifu, chad


def classify_outputs_folders(type):
    if type == "Aesthetic":
        return ["aesthetic", "not_aesthetic"]
    elif type == "Style":
        return ["anime", "other", "real_life", "3d", "manga_like"]
    elif type == "Waifu":
        return ["waifu", "not_waifu"]
    elif type == "ChadPrefix score":
        return ["Chad/Chad_Score_Number-filename.*"]
    elif type == "ChadFolder":
        return ["Chad_Score_Number"]


def output_dir_previews_update(value, classify_type, save_type):
    if value == "":
        return
    if not classify_type == "Chad":
        save_type=""
    folders = classify_outputs_folders(f"{classify_type}{save_type}")
    output_dir_previews = "\n".join([f"- {Path(value)/f}" for f in folders])

    return [f"Output dirs will be created like: \n{output_dir_previews}", gr.update(visible = classify_type == "Chad")]


def progress_str(progress):
    return int(progress * 1000) / 10


def copy_or_move_files(img_path: Path, to: Path, copy, together, img_name = None):
    
    if img_name == None:
        img_name = img_path.stem  # hoge.jpg

    os.makedirs(to, exist_ok=True) # only make dirs when neccessary
    if together:
        for p in img_path.parent.glob(f"{escape(img_path.stem)}.*"):
            if copy:
                shutil.copy2(p, to / f"{img_name}{p.suffix}")
            else:
                if os.path.exists(p):
                    p.rename(to / f"{img_name}{p.suffix}")
                else:
                    print(f"Not found: {p}".encode("utf-8"))
    else:
        if copy:
            shutil.copy2(img_path, to / f"{img_name}{img_path.suffix}")
        else:
            img_path.rename(to / f"{img_name}{img_path.suffix}")


def batch_classify(
    input_dir, output_dir, classify_type, output_style, saving_style, together, basis, threshold
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
        elif classify_type == "Chad":
            classifyer = judge_chad


        for i, f in enumerate(image_paths):
            if f.is_dir():
                continue

            img = Image.open(f)
            f_name = f.stem
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
            
            #Chad has only a score
            if max_label == "Chad" and saving_style == "Folder":
                max_label = repr(round(max_score*100)) 
            elif max_label == "Chad":
                f_name = f"{repr(round(max_score*100))}-{f_name}"
                
            copy_or_move_files(
                f, output_dir / max_label, output_style == "Copy", together, f_name
            )

            print(
                f"Classified {f.name} as {max_label} with {progress_str(max_score)}% confidence".encode("utf-8")
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
                            single_chad_result = gr.Label(label="Chad")
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
                                choices=["Aesthetic", "Style", "Waifu", "Chad"],
                                value="Aesthetic",
                                interactive=True,
                            )
                            save_style_radio = gr.Radio(
                                label="Chad save style",
                                choices=["Prefix score", "Folder"],
                                value="Prefix score",
                                interactive=True,
                                visible = classify_type_radio.value == "Chad"
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
            outputs=[single_aesthetic_result, single_style_result, single_waifu_result, single_chad_result],
        )
        single_start_btn.click(
            fn=judge,
            inputs=image,
            outputs=[single_aesthetic_result, single_style_result, single_waifu_result, single_chad_result],
        )

        output_dir_input.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio, save_style_radio],
            outputs=[output_dir_previews_md, save_style_radio],
        )
        classify_type_radio.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio, save_style_radio],
            outputs=[output_dir_previews_md, save_style_radio],
        )
        save_style_radio.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio, save_style_radio],
            outputs=[output_dir_previews_md, save_style_radio],
        )

        batch_start_btn.click(
            fn=batch_classify,
            inputs=[
                input_dir_input,
                output_dir_input,
                classify_type_radio,
                output_style_radio,
                save_style_radio,
                copy_or_move_captions_together,
                basis_radio,
                absolute_slider,
            ],
            outputs=[status_block],
        )

    return [(ui, "Cafe Aesthetic", "cafe_aesthetic")]


script_callbacks.on_ui_tabs(on_ui_tabs)
