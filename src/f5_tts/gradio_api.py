import json
import pickle
import shutil
import uuid
from pathlib import Path
from importlib.resources import files
import torch
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
import tempfile
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    USING_SPACES = False
except ImportError:
    USING_SPACES = False

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Thư mục lưu checkpoint
DATAS_DIR = Path("datas")
DATAS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = DATAS_DIR / "audios"
AUDIO_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = DATAS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
SPECTROGRAM_DIR = DATAS_DIR / "spectrograms"
SPECTROGRAM_DIR.mkdir(exist_ok=True)

# Cấu hình mặc định gốc
DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# Load models
vocoder = load_vocoder()

def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)

def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

# Hàm tạo UUID từ tên người hoặc random
def generate_uuid(name=None):
    if name and name.strip():
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, name.strip()))
    return str(uuid.uuid4())

def load_checkpoint(checkpoint_path):
    try:
        if not checkpoint_path.endswith(".safetensors"):
            raise ValueError("Only .safetensors checkpoints are supported.")
        ckpt_path = str(cached_path(checkpoint_path))
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
        return load_model(DiT, model_cfg, ckpt_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {str(e)}")
        return None

# Hàm tiền xử lý và lưu checkpoint với kế thừa từ checkpoint gốc
@gpu_decorator
def preprocess_and_save_checkpoint(ref_audio_orig, ref_text, person_name=None, base_checkpoint=None, checkpoint_dir=CHECKPOINT_DIR, show_info=gr.Info):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return None
    
    ref_text = ref_text if ref_text and isinstance(ref_text, str) and ref_text.strip() else ""
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    if not ref_text.strip():
        gr.Warning("Failed to generate valid ref_text. Please enter manually or check audio.")
        return None

    checkpoint_id = str(uuid.uuid4())
    checkpoint_name = f"checkpoint_{checkpoint_id}"
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.ckpts"
    wav_dest = AUDIO_DIR / f"{checkpoint_name}.wav"
    shutil.copy(ref_audio_orig, wav_dest)

    # Đường dẫn mô hình mặc định
    ckpt_path = r"E:\F5-TTS-main\F5-TTS\ckpts\vov_checkpoint\model_350000.safetensors"
    vocab_path = r"E:\F5-TTS-main\F5-TTS\ckpts\vov_checkpoint\vocab.txt"

    # Lưu checkpoint dưới dạng văn bản thuần túy
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write(f"id: {checkpoint_id}\n")
        f.write(f"person_name: {person_name if person_name else 'Unnamed'}\n")
        f.write(f"wav_file: {wav_dest}\n")
        f.write(f"ref_text: {ref_text}\n")
        f.write(f"model_choice: Custom\n")
        f.write(f"base_checkpoint_path: {ckpt_path}\n")
        f.write(f"vocab_path: {vocab_path}\n")
    
    print(f"Checkpoint created at: {checkpoint_path}")
    print(f"ref_audio saved at: {wav_dest}")
    print(f"ref_text: {ref_text}")
    return checkpoint_path

# Hàm sinh ref_text từ âm thanh
@gpu_decorator
def generate_ref_text(ref_audio_orig, show_info=gr.Info):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return ""
    _, ref_text = preprocess_ref_audio_text(ref_audio_orig, "", show_info=show_info)
    if not ref_text.strip():
        gr.Warning("Generated ref_text is empty. Please check audio or enter manually.")
    return ref_text

# Hàm liệt kê checkpoint theo person_name
def list_checkpoints(checkpoint_dir=CHECKPOINT_DIR):
    checkpoints = {}
    for f in checkpoint_dir.glob("*.ckpts"):
        try:
            with open(f, "r", encoding="utf-8") as checkpoint_file:
                lines = checkpoint_file.readlines()
                data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in lines}
                name = data["person_name"]
                base_name = name
                counter = 1
                while name in checkpoints:
                    name = f"{base_name}_{counter}"
                    counter += 1
                checkpoints[name] = str(f)
        except UnicodeDecodeError:
            print(f"Skipping {f}: Old binary format detected.")
            continue
        except Exception as e:
            gr.Warning(f"Failed to load checkpoint {f}: {str(e)}")
    return checkpoints

# Hàm hiển thị thông tin checkpoint
def show_checkpoint_info(checkpoint_name):
    if not checkpoint_name:
        return "No checkpoint selected"
    checkpoints = list_checkpoints()
    checkpoint_path = checkpoints.get(checkpoint_name)
    if not checkpoint_path:
        return "Checkpoint not found"
    
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in f.readlines()}
        info = f"""
Person: {data['person_name']}
Model: {data['model_choice']}
Base: {data.get('base_checkpoint_path', 'None')}
Created: {Path(checkpoint_path).stat().st_ctime}
"""
        return info
    except Exception as e:
        return f"Error loading info: {str(e)}"

# Hàm infer từ checkpoint
@gpu_decorator
def infer_from_checkpoint(checkpoint_person_name, gen_text, remove_silence, cross_fade_duration=0.15, nfe_step=32, speed=1, show_info=gr.Info):
    checkpoints = list_checkpoints()
    checkpoint_path = checkpoints.get(checkpoint_person_name)
    if not checkpoint_path:
        gr.Warning("Please select a valid checkpoint.")
        return None

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        checkpoint_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in lines}
    except Exception as e:
        gr.Warning(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
        return None

    ref_audio = checkpoint_data["wav_file"]
    ref_text = checkpoint_data["ref_text"]
    base_checkpoint_path = checkpoint_data["base_checkpoint_path"]
    vocab_path = checkpoint_data["vocab_path"]

    # Load mô hình từ base_checkpoint_path
    ema_model = load_custom(base_checkpoint_path, vocab_path)
    if torch.cuda.is_available():
        ema_model = ema_model.cuda()
    ema_model.eval()

    print(f"\nInference debug:")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"ref_audio: {ref_audio}")
    print(f"ref_text: {ref_text}")
    print(f"gen_text: {gen_text}")
    print(f"Model loaded from: {base_checkpoint_path}")

    try:
        # Gọi infer_process với đúng thứ tự tham số
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=show_info,
            progress=gr.Progress(),
        )
    except Exception as e:
        gr.Warning(f"Inference failed: {str(e)}")
        return None

    # Xử lý final_wave dựa trên kiểu dữ liệu
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            final_wave = remove_silence_for_generated_wav(f.name)  # Trả về numpy array
            Path(f.name).unlink()  # Xóa file tạm sau khi dùng xong
    else:
        if isinstance(final_wave, torch.Tensor):
            final_wave = final_wave.squeeze().cpu().numpy()  # Chuyển từ tensor sang numpy nếu cần
        elif isinstance(final_wave, np.ndarray):
            final_wave = final_wave.squeeze()  # Chỉ squeeze nếu đã là numpy array
        else:
            raise ValueError(f"Unexpected type for final_wave: {type(final_wave)}")

    return (final_sample_rate, final_wave)

# Giao diện Gradio
with gr.Blocks() as app:
    gr.Markdown(
        f"""
# E2/F5 TTS
This is {"a local web UI" if not USING_SPACES else "an online demo"} for F5-TTS with checkpoint support.
"""
    )
    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        print(f"Switching TTS model to: {new_choice}")
        if new_choice == "Custom":
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
            print(f"Updated tts_model_choice: {tts_model_choice}")
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            print(f"Updated tts_model_choice: {tts_model_choice}")
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"], label="Choose TTS Model", value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"], label="Choose TTS Model", value=DEFAULT_TTS_MODEL
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[2]],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    with gr.Blocks() as app_tts:
        gr.Markdown("# Batched TTS with Checkpoint")
        user_name_input = gr.Textbox(label="User Name (Optional, for UUID)", placeholder="Enter your name here")
        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
        gen_text_input = gr.Textbox(label="Text to Generate", lines=10, placeholder="Enter text to synthesize")
        
        with gr.Row():
            base_checkpoint_dropdown = gr.Dropdown(
                label="Base Checkpoint (Optional)",
                choices=list_checkpoints().keys(),
                value=None,
                interactive=True
            )
            checkpoint_dropdown = gr.Dropdown(
                label="Select Checkpoint",
                choices=list_checkpoints().keys(),
                value=None,
                interactive=True
            )
        
        with gr.Row():
            generate_ref_btn = gr.Button("Generate Ref Text", variant="secondary")
            save_checkpoint_btn = gr.Button("Save Checkpoint", variant="primary")
            load_checkpoint_btn = gr.Button("Load Checkpoints", variant="secondary")
            synthesize_btn = gr.Button("Synthesize", variant="primary")
        
        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(
                label="Reference Text",
                info="Click 'Generate Ref Text' to auto-transcribe or enter manually.",
                lines=2,
            )
            remove_silence = gr.Checkbox(
                label="Remove Silences",
                value=False,
            )
            speed_slider = gr.Slider(
                label="Speed",
                minimum=0.3,
                maximum=2.0,
                value=1.0,
                step=0.1,
            )
            nfe_slider = gr.Slider(
                label="NFE Steps",
                minimum=4,
                maximum=64,
                value=32,
                step=2,
            )
            cross_fade_duration_slider = gr.Slider(
                label="Cross-Fade Duration (s)",
                minimum=0.0,
                maximum=1.0,
                value=0.15,
                step=0.01,
            )

        audio_output = gr.Audio(label="Synthesized Audio")
        checkpoint_output = gr.Textbox(label="Generated Checkpoint Path")
        metadata_output = gr.Textbox(label="Metadata Path")
        checkpoint_info_output = gr.Textbox(label="Checkpoint Info")

        # Sinh ref_text và hiển thị
        generate_ref_btn.click(
            generate_ref_text,
            inputs=[ref_audio_input],
            outputs=[ref_text_input],
        )

        # Lưu checkpoint với base checkpoint
        def save_checkpoint_wrapper(ref_audio, user_name, ref_text, base_checkpoint):
            checkpoint_path = preprocess_and_save_checkpoint(
                ref_audio, ref_text, user_name, 
                base_checkpoint=list_checkpoints().get(base_checkpoint) if base_checkpoint else None
            )
            return str(checkpoint_path) if checkpoint_path else "", ""

        save_checkpoint_btn.click(
            save_checkpoint_wrapper,
            inputs=[ref_audio_input, user_name_input, ref_text_input, base_checkpoint_dropdown],
            outputs=[checkpoint_output, metadata_output],
        )

        # Load checkpoint vào dropdown
        def reload_checkpoints():
            choices = list_checkpoints().keys()
            return gr.update(choices=choices), gr.update(choices=choices)

        load_checkpoint_btn.click(
            reload_checkpoints,
            inputs=[],
            outputs=[base_checkpoint_dropdown, checkpoint_dropdown],
        )

        # Hiển thị thông tin checkpoint
        checkpoint_dropdown.change(
            show_checkpoint_info,
            inputs=[checkpoint_dropdown],
            outputs=[checkpoint_info_output]
        )

        # Synthesize từ checkpoint
        synthesize_btn.click(
            infer_from_checkpoint,
            inputs=[
                checkpoint_dropdown,
                gen_text_input,
                remove_silence,
                cross_fade_duration_slider,
                nfe_slider,
                speed_slider,
            ],
            outputs=[audio_output],
        )

    with gr.Blocks() as app_credits:
        gr.Markdown("""
# Credits
* [mrfakename](https://github.com/fakerybakery) for the original demo
* [RootingInLoad](https://github.com/RootingInLoad) for initial exploration
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation
""")

    gr.TabbedInterface(
        [app_tts, app_credits],
        ["Basic-TTS", "Credits"],
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio")
def main(port, host, share):
    global app
    print("Starting app...")
    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
    )

if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()