import os
from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine import Engine

if is_gradio_available():
    import gradio as gr

def create_ui(demo_mode: bool = False) -> gr.Blocks:
    engine = Engine(demo_mode=demo_mode, pure_chat=False)

    with gr.Blocks(title="LLaMA Board", css=CSS) as demo:
        if demo_mode:
            gr.HTML("<h1><center>LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory</center></h1>")
            gr.HTML(
                '<h3><center>Visit <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">'
                "LLaMA Factory</a> for details.</center></h3>"
            )
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        # Adding a welcome message with instructions
        gr.HTML("<h2>Welcome to Tunor AI's Fine-Tuning Platform</h2>")
        gr.HTML("<p>Follow the steps below to fine-tune your model easily:</p>")

        engine.manager.add_elems("top", create_top())
        lang: "gr.Dropdown" = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("Fine-Tune Model"):
            gr.Markdown("### Step 1: Upload Your Dataset")
            gr.HTML("<p>Drag and drop your dataset file here:</p>")
            dataset_upload = gr.File(file_count="single", label="Upload Dataset")
            
            gr.Markdown("### Step 2: Select Pre-trained Model")
            gr.HTML("<p>Choose a pre-trained model from the list below:</p>")
            model_select = gr.Dropdown(choices=["LLaMA-7B", "LLaMA-13B", "LLaMA-30B"], label="Select Model")
            
            gr.Markdown("### Step 3: Fine-Tuning Configuration")
            gr.HTML("<p>Configure the fine-tuning parameters:</p>")
            engine.manager.add_elems("train", create_train_tab(engine))

        with gr.Tab("Evaluate & Predict"):
            gr.Markdown("### Evaluate and Test Your Model")
            gr.HTML("<p>Provide input data and evaluate the performance of your fine-tuned model:</p>")
            engine.manager.add_elems("eval", create_eval_tab(engine))

        with gr.Tab("Chat with Model"):
            gr.Markdown("### Interact with Your Fine-Tuned Model")
            gr.HTML("<p>Use this chat interface to interact with your fine-tuned model:</p>")
            engine.manager.add_elems("infer", create_infer_tab(engine))

        if not demo_mode:
            with gr.Tab("Export Model"):
                gr.Markdown("### Export Your Fine-Tuned Model")
                gr.HTML("<p>Export your model for deployment:</p>")
                engine.manager.add_elems("export", create_export_tab(engine))

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo

def create_web_demo() -> gr.Blocks:
    engine = Engine(pure_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"])
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo

def run_web_ui() -> None:
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)

def run_web_demo() -> None:
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
