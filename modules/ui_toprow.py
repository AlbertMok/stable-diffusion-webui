import gradio as gr

from modules import shared, ui_prompt_styles
import modules.images

from modules.ui_components import ToolButton


class Toprow:
    """
    Creates a top row UI with prompts,
    generate button, styles,
    extra little buttons for things,
    and enables some functionality related to their operation
    """

    prompt = None
    prompt_img = None
    negative_prompt = None

    button_interrogate = None
    button_deepbooru = None

    interrupt = None
    interrupting = None
    skip = None
    submit = None

    paste = None
    clear_prompt_button = None
    apply_styles = None
    restore_progress_button = None

    token_counter = None
    token_button = None
    negative_token_counter = None
    negative_token_button = None

    ui_styles = None

    submit_box = None

    def __init__(self, is_img2img, is_compact=False, id_part=None):

        # id_part表示当前是在img2img还是txt2img
        if id_part is None:
            id_part = "img2img" if is_img2img else "txt2img"

        self.id_part = id_part
        self.is_img2img = is_img2img
        self.is_compact = is_compact

        if not is_compact:
            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                self.create_classic_toprow()
        else:
            self.create_submit_box()

    def create_classic_toprow(self):

        # 创建prompt输入区域
        self.create_prompts()

        # 另一列是一些按钮和样式
        with gr.Column(scale=1, elem_id=f"{self.id_part}_actions_column"):
            self.create_submit_box()

            self.create_tools_row()

            self.create_styles_ui()

    def create_inline_toprow_prompts(self):
        """
        第一列：创建prompt输入区域
        第二列：创建一些按钮和样式按钮

        """
        if not self.is_compact:
            return

        # 创建prompt 输入区域
        self.create_prompts()

        with gr.Row(elem_classes=["toprow-compact-stylerow"]):
            with gr.Column(elem_classes=["toprow-compact-tools"]):
                self.create_tools_row()
            with gr.Column():
                self.create_styles_ui()

    def create_inline_toprow_image(self):
        if not self.is_compact:
            return

        self.submit_box.render()

    # 创建prompt输入区域
    def create_prompts(self):

        # prompt container，one column
        with gr.Column(
            elem_id=f"{self.id_part}_prompt_container",
            elem_classes=["prompt-container-compact"] if self.is_compact else [],
            scale=6,
        ):
            # prompt textarea
            with gr.Row(
                elem_id=f"{self.id_part}_prompt_row", elem_classes=["prompt-row"]
            ):
                self.prompt = gr.Textbox(
                    label="Prompt",
                    elem_id=f"{self.id_part}_prompt",
                    show_label=True,
                    lines=3,
                    placeholder="Prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)",
                    elem_classes=["prompt"],
                )

                self.prompt_img = gr.File(
                    label="Prompt Image",
                    elem_id=f"{self.id_part}_prompt_image",
                    file_count="single",
                    type="binary",
                    visible=False,
                )

            # negative prompt textarea
            with gr.Row(
                elem_id=f"{self.id_part}_neg_prompt_row", elem_classes=["prompt-row"]
            ):
                self.negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    elem_id=f"{self.id_part}_neg_prompt",
                    show_label=False,
                    lines=3,
                    placeholder="Negative prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)",
                    elem_classes=["prompt"],
                )

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )

    # 创建提交按钮
    def create_submit_box(self):
        with gr.Row(
            elem_id=f"{self.id_part}_generate_box",
            elem_classes=["generate-box"]
            + (["generate-box-compact"] if self.is_compact else []),
            render=not self.is_compact,
        ) as submit_box:

            self.submit_box = submit_box

            self.interrupt = gr.Button(
                "Interrupt",
                elem_id=f"{self.id_part}_interrupt",
                elem_classes="generate-box-interrupt",
                tooltip="End generation immediately or after completing current batch",
            )

            self.skip = gr.Button(
                "Skip",
                elem_id=f"{self.id_part}_skip",
                elem_classes="generate-box-skip",
                tooltip="Stop generation of current batch and continues onto next batch",
            )

            # 中断按钮
            self.interrupting = gr.Button(
                "Interrupting...",
                elem_id=f"{self.id_part}_interrupting",
                elem_classes="generate-box-interrupting",
                tooltip="Interrupting generation...",
            )

            # -------------------------------
            # 生成按钮
            self.submit = gr.Button(
                "Generate",
                elem_id=f"{self.id_part}_generate",
                variant="primary",
                tooltip="Right click generate forever menu",
            )

            def interrupt_function():
                if (
                    not shared.state.stopping_generation
                    and shared.state.job_count > 1
                    and shared.opts.interrupt_after_current
                ):
                    shared.state.stop_generating()
                    gr.Info(
                        "Generation will stop after finishing this image, click again to stop immediately."
                    )
                else:

                    # 中断生成
                    shared.state.interrupt()

            # 跳过按钮的点击事件
            self.skip.click(fn=shared.state.skip)

            # 中断按钮的点击事件
            self.interrupt.click(
                fn=interrupt_function,
                _js='function(){ showSubmitInterruptingPlaceholder("'
                + self.id_part
                + '"); }',
            )

            self.interrupting.click(fn=interrupt_function)

    # 创建工具栏按钮
    def create_tools_row(self):
        with gr.Row(elem_id=f"{self.id_part}_tools"):
            from modules.ui import (
                paste_symbol,
                clear_prompt_symbol,
                restore_progress_symbol,
            )

            # 粘贴按钮
            self.paste = ToolButton(
                value=paste_symbol,
                elem_id="paste",
                tooltip="Read generation parameters from prompt or last generation if prompt is empty into user interface.",
            )

            # 清空按钮
            self.clear_prompt_button = ToolButton(
                value=clear_prompt_symbol,
                elem_id=f"{self.id_part}_clear_prompt",
                tooltip="Clear prompt",
            )

            # 应用样式按钮
            self.apply_styles = ToolButton(
                value=ui_prompt_styles.styles_materialize_symbol,
                elem_id=f"{self.id_part}_style_apply",
                tooltip="Apply all selected styles to prompts.",
            )

            # 如果是img2img，创建两个按钮
            if self.is_img2img:
                self.button_interrogate = ToolButton(
                    "📎",
                    tooltip="Interrogate CLIP - use CLIP neural network to create a text describing the image, and put it into the prompt field",
                    elem_id="interrogate",
                )
                self.button_deepbooru = ToolButton(
                    "📦",
                    tooltip="Interrogate DeepBooru - use DeepBooru neural network to create a text describing the image, and put it into the prompt field",
                    elem_id="deepbooru",
                )

            self.restore_progress_button = ToolButton(
                value=restore_progress_symbol,
                elem_id=f"{self.id_part}_restore_progress",
                visible=False,
                tooltip="Restore progress",
            )

            self.token_counter = gr.HTML(
                value="<span>0/75</span>",
                elem_id=f"{self.id_part}_token_counter",
                elem_classes=["token-counter"],
                visible=False,
            )
            self.token_button = gr.Button(
                visible=False, elem_id=f"{self.id_part}_token_button"
            )
            self.negative_token_counter = gr.HTML(
                value="<span>0/75</span>",
                elem_id=f"{self.id_part}_negative_token_counter",
                elem_classes=["token-counter"],
                visible=False,
            )
            self.negative_token_button = gr.Button(
                visible=False, elem_id=f"{self.id_part}_negative_token_button"
            )

            self.clear_prompt_button.click(
                fn=lambda *x: x,
                _js="confirm_clear_prompt",
                inputs=[self.prompt, self.negative_prompt],
                outputs=[self.prompt, self.negative_prompt],
            )

    # 创建样式UI
    def create_styles_ui(self):
        self.ui_styles = ui_prompt_styles.UiPromptStyles(
            self.id_part, self.prompt, self.negative_prompt
        )
        self.ui_styles.setup_apply_button(self.apply_styles)
