import json
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import (
    create_override_settings_dict,
    parse_generation_parameters,
)
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
from PIL import Image
import gradio as gr


def txt2img_create_processing(
    id_task: str,
    request: gr.Request,
    prompt: str,
    negative_prompt: str,
    prompt_styles,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    height: int,
    width: int,
    enable_hr: bool,
    denoising_strength: float,
    hr_scale: float,
    hr_upscaler: str,
    hr_second_pass_steps: int,
    hr_resize_x: int,
    hr_resize_y: int,
    hr_checkpoint_name: str,
    hr_sampler_name: str,
    hr_scheduler: str,
    hr_prompt: str,
    hr_negative_prompt,
    override_settings_texts,
    *args,
    force_enable_hr=False,
):
    """
    Create a processing object for txt2img task

    """
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    # 创建一个StableDiffusionProcessingTxt2Img类的对象，来执行生成图片的任务
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,  # 传入模型参数
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=(
            None if hr_checkpoint_name == "Use same checkpoint" else hr_checkpoint_name
        ),
        hr_sampler_name=(
            None if hr_sampler_name == "Use same sampler" else hr_sampler_name
        ),
        hr_scheduler=None if hr_scheduler == "Use same scheduler" else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p


def txt2img_upscale(
    id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args
):
    assert len(gallery) > 0, "No image to upscale"
    assert 0 <= gallery_index < len(gallery), f"Bad image index: {gallery_index}"

    p = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    p.batch_size = 1
    p.n_iter = 1
    # txt2img_upscale attribute that signifies this is called by txt2img_upscale
    p.txt2img_upscale = True

    geninfo = json.loads(generation_info)

    image_info = (
        gallery[gallery_index] if 0 <= gallery_index < len(gallery) else gallery[0]
    )
    p.firstpass_image = infotext_utils.image_from_url_text(image_info)

    parameters = parse_generation_parameters(
        geninfo.get("infotexts")[gallery_index], []
    )
    p.seed = parameters.get("Seed", -1)
    p.subseed = parameters.get("Variation seed", -1)

    p.override_settings["save_images_before_highres_fix"] = False

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    new_gallery = []
    for i, image in enumerate(gallery):
        if i == gallery_index:
            geninfo["infotexts"][
                gallery_index : gallery_index + 1
            ] = processed.infotexts
            new_gallery.extend(processed.images)
        else:
            fake_image = Image.new(mode="RGB", size=(1, 1))
            fake_image.already_saved_as = image["name"].rsplit("?", 1)[0]
            new_gallery.append(fake_image)

    geninfo["infotexts"][gallery_index] = processed.info

    return (
        new_gallery,
        json.dumps(geninfo),
        plaintext_to_html(processed.info),
        plaintext_to_html(processed.comments, classname="comments"),
    )


def txt2img(id_task: str, request: gr.Request, *args):
    """
    处理文本到图片生成任务，并返回生成的图片及相关信息
    """

    # 创建一个进程
    # 调用txt2img_create_processing函数，创建一个处理此任务的进程对象p
    p = txt2img_create_processing(id_task, request, *args)

    # 执行图片生成任务
    # with closing(p):确保在代码块执行完毕后，无论成功还是异常终止，p对象都会被正确关闭或销毁
    with closing(p):

        # 这表明系统可能有两种处理图片任务的方法，优先使用第一种（可能因为它更专门），如果第一种没有返回结果，则使用第二种备选方案
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        # 如果此方法返回None，则调用processing.process_images(p)来处理图片
        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    # 处理生成的信息
    generation_info_js = processed.js()

    # 处理日志输出
    if opts.samples_log_stdout:
        print(generation_info_js)

    # 如果设置了不显示图片，则将图片置空
    if opts.do_not_show_images:
        processed.images = []

    # return result
    return (
        processed.images,  # 生成的图像列表
        generation_info_js,  # 处理后的JavaScript表示信息
        plaintext_to_html(processed.info),  # 将生成过程的信息（纯文本）转换为HTML
        plaintext_to_html(
            processed.comments, classname="comments"
        ),  # 将生成过程的评论转换为带有类名“comments”的HTML格式
    )
