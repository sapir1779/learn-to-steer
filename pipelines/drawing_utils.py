import os
from typing import Tuple, Union, List
import numpy as np
from PIL import Image
import cv2
import torch
from utils import find_word_token_inds


def process_raw_attention_maps(attention_maps):
    if attention_maps.dtype == torch.bfloat16:
        attention_maps = attention_maps.to(torch.float32)
    return attention_maps.clone().detach().cpu()

def prompt_to_tokens(prompt, tokenizer, remove_start_token):
    tokens = tokenizer.encode(prompt)
    if remove_start_token:
        tokens = tokens[1 : len(tokens)-1]
    return tokens

def draw_cross_attention_single(attention_maps, text, bg_image=None, 
                                remove_start_token=False, display_img=True,
                                map_size=16):
    attn_maps = process_raw_attention_maps(attention_maps)
    if attn_maps.ndim < 3:
        attn_maps = attn_maps.unsqueeze(dim=0)
        
    draw_on_image = (bg_image is not None)
    images = []
    for i in range(attn_maps.shape[0]):
        image = attn_maps[i, :, :]
        if draw_on_image:
            image = show_image_relevance(image, bg_image, map_size)
        else:
            image = (image - image.min()) / (image.max() - image.min())
            image = 255 * image
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy()
            
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((128, 128)))
        image = text_under_image(image, text[i])
        images.append(image)

    return draw_images(np.stack(images, axis=0), display_img=display_img)

def draw_cross_attention(attention_maps, prompt, tokenizer, bg_image=None, 
                         remove_start_token=False, display_img=True, skip_prefix_tokens=False):
    attn_maps = process_raw_attention_maps(attention_maps)
    tokens = prompt_to_tokens(prompt, tokenizer, remove_start_token)
    prefix_tokens = prompt_to_tokens("a photo of a", tokenizer, remove_start_token) if skip_prefix_tokens else []
    draw_on_image = (bg_image is not None)
    images = []
    total_tokens = len(tokens) - 1 if skip_prefix_tokens else len(tokens)
    for i in range(total_tokens):
        token = tokens[i]
        if skip_prefix_tokens and i < len(prefix_tokens) and token in prefix_tokens:
            continue
            
        image = attn_maps[i, :, :]
        if draw_on_image:
            image = show_image_relevance(image, bg_image)
        else:
            image = (image - image.min()) / (image.max() - image.min())
            image = 255 * image
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy()
            
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((128, 128)))
        image = text_under_image(image, tokenizer.decode(int(token)))
        images.append(image)

    return draw_images(np.stack(images, axis=0), display_img=display_img)


def draw_cross_attention_objects_only(attention_maps, objects, prompt, tokenizer, bg_image=None, 
                                        remove_start_token=False, display_img=True, 
                                        skip_prefix_tokens=False):
    attn_maps = process_raw_attention_maps(attention_maps)
    # remove_start_token = False on flux-dev, else True
    objects_token_inds = [find_word_token_inds(tokenizer, prompt, obj, remove_start_token)[0] for obj in objects]
    draw_on_image = (bg_image is not None)
    images = []
    for i in range(len(objects_token_inds)):
        obj_tokens_inds = objects_token_inds[i]
        obj_text = objects[i]
        
        start = obj_tokens_inds[0]
        end = obj_tokens_inds[-1]
        if start < end:
            image = attn_maps[start: end, :, :].mean(dim=0)
        else:
            image = attn_maps[start, :, :]
            
        if draw_on_image:
            image = show_image_relevance(image, bg_image)
        else:
            image = (image - image.min()) / (image.max() - image.min())
            image = 255 * image
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy()
            
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((128, 128)))
        image = text_under_image(image, obj_text)
        images.append(image)

    return draw_images(np.stack(images, axis=0), display_img=display_img)


def show_image_relevance(image_relevance, image: Image.Image, relevance_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevance_res ** 2, relevance_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevance_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevance_res ** 2, relevance_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    font_scale = 0.65
    thickness = 2
    textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, font_scale, text_color, thickness)
    return img


def draw_images(images: Union[np.ndarray, List], num_rows: int = 1, offset_ratio: float = 0.0, 
                   display_img: bool = True, scale: int = -1) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if scale > 0:
        pil_img = pil_img.resize(tuple(scale * np.array(pil_img.size)))
        
    return pil_img


def concat_images(input_images, display_img=True, output_path="", add_row_indices=False,
                  num_rows=None, num_cols=None, auto_grid=False):
    if len(input_images) == 0:
        return

    images = add_row_indices_to_images(input_images) if add_row_indices else input_images
    width, height = images[0].size
    num_images = len(images)

    if auto_grid:
        num_cols = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_cols))
    else:
        if num_cols is None:
            num_cols = 1
        if num_rows is None:
            num_rows = num_images

    # Pad with blank white images if needed
    blank_image = Image.new('RGB', (width, height), color=(255, 255, 255))
    while len(images) < num_rows * num_cols:
        images.append(blank_image)

    canvas_width = num_cols * width
    canvas_height = num_rows * height
    output_img = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        output_img.paste(img, (col * width, row * height))

    if output_path:
        output_img.save(output_path)

    return output_img


def add_row_indices_to_images(images):
    if len(images) == 0:
        return

    height = images[0].size[1]
    indexed_images = []
    for idx, img in enumerate(images):
        img_np = np.array(img)

        height, width, channels = img_np.shape
        index_width = height

        # Create the index image (white background)
        index_img = np.ones((height, index_width, channels), dtype=np.uint8) * 255

        # Add the row index as text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text = str(idx)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (index_width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(index_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        combined_img = np.concatenate((index_img, img_np), axis=1)
        indexed_images.append(Image.fromarray(combined_img))

    return indexed_images


def create_folders(include_color, root_path, folder_name, create_sub_folders=True):
    output_path = os.path.join(root_path, folder_name) if len(folder_name) > 0 else root_path
    os.makedirs(output_path, exist_ok=True)
    
    if not create_sub_folders:
        return output_path
    
    folder_grey = os.path.join(output_path, "grey")
    os.makedirs(folder_grey, exist_ok=True)
    folder_color = ''
    if include_color:
        folder_color = os.path.join(output_path, "color")
        os.makedirs(folder_color, exist_ok=True)

    return output_path, folder_grey, folder_color


class MapsDrawer:
    def __init__(self, config, skip_prefix_tokens=False, objects=[]):
        self.config = config
        self.include_color = config.attention.include_color
        self.remove_start_token = config.attention.remove_start_token
        self.dump_sources = config.attention.dump_sources
        self.skip_prefix_tokens = skip_prefix_tokens
        self.objects = objects

    def draw_and_append_maps(self, attn_maps, prompt, tokenizer, bg_image, images_grey=None, images_color=None):
        if images_grey is None:
            images_grey = []
        if images_color is None:
            images_color = []
            
        if len(self.objects) > 0:
            output_grey = draw_cross_attention_objects_only(attn_maps, self.objects, prompt, tokenizer, bg_image=None,
                                            remove_start_token=self.remove_start_token, 
                                            display_img=False, skip_prefix_tokens=self.skip_prefix_tokens)
            images_grey.append(output_grey)
            
            if self.include_color:
                output_color = draw_cross_attention_objects_only(attn_maps, self.objects, prompt, tokenizer, bg_image, 
                                                    self.remove_start_token,
                                                    display_img=False, skip_prefix_tokens=self.skip_prefix_tokens)
                images_color.append(output_color)
        else:
            output_grey = draw_cross_attention(attn_maps, prompt, tokenizer, bg_image=None,
                                    remove_start_token=self.remove_start_token, 
                                    display_img=False, skip_prefix_tokens=self.skip_prefix_tokens)
            images_grey.append(output_grey)
            
            if self.include_color:
                output_color = draw_cross_attention(attn_maps, prompt, tokenizer, bg_image, 
                                                    self.remove_start_token,
                                                    display_img=False, skip_prefix_tokens=self.skip_prefix_tokens)
                images_color.append(output_color) 
        
        return images_grey, images_color


class MapsDrawerNoCache:
    def __init__(self, add_row_indices=True, res16=False, avg=False, map_size=16):
        self.include_color = True
        self.remove_start_token = False
        self.add_row_indices = add_row_indices
        self.res16 = res16
        self.avg = avg
        self.map_size = map_size

    def draw_and_append_maps(self, attn_maps, rel_objects, bg_image, images_grey=None, images_color=None):
        if images_grey is None:
            images_grey = []
        if images_color is None:
            images_color = []
        
        output_grey = draw_cross_attention_single(attn_maps, rel_objects, bg_image=None,
                                                  remove_start_token=self.remove_start_token, 
                                                  display_img=False, map_size=self.map_size)
        images_grey.append(output_grey)
        
        if self.include_color:
            output_color = draw_cross_attention_single(attn_maps, rel_objects, bg_image=bg_image,
                                                  remove_start_token=self.remove_start_token, 
                                                  display_img=False, map_size=self.map_size)
            images_color.append(output_color)
        
        return images_grey, images_color
    