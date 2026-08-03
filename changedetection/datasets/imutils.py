import random
import numpy as np
from PIL import Image
import sys
# from scipy import misc
import torch
import torchvision

from PIL import ImageEnhance


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

# --- 以下函数保持不变 ---
def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label

def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label
# --- 以上函数保持不变 ---


# --- 修改 *_bda 函数 ---

def random_fliplr_bda(pre_img, post_img, label_2): # 移除了 label_1
    if random.random() > 0.5:
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
    # 移除了返回 label_1
    return pre_img, post_img, label_2


def random_flipud_bda(pre_img, post_img, label_2): # 移除了 label_1
    if random.random() > 0.5:
        # 移除了 label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
    # 移除了返回 label_1
    return pre_img, post_img, label_2


def random_rot_bda(pre_img, post_img, label_2): # 移除了 label_1
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    # 移除了 label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    # 移除了返回 label_1
    return pre_img, post_img, label_2

def random_crop_bda(pre_img, post_img, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255): # 移除了 loc_label (相当于 label_1)
    # 原来使用 loc_label.shape, 现在假设 clf_label 形状相同
    h, w = clf_label.shape 

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    # 移除了 pad_loc_label
    pad_clf_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # (图像 padding 逻辑不变)
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]
    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]
    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    # 移除了 pad_loc_label 赋值
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    # (get_random_cropbox 内部逻辑不变，但它依赖的 pad_loc_label 已不存在)
    # 注意：这里的 get_random_cropbox 逻辑现在可能有问题，因为它原来基于 loc_label 确保有内容
    # 如果 cropbox 选择逻辑很重要，需要调整 get_random_cropbox 使用 clf_label
    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            # 原来使用 pad_loc_label, 现在改为 pad_clf_label
            temp_label = pad_clf_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 0 and np.max(cnt) / np.sum(cnt) < cat_max_ratio: # 确保至少有一个非忽略像素
                break
        # 如果循环结束还没找到合适的，就返回最后一次随机结果
        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    # 移除了 loc_label 的裁剪
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]
    # 移除了返回 loc_label
    
    # print(f"DEBUG: Cropped pre_img shape: {pre_img.shape}", file=sys.stderr)
    
    return pre_img, post_img, clf_label

# --- 以下函数保持不变 ---
def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()

    return pre_img, post_img, label_cd, label_1, label_2

def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = img.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    # 修正了原始代码可能的笔误
    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = img 

    # 保持 get_random_cropbox 的内部定义
    def get_random_cropbox(cat_max_ratio=0.75):
        # 这个函数原本依赖外部的 pad_image，这里改为依赖 img (因为没有标签)
        # 并且删除了 temp_label 相关的逻辑
        for i in range(10): 
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size
            break # 仅随机裁剪

        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    
    img = pad_image[H_start:H_end, W_start:W_end, :]

    return img


def random_bi_image_crop(pre_img, object, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = object.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # 保持原逻辑 (假设输入 >= crop_size)
    if h >= crop_size and w >= crop_size:
        H_start = random.randrange(0, h - crop_size + 1, 1)
        W_start = random.randrange(0, w - crop_size + 1, 1)
    else: 
        H_start = 0
        W_start = 0
        
    H_end = H_start + crop_size
    W_end = W_start + crop_size

    pre_img = pre_img[H_start:H_end, W_start:W_end, :]
    object = object[H_start:H_end, W_start:W_end]
    
    return pre_img, object


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]
    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    # 保持 get_random_cropbox 的内部定义
    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            # 修正了原始代码可能的笔误
            if len(cnt) > 0 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break
        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    
    return pre_img, post_img, label

def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label_cd = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_1 = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_2 = np.ones((H, W), dtype=np.float32) * ignore_index

    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]
    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label_cd[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_cd
    pad_label_1[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_1
    pad_label_2[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_2

    # 保持 get_random_cropbox 的内部定义
    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label_1[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            # 修正了原始代码可能的笔误
            if len(cnt) > 0 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break
        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label_cd = pad_label_cd[H_start:H_end, W_start:W_end]
    label_1 = pad_label_1[H_start:H_end, W_start:W_end]
    label_2 = pad_label_2[H_start:H_end, W_start:W_end]

    return pre_img, post_img, label_cd, label_1, label_2


def random_crop_dd(pre_img, post_img, loc_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index

    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]
    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label

    # 保持 get_random_cropbox 的内部定义
    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            # 修正了原始代码可能的笔误
            if len(cnt) > 0 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break
        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label

def random_fliplr_dd(pre_img, post_img, label_1):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_1

def random_rot_dd(pre_img, post_img, label_1):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()

    return pre_img, post_img, label_1

def random_flipud_dd(pre_img, post_img, label_1):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_1
# --- 以上函数保持不变 ---