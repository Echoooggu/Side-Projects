import cv2               # 导入 OpenCV 库（用于图像处理）
import os
import numpy as np       # 导入 NumPy 库（用于数值计算，图像数组处理）
#numpy数组：(H, W, i)：一个三元组，每个像素有i个通道（i=1表示只有亮度值，即灰度图像的三位表示；
#                                             (i=3表示BGR，即每个像素都有BGR三个通道
#          (H, W):一个二元组，表示灰色图像，可以升维为三元组
#imread读出来就是numpy数组，其shape属性的形式为三元组或者二元组
import errno
from glob import glob
import poisson
import generate_mask

extensions = ['.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.gif', '.tiff', '.tif', '.raw', '.bmp']
SRC_FOLDER = "input"
OUT_FOLDER = "output"

# Collect all file routes with a specific prefix
def collect_files(prefix, extension_list=extensions):
    filenames = list(set([f for ext in extension_list for f in glob(prefix + '*' + ext)]))
    return filenames

subfolders = os.walk(SRC_FOLDER)
next(subfolders)

for dirpath, dirnames, fnames in subfolders:
    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)
    print("Processing input {i}...".format(i=image_dir))

    source_names = collect_files(os.path.join(dirpath, 'source'))
    target_names = collect_files(os.path.join(dirpath, 'target'))
    mask_names = collect_files(os.path.join(dirpath, 'mask'))

    if len(mask_names) == 0:
        print("⚠️ 未找到掩码，启动交互式掩码绘制界面...")
        mask_path = os.path.join(dirpath, 'mask.png')
        generate_mask.interactive_generate_mask(source_names[0], mask_path)
        mask_names = [mask_path]

    if not len(source_names) == len(target_names) == len(mask_names) == 1:
        print("ERROR" + "\n")
        continue

    source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
    mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)

    mask = np.atleast_3d(mask_img).astype(np.float32)/255.0
    mask = (mask > 0.9).astype(np.float32)
    mask = mask[:, :, 0]
     #debug:
    from PIL import Image

    Image.open(target_names[0])

    # 调整 source 和 mask 的大小以与 target 对齐
    if source_img.shape[:2] != target_img.shape[:2]:
        print("🔄 Resizing source and mask to match target size...")
        target_h, target_w = target_img.shape[:2]
        source_img = cv2.resize(source_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # 按对齐后的尺寸提取掩码 patch 区域
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        print("❌ 掩码为空，跳过该 case\n")
        continue

    top, bottom = np.min(ys), np.max(ys)
    left, right = np.min(xs), np.max(xs)
    padding = 10
    top = max(top - padding, 0)
    bottom = min(bottom + padding, source_img.shape[0] - 1)
    left = max(left - padding, 0)
    right = min(right + padding, source_img.shape[1] - 1)

    source_patch = source_img[top:bottom + 1, left:right + 1]
    target_patch = target_img[top:bottom + 1, left:right + 1]
    mask_patch = mask[top:bottom + 1, left:right + 1]

    channels = source_patch.shape[-1]
    result_patch = [poisson.process(source_patch[:, :, i], target_patch[:, :, i], mask_patch)
                    for i in range(channels)]
    result_patch = cv2.merge(result_patch)

    result = np.copy(target_img)
    result[top:bottom + 1, left:right + 1] = result_patch
    result = np.clip(result, 0, 255).astype(np.uint8)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    cv2.imwrite(os.path.join(output_dir, "result.png"), result)
    print("Finishing processing input{i}.".format(i=image_dir))

