import xml.etree.ElementTree
import os
import numpy as np
from PIL import Image
# import scipy.misc
import imageio
import imgaug as ia
from imgaug import augmenters as iaa


def read_xml_annotation(root, image_name):
    in_file = open(os.path.join(root, image_name))
    tree = xml.etree.ElementTree.parse(in_file)
    root = tree.getroot()
    # 讀取XML檔
    bndboxlist = list()

    for old_object in root.findall('object'):  # 找到root節點下的所有object節點
        bnd_box = old_object.find('bndbox')  # 找到object節點下的bndbox節點

        xmin = int(bnd_box.find('xmin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymin = int(bnd_box.find('ymin').text)
        ymax = int(bnd_box.find('ymax').text)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # 獲取所有座標
    return bndboxlist


def change_xml_annotation(root, image_name, new_target, num, save_root):
    in_file = open(os.path.join(root, str(image_name) + '.xml'))
    tree = xml.etree.ElementTree.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for new_object in xmlroot.findall('object'):
        bnd_box = new_object.find('bndbox')

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bnd_box.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bnd_box.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bnd_box.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bnd_box.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(save_root, str(image_id) + "_aug" + str(num + 1) + '.xml'))


if __name__ == "__main__":
    IMG_DIR = 'KeyData/Images/'   # 原始圖檔資料夾
    XML_DIR = 'KeyData/Annotations/'     # 原始XML資料夾

    AUG_IMG_DIR = 'KeyData/ImagesDA'  # 資料擴增後圖片儲存位置
    AUG_XML_DIR = 'KeyData/AnnotationsDA'     # 資料擴增後XML儲存位置

    files = os.listdir(IMG_DIR)     # 獲取資料夾下圖片檔名

    for f in files:
        image_id = f[:-4]   # 將副檔名去除
        img = Image.open(os.path.join(IMG_DIR, str(image_id) + '.jpg'))     # 打開圖片
        img = np.array(img)     # 將圖片轉成陣列形式

        bndbox = read_xml_annotation(XML_DIR, str(image_id) + '.xml')   # 呼叫read_xml_annotation函式

        # 設定data augmentation圖片參數
        seq = iaa.Sequential([
            # iaa.Fliplr(0),  # 翻轉
            #iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
            iaa.GaussianBlur(sigma=(0, 3)),   # 高斯擾動
            iaa.Multiply((0.9, 1.1)),   # 亮度
            iaa.Affine(
                translate_px={"x": 12, "y": 12},    # 平移
                scale=(0.8, 0.95),    # 縮放
                rotate=(-5, 5)     # 旋轉
            ),
            iaa.ContrastNormalization((0.5, 1.25))      # 對比度
        ])

        AUGLOOP = 39  # 每張圖片處理的數量

        for epoch in range(AUGLOOP):
            seq_det = seq.to_deterministic()  # 將圖片及座標同步改變
            image_aug = seq_det.augment_images([img])[0]    # 對圖片進行處理
            imageio.imwrite(AUG_IMG_DIR + "/" + image_id + '_aug' + str(epoch + 1) + '.jpg', image_aug)   # 儲存處理後的圖片

            old_bndbox = list()
            new_bndbox = list()
            for i in range(len(bndbox)):
                bbs = ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=bndbox[i][0],
                                   y1=bndbox[i][1],
                                   x2=bndbox[i][2],
                                   y2=bndbox[i][3])
                ], shape=img.shape)     # 獲取原始座標及大小

                bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]      # 對座標進行處理

                new_bndbox.append([int(bbs_aug.bounding_boxes[0].x1),
                                   int(bbs_aug.bounding_boxes[0].y1),
                                   int(bbs_aug.bounding_boxes[0].x2),
                                   int(bbs_aug.bounding_boxes[0].y2)])

                old_bndbox.append([bndbox[i][0], bndbox[i][1], bndbox[i][2], bndbox[i][3]])
            print('原始:', old_bndbox)
            print('處理:', new_bndbox)
            print('---------------------------------------------------------------')
            change_xml_annotation(XML_DIR, image_id, new_bndbox, epoch, AUG_XML_DIR)
