import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ['Invoice_Number', 'Invoice_Date']  # 改成自己的类别
abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    # 檢查是否為有效的圖片檔案名稱
    if image_id == 'classes':
        print("Invalid image_id:", image_id)
        return
    
    try:
        in_file = open('C:/Local/wda-yolov7/KeyData/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    except FileNotFoundError:
        print("Error: File not found for image_id:", image_id)
        return

    print("Processing image_id:", image_id)
    out_file = open('C:/Local/wda-yolov7/KeyData/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    # ... (其余的代码保持不变)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    if not os.path.exists('C:/Local/wda-yolov7/KeyData/labels/'):
        os.makedirs('C:/Local/wda-yolov7/KeyData/labels/')
    image_ids = open('C:/Local/wda-yolov7/KeyData/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()

    if not os.path.exists('C:/Local/wda-yolov7/KeyData/dataSet_path/'):
        os.makedirs('C:/Local/wda-yolov7/KeyData/dataSet_path/')
    # 这行路径不需更改，这是相对路径
    list_file = open('dataSet_path/%s.txt' % image_set, 'w')

    # 图片格式为jpg则设置为 .jpg, 如果为png则设置为 .png。否则会出现路径找不到的问题
    for image_id in image_ids:
        list_file.write('C:/Local/wda-yolov7/KeyData/images/%s.jpg\n' % image_id)
        convert_annotation(image_id)
    list_file.close()
