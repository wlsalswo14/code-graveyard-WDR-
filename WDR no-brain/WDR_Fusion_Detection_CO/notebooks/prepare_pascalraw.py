import os
import shutil
import sys
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config

# Pascal VOC PascalRaw 스타일 20 클래스 → 클래스 ID
PASCAL_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_ID = {name: i for i, name in enumerate(PASCAL_CLASSES)}


def convert_pascalraw_to_yolo(voc_root: str, target_base: str):
    """
    Pascal VOC 디렉터리(JPEGImages, Annotations)를 읽어 YOLO 포맷 txt + 이미지 복사.

    voc_root: Pascal VOC 루트 (예: .../VOCdevkit/VOC2007 또는 PascalRaw 압축 풀린 폴더)
    target_base: 출력 루트 — 아래에 images/, labels/ 생성
    """
    print("=====================================")
    print("🚀 Pascal VOC(XML) -> YOLO 포맷 변환")
    print("=====================================")

    voc_images_dir = os.path.join(voc_root, "JPEGImages")
    voc_annots_dir = os.path.join(voc_root, "Annotations")

    if not os.path.isdir(voc_images_dir) or not os.path.isdir(voc_annots_dir):
        print(f"[오류] JPEGImages 또는 Annotations 를 찾을 수 없습니다: {voc_root}")
        return

    target_images = os.path.join(target_base, "images")
    target_labels = os.path.join(target_base, "labels")

    print(f"출력 폴더: {target_base}")
    for d in [target_images, target_labels]:
        os.makedirs(d, exist_ok=True)

    xml_files = [f for f in os.listdir(voc_annots_dir) if f.endswith('.xml')]
    print(f"총 {len(xml_files)}개의 어노테이션(XML) 변환 시작...")

    valid_count = 0
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(voc_annots_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            continue

        w = int(size.find('width').text)
        h = int(size.find('height').text)

        if w == 0 or h == 0:
            continue

        yolo_labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[name]

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            box_w = xmax - xmin
            box_h = ymax - ymin
            box_cx = xmin + (box_w / 2.0)
            box_cy = ymin + (box_h / 2.0)

            norm_cx = box_cx / w
            norm_cy = box_cy / h
            norm_w = box_w / w
            norm_h = box_h / h

            norm_cx = min(max(norm_cx, 0.0), 1.0)
            norm_cy = min(max(norm_cy, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            yolo_labels.append(f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_labels:
            image_name = xml_file.replace('.xml', '.jpg')
            source_img = os.path.join(voc_images_dir, image_name)

            if os.path.exists(source_img):
                shutil.copy(source_img, os.path.join(target_images, image_name))

                label_path = os.path.join(target_labels, xml_file.replace('.xml', '.txt'))
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels) + '\n')

                valid_count += 1

    print(f"✅ 변환 완료! (성공: {valid_count}장)")
    print(f"학습 입력 경로: {target_base} (images/, labels/)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pascal VOC(XML) 데이터를 프로젝트 data/layout(images, labels)로 변환"
    )
    parser.add_argument(
        "--voc_root",
        type=str,
        required=True,
        help="VOC 루트 (JPEGImages, Annotations 하위 폴더 포함)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="YOLO 형식 출력 루트 (기본: <프로젝트>/data/voc_yolo)",
    )
    args = parser.parse_args()

    out = args.output_dir or os.path.join(config.DATA_DIR, "voc_yolo")
    convert_pascalraw_to_yolo(os.path.abspath(args.voc_root), os.path.abspath(out))
