# pyrefly: ignore [missing-import]
import easyocr
# pyrefly: ignore [missing-import]
import cv2 as cv
import re
import numpy as np
import logging
import torch

# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from probabilities import Probability
from huggingface_hub import hf_hub_download
from xycut import bbox2points, recursive_xy_cut, vis_polygons_with_index
from sklearn.cluster import DBSCAN
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ImageToText:
    def __init__(self):
        try:
            self.probability = Probability()
            self.reader = easyocr.Reader(['vi','en'])
            model_path = hf_hub_download(
                repo_id="hantian/yolo-doclaynet",
                filename="yolov8s-doclaynet.pt"
            )
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error("Failed to initialize OCR engine: %s", e, exc_info=True)
            raise RuntimeError(
                f"Could not load required models (EasyOCR / YOLO / SymSpell). "
                f"Check network connectivity and cache paths. Original error: {e}"
            ) from e

    def image_to_text(self, image_path):
        start_time = time.time()
        sorted_boxes = self.split_image(image_path)
        logger.info("%.2fs — Layout split complete", time.time() - start_time)

        image = cv.imread(image_path)
        output_text = ""

        for i, box in enumerate(sorted_boxes):
            image_sorte = image[box[1]:box[3], box[0]:box[2]]
            result_text = self.reader.readtext(image_sorte)
            full_text = ""
            for _, text, _ in result_text:
                full_text += text+"\n"
            full_text = re.sub(r'\n(?![A-Z])', ' ', full_text)


            for text in full_text.split("\n"):
                fixed_text = self.probability.fix_spelling(text)
                fixed_text = fixed_text.capitalize()
                output_text = output_text + fixed_text + "\n"


        logger.info("%.2fs — OCR pipeline complete", time.time() - start_time)
        return output_text

    def split_image(self, image_path):
        results = self.model(image_path, imgsz=1024, conf=0.15, iou=0.4, agnostic_nms=True)
        image = cv.imread(image_path)
        sorted_boxes = []
        for result in results:
            boxes = result.boxes
            list_box = []

            for box in boxes:
                x_box_min = int(box.xyxy[0][0])
                y_box_min = int(box.xyxy[0][1])
                x_box_max = int(box.xyxy[0][2])
                y_box_max = int(box.xyxy[0][3])
                if x_box_max <= x_box_min or y_box_max <= y_box_min:
                    continue

                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                conf = box.conf[0].item()

                if class_id != 6:
                    list_box.append([x_box_min, y_box_min, x_box_max, y_box_max, class_id, label, conf])

            # FIX 2: Guard against empty list_box (no detected regions)
            if not list_box:
                logger.info("No layout regions detected — falling back to full image bounding box")
                sorted_boxes.append([0, 0, image.shape[1], image.shape[0]])
                continue

            avg_height = sum(box[3] - box[1] for box in list_box) / len(list_box)
            centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in list_box])
            clustering = DBSCAN(eps=avg_height * 4, min_samples=1).fit(centers)

            list_label_area = clustering.labels_

            box_area = []
            for cluster_id in range(max(list_label_area) + 1):
                cluster_boxes = [list_box[i] for i, lbl in enumerate(list_label_area) if lbl == cluster_id]
                x1 = min(b[0] for b in cluster_boxes)
                y1 = min(b[1] for b in cluster_boxes)
                x2 = max(b[2] for b in cluster_boxes)
                y2 = max(b[3] for b in cluster_boxes)

                box_area.append([x1, y1, x2, y2])

            random_boxes = np.array(box_area)
            np.random.shuffle(random_boxes)
            res = []

            recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(random_boxes)), res)

            assert len(res) == len(box_area)

            random_boxes = random_boxes[np.array(res)].tolist()
            while random_boxes:
                short_box = random_boxes.pop(0)
                index_short = [i for i, box in enumerate(random_boxes) if self.is_overlapping(short_box, box)]
                if index_short:
                    bbox = [short_box]
                    for i, index in enumerate(index_short):
                        bbox.append(random_boxes.pop(index - i))
                    x1 = min(b[0] for b in bbox)
                    y1 = min(b[1] for b in bbox)
                    x2 = max(b[2] for b in bbox)
                    y2 = max(b[3] for b in bbox)
                    short_box = [x1, y1, x2, y2]
                sorted_boxes.append(short_box)
        return sorted_boxes

    def is_overlapping(self, box1, box2):
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


if __name__ == "__main__":
    img_path = "mau-cv-xin-viec-don-gian-image-1.jpg"
    image_to_text = ImageToText()
    result = image_to_text.image_to_text(img_path)


