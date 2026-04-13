import easyocr
import cv2 as cv
import re
import numpy as np
import torch

from ultralytics import YOLO
from probabilities import Probability
from huggingface_hub import hf_hub_download
from xycut import bbox2points, recursive_xy_cut, vis_polygons_with_index
from sklearn.cluster import DBSCAN
import time

class ImageToText:
    def __init__(self):
        self.probability = Probability()
        gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(['vi','en'], gpu = gpu)
        model_path = hf_hub_download(
            repo_id="hantian/yolo-doclaynet",
            filename="yolov8x-doclaynet.pt"
        )

        self.model = YOLO(model_path)

    def image_to_text(self, image_path):
        start_time = time.time()
        sorted_boxes = self.split_image(image_path)
        print(time.time() - start_time, " Split sucesfull")

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


        print(time.time() - start_time, " OCR sucesfull")
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

            avg_height = sum(box[3] - box[1] for box in list_box) / len(list_box)
            centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in list_box])
            clustering = DBSCAN(eps=avg_height * 4, min_samples=1).fit(centers)

            list_label_area = clustering.labels_

            box_area = []
            label = []
            for label in range(max(list_label_area) + 1):
                label_index = [list_box[i] for i, label_name in enumerate(list_label_area) if label_name == label]
                x1 = min(label[0] for label in label_index)
                y1 = min(label[1] for label in label_index)
                x2 = max(label[2] for label in label_index)
                y2 = max(label[3] for label in label_index)

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
    img_path = "list_image/65d581a5dca4b054b6980098_Budapest.jpg"
    image_to_text = ImageToText()
    result = image_to_text.image_to_text(img_path)


