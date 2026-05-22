"""
generate_test_data.py — Generate synthetic Vietnamese test images for OCR evaluation.

Creates 20 images from Vietnamese job-posting sentences rendered with PIL,
plus a corresponding ground_truth.txt file.
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "test_images"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
FONT_SIZE = 24
IMAGE_WIDTH = 900
PADDING = 20

SENTENCES = [
    "Có kinh nghiệm quản lý đội nhóm gồm 20 nhân viên cấp dưới.",
    "Tốt nghiệp Đại học Bách Khoa Hà Nội ngành Công nghệ Thông tin.",
    "Thành thạo các công cụ Microsoft Office bao gồm Word Excel PowerPoint.",
    "Có khả năng làm việc độc lập và phối hợp nhóm hiệu quả.",
    "Kinh nghiệm làm việc tại các công ty công nghệ hàng đầu Việt Nam.",
    "Kỹ năng giao tiếp tốt bằng tiếng Anh cả nói và viết.",
    "Chịu trách nhiệm phát triển và duy trì hệ thống phần mềm.",
    "Xây dựng chiến lược marketing và triển khai các chiến dịch quảng cáo.",
    "Quản lý ngân sách dự án và đảm bảo tiến độ đúng kế hoạch.",
    "Phân tích dữ liệu và đưa ra báo cáo định kỳ cho ban lãnh đạo.",
    "Tuyển dụng đào tạo và phát triển đội ngũ nhân sự chất lượng cao.",
    "Thiết kế giao diện người dùng thân thiện và tối ưu trải nghiệm.",
    "Lập trình ứng dụng di động trên nền tảng Android và iOS.",
    "Nghiên cứu và ứng dụng các công nghệ mới vào sản phẩm.",
    "Hợp tác với các đối tác trong và ngoài nước để mở rộng thị trường.",
    "Soạn thảo hợp đồng và các văn bản pháp lý liên quan.",
    "Kiểm tra chất lượng sản phẩm theo tiêu chuẩn ISO quốc tế.",
    "Cung cấp hỗ trợ kỹ thuật và giải quyết sự cố cho khách hàng.",
    "Tham gia các hội thảo và khóa đào tạo chuyên môn thường xuyên.",
    "Xây dựng và duy trì mối quan hệ tốt với khách hàng và đối tác.",
]


def load_font(size):
    """Try known font paths in order, fall back to default."""
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    print("WARNING: No TrueType font found, using PIL default (Vietnamese may not render)")
    return ImageFont.load_default()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font = load_font(FONT_SIZE)

    gt_lines = []

    for idx, sentence in enumerate(SENTENCES):
        # Measure text size to auto-calculate image height
        dummy = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), sentence, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img_height = text_height + 2 * PADDING
        img = Image.new("RGB", (IMAGE_WIDTH, img_height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((PADDING, PADDING), sentence, fill="black", font=font)

        filename = f"{idx:02d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        img.save(filepath)
        gt_lines.append(sentence)
        print(f"  Generated: {filepath}")

    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        for line in gt_lines:
            f.write(line + "\n")

    print(f"\nDone! {len(SENTENCES)} images saved to {OUTPUT_DIR}/")
    print(f"Ground truth saved to {gt_path}")


if __name__ == "__main__":
    main()
