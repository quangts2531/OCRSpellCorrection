# OCRSpellCorrection

## 1. Project Introduction & Research Overview

**OCRSpellCorrection** is a research and development project focused on significantly improving the output accuracy of Optical Character Recognition (OCR) models, specifically targeting **EasyOCR**. The core objective is achieved through a comprehensive pipeline that combines input image pre-processing, layout analysis, and output spelling correction.

The project is structured around three main pillars:

### Pillar 1: Spelling Correction (Completed / Implemented)
*   **Description:** Post-processing the raw text extracted by OCR to detect and correct spelling errors.
*   **Reference Paper:** [*Building an Automatic Spelling Correction System Using N-Gram Information Surrounding Error Word in Vietnamese Queries*](https://ieeexplore.ieee.org/abstract/document/10883437)
*   **Algorithm Used:** N-gram approach.
*   **Reasoning:** The N-gram methodology is easily scalable because massive amounts of unlabeled text data can be crawled and utilized to build the language models. This allows for quick deployment without the heavy time constraints and costs associated with manual data labeling.
*   **Key Tasks Done:** Extensive data crawling, building robust 1-gram, 2-gram, and 3-gram language models, and implementing the core spelling correction logic.

### Pillar 2: Illogical Document Layout Reconstruction (Completed / Implemented)
*   **Description:** Reorganizing and reconstructing non-logical, fragmented, or complex document image layouts into a readable, sequential format prior to text extraction.
*   **Reference Papers:** 
    *   [*A Hybrid Approach for Document Layout Analysis in Document images*](https://arxiv.org/abs/2404.17888)
    *   [*PubLayNet: largest dataset ever for document layout analysis*](https://arxiv.org/abs/1908.07836)
*   **Algorithm Used:** XY Cut algorithm paired with a YOLO layout detection model (`hantian/yolo-doclaynet`).

### Pillar 3: Input Image Quality Enhancement (Future Work / Not Yet Coded)
*   **Description:** Utilizing pre-processing techniques to improve the raw input image quality (e.g., denoising, binarization, de-skewing) before passing it to the OCR model. The goal is to maximize the initial character prediction probability of EasyOCR.
*   **Status:** Currently in the research phase; pending implementation.

---

## 2. Environment Variables Configuration

The application uses specific environment variables to manage execution resources and model caching. Below are the required environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HF_HOME` | `/app/.cache/huggingface` | Defines the directory where Hugging Face will cache downloaded models (e.g., the YOLO layout model). |
| `EASYOCR_MODULE_PATH` | `/app/.cache/easyocr` | Defines the directory where EasyOCR will store its downloaded model weights. |
| `CUDA_VISIBLE_DEVICES` | `""` (Empty string) | Controls which GPUs are accessible to PyTorch/YOLO. Set to an empty string to force CPU-only execution. |
| `OMP_NUM_THREADS` | `2` | Specifies the maximum number of threads OpenMP can use for parallel CPU computations. |
| `MKL_NUM_THREADS` | `2` | Specifies the number of threads for the Intel Math Kernel Library (MKL) to optimize CPU performance. |

**Example `.env` file for local setup:**
```env
HF_HOME=./.cache/huggingface
EASYOCR_MODULE_PATH=./.cache/easyocr
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS="2"
MKL_NUM_THREADS="2"
```

---

## 3. Installation & Docker Execution Guide

The most reliable way to run OCRSpellCorrection is using **Docker** and **Docker Compose**, as it encapsulates all required system dependencies (like OpenCV's `libgl1`) and Python packages.

### Prerequisites
*   Docker installed on your system.
*   Docker Compose installed.

### Step-by-Step Execution

1.  **Clone the repository and navigate to the project directory:**
    ```bash
    git clone <repository_url>
    cd OCRSpellCorrection
    ```

2.  **Build and start the application container:**
    Run the following command to build the Docker image and start the Flask web server. The `--build` flag ensures that any changes to your code or dependencies are captured.
    ```bash
    docker-compose up --build -d
    ```

3.  **Access the application:**
    Once the container is running, the OCR web application will be accessible at:
    `http://localhost:5000/`

4.  **View container logs (Optional):**
    If you need to debug or watch the OCR processing logs:
    ```bash
    docker logs -f ocr-app
    ```

### Cleanup Instructions

When you are done testing and want to stop the application and free up system resources, run the following commands:

1.  **Stop and remove the container:**
    ```bash
    docker-compose down
    ```

2.  **Clean up dangling images and unused resources (Optional but recommended):**
    ```bash
    docker system prune -f
    ```

---

## 4. Project Structure Overview

```text
OCRSpellCorrection/
├── craw/                    # Scripts and JSON data for crawling job datasets (Pillar 1).
├── dictionary/              # Code and text files for 1-gram, 2-gram, and 3-gram language models.
├── nginx/                   # Nginx reverse proxy configuration.
├── out/                     # Directory for generated or processed files/images.
├── templates/               # HTML templates (e.g., index.html) for the Flask frontend.
├── app.py                   # Main Flask application entry point handling web requests.
├── image_to_text.py         # Core pipeline: runs YOLO layout detection, XY Cut, EasyOCR, and spelling correction.
├── probabilities.py         # Implements the N-gram probability logic to fix spelling errors.
├── xycut.py                 # Implements the recursive XY Cut algorithm for document layout analysis.
├── Dockerfile               # Docker configuration for building the application image.
├── docker-compose.yml       # Docker Compose file to orchestrate the container deployment.
└── requirements.txt         # List of Python dependencies required for the project.
```