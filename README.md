# license Plate Recognition
## Prerequests Installation
### Darknet
Bounding box detection
1. Download darknet from github
    ```bash
    git clone https://github.com/AlexeyAB/darknet
    ```
1. 


### Tesseract
Tesseract Optical Character Recognition
1. Download tesseract-ocr 
    ```bash
    sudo apt-get install tesseract-ocr
    ```
1. Download python package
    ```bash
    pip install pytesseract
    ```

## Yolo Model Training
File requirements
1. .data - 
1. .names - 
1. train.txt - 
1. val.txt - 
1. .cfg - 
1. .conv.29 - 
1. data - 
    1. .jpg - 
    1. .txt - 

## Implementation
### Pipeline
1. Yolo license plate detection
1. OpenCV image pre-processing
1. OCR detect characters

### license Plate detection

### license Plate Recognition Usage

python license_recognition.py -m [mode] -i [source] --[gpu_usgage]

mode: image / video
soruce: file name (.jpg or .mp4)
gpu_usgage: `--gpu` or `--no-gpu`

#### Import Image
```bash
python license_recognition.py -m image -i res/test1.jpg --gpu
```
#### Import Video
```bash
python license_recognition.py -m video -i res/highway.mp4 --gpu
```

### OpenCV Image Process

### Optical Character Recognition


