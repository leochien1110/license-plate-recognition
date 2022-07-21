import os
import cv2
import time
import imutils
import numpy as np
import random
import colorsys
import argparse
import pytesseract
import easyocr

reader = easyocr.Reader(['en'],gpu=True)

def tesseract_read(img, boxes):
    results =  []
    
    if len(boxes) > 0:
        for (x, y, w, h) in boxes:

            # extract the actual padded ROIq
            roi = img[y:y+h, x:x+w]

            config = ("-l eng --psm 7 --oem 3")
            # tell Tesseract to only OCR alphanumeric characters
            # alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            # config = "-c tessedit_char_whitelist={}".format(alphanumeric)

            text = pytesseract.image_to_string(roi, config=config)

            results.append(((x, y, x+w, y+h), text))
    
    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    output = img.copy()

    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("tesseract OCR TEXT")
        print("========")
        print("Original: {}\n".format(text))
        text_alnum = ''.join(filter(str.isalnum, text))
        print("Modified: {}\n".format(text_alnum))
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text_alnum = "".join([c if ord(c) < 128 else "" for c in text_alnum]).strip()
        
        cv2.rectangle(output, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(output, text_alnum, (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    # show the output image
    cv2.namedWindow("Tesseract Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tesseract Detection", 1280, 720)
    cv2.imshow("Tesseract Detection", output)

def EasyOCR_read(img, boxes,gpu=False):
    results =  []
    print('box len: ', len(boxes))
    if len(boxes) > 0:
        for (x, y, w, h) in boxes:

            # extract the actual padded ROIq
            roi = img[y:y+h, x:x+w]

            # cv2.imshow("cropped license plate", roi)
            # cv2.waitKey(0)

            result = reader.readtext(roi)
            for (bbox, text, prob) in result:
                # display the OCR'd text and associated probability
                print("[INFO] {:.4f}: {}".format(prob, text))
            
            if result:
                text = result[0][1]
            else :
                text = ""

            results.append(((x, y, x+w, y+h), text))
    
    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])
    
    output = img.copy()

    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("EasyOCR TEXT")
        print("========")
        print("Original: {}\n".format(text))
        text_alnum = ''.join(filter(str.isalnum, text))
        print("Modified: {}\n".format(text_alnum))
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text_alnum = "".join([c if ord(c) < 128 else "" for c in text_alnum]).strip()
        
        cv2.rectangle(output, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(output, text_alnum, (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    # show the output image
    cv2.namedWindow("EasyOCR Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EasyOCR Detection", 1280, 720)
    cv2.imshow("EasyOCR Detection", output)
            

def get_yolo_preds(net, img, confidence_threshold, overlapping_threshold, labels=None, frame_resize_width=1280):
    
    yolo_width_height = (416, 416)

    ln = net.getLayerNames()
    # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # CPU
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()] #GPU
    
    # if frame_resize_width:
        # img = imutils.resize(img, width=frame_resize_width)
    (H, W) = img.shape[:2]

    # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, yolo_width_height, swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                # Scale the bboxes back to the original image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))  # top-left corner
                y = int(centerY - (height / 2)) # bot-right corner
                boxes.append([x, y, int(width), int (height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Remove overlapping bounding boxes
    boxes_id = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlapping_threshold)
    # print(boxes)
    # print(len(bboxes))
    # print(bboxes)
    bboxes = []

    if len(boxes_id) > 0:
        for i in boxes_id.flatten():
            bboxes.append(boxes[i])
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            
            tmp = [labels[classIDs[i]], confidences[i], boxes[i]]
            print(tmp)

            # draw bounding box title background
            text_offset_x = x
            text_offset_y = y
            text_color = (255, 255, 255)
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1)[0]
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 80, text_offset_y - text_height + 4))
            cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)

            # draw bounding box title
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    cv2.namedWindow("YOLOv4 Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv4 Object Detection", 1280, 720)
    cv2.imshow("YOLOv4 Object Detection", img)

    print(bboxes)

    return bboxes


if __name__ == '__main__':

    # Environemnt Setup
    parser = argparse.ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-m", "--mode", dest="mode", type=str, required=True, 
                        help="[image] [video] Read from image or video")
    parser.add_argument("-i", "--input", dest="filename", required=True,
                        help="input file: [webcam] = 0 [video file] = path/to/video")
    parser.add_argument("-r", "--resize", type=int, default=1280,
                        help="resize width to speed up the process")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    print("---------User MODE---------")
    print("INPUT MODE: " + args.mode)
    print("INPUT DEST: " + args.filename)
    print("CUDA USAGE: " + str(args.gpu))
    print("IMG  WIDTH: " + str(args.resize))
    print("---------------------------\n")

    # Neural Network files
    with open("model/licence.names", "r", encoding="utf-8") as f:
        labels = f.read().strip().split("\n")

    yolo_cfg_path = "model/yolov4-tiny-licence.cfg"
    yolo_weights_path = "model/yolov4-tiny-licence.weights"

    useCuda = args.gpu

    net = cv2.dnn.readNet(yolo_cfg_path, yolo_weights_path)

    if useCuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if args.mode == "image":
        img = cv2.imread(args.filename)
        if img is None:
            print("Error opening image {}".format(args.filename))
        
        # resize and copy for detecion and recoginition
        img_yolo = imutils.resize(img, width=args.resize)
        img_ocr = imutils.resize(img, width=args.resize)

        boxes = get_yolo_preds(net, img_yolo, 0.6, 0.1, labels)
        # image binarize or hsv, keep white background only
        
        tesseract_read(img_ocr, boxes)
        EasyOCR_read(img_ocr,boxes,gpu=useCuda)
        
        print("Press any key to terminate the program...")
        key = cv2.waitKey(0)

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filename)
        try:
            if not cap.isOpened():
                print("Error opening vido")
                # quit()
            cv2.namedWindow("YOLOv4 Object Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv4 Object Detection", 1280, 720)

            counter = 0
            start = time.time()
            print("Video starting. Press ESC or 'q' to quit...")
            while True:
                
                ret, frame = cap.read()
                if not ret:
                    print("Video read Failed or ended, trying to loop back...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # break
                else:
                    counter += 1
                # resize and copy for detecion and recoginition
                img_yolo = imutils.resize(frame, width=args.resize)
                img_ocr = imutils.resize(frame, width=args.resize)

                boxes = get_yolo_preds(net, img_yolo, 0.6, 0.1, labels)

                tesseract_read(img_ocr, boxes)
                EasyOCR_read(img_ocr,boxes,gpu=useCuda)

                end = time.time()
                fps = counter / (end - start)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
        