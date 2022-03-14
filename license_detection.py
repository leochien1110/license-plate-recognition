import os
import cv2
import time
import imutils
import numpy as np
import random
import colorsys
import argparse

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def get_yolo_preds(net, img, confidence_threshold, overlapping_threshold, labels=None, frame_resize_width=1280):
    
    yolo_width_height = (416, 416)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    if frame_resize_width:
        img = imutils.resize(img, width=frame_resize_width)
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
    bboxes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlapping_threshold)
    if len(bboxes) > 0:
        for i in bboxes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])

            # draw bounding box title background
            text_offset_x = x
            text_offset_y = y
            text_color = (255, 255, 255)
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1)[0]
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 80, text_offset_y - text_height + 4))
            cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)

            # draw bounding box title
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img


if __name__ == '__main__':

    # Environemnt Setup
    parser = argparse.ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-m", "--mode", dest="mode", type=str, required=True, 
                        help="[image] [video] Read from image or video")
    parser.add_argument("-i", "--input", dest="filename", required=True,
                        help="input file: [webcam] = 0 [video file] = path/to/video")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    print("---------User MODE---------")
    print("INPUT MODE: " + args.mode)
    print("INPUT DEST: " + args.filename)
    print("CUDA USAGE: " + str(args.gpu))
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
        cv2.namedWindow("YOLOv4 Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv4 Object Detection", 1920, 1080)

        img = get_yolo_preds(net, img, 0.6, 0.1, labels, 1280)
        
        cv2.imshow("YOLOv4 Object Detection", img)

        print("Press any key to terminate the program...")
        key = cv2.waitKey(0)

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filename)
        try:
            if not cap.isOpened():
                print("Error opening vido")
                # quit()
            cv2.namedWindow("YOLOv4 Object Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv4 Object Detection", 1920, 1080)

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
                
                frame = get_yolo_preds(net, frame, 0.6, 0.1, labels, 1280)

                end = time.time()
                fps = counter / (end - start)
                cv2.putText(frame, "FPS: {:.4}".format(fps), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                cv2.imshow("YOLOv4 Object Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
        