import cv2
import os
# from roboflow import Roboflow
# import supervision as sv

from ultralytics import YOLO

import hand_track as ht

# Read web cam
cap = cv2.VideoCapture(0)
# Chande resolution 1280x720
cap.set(3, 1280)
cap.set(4, 720)
# Counter
counter = 0
last_key_pressed = 0

# Declare detector
detector = ht.handDetector(conf_detetion=0.9)

# Read model
# rf = Roboflow(api_key="PlQ8PJLUTIKArvGxoLEI")
# project = rf.workspace().project("lsc-9cbqe")
# model = project.version(1).model
model = YOLO('runs/segment/train10/weights/best.pt')

while True:
    # Read cap
    ret, frame = cap.read()

    # Get hand data
    frame = detector.findHands(frame, draw=False)
    frame = cv2.flip(frame, 1)
    # Get position only a hand
    _, bbox, hand = detector.find_position(frame, hand_num=0, draw_points=False, draw_box=False, color=[0, 255,
                                                                                                           0])
    # Read keyboard
    key = cv2.waitKey(1)

    # If found some hand
    if hand == 1:
        # Get data from inside rectangule
        x_min, y_min, x_max, y_max = bbox
        # Draw rectangule with margin
        x_min = x_min - 40
        y_min = y_min - 40
        x_max = x_max + 40
        y_max = y_max + 40
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), [0, 255, 0], 2)

        # Make frame crop
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.any():
            crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
            # if last_key_pressed or (key == 13):
            # Rescale for best behavior en Yolo 8
            model_results = model.predict('/home/jaidiver/Desktop/sign_languaje/LSC.v2i.yolov8 ('
                                          '2)/train/images/1_3_jpg.rf.c32219fe10ee121366cc0812522b3326.jpg', conf=0.55)
            # image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # model_results = model.predict(image).json()

            # labels = [item["class"] for item in model_results["predictions"]]

            # detections = sv.Detections.from_roboflow(model_results)

            # label_annotator = sv.LabelAnnotator()
            # mask_annotator = sv.MaskAnnotator()

            # image = crop
            #
            # annotated_image = mask_annotator.annotate(
            #     scene=image, detections=detections)
            # annotated_image = label_annotator.annotate(
            #     scene=annotated_image, detections=detections, labels=labels)

            # cv2.imshow("HAND CROP", annotated_image)
            # sv.plot_image(image=annotated_image, size=(16, 16))
            if len(model_results) != 0:
                # for model_result in model_results:
                #     masks = model_result.masks
                #     coordinates = masks

                annotations = model_results[0].plot()
                crop = annotations
            cv2.imshow("HAND CROP", crop)

            # When Press some alphabet letter (a,b,c,..., z)
            if last_key_pressed or (96 < key < 123):
                if not last_key_pressed:
                    # Save last key pressed in uppercase
                    last_key_pressed = key - 32
                # Save images to data set
                letter_uppercase = chr(last_key_pressed)
                # Create folder data
                name = f"Letter_{letter_uppercase}"
                direction = './data'
                folder = direction + '/' + name

                # If folder had not been created
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    print('Folder created: ', folder)

                path = f"{folder}/{letter_uppercase}_{counter}.jpg"
                print(f"Save {letter_uppercase} symbol  in {path} successfull.")
                cv2.imwrite(path, crop)
                counter = counter + 1  # Increased counter

    # Adjust frame position and show FPS
    cv2.imshow("SING LANGUAGE", frame)

    if key == 27 or counter == 100:
        break
cap.release()
cv2.destroyAllWindows()
