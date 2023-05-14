import os
import cv2 as cv
import mediapipe as mp

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# read image
img = cv.imread(r'data\testImg.png')

H, W, _ = img.shape

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    cv.imshow('image', img)
    cv.waitKey(0)


# save image
cv.imwrite(os.path.join(output_dir, 'output.png'), img)