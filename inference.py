# inference.py
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import os

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")
yolo_model.eval()

# Load trained classifier
class FusionModel(nn.Module):
    def __init__(self, gait_input_dim=6):
        super(FusionModel, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.gait_net = nn.Sequential(
            nn.Linear(gait_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img, gait):
        img_feat = self.cnn(img)
        gait_feat = self.gait_net(gait)
        combined = torch.cat((img_feat, gait_feat), dim=1)
        out = self.fc(combined)
        return out.squeeze(1)

# Load classifier model
clf_model = FusionModel()
clf_model.load_state_dict(torch.load("fusion_model.pt", map_location='cpu'))
clf_model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_gait_features(landmarks):
    try:
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        shoulder_width = np.linalg.norm([ls.x - rs.x, ls.y - rs.y])
        hip_width = np.linalg.norm([lh.x - rh.x, lh.y - rh.y])
        mid_shoulder = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2]
        mid_hip = [(lh.x + rh.x) / 2, (lh.y + rh.y) / 2]
        torso_height = np.linalg.norm([mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1]])
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]
        body_tilt = abs(np.arctan2(dx, dy) * 180 / np.pi)
        left_hand_height = abs(lw.y - ls.y)
        right_hand_height = abs(rw.y - rs.y)

        return [shoulder_width, hip_width, torso_height, body_tilt, left_hand_height, right_hand_height]
    except:
        return None

def run_inference(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = os.path.join("/tmp", "output.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    prev_gray = None
    prev_centers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = yolo_model(frame)[0]
        boxes = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                 for box in results.boxes if int(box.cls[0]) == 0]

        motion_boxes = []
        if prev_gray is not None and len(prev_centers) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, np.array(prev_centers, dtype=np.float32), None)
            for i, (new, old) in enumerate(zip(next_pts, prev_centers)):
                if status[i] == 1:
                    dist = np.linalg.norm(new - old)
                    if dist > 2 and i < len(boxes):
                        motion_boxes.append(boxes[i])
        else:
            motion_boxes = boxes

        prev_gray = frame_gray
        if boxes:
            prev_centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes], dtype=np.float32)
        else:
            prev_centers = np.array([], dtype=np.float32).reshape(0, 2)

        for x1, y1, x2, y2 in motion_boxes:
            person_crop = frame_rgb[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            result = pose.process(person_crop)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                gait_feats = extract_gait_features(landmarks)
                if gait_feats:
                    img = Image.fromarray(person_crop)
                    img_tensor = transform(img).unsqueeze(0)
                    gait_tensor = torch.tensor(gait_feats, dtype=torch.float).unsqueeze(0)

                    with torch.no_grad():
                        pred = clf_model(img_tensor, gait_tensor)
                        label = "Distracted" if pred.item() > 0.5 else "Not Distracted"

                    color = (0, 0, 255) if label == "Distracted" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_video_path
