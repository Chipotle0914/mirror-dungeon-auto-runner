import torch
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab
import time
import easyocr
import keyboard 
import sys

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/mirror_dungeon_train7/weights/best.pt')
model.conf = 0.7 

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=True)

# Set up all Macros
FIGHT = "fight_stage"
QUESTION = "question_stage"
COMPLETED = "completed_stage"
TRAIN = "train"
SHOP = "shop"
ACQUIRE_EGO = "acquire_ego_gift"
REWARD_MONEY = "reward_money"
REWARD_GAMBLE = "reward_gamble"
REWARD_RESOURCE = "reward_ego_resource"
REWARD_RANDOM = "reward_random_ego"
REWARD_STAR = "reward_star"
A_CHOICES = "A_choices"
B_CHOICES = "B_choices"
ENTER_REGION = (0.7547, 0.6611, 0.9891, 0.8546)


#move cursor to targeted class and click
def yolo_detect_click(target_class: str, click_num: int = 1):
    print(f"🔍 Searching for '{target_class}' using YOLO...")
    
    # Capture screenshot
    screenshot = np.array(ImageGrab.grab())
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model(screenshot_rgb)
    detections = results.pandas().xyxy[0]

    # Filter for target class
    match = detections[detections['name'] == target_class]

    if match.empty:
        print(f"❌ '{target_class}' not found.")
        return False

    # Get first matching box
    x1, y1, x2, y2 = match.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']]
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    print(f"✅ Found '{target_class}' at ({center_x}, {center_y})")

    pyautogui.moveTo(center_x, center_y, duration=0.2)
    
    #click for click_times
    for _ in range(click_times):
        pyautogui.click()
        time.sleep(0.2)
    print(f"🖱️ Clicked on '{target_class}'")

    return True

#scan a area of screen with easyorc to find specic text
def scan_box_click_text(target, region):

    x1_ratio, y1_ratio, x2_ratio, y2_ratio = region
    screen_w, screen_h = pyautogui.size()

    # Convert ratios to absolute screen coordinates
    x1 = int(screen_w * x1_ratio)
    y1 = int(screen_h * y1_ratio)
    x2 = int(screen_w * x2_ratio)
    y2 = int(screen_h * y2_ratio)

    print(f"📦 Scanning box region: ({x1}, {y1}) to ({x2}, {y2})")

    img = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(img)
    print(results)
    for (bbox, text, conf) in results:
        if target.lower() in text.lower():
            center_x = int((bbox[0][0] + bbox[2][0]) / 2) + x1
            center_y = int((bbox[0][1] + bbox[2][1]) / 2) + y1
            print(f"✅ Found '{text}' at ({center_x}, {center_y}) with confidence {conf:.2f}")
            pyautogui.moveTo(center_x, center_y, duration=0.2)
           # pyautogui.click()
            return True

    print(f"❌ '{target}' not found in region.")
    return False


def click_enter():
    scan_box_click_text("enter", ENTER_REGION)


#reset the view of the level
def reset_view():
    screen_w, screen_h = pyautogui.size()
    #center bottom coordinates
    x_ratio = 0.4703
    y_ratio = 0.6333

    x = x_ratio * screen_w
    y = y_ratio * screen_h

    pyautogui.moveTo(x, y, duration=0.2)
    for _ in range(3):
        pyautogui.scroll(-500)
        time.sleep(0.2)

time.sleep(2)
#reset_view()
#yolo_detect_click(FIGHT)

def test_features():
    click_enter()


if __name__ == "__main__":
    time.sleep(2)
    test_features()