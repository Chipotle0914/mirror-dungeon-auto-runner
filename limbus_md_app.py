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
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/mirror_dungeon_train8/weights/best.pt')
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
TO_BATTLE_REGION = (0.8047, 0.7306, 0.9870, 0.9620)
CONFIRM_REGION = (0.0854, 0.5741, 0.9547, 0.9194)
SELECT_REGION = (0.7646, 0.7972, 0.9943, 0.9889)
REWARD_REGION = (0.0745, 0.1028, 0.8000, 0.2472)
P_ENTER_REGION = (0.6245, 0.6491, 0.9307, 0.8935)
SKIP_REGION = (0.3802, 0.1731, 0.5266, 0.7037)
COMMENCE_CONTINUE_PROCEED_REGION = (0.7438, 0.7713, 0.9927, 0.9667)
SKILL_CHECK_REGION = (0.0042, 0.8185, 0.7849, 0.8759)

DEFAULT_PRIOR = [REWARD_STAR, REWARD_MONEY, REWARD_GAMBLE, REWARD_RANDOM, REWARD_RESOURCE]
#move cursor to targeted class and click
def yolo_detect_click(target_class: str, click_num: int = 1, top_most: bool = False):
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

    ####DEBUG
    if target_class == ACQUIRE_EGO:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        debug_img_path = f"debug_acquire_ego_{timestamp}.png"
        cv2.imwrite(debug_img_path, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Saved debug screenshot to {debug_img_path}")
    ####DEBUG
    # Get target box
    if top_most:
        match = match.sort_values(by='ymin').iloc[0]
        print("↕️ Top-most mode enabled")
    else:
        match = match.iloc[0]

    # Get first matching box
    x1, y1, x2, y2 = match[['xmin', 'ymin', 'xmax', 'ymax']]
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    print(f"✅ Found '{target_class}' at ({center_x}, {center_y})")

    pyautogui.moveTo(center_x, center_y, duration=0.2)
    
    #click for click_num
    for _ in range(click_num):
        pyautogui.click()
        time.sleep(0.2)
    print(f"🖱️ Clicked on '{target_class}'")

    return True

#scan a area of screen with easyorc to find specic text
def scan_box_click_text(target, region, moveTo_flag = 1,click_flag = 1):

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
            
            if moveTo_flag:
                pyautogui.moveTo(center_x, center_y, duration=0.2)
            
            for _ in range(click_flag):
                pyautogui.click()
                time.sleep(0.2)
                print(f"🖱️ Clicked on '{target}'")
            return True

    print(f"❌ '{target}' not found in region.")
    return False


#clicking functions
def click_enter():
    scan_box_click_text("enter", ENTER_REGION)


def click_to_battle():
    scan_box_click_text("battle", TO_BATTLE_REGION)

def click_confirm():
    return scan_box_click_text("confirm", CONFIRM_REGION)

def check_confirm():
    return scan_box_click_text("confirm", CONFIRM_REGION, 0, 0)

def click_select():
    return scan_box_click_text("select", SELECT_REGION)

def check_reward():
    return scan_box_click_text("reward", REWARD_REGION, 0, 0)

def check_p_enter():
    return scan_box_click_text("win", P_ENTER_REGION, 0, 0) or scan_box_click_text("damage", P_ENTER_REGION, 0, 0)

def click_skip_5_times():
    return scan_box_click_text("skip", SKIP_REGION, 1, 5)

def click_continue():
    return scan_box_click_text("continue", COMMENCE_CONTINUE_PROCEED_REGION)

def click_commence():
    return scan_box_click_text("commence", COMMENCE_CONTINUE_PROCEED_REGION)

def click_proceed():
    return scan_box_click_text("proceed", COMMENCE_CONTINUE_PROCEED_REGION)

def check_skil_check():
    for best_skill in ["very high", "high", "normal", "low", "very low"]:
        found = scan_box_click_text(best_skill, SKILL_CHECK_REGION, 0, 0)
        if found:
            print(f"🎯 Skill check level detected: {best_skill}")
            return True 
    return False

def click_best_skill_check():
    for best_skill in ["very high", "high", "normal", "low", "very low"]:
        found = scan_box_click_text(best_skill, SKILL_CHECK_REGION)
        if found:
            print(f"🎯 Skill check level clicked: {best_skill}")
            return True 
    sys.exit("❌ No skill check to click on screen.")

def click_reward_prior():
    for target in DEFAULT_PRIOR:
        found = yolo_detect_click(target)
        if found:
            print(f"🎯 Picked reward: {target}")
            return True
    print("❌ No preferred rewards found.")
    return False

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


def process_fight():
    # start the battle
    click_to_battle()
    #fight until encounter end of fights scenarios
    while True:
        #Scen1: go back to train
        if yolo_detect_click(TRAIN, 0):
            time.sleep(1)
            if yolo_detect_click(TRAIN, 0):
                break
            else:
                continue
        #Scen2: pick reward choice
        elif check_reward():
            #wait for cards to load
            time.sleep(1.5)
            #choose based on priority
            if click_reward_prior():
                time.sleep(1)
                click_confirm()
                time.sleep(3)
                continue
            else:
                #the 33 percent where we still are on the reward screen, but we need to click confirm
                if check_confirm():
                    click_confirm()
                    time.sleep(1)
                    continue
        #Scen3: accept ego gift by select + confirm
        elif yolo_detect_click(ACQUIRE_EGO):
            if click_select():
                #it takes time for confirm button to show
                time.sleep(1.5)
                if click_confirm():
                    time.sleep(2)
                    continue
                else:
                    sys.exit("ACQUIRE_EGO Error: confirm")

            else:
                sys.exit("ACQUIRE_EGO Error: select")
        #Scen4: accept ego gift by pressing confirm(or enter)
        elif check_confirm():
            if not(check_reward()):
                click_confirm()
                time.sleep(1)
                continue

            
        else:
            if check_p_enter():
                pyautogui.press('p')
                time.sleep(0.2)
                pyautogui.press('enter')
                time.sleep(3)
            else:
                continue

def process_question():
    #enter qustion
    while True:
        #Scen1: go back to train      
        if yolo_detect_click(TRAIN, 0):
            time.sleep(1.5)
            if yolo_detect_click(TRAIN, 0):
                break
            else:
                continue
        #Scen3: click confirm for ego gift after the question stage
        elif check_confirm():
            click_confirm()
            time.sleep(1)
            continue
        # click proceed
        elif click_proceed():
            click_skip_5_times()
            time.sleep(1.5)
            continue
        #click commence
        elif click_commence():
            time.sleep(3)
            click_skip_5_times()
            time.sleep(1.5)
            continue
        #click continue(continue is always the last step)
        elif click_continue():
            time.sleep(1)
            continue
        #Scen2: check A choices first
        elif yolo_detect_click(A_CHOICES, top_most=True):
            click_skip_5_times()
            time.sleep(1)
            continue
        #Scen2: check B choices Second
        elif yolo_detect_click(B_CHOICES, top_most=True):
            click_skip_5_times()
            time.sleep(1)
            continue
       
        #skill check
        elif check_skil_check():
            click_best_skill_check()
            time.sleep(1)
            continue
        else:
            click_skip_5_times()
            time.sleep(1)


def test_features():
    click_enter()
    time.sleep(2)
    #process_question()
    process_fight()
if __name__ == "__main__":
    time.sleep(2)
    test_features()