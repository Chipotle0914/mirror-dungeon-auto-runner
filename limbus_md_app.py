import torch
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab
import time
import easyocr
import keyboard 
import sys
#yepeeee
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/mirror_dungeon_train10/weights/best.pt')
model.conf = 0.75 

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
GOOD_PACK = "good_pack"
BAD_PACK = "bad_pack"
ENTER_REGION = (0.7547, 0.6611, 0.9891, 0.8546)
TO_BATTLE_REGION = (0.8047, 0.7306, 0.9870, 0.9620)
TO_BATTLE_BACKUP_REGION = (0.7786, 0.5694, 0.9938, 0.9528)
CONFIRM_REGION = (0.0854, 0.5741, 0.9547, 0.9194)
SELECT_REGION = (0.7646, 0.7972, 0.9943, 0.9889)
REWARD_REGION = (0.0745, 0.1028, 0.8000, 0.2472)
P_ENTER_REGION = (0.6245, 0.6491, 0.9307, 0.8935)
SKIP_REGION = (0.3802, 0.1731, 0.5266, 0.7037)
COMMENCE_CONTINUE_PROCEED_REGION = (0.7438, 0.7713, 0.9927, 0.9667)
LEAVE_REGION = (0.7536, 0.7917, 0.9880, 0.9630)
SHOP_COMFIRM_REGION = (0.4167, 0.4620, 0.7792, 0.8481)
SKILL_CHECK_REGION = (0.0042, 0.8185, 0.7849, 0.8759)
ENDING_SCREEN_TOP_LEFT_REGION = (0.0005, 0.0907, 0.3203, 0.2815)
ENDING_SCREEN_VICTORY_REGION = (0.7438, 0.0620, 0.9547, 0.2583)
ENDING_SCREEN_CONFIRM = (0.6839, 0.6769, 0.9635, 0.9500)
BOSS_X_OFFSET = 0.2005
REFRESH_REGION = (0.6786, 0.0000, 0.9885, 0.1481)
DEFAULT_PRIOR = [REWARD_STAR, REWARD_MONEY, REWARD_RANDOM, REWARD_GAMBLE, REWARD_RESOURCE]

#move cursor to targeted class and click
def yolo_detect_click(target_class: str, click_num: int = 1, top_most: bool = False, drag_down: bool = False, move_to: bool = True):
    print(f"🔍 Searching for '{target_class}' using YOLO...")
    
    # Capture screenshot
    screenshot = np.array(ImageGrab.grab())
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model(screenshot_rgb)
    detections = results.pandas().xyxy[0]

      # Filter for target class and confidence threshold
    match = detections[
        (detections['name'] == target_class) &
        (detections['confidence'] >= model.conf)
    ]

    if match.empty:
        print(f"❌ '{target_class}' not found.")
        return False

    ####DEBUG
    match = match.sort_values(by='confidence', ascending=False)
    det_conf = match.iloc[0]['confidence']

    if target_class in [GOOD_PACK, BAD_PACK, REWARD_STAR, REWARD_STAR, REWARD_MONEY, REWARD_RANDOM, REWARD_GAMBLE, REWARD_RESOURCE]:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        debug_img_path = f"debug_{target_class}_{timestamp}.png"
        cv2.imwrite(debug_img_path, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Saved debug screenshot to {debug_img_path}")
        print(f"🔎 Confidence for '{target_class}': {det_conf:.2f} (threshold: {model.conf})")
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

    # 🔍 SHOP-specific condition: check if left of TRAIN
    if target_class == SHOP:
        train_match = detections[
            (detections['name'] == TRAIN) & 
            (detections['confidence'] >= model.conf)
        ]
        if train_match.empty:
            print("⚠️ SHOP found but TRAIN not detected — skipping SHOP.")
            return False
        train = train_match.sort_values(by='confidence', ascending=False).iloc[0]
        train_x1, train_x2 = train[['xmin', 'xmax']]
        train_center_x = int((train_x1 + train_x2) / 2)

        if center_x < train_center_x:
            print(f"❌ Ignored SHOP — not left of TRAIN (SHOP X: {center_x}, TRAIN X: {train_center_x})")
            return False
        else:
            print(f"✅ SHOP is left of TRAIN → proceeding (SHOP X: {center_x}, TRAIN X: {train_center_x})")

    print(f"✅ Found '{target_class}' at ({center_x}, {center_y})")

    if move_to:
        pyautogui.moveTo(center_x, center_y, duration=0.2)
    
    #drag down
    if drag_down:
        pyautogui.mouseDown(center_x, center_y)
        pyautogui.moveTo(center_x, center_y + 400, duration=0.2)
        pyautogui.mouseUp()
        print("Successfully Selected Theme pack!")
    else:
    #click for click_num
        for _ in range(click_num):
            pyautogui.click()
            time.sleep(0.3)
        print(f"🖱️ Clicked on '{target_class}'")

    return True

#helper functinon for scan_box_click_text
def word_centers_x(text_line: str, words: list, box_left: float, box_width: float) -> list:
    """
    Map each token's center character position in text_line to an x coord inside the bbox.
    Robust to uneven token lengths and punctuation (e.g., '[', 'low]').
    """
    centers = []
    cursor = 0
    L = max(1, len(text_line))  # avoid div-by-zero

    for i, w in enumerate(words):
        # Find this token at/after the current cursor
        start = text_line.find(w, cursor)
        if start == -1:
            # Fallback: try from beginning (OCR spacing/merging can shift)
            start = text_line.find(w)
            if start == -1:
                # Last resort: equal-slot approximation for this position
                segments = max(1, len(words))
                frac = (i + 0.5) / segments
                centers.append(box_left + box_width * frac)
                continue

        end = start + len(w)
        center_char = (start + end) / 2.0
        frac = center_char / L  # 0..1 across the full text line
        x = box_left + box_width * frac
        centers.append(x)

        # Advance cursor to just after this token
        cursor = end

    return centers




# scan an area of screen with easyocr to find specific text
# scan an area of screen with easyocr to find specific text
def scan_box_click_text(target, region, moveTo_flag=1, click_flag=1, y_offset=0):
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = region
    screen_w, screen_h = pyautogui.size()

    # Convert ratios to absolute screen coordinates
    x1 = int(screen_w * x1_ratio)
    y1 = int(screen_h * y1_ratio)
    x2 = int(screen_w * x2_ratio)
    y2 = int(screen_h * y2_ratio)

    # Convert vertical offset (fraction of screen height) to pixels
    y_offset = y_offset * screen_h

    print(f"📦 Scanning box region: ({x1}, {y1}) to ({x2}, {y2})")

    img = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(img)
    print(results)

    for (bbox, text, conf) in results:
        text_lower = text.lower().strip()
        target_lower = target.lower().strip()

        if target_lower in text_lower:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            box_left = min(xs)
            box_right = max(xs)
            box_width = box_right - box_left
            box_height_center = (min(ys) + max(ys)) / 2

            # Tokenize OCR line and target
            words = text_lower.split()
            print("all detected words:", words)
            target_arr = target_lower.split()

            # ---- Determine idx (which token to click) ----
            idx = None
            if len(target_arr) > 1:
                # Two-word contiguous match: [first, second]
                first, second = target_arr[0], target_arr[1]
                for i in range(len(words) - 1):
                    if first in words[i] and second in words[i + 1]:
                        idx = i  # click starting word of the pair
                        break
            else:
                # Single word: first token that contains it
                single = target_arr[0]
                for i, w in enumerate(words):
                    if single in w:
                        idx = i
                        break

            # If we didn't find the sequence/word in this OCR region, check next result
            if idx is None:
                continue

            # Clamp idx just in case
            idx = max(0, min(idx, len(words) - 1))
            print(f"🎯 Matched token index: {idx} (token='{words[idx]}')")

            # ---- Map token centers to x coords and choose click_x ----
            centers_x = word_centers_x(text_lower, words, box_left, box_width)
            if idx < len(centers_x):
                click_x = centers_x[idx]
            else:
                # Fallback: equal-slot center
                frac = (idx + 0.5) / max(1, len(words))
                click_x = box_left + box_width * frac

            click_y = box_height_center

            # Convert to absolute screen coords
            click_x += x1
            click_y += y1

            # Apply vertical offset
            click_y += y_offset

            print(f"✅ Found '{text}' → clicking at ({click_x:.0f}, {click_y:.0f}) | conf={conf:.2f}")

            if moveTo_flag:
                pyautogui.moveTo(click_x, click_y, duration=0.2)

            for _ in range(click_flag):
                pyautogui.click()
                time.sleep(0.3)
                print(f"🖱️ Clicked on '{target}'")
            return True

    print(f"❌ '{target}' not found in region.")
    return False


#clicking functions
def click_enter():
    scan_box_click_text("enter", ENTER_REGION)


def click_to_battle():
    return scan_box_click_text("battle", TO_BATTLE_REGION)

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

def check_leave():
    return scan_box_click_text("leave", LEAVE_REGION, 0, 0)

def check_ending():
    #scan_box_click_text("confirm", ENDING_SCREEN_CONFIRM)
    if scan_box_click_text("contributed", ENDING_SCREEN_TOP_LEFT_REGION, 1, 0):
        scan_box_click_text("confirm", ENDING_SCREEN_CONFIRM)
        return True
    else:
        return False

def check_skip():
    return scan_box_click_text("skip", SKIP_REGION, 0, 0)

def click_skip_5_times():
    return scan_box_click_text("skip", SKIP_REGION, 1, 5)

def click_continue():
    return scan_box_click_text("continue", COMMENCE_CONTINUE_PROCEED_REGION)

def click_commence():
    return scan_box_click_text("commence", COMMENCE_CONTINUE_PROCEED_REGION)

def click_proceed():
    return scan_box_click_text("proceed", COMMENCE_CONTINUE_PROCEED_REGION)

def click_leave():
    return scan_box_click_text("leave", LEAVE_REGION)

def click_shop_confirm():
    return scan_box_click_text("confirm", SHOP_COMFIRM_REGION, y_offset = -0.0224)

def check_to_battle():
    return scan_box_click_text("battle", TO_BATTLE_REGION, 0, 0)

def click_refresh():
    return scan_box_click_text("refresh", REFRESH_REGION)

def click_boss():
    screen_w, screen_h= pyautogui.size()
    x, y = pyautogui.position()
    #move to boss room
    target_x = x + int(BOSS_X_OFFSET * screen_w)
    #adjust slighly downwards to make sure it always hits the boss room
    y += (0.0324 * screen_h)
    pyautogui.moveTo(target_x, y, duration=0.2)
    pyautogui.click()

def check_skill_check():
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
        if yolo_detect_click(target):
            print(f"🎯 Picked reward: {target}")
            return target
    print("❌ No preferred rewards found.")
    return None
#reset the view of the level
def reset_view():
    screen_w, screen_h = pyautogui.size()
    #center bottom coordinates
    x_ratio = 0.4703
    y_ratio = 0.6333

    x = x_ratio * screen_w
    y = y_ratio * screen_h

    pyautogui.moveTo(x, y, duration=0.2)
    for _ in range(1):
        pyautogui.scroll(-50)
        time.sleep(0.2)

def move_away():
    screen_w, screen_h = pyautogui.size()
    #center bottom coordinates
    x_ratio = 0.1094
    y_ratio = 0.8759

    x = x_ratio * screen_w
    y = y_ratio * screen_h

    pyautogui.moveTo(x, y, duration=0.2)



def process_fight(boss_fight=False, skip_to_battle=False):
    # start the battle
    
    #skip battle if it misread fight stage as question
    if not skip_to_battle:
        while True:
            if click_to_battle() or scan_box_click_text("clear", TO_BATTLE_BACKUP_REGION, 1, 0, 0.1537):
                pyautogui.click()
                time.sleep(0.3)
                break
    
    #time sleep here so it stop accidetnally misread themepacks and ego gifts
    time.sleep(1.5)
    #fight until encounter end of fights scenarios
    while True:
            
        # 🛑 Exit for regular fights: check for train
        if not boss_fight and yolo_detect_click(TRAIN, 0):
            time.sleep(1)
            if yolo_detect_click(TRAIN, 0):
                break
            else:
                continue
        # 🛑 Exit for boss fight: check theme packs
        # some how it was possible to find a pack during a fight
        if boss_fight and (yolo_detect_click(GOOD_PACK, 0, move_to = False) or yolo_detect_click(BAD_PACK, 0, move_to = False)):
            #do a double check
            time.sleep(1.5)
            if not (yolo_detect_click(GOOD_PACK, 0, move_to = False) or yolo_detect_click(BAD_PACK, 0, move_to = False)):
                continue
            print("🛑 Boss fight ended — good or bad pack found.")
            break
            
        # Scen2: pick reward choice
        elif check_reward():
            time.sleep(3)  # wait for cards to load
            
            picked_class = click_reward_prior()  # now returns class name or None

            if picked_class:
                time.sleep(1)

                # Special case: gamble or random ego gift might trigger extra confirm
                if picked_class in [REWARD_GAMBLE, REWARD_RANDOM]:
                    click_confirm()
                    time.sleep(1)
                    print(f"⏳ Waiting for confirm prompt due to special reward: {picked_class}")
                    if check_confirm():
                        click_confirm()
                        time.sleep(3)
                        move_away()
                        continue
                else:
                    click_confirm()
                    time.sleep(3)
                    move_away()
                    continue

            # Fallback confirm if no reward was picked
            elif check_confirm():
                click_confirm()
                time.sleep(1)
                continue
        # Scen3: accept ego gift by select + confirm
        #adding continues because sometimes the ego won't be clicked
        elif yolo_detect_click(ACQUIRE_EGO):
            time.sleep(0.8)  # make sure ego is selected
            if not click_select():
                continue

            time.sleep(1.5)  # wait for confirm button to show
            if not click_confirm():
                #confirm didn't show after the click
                pyautogui.mouseDown(button='left')
                time.sleep(0.001)
                pyautogui.mouseUp(button='left') 
                if check_confirm():
                    click_confirm()
                    print("Second Select click: Success✅")
                    continue
                print("Second Select click: failed ❌")
                print("Third Select click: attempting")
                #try moving away then come back to click
                move_away()
                scan_box_click_text("select", SELECT_REGION, click_flag = 0, moveTo_flag = 1)
                pyautogui.mouseDown(button='left')
                time.sleep(0.001)
                pyautogui.mouseUp(button='left')
                continue
            move_away()
            time.sleep(3)
            continue
        #Scen4: accept ego gift by pressing confirm(or enter)
        elif check_confirm():
            if not(check_reward()):
                #beofore click check if it's final stage
                   #Exit for ending the run
                if boss_fight and check_ending():
                    print("CONGRATS FINISHING A WHOLE RUN!")
                    return True
                else:
                    click_confirm()
                    time.sleep(1)
                    move_away()
                    continue

        #Scen5: deal with skill check boss
        elif check_skip():
            #process skill check without choice A, B
            if not click_skip_5_times():
                time.sleep(1)
                click_skip_5_times()
            #do skil check
            if check_skill_check():
                click_best_skill_check()
                time.sleep(1)
            #click commence
            if click_commence():
                time.sleep(3)
            else:
                sys.exit("During battle, can't find commence")
            if not click_skip_5_times():
                time.sleep(1)
                click_skip_5_times()
            #click continue
            if click_continue():
                time.sleep(1.5)
                continue
            else:
                sys.exit("During battle, can't find continue")
            

        else:
            if check_p_enter():
                screen_w, screen_h = pyautogui.size()
                #center bottom coordinates
                x_ratio = 0.1094
                y_ratio = 0.8759

                x = x_ratio * screen_w
                y = y_ratio * screen_h

                pyautogui.moveTo(x, y, duration=0.2)
                #to continue with boss fight that has skill check
                pyautogui.click()
                pyautogui.press('p')
                time.sleep(0.2)
                pyautogui.press('enter')
                time.sleep(3)
            else:
                continue

def process_question():
    #always start from clicking skip
    click_skip_5_times()
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
            if not click_skip_5_times():
                #check again
                time.sleep(1)
                click_skip_5_times()
            
            time.sleep(1.5)
            continue
        #click commence
        elif click_commence():
            time.sleep(3)
            if not click_skip_5_times():
                #check again
                time.sleep(1)
                click_skip_5_times()
            time.sleep(1.5)
            continue
        #click continue(continue is always the last step)
        elif click_continue():
            time.sleep(1)
            continue
        #Scen2: check A choices first
        elif yolo_detect_click(A_CHOICES, top_most=True):
            if not click_skip_5_times():
                #check again
                time.sleep(1)
                click_skip_5_times()
            time.sleep(1)
            continue
        #Scen2: check B choices Second
        elif yolo_detect_click(B_CHOICES, top_most=True):
            if not click_skip_5_times():
                #check again
                time.sleep(1)
                click_skip_5_times()
            time.sleep(1)
            continue
       
        #skill check
        elif check_skill_check():
            click_best_skill_check()
            time.sleep(1)
            continue
        #if choice A/B leads to battle
        elif check_to_battle():
            process_fight()
            break
        #misread any fight stages as question
        elif check_p_enter():
            process_fight(skip_to_battle=True)
            break
        
        elif check_leave():
            break
        else:
            click_skip_5_times()
            time.sleep(1)


def process_start_to_shop():
    #auto run util shop
    while True:
        if yolo_detect_click(SHOP) or check_leave():
            print("MD run till shop successful!")
            break
        #check twice
        elif yolo_detect_click(SHOP):
            print("MD run till shop successful!")
            break
        elif yolo_detect_click(QUESTION):
            time.sleep(1)
            click_enter()
            time.sleep(1.5)
            process_question()
            reset_view()
            time.sleep(0.5)
        elif yolo_detect_click(FIGHT):
            time.sleep(1)
            click_enter()
            time.sleep(0.8)
            process_fight()
            reset_view()
            time.sleep(1) 

def process_shop_boss_packs():
    #enter shop
    click_enter()
    time.sleep(1)
    #click leave
    click_leave()
    time.sleep(1)
    #click confirm
    click_shop_confirm()
    time.sleep(2)
    #find train
    yolo_detect_click(TRAIN, 0)
    #move right to click on boss
    click_boss()
    #wait for boss
    time.sleep(1)
    #click enter
    click_enter()
    #process boss fight
    if process_fight(boss_fight = True):
        return True

    #wait for packs to load
    time.sleep(2)
   #if all packs are bad scenario
    count = 0
    while count < 3 and ((not yolo_detect_click(GOOD_PACK, 0, move_to = False)) and yolo_detect_click(BAD_PACK, 0, move_to = False)):
        click_refresh()
        time.sleep(3)
        count += 1

    move_away()

    yolo_detect_click(GOOD_PACK, drag_down = True)
    time.sleep(3)
    return False

#def train_testing():



if __name__ == "__main__":
    #whether to stop at first shop
    shop_flag = int(input("Enter 1 to stop after entering shop, 0 to continue: "))

    while True:
        if scan_box_click_text("enter", ENTER_REGION, 0, 0):
            print("processing boss until next level!!!")  
            if process_shop_boss_packs():
                print("We DONE!!!")
                break           # Run post-shop boss path and process boss fight
            time.sleep(2) 
        else:
            print("processing until shop!!!")
            process_start_to_shop()  # Run through Mirror Dungeon normally until shop
            if shop_flag:
                print("User wants to stop at shop")
                break
            time.sleep(2)




       
        