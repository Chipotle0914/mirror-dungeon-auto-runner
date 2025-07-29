import easyocr
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import time
from pynput import mouse
from datetime import datetime

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=True)   

def find_and_hover_text_fullscreen(target):
    """Scan full screen for target word, move mouse to it, and return coordinates."""
    print(f"🔍 Scanning full screen for '{target}'...")

    # Grab full screen
    img = np.array(ImageGrab.grab())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # OCR detection
    results = reader.readtext(img)

    for (bbox, text, conf) in results:
        if target.lower() in text.lower():
            # Calculate center of bounding box
            x = int((bbox[0][0] + bbox[2][0]) / 2)
            y = int((bbox[0][1] + bbox[2][1]) / 2)
            print(f"✅ Found '{text}' at ({x}, {y}) with confidence {conf:.2f}")
            pyautogui.moveTo(x, y, duration=0.2)
            return (x, y)

    print(f"❌ '{target}' not found.")
    return None
def scan_relative_region_for_text_easyocr(target, x_ratio, y_ratio, offset=0.05):
    """
    Scans a square box (default 5%) around a reference point defined by ratio.
    """
    screen_w, screen_h = pyautogui.size()

    x1 = int(screen_w * (x_ratio - offset))
    y1 = int(screen_h * (y_ratio - offset))
    x2 = int(screen_w * (x_ratio + offset))
    y2 = int(screen_h * (y_ratio + offset))

    print(f"📷 Scanning around ({x_ratio:.4f}, {y_ratio:.4f}) → Box: ({x1}, {y1}) to ({x2}, {y2})")

    img = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(img)
    print(results)
    for (bbox, text, conf) in results:
        if target.lower() in text.lower():
            cx = int((bbox[0][0] + bbox[2][0]) / 2) + x1
            cy = int((bbox[0][1] + bbox[2][1]) / 2) + y1
            print(f"✅ Found '{text}' at ({cx}, {cy}) — confidence: {conf:.2f}")
            return (cx, cy)

    print(f"❌ '{target}' not found.")
    return None


def click_by_ratio(x_ratio, y_ratio):
    """
    Clicks a position on the screen based on (x_ratio, y_ratio),
    which are values between 0 and 1 representing relative screen position.
    """
    screen_w, screen_h = pyautogui.size()
    actual_x = int(screen_w * x_ratio)
    actual_y = int(screen_h * y_ratio)

    print(f"🖱️ Clicking at ({actual_x}, {actual_y}) from ratio ({x_ratio:.4f}, {y_ratio:.4f})")
    pyautogui.moveTo(actual_x, actual_y, duration=0.2)
    #pyautogui.click()


def scan_box_region_for_text_easyocr_ratio(target, x1_ratio, y1_ratio, x2_ratio, y2_ratio):
    """
    Scans a rectangular region defined by (x1_ratio, y1_ratio) to (x2_ratio, y2_ratio),
    where all values are ratios between 0 and 1, relative to screen size.
    """
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
            # pyautogui.click()  # Uncomment if you want it to click
            return (center_x, center_y)

    print(f"❌ '{target}' not found in region.")
    return None
#time.sleep(2)
#find_and_hover_text_fullscreen("skip")
#click_by_ratio( 0.8807, 0.8954)

#scan_box_region_for_text_easyocr("normal", 18, 884, 1407, 939)

def click_at_position(x, y, move_duration=0.2):
    pyautogui.moveTo(x, y, duration=move_duration)
    #pyautogui.click()
def on_click(x, y, button, pressed):
    if pressed:
        current_time = datetime.now().strftime("%H:%M:%S")

        if button.name == 'left':
            screen_w, screen_h = pyautogui.size()
            x_ratio = x / screen_w
            y_ratio = y / screen_h
            print(f"🖱️ Left-click at ({x}, {y}) | Ratio: ({x_ratio:.4f}, {y_ratio:.4f}) | Time: {current_time}")

        elif button.name == 'right':
            print(f"🛑 Right-click detected at {current_time} — exiting.")
            return False  # Stop listener
#pyautogui.moveTo(1681, 886, duration=0.2)
screen_w, screen_h = pyautogui.size()
#click_by_ratio( 0.8807, 0.8954)
#889, 467
print("📍 Left-click to print coordinates.")
print("🛑 Right-click to stop.")
"""
def skill_check():
    print("🔄 Checking for skill check level...")

    x1_ratio, y1_ratio = 0.0042, 0.8185
    x2_ratio, y2_ratio = 0.7849, 0.8759

    for level in ["very high", "high", "normal", "low", "very low"]:
        result = scan_box_region_for_text_easyocr_ratio(level, x1_ratio, y1_ratio, x2_ratio, y2_ratio)
        if result:
            print(f"🎯 Skill check level detected: {level.upper()}")
            return  # Exit after first match

    print("❌ No skill check on screen.")
    sys.exit()
skill_check()
"""
if (comm_cooords := scan_box_region_for_text_easyocr_ratio("commence", 0.7979, 0.8398, 0.9682, 0.9528)):
    click_at_position(*comm_cooords)
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
