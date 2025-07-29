import pytesseract
from pytesseract import Output
from PIL import ImageGrab
import cv2
import numpy as np
import pyautogui
import keyboard
import time
import sys
# Uncomment and set path if Tesseract isn't in your PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def jump_to_text(target="Enter"):
    print(f"🔍 Scanning for '{target}'...")
    img = np.array(ImageGrab.grab())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    for i, word in enumerate(data['text']):
        if target.lower() in word.lower():
            x = data['left'][i] + data['width'][i] // 2
            y = data['top'][i] + data['height'][i] // 2
            print(f"✅ Found '{word}' at ({x}, {y}) — clicking.")
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()
            return

    print(f"❌ '{target}' not found on screen.")
def main_loop():
    print("Press Ctrl + G to scan for 'Enter'. Press Ctrl + Q to quit.")
    while True:
        if keyboard.is_pressed("ctrl+g"):
            jump_to_text("Enter")
            time.sleep(1)  # prevent repeated triggers
        elif keyboard.is_pressed("ctrl+q"):
            print("👋 Exiting program.")
            sys.exit()

if __name__ == "__main__":
    main_loop()
