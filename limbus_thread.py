# limbus_thread.py

# --- stdlib
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

# --- 3rd party
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab
import keyboard 
import time

# --- local imports (from your limbus_md_app.py / mirror_dungeon_bot_refactor.py)
from limbus_md_app import Region, OcrAgent, Bot # <-- this gives you click_text()

ENTER_REGION = Region(0.1969, 0.6269, 0.3844, 0.6833)
DIFF_REGION = Region(0.3016, 0.6000, 0.5938, 0.6694)
BATTLE_REGION = Region(0.8057, 0.7528, 0.9698, 0.8556)
CON_REGION = Region(0.7464, 0.6926, 0.9484, 0.8481)
P_ENTER_REGION = Region(0.0000, 0.6704, 0.9995, 0.9204)
orc = OcrAgent()
bot = Bot(model_path='limbus_train_model/mirror_dungeon_train16/weights/best.pt' , yolo_conf=0.0)

def enter_thread():
    orc.click_text("enter", ENTER_REGION); time.sleep(2)
    orc.click_text("difficult", DIFF_REGION ); time.sleep(2)
    orc.click_text("battle", BATTLE_REGION ); time.sleep(2)
    time.sleep(4)


def end_thread():
    orc.click_text("confirm", CON_REGION); time.sleep(1)
    

    


if __name__ == "__main__":
    
    run_num   = int(input("Enter how many runs: "))
    print("⏸ Waiting for Ctrl+G to start ...")
    keyboard.wait("ctrl+g")
    print("▶️ Starting!")

    while run_num > 0:
        while True:
            if orc.click_text("enter", ENTER_REGION, clicks=0, move_to=False):
                break
        
        enter_thread()
        
        while True:
            if orc.click_text("confirm", CON_REGION, clicks=0, move_to=False):
                break
            
            if bot.check_p_enter():
                bot.move_away()
                pyautogui.click()
                pyautogui.press('p'); time.sleep(0.2); pyautogui.press('enter'); time.sleep(3.0)
            elif bot.check_skip():
                #process skill check
                bot.click_skip(); time.sleep(3.0)
                bot.click_best_skill_check(); time.sleep(1)
                bot.click_commence(); time.sleep(3.5)
                bot.click_skip(); time.sleep(1)
                bot.click_continue()
        end_thread()
        run_num -= 1
                
            