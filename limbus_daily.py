# OCR (EasyOCR)
import easyocr

# Mouse movement, clicking, keyboard pressing, screenshots
import pyautogui

# Keyboard library (optional, better for key hold/release or global hotkeys)
import keyboard

# Image processing / template matching
import cv2
import numpy as np

# Delay pauses
import time

# Better screenshot capture performance (optional)
from mss import mss

from typing import Optional, Tuple, List, Iterable
from dataclasses import dataclass

def sleep_s(sec: float) -> None:
    time.sleep(sec)
def log(msg: str) -> None:
    print(msg, flush=True)

@dataclass
class Region:
    x1: float; y1: float; x2: float; y2: float  # ratios (0..1)

    def abs_coords(self) -> Tuple[int, int, int, int]:
        sw, sh = pyautogui.size()
        return (int(self.x1 * sw), int(self.y1 * sh),
                int(self.x2 * sw), int(self.y2 * sh))

class OcrAgent:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.sct = mss() 

    @staticmethod
    def _token_centers(text_line: str, tokens: List[str], left: float, width: float) -> List[float]:
        centers = []
        cursor = 0
        L = max(1, len(text_line))
        for i, w in enumerate(tokens):
            start = text_line.find(w, cursor)
            if start == -1:
                start = text_line.find(w)
                if start == -1:
                    seg = max(1, len(tokens))
                    frac = (i + 0.5) / seg
                    centers.append(left + width * frac)
                    continue
            end = start + len(w)
            center_char = (start + end) / 2.0
            frac = center_char / L
            centers.append(left + width * frac)
            cursor = end
        return centers

    def click_text(
        self,
        target: str,
        region: Region,
        move_to: bool = True,
        clicks: int = 1,
        y_offset_frac: float = 0.0
    ) -> bool:
        (x1, y1, x2, y2) = region.abs_coords()
        xoff = 0
        yoff = int(y_offset_frac * pyautogui.size()[1])

        log(f"📦 Scanning box region: ({x1}, {y1}) to ({x2}, {y2})")
        monitor = {
            "top": y1,
            "left": x1,
            "width": x2 - x1,
            "height": y2 - y1
        }
        img = np.array(self.sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) 

        results = self.reader.readtext(img)
        log(str(results))

        target_lower = target.lower().strip()

        for (bbox, text, conf) in results:
            text_lower = text.lower().strip()
            if target_lower in text_lower:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                left = min(xs); right = max(xs)
                width = right - left
                y_center = (min(ys) + max(ys)) / 2

                tokens = text_lower.split()
                log(f"all detected words: {tokens}")
                target_tokens = target_lower.split()

                idx = None
                if len(target_tokens) > 1:
                    a, b = target_tokens[0], target_tokens[1]
                    for i in range(len(tokens) - 1):
                        if a in tokens[i] and b in tokens[i + 1]:
                            idx = i
                            break
                else:
                    single = target_tokens[0]
                    for i, tk in enumerate(tokens):
                        if single in tk:
                            idx = i
                            break
                if idx is None:
                    continue

                idx = max(0, min(idx, len(tokens) - 1))
                log(f"🎯 Matched token index: {idx} (token='{tokens[idx]}')")

                centers_x = self._token_centers(text_lower, tokens, left, width)
                click_x_local = centers_x[idx] if idx < len(centers_x) else left + width * ((idx + 0.5) / max(1, len(tokens)))

                click_x = x1 + int(click_x_local) + xoff
                click_y = y1 + int(y_center) + yoff

                log(f"✅ Found '{text}' → clicking at ({click_x}, {click_y}) | conf={conf:.2f}")
                if move_to:
                    pyautogui.moveTo(click_x, click_y, duration=0.2)
                for _ in range(max(0, clicks)):
                    pyautogui.click()
                    sleep_s(0.3)
                log(f"🖱️ Clicked on '{target}'" if clicks > 0 else f"👀 Hovered '{target}'")
                return True

        log(f"❌ '{target}' not found in region.")
        return False

    def click_points(
            self,
            points: Iterable[Tuple[float, float]],
            clicks: int = 1,
            move_duration: float = 0.2,
            delay_between: float = 0.3
    ) -> None:
        sw, sh = pyautogui.size()

        for i, (xr, yr) in enumerate(points):
            x = int(xr * sw)
            y = int(yr * sh)

            log(f"🖱️ Clicking point {i+1}: ({xr:.4f}, {yr:.4f}) → ({x}, {y})")

            pyautogui.moveTo(x, y, duration=move_duration)
            for _ in range(max(1, clicks)):
                pyautogui.click()
                sleep_s(delay_between)


DRIVE_REGION = Region(0.6448, 0.8426, 0.7146, 0.9426)

ENTER_REGION = Region(0.7776, 0.6278, 0.9427, 0.6880)
TO_BATTLE_REGION = Region(0.8068, 0.7556, 0.9745, 0.8583)
CONFIRM_REGION = Region(0.7797, 0.7102, 0.9464, 0.8426)
THREAD_REGION = Region(0.0411, 0.4139, 0.1755, 0.5130)
THREAD_ENTER_REGION = Region(0.2188, 0.6435, 0.3630, 0.6889)
DIFFICULTY_REGION = Region(0.3792, 0.6213, 0.4589, 0.6639)
WIN_REGION = Region(0.0000, 0.7148, 0.9995, 0.8759)
WINDOW_REGION = Region(0.4828, 0.8500, 0.5500, 0.9398)
THREAD_TAB_REGION = Region(0.0479, 0.4204, 0.1724, 0.5065)
MISSION_REGION = Region(0.2349, 0.0398, 0.3469, 0.1093)
BATTLE_PASS_REGION = Region(0.1234, 0.0389, 0.1234, 0.0389)
CLAIM_ALL_REGION = Region(0.5047, 0.7648, 0.5047, 0.7648)
PASS_POINTS = [(0.8495, 0.3287)]
GET_DAILY_POINTS = [(0.4000, 0.3194), (0.4000, 0.4389), (0.4036, 0.5593), (0.4031, 0.6815), (0.3990, 0.7870)]
LUX_POINTS = [(0.3484, 0.2472)]
BACK_ARROW_POINTS = [(0.0734, 0.0667)]

program = OcrAgent()

program.click_text("drive", DRIVE_REGION)
sleep_s(0.5)
program.click_points(LUX_POINTS)
sleep_s(0.5)
program.click_text("enter", ENTER_REGION)
sleep_s(0.5)
program.click_text("battle", TO_BATTLE_REGION)

#exp fight

while True:
    if program.click_text("confirm", CONFIRM_REGION, move_to=False, clicks=0):
        break
    else:
        if program.click_text("win", WIN_REGION, move_to=False, clicks=0):
            pyautogui.press('p'); sleep_s(0.2); pyautogui.press('enter'); sleep_s(3.0)

#exit exp, swap to thread

program.click_text("confirm", CONFIRM_REGION)
program.click_text("thread", THREAD_TAB_REGION)

count = 3

while count > 0:
    if program.click_text("enter", THREAD_ENTER_REGION):
        sleep_s(1)
        program.click_text("difficulty", DIFFICULTY_REGION)
        sleep_s(1)
        program.click_text("battle", TO_BATTLE_REGION)
        while True:
            if program.click_text("confirm", CONFIRM_REGION, move_to=False, clicks=0):
                break
            else:
                if program.click_text("win", WIN_REGION, move_to=False, clicks=0):
                    pyautogui.press('p'); sleep_s(0.2); pyautogui.press('enter'); sleep_s(3.0)
        program.click_text("confirm", CONFIRM_REGION)
        count -= 1

#last page
program.click_points(BACK_ARROW_POINTS)
sleep_s(0.5)
program.click_text("window", WINDOW_REGION)
sleep_s(0.5)
program.click_points(PASS_POINTS)
sleep_s(0.5)
program.click_text("pass", MISSION_REGION)
sleep_s(0.5)
program.click_points(GET_DAILY_POINTS)
sleep_s(0.5)
program.click_text("pass", BATTLE_PASS_REGION)
sleep_s(0.5)
program.click_text("claim", CLAIM_ALL_REGION)
sleep_s(1)
pyautogui.press('enter')