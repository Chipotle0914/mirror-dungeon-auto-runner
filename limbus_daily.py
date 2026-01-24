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
        y_offset_frac: float = 0.0,
        pick_lowest: bool = False,
    ) -> bool:
        (x1, y1, x2, y2) = region.abs_coords()
        xoff = 0
        yoff = int(y_offset_frac * pyautogui.size()[1])

        log(f"📦 Scanning box region: ({x1}, {y1}) to ({x2}, {y2})")
        log(f"🎯 Target text to find: '{target}'")

        monitor = {
            "top": y1,
            "left": x1,
            "width": x2 - x1,
            "height": y2 - y1
        }

        img = np.array(self.sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        results = self.reader.readtext(
            gray,
            allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        log(f"🧾 Raw OCR results: {results}")

        target_lower = target.lower().strip()
        matches = []

        for (bbox, text, conf) in results:
            text_lower = text.lower().strip()
            tokens = text_lower.split()

            log(f"🔍 OCR line: '{text_lower}' | tokens={tokens} | conf={conf:.2f}")

            if target_lower not in text_lower:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            left = min(xs)
            right = max(xs)
            width = max(1, right - left)
            y_center = (min(ys) + max(ys)) / 2

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
                log("⚠️ Target substring found, but no token index matched")
                continue

            idx = max(0, min(idx, len(tokens) - 1))
            log(f"🎯 Matched token index={idx}, token='{tokens[idx]}'")

            centers_x = self._token_centers(text_lower, tokens, left, width)
            click_x_local = (
                centers_x[idx]
                if idx < len(centers_x)
                else left + width * ((idx + 0.5) / max(1, len(tokens)))
            )

            click_x = x1 + int(click_x_local) + xoff
            click_y = y1 + int(y_center) + yoff

            matches.append((click_y, click_x, text, conf))

        if not matches:
            log(f"❌ '{target}' not found in region.")
            return False

        if pick_lowest:
            click_y, click_x, text, conf = max(matches, key=lambda m: m[0])
        else:
            click_y, click_x, text, conf = matches[0]

        log(f"✅ FINAL MATCH: '{text}' | conf={conf:.2f}")
        log(f"🖱️ Clicking at screen coords ({click_x}, {click_y})")

        if move_to:
            pyautogui.moveTo(click_x, click_y, duration=0.2)

        for _ in range(max(0, clicks)):
            pyautogui.click()
            sleep_s(0.3)

        log(f"🖱️ Clicked '{target}'" if clicks > 0 else f"👀 Hovered '{target}'")
        return True


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
            for _ in range(clicks):
                pyautogui.click()
                sleep_s(delay_between)
    def move_point(self, point: Tuple[float, float], duration: float = 0.2):
        sw, sh = pyautogui.size()
        x = int(point[0] * sw)
        y = int(point[1] * sh)
        log(f"🖱️ Moving mouse to ({point[0]:.4f}, {point[1]:.4f}) → ({x}, {y})")
        pyautogui.moveTo(x, y, duration=duration)

    def click_skip_5(self):
        self.click_text("skip", SKIP_REGION, clicks=5)


    def drag_down_half_screen_at_point(
        self,
        point: Tuple[float, float],
        hold_sec: float = 0.10,
        move_duration: float = 0.25,
    ) -> None:
        """
        Move to a ratio point, click+hold, drag down half the screen, release.
        point: (x_ratio, y_ratio) in [0..1]
        """
        sw, sh = pyautogui.size()
        x = int(point[0] * sw)
        y = int(point[1] * sh)

        drag_to_y = int(min(sh - 2, y + sh * 0.5))  # half-screen down, clamp
        drag_to_x = x

        log(f"🖱️ Drag down half-screen from ({point[0]:.4f},{point[1]:.4f}) → ({x},{y}) to ({drag_to_x},{drag_to_y})")

        pyautogui.moveTo(x, y, duration=move_duration)
        pyautogui.mouseDown(button="left")
        sleep_s(hold_sec)

        pyautogui.moveTo(drag_to_x, drag_to_y, duration=move_duration)
        pyautogui.mouseUp(button="left")

COMMENCE_AND_CONTINUE_REGION = Region(0.7703, 0.7917, 0.9865, 0.9796)
SKILL_CHECK_REGION = Region(0.0000, 0.7843, 0.9995, 0.9991)
SKIP_REGION = Region(0.4432, 0.4093, 0.4932, 0.4472)
DRIVE_REGION = Region(0.6448, 0.8426, 0.7146, 0.9426)
ENTER_REGION = Region(0.7776, 0.6278, 0.9427, 0.6880)
TO_BATTLE_REGION = Region(0.8039, 0.7257, 0.9961, 0.9653)
CONFIRM_REGION = Region(0.5961, 0.6361, 0.9898, 0.9806)
THREAD_REGION = Region(0.0411, 0.4139, 0.1755, 0.5130)
THREAD_ENTER_REGION = Region(0.2188, 0.6435, 0.3630, 0.6889)
DIFFICULTY_REGION = Region(0.2870, 0.2722, 0.7000, 0.7389)
WIN_REGION = Region(0.0000, 0.7148, 0.9995, 0.8759)
WINDOW_REGION = Region(0.4828, 0.8500, 0.5500, 0.9398)
THREAD_TAB_REGION = Region(0.0479, 0.4204, 0.1724, 0.5065)
MISSION_REGION = Region(0.2349, 0.0398, 0.3469, 0.1093)
BATTLE_PASS_REGION = Region(0.1255, 0.0417, 0.2286, 0.1028)
CLAIM_ALL_REGION = Region(0.5312, 0.6037, 0.8578, 0.9250)
USE_LUNACY_REGION = Region(0.4453, 0.2815, 0.5474, 0.3398)
MAKE_MODULES_REGION = Region(0.3411, 0.2944, 0.4437, 0.3398)
LUNACY_CONFIRM_REIGON = Region(0.5016, 0.7009, 0.6849, 0.7750)
PASS_POINTS = [(0.8495, 0.3287)]
GET_DAILY_POINTS = [(0.4000, 0.3194), (0.4000, 0.4389), (0.4036, 0.5593), (0.4031, 0.6815), (0.3990, 0.7870)]
LUX_POINTS = [(0.3484, 0.2472)]
BACK_ARROW_POINTS = [(0.0734, 0.0667)]
MODULES_POINTS = [(0.3042, 0.9111)]
MODULES_MAX_ARROW = [(0.6276, 0.4602)]
BAR_TO_SCROLL_DOWN = (0.6948, 0.3815)

def ask_skip_modules() -> bool:
    """
    Ask user whether to skip the module-making process.
    Returns True if user wants to skip.
    """
    while True:
        ans = input("Skip module making? (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")

program = OcrAgent()

def process_mid_fight_q():
    program.click_skip_5()
    sleep_s(1)
    for level in ["very high", "high", "normal", "low", "very low"]:
        if program.click_text(level, SKILL_CHECK_REGION):
            break
    sleep_s(1)
    program.click_text("commence", COMMENCE_AND_CONTINUE_REGION)
    sleep_s(2)
    program.click_skip_5()
    sleep_s(1)
    program.click_text("continue", COMMENCE_AND_CONTINUE_REGION)

def process_modules():
    program.click_points(MODULES_POINTS)
    sleep_s(1)
    program.click_text("use", USE_LUNACY_REGION)
    sleep_s(0.3)
    program.click_text("confirm", LUNACY_CONFIRM_REIGON, clicks=2)
    sleep_s(0.5)
    program.click_text("modules", MAKE_MODULES_REGION)
    sleep_s(0.5)
    program.click_points(MODULES_MAX_ARROW)
    sleep_s(0.5)
    program.click_text("confirm", LUNACY_CONFIRM_REIGON)
    sleep_s(0.5)
    program.click_points(MODULES_POINTS)
    sleep_s(0.5)
    """
#make modules first twice!
if not ask_skip_modules():
    process_modules()
else:
    log("⏭️ Skipping module-making step by user choice.")
    """
process_modules()
program.click_text("drive", DRIVE_REGION)
sleep_s(0.5)
program.click_points(LUX_POINTS)
sleep_s(0.5)
program.click_text("enter", ENTER_REGION)
sleep_s(1)
program.click_text("to", TO_BATTLE_REGION)

#exp fight

while True:
    if program.click_text("confirm", CONFIRM_REGION, move_to=False, clicks=0):
        break
    else:
        if program.click_text("win", WIN_REGION, move_to=False, clicks=0):
            pyautogui.press('p'); sleep_s(0.2); pyautogui.press('enter'); 
            program.move_point((0.0792, 0.9352))
            sleep_s(3.0)

#exit exp, swap to thread

program.click_text("confirm", CONFIRM_REGION)

program.click_text("thread", THREAD_TAB_REGION)

count = 3

while count > 0:
    if program.click_text("enter", THREAD_ENTER_REGION):
        sleep_s(1)
        #scroll to the bottom 
        program.drag_down_half_screen_at_point(BAR_TO_SCROLL_DOWN)
        

        program.click_text("diff", DIFFICULTY_REGION, pick_lowest=True)
        sleep_s(1)
        program.click_text("to", TO_BATTLE_REGION)
        while True:
            if program.click_text("confirm", CONFIRM_REGION):
                break
            elif program.click_text("skip", SKIP_REGION, move_to=False, clicks=0):
                process_mid_fight_q()
            else:
                if program.click_text("win", WIN_REGION, move_to=False, clicks=0):
                    pyautogui.click()
                    pyautogui.press('p'); sleep_s(0.2); pyautogui.press('enter'); 
                    program.move_point((0.0792, 0.9352))
                    sleep_s(3.0)
        count -= 1

#last page
sleep_s(1)
program.click_points(BACK_ARROW_POINTS)
sleep_s(0.5)
program.click_text("window", WINDOW_REGION)
sleep_s(1)
program.click_points(PASS_POINTS)
sleep_s(0.5)
program.click_text("pass", MISSION_REGION)
sleep_s(0.5)
program.click_points(GET_DAILY_POINTS)
sleep_s(0.5)
program.click_text("pass", BATTLE_PASS_REGION)
sleep_s(0.5)
program.click_text("claim", CLAIM_ALL_REGION)
sleep_s(1.5)
pyautogui.press('enter')