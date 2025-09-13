# mirror_dungeon_bot_refactor.py
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import torch
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab
import easyocr


# =========================
# Config & Small Utilities
# =========================

@dataclass(frozen=True)
class Region:
    x1: float; y1: float; x2: float; y2: float  # ratios (0..1)

    def abs_coords(self) -> Tuple[int, int, int, int]:
        sw, sh = pyautogui.size()
        return (int(self.x1 * sw), int(self.y1 * sh),
                int(self.x2 * sw), int(self.y2 * sh))

def sleep_s(sec: float) -> None:
    time.sleep(sec)

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def log(msg: str) -> None:
    print(msg, flush=True)

def wait_until(cond, timeout: float, interval: float = 0.2) -> bool:
    """Poll a predicate until True or timeout."""
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        sleep_s(interval)
    return False


# =========================
# Constants (names unchanged)
# =========================

FIGHT            = "fight_stage"
QUESTION         = "question_stage"
COMPLETED        = "completed_stage"
TRAIN            = "train"
SHOP             = "shop"
ACQUIRE_EGO      = "acquire_ego_gift"
REWARD_MONEY     = "reward_money"
REWARD_GAMBLE    = "reward_gamble"
REWARD_RESOURCE  = "reward_ego_resource"
REWARD_RANDOM    = "reward_random_ego"
REWARD_STAR      = "reward_star"
A_CHOICES        = "A_choices"
B_CHOICES        = "B_choices"
GOOD_PACK        = "good_pack"
BAD_PACK         = "bad_pack"

# Regions
ENTER_REGION      = Region(0.7547, 0.6611, 0.9995, 0.9991)
TO_BATTLE_REGION  = Region(0.8047, 0.7306, 0.9870, 0.9620)
TO_BATTLE_BACKUP  = Region(0.7786, 0.5694, 0.9938, 0.9528)
CONFIRM_REGION    = Region(0.0000, 0.3444, 0.9995, 0.9972)
SELECT_REGION     = Region(0.7625, 0.7444, 0.9943, 0.9889)
REWARD_REGION     = Region(0.0745, 0.1028, 0.8000, 0.2472)
P_ENTER_REGION    = Region(0.0516, 0.6028, 0.9895, 0.9375)
SKIP_REGION       = Region(0.3802, 0.1731, 0.5266, 0.7037)
COMMENCE_REGION   = Region(0.7438, 0.7713, 0.9927, 0.9667)  # commence/continue/proceed
LEAVE_REGION      = Region(0.7552, 0.7222, 0.9958, 0.9731)
SHOP_CONFIRM      = Region(0.4167, 0.4620, 0.7792, 0.8481)
SKILL_CHECK_REGION= Region(0.0042, 0.8185, 0.7849, 0.8759)
END_TOP_LEFT      = Region(0.0005, 0.0907, 0.3203, 0.2815)
END_CONFIRM       = Region(0.6839, 0.6769, 0.9635, 0.9500)
REFRESH_REGION    = Region(0.6786, 0.0000, 0.9885, 0.1481)

DRIVE_REGION      = Region(0.4411, 0.7370, 0.9995, 0.9991)
MD_REGION         = Region(0.2417, 0.2639, 0.5078, 0.5463)
ENTER_MD_REGION   = Region(0.4938, 0.6269, 0.9714, 0.7917)
BEG_EGO_REGION    = Region(0.0870, 0.1713, 0.5906, 0.7778)
REFUSE_GIFT_REGION= Region(0.6120, 0.7333, 0.9141, 0.9102)
CLAIM_REWARD_REGION = Region(0.5828, 0.6435, 0.9682, 0.9194)

# Hardcoded click points (ratios)
BEG_BUFF_HARDCODE   = [(0.1922, 0.3454), (0.3495, 0.3454), (0.5010, 0.3546), (0.6568, 0.3565)]
BEG_TOP_2_EGO       = [(0.7438, 0.3620), (0.7411, 0.5074)]

DEFAULT_PRIOR: List[str] = [REWARD_STAR, REWARD_MONEY, REWARD_RANDOM, REWARD_GAMBLE, REWARD_RESOURCE]
REQUIRE_RIGHT_OF_TRAIN = {FIGHT, QUESTION, SHOP}
BOSS_X_OFFSET = 0.2005

_TRAIN_RELAX_CONF = 0.60
_LAST_TTL_SEC     = 2.0


# =========================
# Vision Helpers (YOLO + OCR)
# =========================

class YoloDetector:
    def __init__(self, model_path: str, conf: float = 0.85):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf
        self._last_train = {"cx": None, "t": 0.0}

    def _grab_rgb(self) -> np.ndarray:
        return np.array(ImageGrab.grab())

    def _update_train_memory(self, det_df) -> None:
        now = time.time()
        train_now = det_df[
            (det_df["name"] == TRAIN) &
            (det_df["confidence"] >= min(_TRAIN_RELAX_CONF, float(getattr(self.model, "conf", 0.25))))
        ].sort_values(by="confidence", ascending=False)
        if not train_now.empty:
            x1, x2 = train_now.iloc[0][["xmin", "xmax"]]
            self._last_train = {"cx": int((x1 + x2) / 2), "t": now}

    def detect_and_optionally_click(
        self,
        target_class: str,
        click_num: int = 1,
        top_most: bool = False,
        drag_down: bool = False,
        move_to: bool = True,
        require_right_of_train: bool = True,
        fallback_if_no_train: bool = True,
        save_train_shot: bool = False,
    ) -> bool:
        log(f"🔍 Searching for '{target_class}' using YOLO...")
        rgb = self._grab_rgb()
        results = self.model(rgb)
        det = results.pandas().xyxy[0]

        # cache TRAIN cx
        self._update_train_memory(det)

        # filter matches
        matches = det[(det["name"] == target_class) & (det["confidence"] >= float(getattr(self.model, "conf", 0.25)))]
        if matches.empty:
            log(f"❌ '{target_class}' not found.")
            return False
        matches = matches.sort_values(by="confidence", ascending=False)

        # right-of-TRAIN soft rule
        valid = []
        if require_right_of_train and (target_class in REQUIRE_RIGHT_OF_TRAIN):
            cx_train = None
            # prefer current frame
            tn = det[(det["name"] == TRAIN)].sort_values(by="confidence", ascending=False)
            if not tn.empty:
                x1, x2 = tn.iloc[0][["xmin", "xmax"]]
                cx_train = int((x1 + x2) / 2)
            else:
                last = self._last_train
                if last["cx"] is not None and (time.time() - last["t"] <= _LAST_TTL_SEC):
                    cx_train = last["cx"]

            if cx_train is None:
                msg = f"⚠️ TRAIN not detected; cannot verify '{target_class}' right-of-TRAIN"
                if fallback_if_no_train:
                    log(msg + " → relaxing constraint.")
                    valid = list(matches.itertuples(index=False))
                else:
                    log(msg + " → skipping.")
                    Bot.move_away_static()
                    return False
            else:
                for _, row in matches.iterrows():
                    cx = int((row["xmin"] + row["xmax"]) / 2)
                    if cx > cx_train:
                        valid.append(row)
                if not valid:
                    log(f"❌ '{target_class}' found, but none to the RIGHT of TRAIN (cx_train={cx_train}).")
                    return False
                else:
                    log(f"🧭 Right-of-TRAIN check OK using cx_train={cx_train}.")
        else:
            valid = list(matches.itertuples(index=False))
        """
        if save_train_shot and target_class == TRAIN:
            Bot.save_full_screenshot("train")
        """
        # choose candidate
        if top_most:
            log("↕️ Top-most mode enabled")
            chosen = min(valid, key=lambda r: getattr(r, "ymin", r.ymin) if hasattr(r, "ymin") else r.ymin)
        else:
            chosen = valid[0]

        x1, y1, x2, y2 = (chosen.xmin, chosen.ymin, chosen.xmax, chosen.ymax)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        conf = float(chosen.confidence)
        log(f"✅ Found '{target_class}' at ({cx}, {cy}) | conf={conf:.2f}")

        # act
        if move_to:
            pyautogui.moveTo(cx, cy, duration=0.2)

        if drag_down:
            pyautogui.mouseDown(cx, cy)
            pyautogui.moveTo(cx, cy + 400, duration=0.2)
            pyautogui.mouseUp()
            log("Successfully Selected Theme pack!")
        else:
            for _ in range(max(0, click_num)):
                pyautogui.click()
                sleep_s(0.3)
            log(f"🖱️ Clicked on '{target_class}'" if click_num > 0 else f"👀 Detected '{target_class}' (no click)")

        return True


class OcrAgent:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

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
        img = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


# =========================
# Bot (state & flows)
# =========================

class Bot:
    def __init__(self, model_path: str, yolo_conf: float = 0.85):
        self.det = YoloDetector(model_path, conf=yolo_conf)
        self.ocr = OcrAgent()

    # ---- Static helpers (no self state) ----
    @staticmethod
    def save_full_screenshot(prefix: str) -> str:
        path = f"{prefix}_{now_ts()}.png"
        ImageGrab.grab().save(path)
        log(f"🖼️ Saved screenshot to {path}")
        return path

    @staticmethod
    def move_away_static() -> None:
        sw, sh = pyautogui.size()
        x = int(0.1094 * sw); y = int(0.8759 * sh)
        pyautogui.moveTo(x, y, duration=0.2)

    @staticmethod
    def reset_view_static() -> None:
        sw, sh = pyautogui.size()
        x = int(0.4703 * sw); y = int(0.6333 * sh)
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.scroll(-50)
        sleep_s(0.2)

    # ---- OCR wrappers ----
    def click_enter(self) -> bool:          return self.ocr.click_text("enter", ENTER_REGION)
    def click_confirm(self) -> bool:        return self.ocr.click_text("confirm", CONFIRM_REGION)
    def click_select(self) -> bool:         return self.ocr.click_text("select", SELECT_REGION)
    def click_to_battle(self) -> bool:      return self.ocr.click_text("battle", TO_BATTLE_REGION)
    def check_to_battle(self) -> bool:      return self.ocr.click_text("battle", TO_BATTLE_REGION, move_to=False, clicks=0)
    def check_reward(self) -> bool:         return self.ocr.click_text("reward", REWARD_REGION, move_to=False, clicks=0)
    def check_confirm(self) -> bool:        return self.ocr.click_text("confirm", CONFIRM_REGION, move_to=False, clicks=0)
    def check_skip(self) -> bool:           return self.ocr.click_text("skip", SKIP_REGION, move_to=False, clicks=0)
    def click_skip(self, n: int = 5) -> bool: return self.ocr.click_text("skip", SKIP_REGION, clicks=n)
    def click_continue(self) -> bool:       return self.ocr.click_text("continue", COMMENCE_REGION)
    def click_commence(self) -> bool:       return self.ocr.click_text("commence", COMMENCE_REGION)
    def click_proceed(self) -> bool:        return self.ocr.click_text("proceed", COMMENCE_REGION)
    def click_leave(self) -> bool:          return self.ocr.click_text("leave", LEAVE_REGION)
    def click_shop_confirm(self) -> bool:   return self.ocr.click_text("confirm", SHOP_CONFIRM, y_offset_frac=-0.0224)
    def click_refresh(self) -> bool:        return self.ocr.click_text("refresh", REFRESH_REGION)
    def click_claim(self) -> bool:          return self.ocr.click_text("claim", CLAIM_REWARD_REGION)

    def check_p_enter(self) -> bool:
        return (self.ocr.click_text("win", P_ENTER_REGION, move_to=False, clicks=0) or
                self.ocr.click_text("damage", P_ENTER_REGION, move_to=False, clicks=0))

    def check_leave(self) -> bool:          return self.ocr.click_text("leave", LEAVE_REGION, move_to=False, clicks=0)

    def check_ending(self) -> bool:
        if self.ocr.click_text("contributed", END_TOP_LEFT, move_to=True, clicks=0):
            self.ocr.click_text("confirm", END_CONFIRM)
            return True
        return False

    def click_drive(self) -> bool:          return self.ocr.click_text("drive", DRIVE_REGION)
    def check_drive(self) -> bool:          return self.ocr.click_text("drive", DRIVE_REGION, move_to=False, clicks=0)
    def click_md(self) -> bool:             return self.ocr.click_text("mirror", MD_REGION)
    def click_enter_md(self) -> bool:       return self.ocr.click_text("enter", ENTER_MD_REGION)
    def pick_ego_type(self, se_type: str) -> bool:
        return self.ocr.click_text(se_type, BEG_EGO_REGION)

    # ---- Mouse helpers ----
    def move_away(self) -> None: Bot.move_away_static()
    def reset_view(self) -> None: Bot.reset_view_static()

    def click_list_points(self, points: Iterable[Tuple[float, float]], delay: float = 0.2) -> None:
        sw, sh = pyautogui.size()
        for (xr, yr) in points:
            pyautogui.moveTo(int(xr * sw), int(yr * sh), duration=0.5)
            pyautogui.click()
            sleep_s(delay)

    def click_boss(self) -> None:
        sw, sh = pyautogui.size()
        x, y = pyautogui.position()
        target_x = x + int(BOSS_X_OFFSET * sw)
        y += int(0.0324 * sh)
        pyautogui.moveTo(target_x, y, duration=0.2)
        pyautogui.click()

    # ---- YOLO wrappers ----
    def yolo_click(self, cls: str, **kwargs) -> bool:
        return self.det.detect_and_optionally_click(cls, **kwargs)

    # ---- Higher-level helpers ----
    def click_reward_prior(self) -> Optional[str]:
        for target in DEFAULT_PRIOR:
            if self.yolo_click(target):
                log(f"🎯 Picked reward: {target}")
                return target
        log("❌ No preferred rewards found.")
        return None

    # =========================
    # Flows
    # =========================

    def start_md(self, se_type: str) -> bool:
        self.click_drive();     sleep_s(1.5)
        self.click_md();        sleep_s(1.5)
        self.click_enter_md();  sleep_s(1.5)
        self.click_enter_md();  sleep_s(1.0)
        self.click_confirm();   sleep_s(1.0)
        self.click_confirm();   sleep_s(1.5)

        self.click_list_points(BEG_BUFF_HARDCODE); sleep_s(0.8)
        self.click_enter();                        sleep_s(0.8)
        self.click_confirm();                      sleep_s(1.5)

        self.pick_ego_type(se_type)
        self.click_list_points(BEG_TOP_2_EGO)
        self.click_select();                       sleep_s(0.8)
        self.click_confirm();                      sleep_s(0.8)
        self.click_confirm();                      sleep_s(1.0)
        self.ocr.click_text("refuse", REFUSE_GIFT_REGION); sleep_s(0.8)
        self.click_confirm();                      sleep_s(0.8)
        self.move_away();                          sleep_s(2.0)

        # wait for GOOD_PACK (can load late)
        while True:
            if self.yolo_click(GOOD_PACK, move_to=False, clicks=0):
                break
            sleep_s(1.0)
        self.yolo_click(GOOD_PACK, drag_down=True)
        return True

    def process_fight(self, boss_fight: bool = False, skip_to_battle: bool = False) -> bool:
        if not skip_to_battle:
            while True:
                if (self.click_to_battle() or
                    self.ocr.click_text("clear", TO_BATTLE_BACKUP, move_to=True, clicks=0, y_offset_frac=0.1537)):
                    pyautogui.click(); sleep_s(0.3)
                    break

        sleep_s(1.5)  # reduce misreads

        while True:
            # Exit for regular fights: train seen twice
            if (not boss_fight) and self.yolo_click(TRAIN, click_num=0, move_to=False):
                sleep_s(1.0)
                if self.yolo_click(TRAIN, click_num=0):
                    if self.check_p_enter():
                        continue
                    break
                else:
                    continue

            # Exit for boss: pack appeared (double-check)
            if boss_fight and (self.yolo_click(GOOD_PACK, click_num=0, move_to=False) or
                               self.yolo_click(BAD_PACK,  click_num=0, move_to=False)):
                sleep_s(1.5)
                if not (self.yolo_click(GOOD_PACK, click_num=0, move_to=False) or
                        self.yolo_click(BAD_PACK,  click_num=0, move_to=False)):
                    continue
                log("🛑 Boss fight ended — good or bad pack found.")
                break

            # Reward cards
            if self.check_reward():
                sleep_s(3.0)
                picked = self.click_reward_prior()
                if picked:
                    sleep_s(1.0)
                    if picked in [REWARD_GAMBLE, REWARD_RANDOM]:
                        self.click_confirm(); sleep_s(1.0)
                        log(f"⏳ Waiting for confirm prompt due to special reward: {picked}")
                        if self.check_confirm():
                            self.click_confirm(); sleep_s(3.0)
                            self.move_away();     continue
                    else:
                        self.click_confirm(); sleep_s(3.0)
                        self.move_away();     continue
                elif self.check_confirm():
                    self.click_confirm(); sleep_s(1.0); continue

            # Ego gift: select + confirm, with failsafes
            elif self.yolo_click(ACQUIRE_EGO):
                sleep_s(0.8)
                if not self.click_select():
                    continue
                sleep_s(1.5)
                if not self.click_confirm():
                    pyautogui.mouseDown(button='left'); sleep_s(0.001); pyautogui.mouseUp(button='left')
                    if self.check_confirm():
                        self.click_confirm(); log("Second Select click: Success ✅"); continue
                    log("Second Select click: failed ❌\nThird Select click: attempting")
                    self.move_away()
                    self.ocr.click_text("select", SELECT_REGION, move_to=True, clicks=0)
                    pyautogui.mouseDown(button='left'); sleep_s(0.001); pyautogui.mouseUp(button='left')
                    continue
                self.move_away(); sleep_s(3.0); continue

            # Plain confirm (and maybe final)
            elif self.check_confirm():
                if not self.check_reward():
                    if boss_fight and self.check_ending():
                        return True
                    else:
                        self.click_confirm(); sleep_s(1.0); self.move_away(); continue

            # Skill check flow
            elif self.check_skip():
                self.process_question(mid_fight=True); continue

            else:
                # “P / Enter” nudge for special boss UI
                if self.check_p_enter():
                    sw, sh = pyautogui.size()
                    pyautogui.moveTo(int(0.1094 * sw), int(0.8759 * sh), duration=0.2)
                    pyautogui.click()
                    pyautogui.press('p'); sleep_s(0.2); pyautogui.press('enter'); sleep_s(3.0)
                else:
                    continue

        return False

    def process_question(self, mid_fight: bool = False) -> None:
        self.click_skip(n=5)
        while True:

            if mid_fight and self.check_p_enter():
                break
            # Exit to TRAIN (double seen)
            if self.yolo_click(TRAIN, click_num=0, move_to=False):
                sleep_s(1.5)
                if self.yolo_click(TRAIN, click_num=0): break
                else: continue

            elif self.check_confirm():
                self.click_confirm(); sleep_s(1.0); continue

            elif self.click_proceed():
                if not self.click_skip(n=5): sleep_s(1.0); self.click_skip(n=5)
                sleep_s(1.5); continue

            elif self.click_commence():
                sleep_s(3.0)
                if not self.click_skip(n=5): sleep_s(1.0); self.click_skip(n=5)
                sleep_s(1.5); continue

            elif self.click_continue():
                sleep_s(1.0); continue

            elif self.yolo_click(A_CHOICES, top_most=True):
                if not self.click_skip(n=5): sleep_s(1.0); self.click_skip(n=5)
                sleep_s(1.0); continue

            elif self.yolo_click(B_CHOICES, top_most=True):
                if not self.click_skip(n=5): sleep_s(1.0); self.click_skip(n=5)
                sleep_s(1.0); continue

            elif self.check_skill_check():
                self.click_best_skill_check(); sleep_s(1.0); continue

            elif self.check_to_battle():
                self.process_fight(); break

            elif self.check_p_enter():
                self.process_fight(skip_to_battle=True); break

            elif self.check_leave():
                break

            else:
                self.click_skip(n=5); sleep_s(1.0)

    def process_start_to_shop(self) -> None:
        while True:
            if self.yolo_click(SHOP) or self.check_leave():
                log("MD run till shop successful!")
                break
            elif self.yolo_click(SHOP):
                log("MD run till shop successful!")
                break
            elif self.yolo_click(QUESTION):
                sleep_s(1.0); self.click_enter(); sleep_s(1.5)
                self.process_question(); self.reset_view(); sleep_s(0.5)
            elif self.yolo_click(FIGHT):
                sleep_s(1.0); self.click_enter(); sleep_s(0.8)
                self.process_fight(); self.reset_view(); sleep_s(1.0)

    def process_shop_boss_packs(self) -> bool:
        self.click_enter();          sleep_s(1.0)
        self.click_leave();          sleep_s(1.0)
        self.click_shop_confirm();   sleep_s(2.0)
        self.yolo_click(TRAIN, click_num=0)
        self.click_boss();           sleep_s(1.0)
        self.click_enter()

        if self.process_fight(boss_fight=True):
            return True

        sleep_s(2.0)
        count = 0
        while count < 3 and ((not self.yolo_click(GOOD_PACK, click_num=0, move_to=False)) and
                              self.yolo_click(BAD_PACK,  click_num=0, move_to=False)):
            self.click_refresh(); sleep_s(3.0); count += 1

        self.move_away()
        self.yolo_click(GOOD_PACK, drag_down=True); sleep_s(3.0)
        return False

    # ---- Subroutines used in flows ----

    def check_skill_check(self) -> bool:
        for level in ["very high", "high", "normal", "low", "very low"]:
            if self.ocr.click_text(level, SKILL_CHECK_REGION, move_to=False, clicks=0):
                log(f"🎯 Skill check level detected: {level}")
                return True
        return False

    def click_best_skill_check(self) -> bool:
        for level in ["very high", "high", "normal", "low", "very low"]:
            if self.ocr.click_text(level, SKILL_CHECK_REGION, move_to=True, clicks=1):
                log(f"🎯 Skill check level clicked: {level}")
                return True
        sys.exit("❌ No skill check to click on screen.")

    # ---- End game ----
    def end_md(self) -> None:
        self.click_claim(); sleep_s(0.8)
        self.click_claim()
        for _ in range(3):
            self.click_confirm(); sleep_s(1.0)
            self.move_away();     sleep_s(0.5)
        sleep_s(5.0)


# =========================
# Main
# =========================

def main():
    MODEL_PATH = 'limbus_train_model/mirror_dungeon_train16/weights/best.pt'  
    bot = Bot(MODEL_PATH, yolo_conf=0.85)

    shop_flag = 0
    run_num   = int(input("Enter how many runs: "))
    skip_start= int(input("May resume in the run: "))
    se_type   = input("Enter status effect of your team: ")

    while run_num > 0:
        if skip_start == 0:
            sleep_s(2.0)
            bot.start_md(se_type)
            sleep_s(3.0)

        #set stkip_start back to 0 so after resume, it'll re-start the run probably
        skip_start = 0
        while True:
            # “enter” means we’re at the shop door/boss path
            if bot.ocr.click_text("enter", ENTER_REGION, move_to=False, clicks=0):
                log("processing boss until next level!!!")
                if bot.process_shop_boss_packs():
                    log("We DONE!!!")
                    bot.move_away(); sleep_s(2.0)
                    break
                sleep_s(2.0)
            else:
                log("processing until shop!!!")
                bot.process_start_to_shop()
                if shop_flag:
                    log("User wants to stop at shop")
                    break
                sleep_s(2.0)

        bot.end_md()
        run_num -= 1

        # wait until “drive” visible again before next run
        wait_until(bot.check_drive, timeout=120.0, interval=1.0)  # generous timeout
        sleep_s(1.0)


if __name__ == "__main__":
    main()
