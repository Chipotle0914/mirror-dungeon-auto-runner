import easyocr
import pytesseract
from pytesseract import Output
from PIL import ImageGrab
import pyautogui
import cv2
import numpy as np
import time
import keyboard 
import sys

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=True)

# Set your Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def find_text_coordinates(target, crop_right_percentage, crop_bottom_percentage, offset_x = 0, offset_y = 0):
    print(f"🔍 Scanning right side for '{target}'...")
    # Take a screenshot of the full screen
    img = np.array(ImageGrab.grab())
    
    # Get the width and height of the image
    height, width, _ = img.shape
    # Convert percentages to decimal (e.g., 40% -> 0.4)
    crop_right_percentage *= 0.01
    crop_bottom_percentage *= 0.01
    #calculate the offset based on full width, full height
    offset_x *= width
    offset_y *= height
    # Calculate the crop positions
    crop_right = 1 - crop_right_percentage  # For right cropping, keep the portion to the left
    crop_bottom = 1 - crop_bottom_percentage  # For bottom cropping, keep the top portion

    # If crop percentages are 0, we should keep the full width or height
    if crop_right_percentage == 0:
        right_side_img = img  # No right crop
    else:
        right_side_img = img[:, int(crop_right * width):]  # Crop the left side

    if crop_bottom_percentage == 0:
        cropped_img = right_side_img  # No bottom crop
    else:
        cropped_img = right_side_img[int(crop_bottom * height):, :] # Crop the top side

    # Convert the cropped image to grayscale for better OCR performance
    
    # Convert the cropped image to grayscale for better OCR performance
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to boost contrast
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
 
    if(target == "Difficulty"):
    
        # Apply median blur to reduce noise
        gray = cv2.medianBlur(gray, 3)

        # Sharpen image (optional, helps edge clarity)
        kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

   
    """
    # Save the cropped image as a PNG to inspect it visually
    
    print("Cropped image saved as 'cropped_image.png'")
    """
    debug_filename = f'cropped_debug_{target.lower()}.png'
    cv2.imwrite(debug_filename, gray)
    # Use pytesseract to extract data (text + bounding box)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)

    rightmost_x = -1
    rightmost_y = -1
    found_word = None
    print(data["text"])  # To see what text is detected

    for i, word in enumerate(data['text']):
        if word.strip() and target.lower() in word.lower():
            # Get the coordinates of the bounding box
            x = data['left'][i] + data['width'][i] // 2  # Horizontal center of the bounding box
            y = data['top'][i] + data['height'][i] // 2  # Vertical center of the bounding box

            if x > rightmost_x:
                rightmost_x = x
                rightmost_y = y
                found_word = word

    if found_word:  
        if crop_bottom_percentage != 0:
            rightmost_y += (crop_bottom * height)
        if crop_right_percentage != 0:
            rightmost_x += (crop_right * width)
        
        #calculate with offset
        rightmost_x += offset_x
        rightmost_y += offset_y
        print(f"✅ Right-most '{found_word}' found at ({rightmost_x}, {rightmost_y}) — clicking")
        return(rightmost_x, rightmost_y)
        
    print(f"❌ '{target}' not found on screen.")
    return None

def click_at_position(x, y, click_amount=1, move_duration=0.2):

    if click_amount < 0:
        print("click amount should at least be 1")
        return

    while click_amount > 0:
        pyautogui.moveTo(x, y, duration=move_duration)
        pyautogui.click()
        click_amount -= 1
        time.sleep(0.3)


def move_mouse_away():
    # Get the screen width and height
    screen_width, screen_height = pyautogui.size()

    # Calculate the center of the screen
    center_x = screen_width // 2
    bottom_y = screen_height // 6
    pyautogui.moveTo(center_x, bottom_y, duration=0.2)
    return (center_x, bottom_y)

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
            return (center_x, center_y)

    print(f"❌ '{target}' not found in region.")
    return None

def skill_check():
    print("🔄 Checking for skill check level...")

    x1_ratio, y1_ratio = 0.0042, 0.8185
    x2_ratio, y2_ratio = 0.7849, 0.8759

    for level in ["very high", "high", "normal", "low", "very low"]:
        coords = scan_box_region_for_text_easyocr_ratio(level, x1_ratio, y1_ratio, x2_ratio, y2_ratio)
        if coords:
            print(f"🎯 Skill check level detected: {level.upper()}")
            click_at_position(*coords)
            return  # Exit after first match

    print("❌ No skill check on screen.")
    sys.exit()




def perform_actions():
    #program starts in 3 seconds
    print("program starts in 3 seconds! please swtich to limbusCompany before it begeins!")
    time.sleep(3)


    #click on exp enter
    while True:
        if (reward_coords := find_text_coordinates("reward", 0, 0, 0, 0.0667)):
            click_at_position(*reward_coords)
            time.sleep(2)
            break
    #move cursor out of the way(avoid detection erorrs)
    move_mouse_away()

    #click on level 50
    if(diff_coords := scan_box_region_for_text_easyocr_ratio("enter", 0.2667, 0.5861, 0.7109, 0.7667)):
        click_at_position(*diff_coords)
    else:
        sys.exit()
    
    move_mouse_away()
    #now click on battle to enter the fight!
    if (sin_coords := find_text_coordinates("sin", 50, 0, 0, 0.6389)):
        click_at_position(*sin_coords)
        time.sleep(2)
    elif (battle_coords := find_text_coordinates("battle", 50, 0, 0)):
        click_at_position(*battle_coords)
    else:
        sys.exit()

    flag = 0

    #now we are in the a fight!!!
    while True:
        coords = find_text_coordinates("reward", 0, 0, 0, 0.0667)

        if coords:
            print("Confirm found and clicked.")
            time.sleep(2)
            flag = 1
            break
        elif (skip_coords := scan_relative_region_for_text_easyocr("skip", 0.467, 0.4389)):
            #multiple clicks to comfirm the skill check proceed
            click_at_position(*skip_coords, 5)
            #prcoeed to skill check
            skill_check()
            #wait for commence button to comeout of the screen
            time.sleep(1)
            #click commence
            if (comm_cooords := scan_box_region_for_text_easyocr_ratio("commence", 0.7979, 0.8398, 0.9682, 0.9528)):
                click_at_position(*comm_cooords)
            else:
                print("Can't find commence button!")
                sys.exit()
            #wait for continue button to comeout of the screen
            time.sleep(3)
            #multiple clicks to comfirm the skill check proceed
            click_at_position(*skip_coords, 5)
            #click cotinue
            if (contin_cooords := scan_box_region_for_text_easyocr_ratio("continue", 0.7979, 0.8398, 0.9682, 0.9528)):
                click_at_position(*contin_cooords)
            else:
                print("Can't find continue button!")
                sys.exit()
        else:
            print("Still scanning...")
            top_mid = move_mouse_away()
            click_at_position(*top_mid)
            pyautogui.press("p")
            time.sleep(0.3)
            pyautogui.press("enter")
    return flag

def start_program():

    print("🔢 How many times do you want the program to run?")
    
    # Get number of runs from the user
    while True:
        try:
            total_runs = int(input("Enter a number: "))
            if total_runs <= 0:
                print("Please enter a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")
    runs = 0
    print("🔍 Press Ctrl+G to start the process.")
    
    # Wait for Ctrl+G once before starting the loop
    keyboard.wait("ctrl+g")
    print("▶️ Ctrl+G detected! Starting infinite process...")
    
    try:
        while runs < total_runs:
            if perform_actions():
                runs += 1
                print(f"✅ Run {runs}/{total_runs} completed")
            else:
                print("❌ Something went wrong.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Program stopped by Ctrl+C.")


start_program()     