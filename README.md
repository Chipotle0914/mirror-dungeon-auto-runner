# mirror_dungeon_auto

### 0) Before You Start: Get Python able to Run in VS Code

### 1) Install Dependencies


# (Optional) Update pip
```
python -m pip install --upgrade pip
```

# Core packages
```
pip install opencv-python
pip install pyautogui
pip install easyocr
pip install pillow
pip install keyboard
pip install --upgrade numpy pandas
pip install seaborn
pip install ultralytics
```
## PyTorch (pick ONE line)

### If you have an NVIDIA GPU, choose the CUDA that matches ```nvidia-smi```
### If you don’t have NVIDIA, pick CPU-only.

# NVIDIA CUDA 12.4
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
# NVIDIA CUDA 12.1
```
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
# NVIDIA CUDA 11.8
```
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
# CPU-only (no NVIDIA)
```
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2) Game Setup (Important)
- Fullscreen (or borderless fullscreen)
- Windows Display Scale: 100%
- Turn off overlays (Steam, FPS, GPU, chat) that cover the UI.

### 3) Run the programms!
```
python .\limbus_md_app.py
```
Prompts for:

- how many runs
   
- whether to resume:  you can only resume from two screens, one is any stage but haven't click on enter, the other is either after clicking on shop or final boss and you are at the screen with the enter button on the button right; 1 to resume, 0 to start from the beginning

- team status effect (for starting EGO pick)


```
python .\limbus_thread.py
```
Prompts for:

- how many runs
- press control + g to start(make sure you are at the thread screen before you press it)

```
For doing dailies which includes the following
- Do exp level once
- Do thread 3 times
- used up 78 lunacy(26 + 52) total to gain Enkephalin then convert all to Modules
NOTE: Current version does allow you to preselect a team of 6 and no more or else it'll causes the program to crash
```
Command to set up for .\limbus_daily.py
```
pip install easyocr pyautogui keyboard opencv-python numpy mss torch torchvision pillow
```
