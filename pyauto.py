import pyautogui as pa
import time
x,y=pa.position()
pa.click(x,y)
for _ in range(3):
    pa.press("left")
    time.sleep(3)
