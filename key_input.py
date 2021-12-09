# adapted from https://github.com/Sentdex/pygta5/blob/master/getkeys.py
import win32api as wapi
import time
import ast


# this is a list of characters to check when pressing in key_check()
keyList = ["\b"]
for char in "TFGHXCMQqpPYUN":
    keyList.append(char)

def key_check():
    keys = []

    for key in keyList:
        # the ord() function returns an integer representing the Unicode character
        # chr() goes opposite way
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    # doesn't work to catch shift...
    return keys


# https://stackoverflow.com/questions/3698635/getting-cursor-position-in-python
from ctypes import windll, Structure, c_long, byref

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

def mouse_check():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt.x, pt.y

def mouse_l_click_check(previous_status):
    held_down=0; clicked=0;
    current_status = wapi.GetKeyState(0x01)

    if current_status < 0:
        held_down = 1 # held down click
    elif current_status != previous_status and not previous_status < 0:
        clicked = 1 # just tapped this

    return current_status, clicked, held_down

def mouse_r_click_check(previous_status):
    held_down=0; clicked=0;
    current_status = wapi.GetKeyState(0x02)

    if current_status < 0:
        held_down = 1 # held down click
    elif current_status != previous_status and not previous_status < 0:
        clicked = 1 # just tapped this

    return current_status, clicked, held_down

def mouse_log_test():
    loop_fps=20
    previous_status_l = wapi.GetKeyState(0x01)
    previous_status_r = wapi.GetKeyState(0x02)

    while True:
        loop_start_time = time.time() # this is in seconds

        current_status_l, clicked_l, held_down_l = mouse_l_click_check(previous_status_l)
        current_status_r, clicked_r, held_down_r = mouse_r_click_check(previous_status_r)
        print('l_click', clicked_l, ' l_held', held_down_l,
            ' | r_click', clicked_r, ' r_held', held_down_r)
        previous_status_l = current_status_l
        previous_status_r = current_status_r
        # time.sleep(0.1)

        # wait until end of time step
        while time.time() < loop_start_time + 1/loop_fps:
            pass

    return

if __name__ == "__main__":
    mouse_log_test()

