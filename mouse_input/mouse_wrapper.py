import ctypes
import time
from ctypes import wintypes

INPUT_MOUSE = 0

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP   = 0x0040
MOUSEEVENTF_WHEEL      = 0x0800

WHEEL_DELTA = 120

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]

class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUTunion),
    ]

# load SendInput
_user32 = ctypes.WinDLL("user32", use_last_error=True)

_SendInput = _user32.SendInput
_SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_SendInput.restype = wintypes.UINT

def _send_mouse(flags, dx=0, dy=0, data=0):
    inp = INPUT(
        type=INPUT_MOUSE,
        union=_INPUTunion(
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=data,
                dwFlags=flags,
                time=0,
                dwExtraInfo=None,
            )
        ),
    )

    sent = _SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())
    
def move_rel(dx, dy):
    _send_mouse(MOUSEEVENTF_MOVE, dx=dx, dy=dy)

def left_click(delay=0.015):
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    time.sleep(delay)
    _send_mouse(MOUSEEVENTF_LEFTUP)

def right_click(delay=0.015):
    _send_mouse(MOUSEEVENTF_RIGHTDOWN)
    time.sleep(delay)
    _send_mouse(MOUSEEVENTF_RIGHTUP)

def middle_click(delay=0.015):
    _send_mouse(MOUSEEVENTF_MIDDLEDOWN)
    time.sleep(delay)
    _send_mouse(MOUSEEVENTF_MIDDLEUP)

def scroll(steps=-1):
    _send_mouse(MOUSEEVENTF_WHEEL, data=(steps * WHEEL_DELTA))

def double_click(click_delay=0.015, between_delay=0.15):
    left_click(click_delay)
    time.sleep(between_delay)
    left_click(click_delay)

def click_down(button="left"):
    if button == "right":
        _send_mouse(MOUSEEVENTF_RIGHTDOWN)
    else:
        _send_mouse(MOUSEEVENTF_LEFTDOWN)

def click_up(button="left"):
    if button == "right":
        _send_mouse(MOUSEEVENTF_RIGHTUP)
    else:
        _send_mouse(MOUSEEVENTF_LEFTUP)
