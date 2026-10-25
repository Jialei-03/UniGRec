import os


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except ValueError:
        index = len(color_set) - 1
    prev_log = "\033[" + ("1;3" if highlight else "0;3") + str(index) + "m"
    return prev_log + log + "\033[0m"
