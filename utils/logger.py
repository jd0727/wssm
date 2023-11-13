import os
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(fmt='%(asctime)s %(module)s[%(funcName)s] %(levelname)s : %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

LOGGER = logging.getLogger(name='Detect')
LOGGER.setLevel(logging.DEBUG)
print_ori = print
# print = LOGGER.info
print = print


def logsys():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(FORMATTER)
    LOGGER.addHandler(handler)
    return None


def logfile(log_pth, new_log=False):
    if not os.path.exists(os.path.dirname(log_pth)):
        os.makedirs(os.path.dirname(log_pth))
    if new_log and os.path.isfile(log_pth):
        os.remove(log_pth)
    handler = TimedRotatingFileHandler(log_pth, when='D', encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(FORMATTER)
    LOGGER.addHandler(handler)
    return handler


def rmv_handler(handler):
    LOGGER.removeHandler(handler)
    return True


def rmv_file_handler(file_pth):
    for handler in LOGGER.handlers:
        if isinstance(handler, TimedRotatingFileHandler) and handler.baseFilename == file_pth:
            LOGGER.removeHandler(handler)
    return True


logsys()


class STYLE:
    class FORE:
        BLACK = 30
        RED = 31
        GREEN = 32
        YELLO = 33
        BLUE = 34
        PURPLE = 35
        CYAN = 36
        WHITE = 37

    class BACK:
        BLACK = 40
        RED = 41
        GREEN = 42
        YELLO = 43
        BLUE = 44
        PURPLE = 45
        CYAN = 46
        WHITE = 47

    class MODE:
        NORMAL = 0  # 终端默认设置
        BOLD = 1  # 高亮显示
        UNDERLINE = 4  # 使用下划线
        BLINK = 5  # 闪烁
        INVERT = 7  # 反白显示
        HIDE = 8  # 不可见

    class DEFAULT:
        END = 0


def style_str(string, mode=STYLE.MODE.NORMAL, fore=STYLE.FORE.BLACK, back=STYLE.BACK.RED):
    style = ';'.join(['%s' % s for s in [mode, fore, back]])
    style = '\033[%sm' % style if style else ''
    end = '\033[%sm' % STYLE.DEFAULT.END if style else ''
    return '%s%s%s' % (style, string, end)


if __name__ == '__main__':
    handler = logfile('D://DeskTop//xx.txt')
    print(style_str('黑色', mode=STYLE.MODE.NORMAL, fore=STYLE.FORE.BLACK, back=STYLE.BACK.RED))
