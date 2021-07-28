import matplotlib.pyplot as plt
from matplotlib import rc


def setup_tex():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'CMU Serif'
    rc('text', **{'usetex': True})


def esc(s: str):
    CHARS = {
        '&':  r'\&',
        '%':  r'\%',
        '$':  r'\$',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\~',
        '^':  r'\^',
        '\\': r'\\',
    }
    return "".join([CHARS.get(c, c) for c in s])
