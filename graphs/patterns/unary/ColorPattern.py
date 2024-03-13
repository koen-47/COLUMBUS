import matplotlib.colors as mcolors


class ColorPattern:
    def __init__(self):
        self.keywords = [color.replace("tab:", "") for color in list(mcolors.TABLEAU_COLORS.keys())]

