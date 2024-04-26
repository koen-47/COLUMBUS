import nltk
import matplotlib.colors as mcolors
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

nltk.download("popular", quiet=True)

df = pd.read_csv("../saved/ladec_raw.csv")
df = df[df["isCommonstim"] == 1]
df[["c1", "c2", "stim", "isPlural"]].to_csv("../saved/ladec_common_raw_small.csv", index=False)

basic_colors = [color.replace("tab:", "") for color in list(mcolors.TABLEAU_COLORS.keys())]
color_compounds = df[df["c1"].isin(basic_colors) | df["c2"].isin(basic_colors)]
color_compounds_zip = zip(color_compounds["c1"], color_compounds["c2"])

color_graphs = [(c1, "color", c2) for c1, c2 in color_compounds_zip if c1 in basic_colors] + \
               [(c2, "color", c1) for c1, c2 in color_compounds_zip if c2 in basic_colors]
print(len(color_graphs))


def create_color_rebus(text, color):
    W, H = (200, 200)
    font = ImageFont.truetype("arial.ttf", 32)
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    draw.text(((W-w)/2, (H-h)/2), text, font=font, fill=color)
    return image


# for color, _, word in color_graphs:
#     color_rebus_img = create_color_rebus(word, color)
#     color_rebus_img.save(f"../results/{word}_{color}.png", "PNG")
