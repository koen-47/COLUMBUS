# import matplotlib
# import matplotlib.pyplot as plt
#
# matplotlib.rcParams["text.usetex"] = True
#
# # Create a plot
# fig, ax = plt.subplots(figsize=(4, 4))
#
# font_size = 40
# font_weight = "bold"
#
#
# def render_inside_outside_text(inside_word, outside_word):
#     # if len(outside_word) % 2 == 1:
#     #     outside_word = f"{outside_word} {outside_word}"
#     # outside_left = outside_word[:int(len(outside_word)/2)]
#     # outside_right = outside_word[int(len(outside_word)/2):]
#     #
#     # x, y = 0.5, 0.5
#     # ax.text(x, y, inside_word, fontsize=font_size, fontweight="bold", fontfamily="Consolas", color="black", ha="center",
#     #         va="center")
#     #
#     # x_left = x - (0.005 * (len(inside_word) / 2) * font_size * len(outside_left))
#     # ax.text(x_left, 0.5, outside_left, fontsize=font_size, fontweight="bold", fontfamily="Consolas", color="black", ha="center",
#     #         va="center")
#     #
#     # x_right = x + (0.005 * (len(inside_word) / 2) * font_size * len(outside_right))
#     # ax.text(x_right, 0.5, outside_right, fontsize=font_size, fontweight="bold", fontfamily="Consolas", color="black", ha="center",
#     #         va="center")
#
#     plt.text(0.5, 0.5, r'$\mathrm{asdf}$', fontsize=12)
#
# render_inside_outside_text("ALL", "ALL")
#
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.axis('off')
#
# # plt.tight_layout()
# plt.show()

# import matplotlib
# matplotlib.use('ps')
# from matplotlib import rc
#
# rc('text',usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.ylabel(r'\textcolor{red}{Today} '+
#            r'\textcolor{green}{is} '+
#            r'\textcolor{blue}{cloudy.}')
# plt.savefig('test.ps')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# plt.axvline(x=0.5)

# inside_word = "MOOOOOOOOOOOOOOON"
# outside_word = "ONCE"
# x = 0.5 - 0.019 * (len(inside_word + outside_word) / 2)
#
# text = ax.text(x, 0.5, "ON", color="black", ha="center", va="center", weight="bold")
# text = ax.annotate(inside_word, xycoords=text, xy=(1, 0), verticalalignment="bottom", color="blue", weight="bold")
# text = ax.annotate("CE", xycoords=text, xy=(1, 0), verticalalignment="bottom", color="black", weight="bold")

# text = ax.text(0.5, 0.5, "MOOOOOOOOOOOOOOOOON", color="blue", ha="center", va="center", weight="bold", fontsize=20, fontfamily="Consolas")
# ax.annotate("ALL", xycoords=text, xy=(0, 0), va="bottom", ha="right", color="black", weight="bold", fontsize=20, fontfamily="Consolas")
# ax.annotate("ALL", xycoords=text, xy=(1, 0), va="bottom", color="black", weight="bold", fontsize=20, fontfamily="Consolas")
#
#
# plt.show()

# text = "EFFECT → ___ ←"
# text = " ↓ \nA I R\n ↑ \n ↓ \nA I R\n ↑ "
# text = "→ SECRET\n  SECRET\n  SECRET"
# text = "+---------+\n" + "| CHATTER |\n" + "+---------+\n"
# text = "PUNCH\n" \
#        "───────────\n" \
#        "1111 WEIGHT\n"
#
# # text = "N\nE\nA\nL\nC"
#
#
# ax.text(0.5, 0.5, text, color="black", ha="center", va="center", weight="bold", fontsize=40, fontfamily="Consolas")
# plt.show()


import numpy as np

points = np.linspace(0.25, 0.75, 1).tolist()
print(points)
for p in points:
    ax.text(0.5, p, "TEST", color="black", ha="center", va="center", weight="bold", fontsize=40, fontfamily="Consolas")
plt.show()



