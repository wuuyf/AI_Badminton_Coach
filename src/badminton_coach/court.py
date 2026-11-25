"""Draw a badminton singles court in ShuttleSet's court-coordinate frame.

Estimated from the data (not from an official spec): the frame spans the full court,
x about 0-355 across and y about 84-850 along, net at y = 467.
That is roughly 57.5 units per metre for a 6.10 m x 13.40 m court.
"""
import matplotlib.pyplot as plt
import numpy as np

NET_Y = 467.0
UNITS_PER_M = 57.5
WIDTH = 6.10 * UNITS_PER_M            # doubles width
HALF_LEN = 6.70 * UNITS_PER_M
X0 = 0.0
Y_TOP, Y_BOT = NET_Y - HALF_LEN, NET_Y + HALF_LEN

LINE = "#ffffff"
COURT = "#2e7d5b"
COLORS = {"A": "#4c9be8", "B": "#f2c14e", "pred": "#ff7f0e", "true": "#e45756"}


def m(v: float) -> float:
    return v * UNITS_PER_M


def draw_court(ax=None, figsize=(4.2, 8)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.add_patch(plt.Rectangle((X0, Y_TOP), WIDTH, 2 * HALF_LEN, color=COURT, zorder=0))
    lw = 1.6
    # outer (doubles) boundary and singles side lines
    ax.plot([X0, X0 + WIDTH, X0 + WIDTH, X0, X0], [Y_TOP, Y_TOP, Y_BOT, Y_BOT, Y_TOP], c=LINE, lw=lw)
    for x in (X0 + m(0.46), X0 + WIDTH - m(0.46)):
        ax.plot([x, x], [Y_TOP, Y_BOT], c=LINE, lw=lw)
    for sign in (-1, 1):
        short = NET_Y + sign * m(1.98)                      # short service line
        long_d = NET_Y + sign * (HALF_LEN - m(0.76))        # doubles long service line
        ax.plot([X0, X0 + WIDTH], [short, short], c=LINE, lw=lw)
        ax.plot([X0, X0 + WIDTH], [long_d, long_d], c=LINE, lw=lw, alpha=0.6)
        end = Y_TOP if sign < 0 else Y_BOT
        ax.plot([X0 + WIDTH / 2] * 2, [short, end], c=LINE, lw=lw)   # centre line
    ax.plot([X0 - 8, X0 + WIDTH + 8], [NET_Y, NET_Y], c="#dddddd", lw=3.5, zorder=3)  # net
    ax.set_xlim(X0 - 40, X0 + WIDTH + 40)
    ax.set_ylim(Y_BOT + 40, Y_TOP - 40)                    # top of the figure = small y
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_rally(ax, xs, ys, hitters, types, first_player):
    """Numbered LANDING points of the observed strokes, linked in order.

    Colour = who HIT the stroke, so a player's dots appear on the opponent's half.
    Each player's name is written at the end of the half they defend.
    """
    ax.plot(xs, ys, c="white", lw=1, alpha=0.5, ls="--", zorder=4)
    placed = []
    for i, (x, y, p, t) in enumerate(zip(xs, ys, hitters, types), 1):
        c = COLORS["A"] if p == first_player else COLORS["B"]
        ax.scatter(x, y, s=260, c=c, edgecolors="black", lw=0.8, zorder=5)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, weight="bold", zorder=6)
        # label under the dot; move it above if another label is already close by
        ly = y + 24
        if any(abs(x - lx) < 95 and abs(ly - py) < 22 for lx, py in placed):
            ly = y - 24
        ax.text(x, ly, t, fontsize=7, color="white", va="center", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000088", ec="none"))
        placed.append((x, ly))

    # stroke 1 lands on the receiver's half -> the server defends the other half
    other = next((h for h in hitters if h != first_player), "opponent")
    server_top = ys[0] > NET_Y
    for name, top, col in [(first_player, server_top, COLORS["A"]), (other, not server_top, COLORS["B"])]:
        y = Y_TOP - 18 if top else Y_BOT + 18
        ax.text(X0 + WIDTH / 2, y, f"{name}'s side", ha="center", va="center", fontsize=8,
                weight="bold", color="black", bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))


def draw_prediction(ax, x, y, radius, true_xy=None):
    """Predicted landing (orange) with an error circle; optionally the true landing (red X)."""
    ax.add_patch(plt.Circle((x, y), radius, color=COLORS["pred"], alpha=0.25, zorder=4))
    ax.scatter(x, y, s=180, marker="*", c=COLORS["pred"], edgecolors="black", lw=0.8, zorder=7)
    if true_xy is not None:
        ax.scatter(*true_xy, s=120, marker="X", c=COLORS["true"], edgecolors="black", lw=0.8, zorder=7)


if __name__ == "__main__":       # quick visual check: python -m badminton_coach.court
    ax = draw_court()
    draw_rally(ax, np.array([147, 115, 109, 237, 221]), np.array([306, 571, 415, 769, 305]),
               ["A", "B", "A", "B", "A"], ["short service", "net shot", "net shot", "lob", "smash"], "A")
    draw_prediction(ax, 120, 560, 114, true_xy=(90, 520))
    plt.savefig("results/figures/court_preview.png", dpi=150, bbox_inches="tight")
    print("Saved results/figures/court_preview.png")
