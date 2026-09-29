"""Drawn test images with known answers, for Lichen's image questions.

`cases(random.Random(7))` gives 84 (task, image, question, answer) tuples: the colour of a
shape, the kind of shape, a count of 1 to 6 dots, the word on a stamp, the trend of a line
chart, a customer message drawn as a picture (which team should handle it), whether a red
object is present (a noul), and how full a container is (a score). The answer of a noul is a
bool, of a score its level index, of a choice its key. Needs Pillow.
"""
import math

from PIL import Image, ImageDraw, ImageFont

COLOURS = {"red": (215, 35, 35), "green": (35, 150, 55), "blue": (35, 75, 215),
           "yellow": (235, 200, 20), "purple": (125, 45, 165)}
TICKETS = {
    "billing": ["I was charged twice for my subscription this month. Please refund one of the charges.",
                "My invoice shows the wrong company address. Can you send a corrected one?",
                "The refund you promised two weeks ago still has not reached my card.",
                "Why was my card declined when I paid the annual renewal?",
                "Payouts to my bank account have failed since Monday."],
    "technical": ["The dashboard shows a blank white page after I log in, in Chrome and in Firefox.",
                  "Your API returns 502 errors for every request since this morning.",
                  "The webhook for order.created stopped firing after your last update.",
                  "I can't connect the Slack integration. It says 'invalid redirect URI'.",
                  "The mobile app crashes as soon as I open the settings screen."],
    "sales": ["We would like a quote for 250 seats on the enterprise plan.",
              "What is the difference in price between Pro and Business?",
              "Can I move up to the Team plan in the middle of the month?",
              "We are a nonprofit. Do you give a discount to new accounts?",
              "I want to talk to someone about adding 40 more users to our contract."],
}
FILL = ["empty", "about a quarter full", "about half full", "about three quarters full", "full"]


def font(size):
    return ImageFont.load_default(size=size)


def shape(d, kind, box, colour):
    x0, y0, x1, y1 = box
    if kind == "circle":
        d.ellipse(box, fill=colour)
    elif kind == "square":
        d.rectangle(box, fill=colour)
    else:
        d.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=colour)


def blank(w=448, h=448):
    img = Image.new("RGB", (w, h), "white")
    return img, ImageDraw.Draw(img)


def cases(rng):
    """(task, image, question, truth) for every test image."""
    out = []
    names = list(COLOURS)
    for i in range(10):
        c = names[i % len(names)]
        img, d = blank()
        s = rng.randint(150, 260)
        x, y = rng.randint(20, 428 - s), rng.randint(20, 428 - s)
        shape(d, rng.choice(["circle", "square", "triangle"]), (x, y, x + s, y + s), COLOURS[c])
        out.append(("colour", img, {"type": "choice", "instructions": "What colour is the shape in the image?",
                                    "criteria": {n: "" for n in names}}, c))
    for i in range(9):
        k = ["circle", "square", "triangle"][i % 3]
        img, d = blank()
        s = rng.randint(150, 260)
        x, y = rng.randint(20, 428 - s), rng.randint(20, 428 - s)
        shape(d, k, (x, y, x + s, y + s), COLOURS[rng.choice(names)])
        out.append(("shape", img, {"type": "choice", "instructions": "What shape is in the image?",
                                   "criteria": {"circle": "", "square": "", "triangle": ""}}, k))
    words = ["one", "two", "three", "four", "five", "six"]
    for i in range(12):
        n = i % 6 + 1
        img, d = blank()
        dots = []
        while len(dots) < n:
            r = rng.randint(20, 34)
            x, y = rng.randint(r + 10, 438 - r), rng.randint(r + 10, 438 - r)
            if all(math.hypot(x - a, y - b) > r + q + 12 for a, b, q in dots):
                dots.append((x, y, r))
        for x, y, r in dots:
            d.ellipse((x - r, y - r, x + r, y + r), fill="black")
        out.append(("count", img, {"type": "choice", "instructions": "How many black dots are in the image?",
                                   "criteria": {w: "" for w in words}}, words[n - 1]))
    for i in range(9):
        word = ["APPROVED", "REJECTED", "PENDING"][i % 3]
        img, d = blank(560, 420)
        for row in range(14):
            d.rectangle((40, 30 + row * 26, 40 + rng.randint(280, 480), 40 + row * 26), fill=(200, 200, 200))
        stamp = Image.new("RGBA", (420, 110), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp)
        ink = {"APPROVED": (30, 130, 50), "REJECTED": (190, 30, 30), "PENDING": (40, 60, 170)}[word]
        sd.rectangle((4, 4, 416, 106), outline=ink, width=6)
        sd.text((210, 55), word, font=font(64), fill=ink, anchor="mm")
        stamp = stamp.rotate(rng.uniform(-15, 15), expand=True)
        img.paste(stamp, (rng.randint(20, 120), rng.randint(60, 220)), stamp)
        out.append(("stamp", img, {"type": "choice", "instructions": "What does the stamp on this document say?",
                                   "criteria": {"approved": "", "rejected": "", "pending": ""}}, word.lower()))
    for i in range(9):
        trend = ["rising", "falling", "flat"][i % 3]
        img, d = blank(520, 340)
        d.line((50, 20, 50, 300, 500, 300), fill="black", width=2)
        slope = {"rising": 1, "falling": -1, "flat": 0}[trend] * rng.uniform(12, 20)
        start = {"rising": 70, "falling": 250, "flat": 160}[trend]
        pts = [(60 + j * 38, 300 - (start + slope * j + rng.uniform(-10, 10))) for j in range(12)]
        d.line(pts, fill=(35, 75, 215), width=4)
        d.text((260, 318), "month", font=font(16), fill="black", anchor="mm")
        out.append(("chart", img, {"type": "choice", "instructions": "What is the trend of the line in this chart?",
                                   "criteria": {"rising": "", "falling": "", "flat": ""}}, trend))
    for team, messages in TICKETS.items():
        for m in messages:
            img, d = blank(640, 260)
            d.rounded_rectangle((20, 20, 620, 240), radius=18, fill=(236, 240, 246))
            d.text((44, 36), "Customer message", font=font(18), fill=(90, 90, 90))
            lines, line = [], ""
            for w in m.split():
                if len(line) + len(w) > 44:
                    lines.append(line)
                    line = ""
                line = f"{line} {w}".strip()
            d.multiline_text((44, 76), "\n".join(lines + [line]), font=font(26), fill="black", spacing=10)
            out.append(("ticket", img, {"type": "choice", "instructions": "Which team should handle this?",
                                        "criteria": {"billing": "Payments, invoicing, refunds",
                                                     "technical": "Bugs, outages, integrations",
                                                     "sales": "Pricing, upgrades, new accounts"}}, team))
    others = [n for n in names if n != "red"]
    for i in range(10):
        img, d = blank()
        colours = rng.sample(others, 3)
        if i % 2 == 0:
            colours[rng.randrange(3)] = "red"
        for j, c in enumerate(colours):
            s = rng.randint(80, 120)
            x, y = 20 + j * 140, rng.randint(20, 428 - s)
            shape(d, rng.choice(["circle", "square", "triangle"]), (x, y, x + s, y + s), COLOURS[c])
        out.append(("red", img, {"type": "noul", "instructions": "Is there a red object in the image?"},
                    "red" in colours))
    for i in range(10):
        level = i % 5
        img, d = blank(300, 448)
        d.rectangle((100, 40, 200, 420), outline="black", width=4)
        top = 418 - level * 94
        if level:
            d.rectangle((103, top, 197, 417), fill=(35, 75, 215))
        out.append(("fill", img, {"type": "score", "instructions": "How full is the container in the image?",
                                  "criteria": FILL}, level))
    return out
