import csv
import math
from PIL import Image, ImageDraw, ImageFont

def load_font(font_p, size):
    return ImageFont.truetype(font_p, size=size)

def get_bbox(txt, font):
    img = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(img)
    return draw.textbbox((0, 0), txt, font=font)

def make_sym(sym, font, thr):
    box = get_bbox(sym, font)
    wid = box[2] - box[0] + 40
    hei = box[3] - box[1] + 40
    img = Image.new("L", (wid, hei), 255)
    draw = ImageDraw.Draw(img)
    draw.text((20 - box[0], 20 - box[1]), sym, fill=0, font=font)

    pix = img.load()
    xs = []
    ys = []
    for y in range(img.height):
        for x in range(img.width):
            if pix[x, y] < thr:
                xs.append(x)
                ys.append(y)
    crop = img.crop((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return crop.point(lambda val: 0 if val < thr else 255, mode="1").convert("L")

def get_pts(img):
    pix = img.load()
    return [
        (x, y)
        for y in range(img.height)
        for x in range(img.width)
        if pix[x, y] == 0
    ]

def cnt_blk(img, x0, y0, x1, y1):
    pix = img.load()
    cnt = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if pix[x, y] == 0:
                cnt += 1
    return cnt

def get_prof(img):
    pix = img.load()
    prof_x = [
        sum(1 for y in range(img.height) if pix[x, y] == 0)
        for x in range(img.width)
    ]
    prof_y = [
        sum(1 for x in range(img.width) if pix[x, y] == 0)
        for y in range(img.height)
    ]
    return prof_x, prof_y

def get_feat(sym, img):
    pts = get_pts(img)
    mass = len(pts)
    wid, hei = img.size
    mid_x = wid // 2
    mid_y = hei // 2
    quads = [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, wid, mid_y),
        (0, mid_y, mid_x, hei),
        (mid_x, mid_y, wid, hei),
    ]
    q_w = [cnt_blk(img, *quad) for quad in quads]
    q_a = [max(1, (x1 - x0) * (y1 - y0)) for x0, y0, x1, y1 in quads]
    q_d = [q_w[i] / q_a[i] for i in range(4)]

    cen_x = sum(x for x, _ in pts) / mass
    cen_y = sum(y for _, y in pts) / mass
    iner_x = sum((y - cen_y) ** 2 for _, y in pts)
    iner_y = sum((x - cen_x) ** 2 for x, _ in pts)

    row = {
        "symbol": sym,
        "unicode": f"U+{ord(sym):04X}",
        "width": wid,
        "height": hei,
        "mass": mass,
        "center_x": cen_x,
        "center_y": cen_y,
        "center_x_norm": cen_x / (wid - 1) if wid > 1 else 0.0,
        "center_y_norm": cen_y / (hei - 1) if hei > 1 else 0.0,
        "inertia_x": iner_x,
        "inertia_y": iner_y,
        "inertia_x_norm": iner_x / (mass * hei * hei),
        "inertia_y_norm": iner_y / (mass * wid * wid),
    }
    for i, val in enumerate(q_w, 1):
        row[f"q{i}_weight"] = val
    for i, val in enumerate(q_d, 1):
        row[f"q{i}_density"] = val
    return row

def draw_bar(vals, title, orient, path):
    ch_w = max(520, len(vals) * 12 + 100)
    ch_h = 360
    mar_l = 58
    mar_t = 40
    mar_r = 26
    mar_b = 46
    plt_w = ch_w - mar_l - mar_r
    plt_h = ch_h - mar_t - mar_b
    max_v = max(vals) if vals else 1
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 14)
    small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 11)

    img = Image.new("RGB", (ch_w, ch_h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((mar_l, mar_t, mar_l + plt_w, mar_t + plt_h), outline=(40, 40, 40))
    draw.text((mar_l, 12), title, fill=(0, 0, 0), font=font)

    step = max(1, math.ceil(max_v / 5))
    for tick in range(0, max_v + 1, step):
        if orient == "vert":
            y = mar_t + plt_h - round(tick / max_v * plt_h)
            draw.line((mar_l - 4, y, mar_l + plt_w, y), fill=(225, 225, 225))
            draw.text((8, y - 7), str(tick), fill=(0, 0, 0), font=small)
        else:
            x = mar_l + round(tick / max_v * plt_w)
            draw.line((x, mar_t, x, mar_t + plt_h + 4), fill=(225, 225, 225))
            draw.text((x - 6, ch_h - 34), str(tick), fill=(0, 0, 0), font=small)

    if orient == "vert":
        bar_w = max(2, plt_w // max(1, len(vals)))
        for i, val in enumerate(vals):
            x0 = mar_l + i * bar_w
            x1 = x0 + max(1, bar_w - 1)
            y0 = mar_t + plt_h - round(val / max_v * plt_h)
            draw.rectangle((x0, y0, x1, mar_t + plt_h), fill=(35, 95, 155))
        lab_st = max(1, math.ceil(len(vals) / 12))
        for i in range(0, len(vals), lab_st):
            x = mar_l + i * bar_w
            draw.text((x - 4, ch_h - 32), str(i), fill=(0, 0, 0), font=small)
    else:
        bar_h = max(2, plt_h // max(1, len(vals)))
        for i, val in enumerate(vals):
            y0 = mar_t + i * bar_h
            y1 = y0 + max(1, bar_h - 1)
            x1 = mar_l + round(val / max_v * plt_w)
            draw.rectangle((mar_l, y0, x1, y1), fill=(95, 115, 45))
        lab_st = max(1, math.ceil(len(vals) / 12))
        for i in range(0, len(vals), lab_st):
            y = mar_t + i * bar_h
            draw.text((24, y - 6), str(i), fill=(0, 0, 0), font=small)

    draw.text((ch_w // 2 - 18, ch_h - 18), "index", fill=(0, 0, 0), font=small)
    img.save(path)


def save_imgs(sym, img, prof_x, prof_y, out_dir):
    stem = f"{ord(sym):04X}"
    sym_png = f"{out_dir}/symbols/{stem}.png"
    prof_xp = f"{out_dir}/profiles/{stem}_x.png"
    prof_yp = f"{out_dir}/profiles/{stem}_y.png"

    img.save(sym_png)
    draw_bar(prof_x, f"{sym} ({stem}) profile X", "vert", prof_xp)
    draw_bar(prof_y, f"{sym} ({stem}) profile Y", "horiz", prof_yp)


def write_csv(rows, out_dir):
    cols = ["symbol", "unicode", "width", "height", "mass",
        "q1_weight", "q2_weight", "q3_weight", "q4_weight",
        "q1_density", "q2_density", "q3_density", "q4_density",
        "center_x", "center_y", "center_x_norm", "center_y_norm",
        "inertia_x", "inertia_y", "inertia_x_norm", "inertia_y_norm"]
    with open(f"{out_dir}/features.csv", "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols, delimiter=";")
        writer.writeheader()
        for row in rows:
            fmt = {}
            for key, val in row.items():
                if isinstance(val, float):
                    fmt[key] = f"{val:.6f}".replace(".", ",")
                else:
                    fmt[key] = val
            writer.writerow(fmt)

def main():
    out_dir = "output"
    font_p = r"C:\Windows\Fonts\times.ttf"
    size = 14
    thr = 128
    chars = "АӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФХҺЦЧШЩЪЫІЬЭЮЯ"

    font = load_font(font_p, size)
    rows = []
    for sym in chars:
        img = make_sym(sym, font, thr)
        prof_x, prof_y = get_prof(img)
        save_imgs(sym, img, prof_x, prof_y, out_dir)
        rows.append(get_feat(sym, img))
    write_csv(rows, out_dir)

if __name__ == "__main__":
    main()
