import csv
from PIL import Image, ImageDraw, ImageFont

def open_rgb(path):
    img = Image.open(f"input/{path}")
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")

def gray_img(img):
    src = img.tobytes()
    res = bytearray()
    for idx in range(0, len(src), 3):
        r = src[idx]
        g = src[idx + 1]
        b = src[idx + 2]
        res.append((299 * r + 587 * g + 114 * b + 500) // 1000)
    return Image.frombytes("L", img.size, bytes(res))

def bin_img(gray, thr):
    res = bytearray()
    for val in gray.tobytes():
        res.append(0 if val < thr else 255)
    return Image.frombytes("L", gray.size, bytes(res))

def bbox(img):
    pix = img.load()
    xs = []
    ys = []
    for y in range(img.height):
        for x in range(img.width):
            if pix[x, y] == 0:
                xs.append(x)
                ys.append(y)
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1

def crop_fg(img, pad):
    x0, y0, x1, y1 = bbox(img)
    box = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(img.width, x1 + pad),
        min(img.height, y1 + pad),
    )
    return img.crop(box)

def profs(img):
    src = img.tobytes()
    wid, hgt = img.size
    vert = []
    for x in range(wid):
        cnt = 0
        for y in range(hgt):
            if src[y * wid + x] == 0:
                cnt += 1
        vert.append(cnt)
    horiz = []
    for y in range(hgt):
        row = src[y * wid : (y + 1) * wid]
        horiz.append(sum(1 for val in row if val == 0))
    return horiz, vert

def runs(vals, cut):
    res = []
    start = None
    for idx, val in enumerate(vals):
        if val > cut and start is None:
            start = idx
        elif val <= cut and start is not None:
            res.append((start, idx))
            start = None
    if start is not None:
        res.append((start, len(vals)))
    return res

def seg_chars(img, cut):
    _, vert = profs(img)
    xruns = runs(vert, cut)
    pix = img.load()
    boxes = []
    for x0, x1 in xruns:
        ys = []
        for y in range(img.height):
            for x in range(x0, x1):
                if pix[x, y] == 0:
                    ys.append(y)
                    break
        if ys:
            boxes.append((x0, min(ys), x1, max(ys) + 1))
    return merge_cmp(boxes, img.height)

def merge_cmp(boxes, hgt):
    res = []
    mid = hgt // 2
    for box in boxes:
        x0, y0, x1, y1 = box
        wid = x1 - x0
        if res:
            px0, py0, px1, py1 = res[-1]
            gap = x0 - px1
            over = min(y1, py1) - max(y0, py0)
            intern = wid <= 6 and gap <= 4 and y0 < mid and over > 0
            if intern:
                res[-1] = (px0, min(py0, y0), x1, max(py1, y1))
                continue
        res.append(box)
    return res

def draw_prof(vals, title, kind, path):
    ch_w = max(500, len(vals) * 3 + 100)
    ch_h = 260
    left = 46
    top = 34
    right = 24
    bottom = 34
    pl_w = ch_w - left - right
    pl_h = ch_h - top - bottom
    max_v = max(max(vals), 1)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 13)
    small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 10)
    img = Image.new("RGB", (ch_w, ch_h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((left, top, left + pl_w, top + pl_h), outline=(35, 35, 35))
    draw.text((left, 10), title, fill=(0, 0, 0), font=font)
    step = max(1, max_v // 4)
    for tick in range(0, max_v + 1, step):
        y = top + pl_h - round(tick / max_v * pl_h)
        draw.line((left - 3, y, left + pl_w, y), fill=(225, 225, 225))
        draw.text((6, y - 6), str(tick), fill=(0, 0, 0), font=small)
    if kind == "vert":
        col_w = pl_w / max(1, len(vals))
        for idx, val in enumerate(vals):
            x0 = left + int(idx * col_w)
            x1 = left + max(x0 - left + 1, int((idx + 1) * col_w))
            y0 = top + pl_h - round(val / max_v * pl_h)
            draw.rectangle((x0, y0, x1, top + pl_h), fill=(38, 94, 150))
    else:
        row_h = pl_h / max(1, len(vals))
        for idx, val in enumerate(vals):
            y0 = top + int(idx * row_h)
            y1 = top + max(y0 - top + 1, int((idx + 1) * row_h))
            x1 = left + round(val / max_v * pl_w)
            draw.rectangle((left, y0, x1, y1), fill=(90, 120, 55))
    img.save(path)

def draw_boxes(img, boxes):
    res = img.convert("RGB")
    draw = ImageDraw.Draw(res)
    colors = [(220, 40, 40), (30, 120, 210), (50, 155, 70), (190, 90, 20)]
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colors[idx % len(colors)], width=1)
    return res

def text_box(txt, font):
    img = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(img)
    return draw.textbbox((0, 0), txt, font=font)

def rend_sym(sym, font, thr):
    box = text_box(sym, font)
    wid = box[2] - box[0] + 24
    hgt = box[3] - box[1] + 24
    img = Image.new("L", (wid, hgt), 255)
    draw = ImageDraw.Draw(img)
    draw.text((12 - box[0], 12 - box[1]), sym, fill=0, font=font)
    return crop_fg(bin_img(img, thr), 2)

def save_segs(img, boxes, out_dir):
    seg_dir = f"{out_dir}/segments"
    for idx, box in enumerate(boxes, 1):
        x0, y0, x1, y1 = box
        crop = img.crop((
            max(0, x0 - 1),
            max(0, y0 - 1),
            min(img.width, x1 + 1),
            min(img.height, y1 + 1)
        ))
        crop.save(f"{seg_dir}/{idx:02d}.png")

def csv_boxes(boxes, out_dir):
    path = f"{out_dir}/segments.csv"
    with open(path, "w", encoding="utf-8", newline="") as file:
        wrt = csv.writer(file, delimiter=";")
        wrt.writerow(["index", "x0", "y0", "x1", "y1", "width", "height"])
        for idx, box in enumerate(boxes, 1):
            x0, y0, x1, y1 = box
            wrt.writerow([idx, x0, y0, x1, y1, x1 - x0, y1 - y0])

def csv_prof(name, horiz, vert, out_dir):
    path = f"{out_dir}/{name}_profiles.csv"
    with open(path, "w", encoding="utf-8", newline="") as file:
        wrt = csv.writer(file, delimiter=";")
        wrt.writerow(["axis", "index", "value"])
        for idx, val in enumerate(horiz):
            wrt.writerow(["horizontal_y", idx, val])
        for idx, val in enumerate(vert):
            wrt.writerow(["vertical_x", idx, val])

def gen_alph(alph, out_dir, thr):
    sym_dir = f"{out_dir}/symbols"
    prof_dir = f"{out_dir}/profiles"
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 14)
    for sym in alph:
        code = f"{ord(sym):04X}"
        img = rend_sym(sym, font, thr)
        img.save(f"{sym_dir}/{code}.png")
        horiz, vert = profs(img)
        draw_prof(horiz, f"{sym} horizontal profile", "horiz", f"{prof_dir}/{code}_horizontal.png")
        draw_prof(vert, f"{sym} vertical profile", "vert", f"{prof_dir}/{code}_vertical.png")

def save_imgs(src_nm, out_dir):
    thr = 160
    cut = 1
    src = open_rgb(src_nm)
    src.save(f"{out_dir}/source.png")
    gray = gray_img(src)
    img = crop_fg(bin_img(gray, thr), 0)
    horiz, vert = profs(img)
    boxes = seg_chars(img, cut)
    img.save(f"{out_dir}/binary.bmp")
    img.save(f"{out_dir}/binary.png")
    draw_prof(horiz, "Horizontal text profile", "horiz", f"{out_dir}/profile_horizontal.png")
    draw_prof(vert, "Vertical text profile", "vert", f"{out_dir}/profile_vertical.png")
    draw_boxes(img, boxes).save(f"{out_dir}/segmented.png")
    save_segs(img, boxes, out_dir)
    csv_boxes(boxes, out_dir)
    csv_prof("text", horiz, vert, out_dir)

def proc_img(src_nm):
    save_imgs(src_nm, "output")

def main():
    alph = (
        "АӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФХҺЦЧШЩЪЫІЬЭЮЯ"
    )
    proc_img("1.png")
    gen_alph(alph, "output", 160)

if __name__ == "__main__":
    main()
