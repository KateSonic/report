import csv
import math
from PIL import Image, ImageDraw, ImageFont

def get_data():
    alph = "АӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФХҺЦЧШЩЪЫІЬЭЮЯ"
    return {
        "alph": alph,
        "text": "ОСЫ ЖҰМСАҚ ФРАНЦУЗ ОРАМДАРЫН КӨБІРЕК ЖЕП ШАЙ ІШІҢІЗ",
        "font": r"C:\Windows\Fonts\times.ttf",
        "size": 14,
        "exp_sz": 18,
        "top": 10}

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

def load_font(path, size):
    return ImageFont.truetype(path, size=size)

def rend_sym(ch, size, cfg):
    font = load_font(cfg["font"], size)
    box = text_box(ch, font)
    img = Image.new("L", (box[2] - box[0] + 24, box[3] - box[1] + 24), 255)
    draw = ImageDraw.Draw(img)
    draw.text((12 - box[0], 12 - box[1]), ch, fill=0, font=font)
    return crop_fg(bin_img(img, 160), 0)

def rend_txt(text, size, cfg, track=8):
    parts = []
    for ch in text:
        parts.append(None if ch == " " else rend_sym(ch, size, cfg))
    height = max(part.height for part in parts if part is not None)
    width = sum((track * 4) if part is None else part.width + track for part in parts)
    img = Image.new("L", (width, height), 255)
    x = 0
    for part in parts:
        if part is None:
            x += track * 4
        else:
            img.paste(part, (x, height - part.height))
            x += part.width + track
    return img

def save_segs(img, boxes, out_dir):
    for num, (x0, y0, x1, y1) in enumerate(boxes, 1):
        seg = img.crop((
            max(0, x0 - 1),
            max(0, y0 - 1),
            min(img.width, x1 + 1),
            min(img.height, y1 + 1),
        ))
        seg.save(f"{out_dir}/segments/{num:02d}.png")

def save_imgs(name, img, boxes, out_dir):
    stem = name.rsplit(".", 1)[0]
    bin_png = f"{out_dir}/{stem}_binary.png"
    seg_png = f"{out_dir}/{stem}_segmented.png"
    img.save(bin_png)
    marked = draw_boxes(img, boxes)
    marked.save(seg_png)
    if stem in ("main", "experiment"):
        img.save(f"{out_dir}/binary.png")
        marked.save(f"{out_dir}/segmented.png")
    save_segs(img, boxes, out_dir)

def black_pts(img):
    pix = img.load()
    pts = []
    for y in range(img.height):
        for x in range(img.width):
            if pix[x, y] == 0:
                pts.append((x, y))
    return pts

def feat_vec(img):
    pts = black_pts(img)
    if not pts:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mass = len(pts)
    width, height = img.size
    cx = sum(x for x, _ in pts) / mass
    cy = sum(y for _, y in pts) / mass
    ix = sum((y - cy) ** 2 for _, y in pts)
    iy = sum((x - cx) ** 2 for x, _ in pts)
    return (
        float(mass),
        cx / (width - 1) if width > 1 else 0.0,
        cy / (height - 1) if height > 1 else 0.0,
        ix / (mass * height * height),
        iy / (mass * width * width),
    )

def feat_row(num, ch, img):
    _, cx, cy, ix, iy = feat_vec(img)
    pts = black_pts(img)
    mass = len(pts)
    return {
        "index": "" if num is None else num,
        "symbol": ch,
        "unicode": "" if not ch else f"U+{ord(ch):04X}",
        "width": img.width,
        "height": img.height,
        "mass": mass,
        "mass_norm": mass / max(1, img.width * img.height),
        "center_x_norm": cx,
        "center_y_norm": cy,
        "inertia_x_norm": ix,
        "inertia_y_norm": iy,
    }

def dist(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def sim(val):
    return 1.0 / (1.0 + val)

def norm_vec(vec, ranges):
    res = []
    for val, (lo, hi) in zip(vec, ranges):
        span = hi - lo
        res.append((val - lo) / span if span else 0.0)
    return tuple(res)

def refs(size, cfg):
    res = {}
    for ch in cfg["alph"]:
        res[ch] = feat_vec(rend_sym(ch, size, cfg))
    return res

def classify(seg_vecs, ref_vecs):
    vals = list(ref_vecs.values())
    ranges = []
    for i in range(5):
        ranges.append((min(vec[i] for vec in vals), max(vec[i] for vec in vals)))
    n_refs = {}
    for ch, vec in ref_vecs.items():
        n_refs[ch] = norm_vec(vec, ranges)
    res = []
    for vec in seg_vecs:
        n_vec = norm_vec(vec, ranges)
        hyps = []
        for ch, ref in n_refs.items():
            hyps.append((ch, sim(dist(n_vec, ref))))
        hyps.sort(key=lambda item: item[1], reverse=True)
        res.append(hyps)
    return res

def clean_txt(text, alph):
    return "".join(ch for ch in text if ch in alph)

def cmp_txt(expect, recog):
    limit = min(len(expect), len(recog))
    subs = sum(1 for i in range(limit) if expect[i] != recog[i])
    errs = subs + abs(len(expect) - len(recog))
    acc = 100.0 * (len(expect) - errs) / len(expect) if expect else 0.0
    return errs, max(0.0, acc)

def write_csv(path, rows):
    cols = ["index", "symbol", "unicode", "width",
        "height", "mass", "mass_norm",
        "center_x_norm", "center_y_norm", "inertia_x_norm", "inertia_y_norm"]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols, delimiter=";")
        writer.writeheader()
        for row in rows:
            fmt = {}
            for key, val in row.items():
                fmt[key] = f"{val:.6f}".replace(".", ",") if isinstance(val, float) else val
            writer.writerow(fmt)

def write_hyp(path, hyps):
    lines = []
    for num, row in enumerate(hyps, 1):
        vals = ", ".join(f'("{ch}", {score:.4f})' for ch, score in row)
        lines.append(f"{num}: [{vals}]")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

def write_top(path, hyps, top):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["index", "rank", "symbol", "similarity"])
        for num, row in enumerate(hyps, 1):
            for rank, (ch, score) in enumerate(row[:top], 1):
                writer.writerow([num, rank, ch, f"{score:.6f}".replace(".", ",")])

def run_case(size, out_dir, name, cfg):
    img = rend_txt(cfg["text"], size, cfg)
    boxes = seg_chars(img, 1)
    save_imgs(name, img, boxes, out_dir)
    segs = []
    for x0, y0, x1, y1 in boxes:
        segs.append(img.crop((x0, y0, x1, y1)))
    ref_vecs = refs(size, cfg)
    seg_vecs = [feat_vec(seg) for seg in segs]
    hyps = classify(seg_vecs, ref_vecs)
    recog = "".join(row[0][0] for row in hyps)
    expect = clean_txt(cfg["text"], cfg["alph"])
    errs, acc = cmp_txt(expect, recog)
    return {
        "img": img,
        "boxes": boxes,
        "segs": segs,
        "hyps": hyps,
        "recog": recog,
        "expect": expect,
        "errs": errs,
        "acc": acc
    }

def save_case(res, out_dir, cfg):
    ref_rows = []
    for ch in cfg["alph"]:
        ref_rows.append(feat_row(None, ch, rend_sym(ch, cfg["size"], cfg)))
    seg_rows = []
    for num, seg in enumerate(res["segs"], 1):
        seg_rows.append(feat_row(num, "", seg))
    write_csv(f"{out_dir}/reference_features.csv", ref_rows)
    write_csv(f"{out_dir}/segment_features.csv", seg_rows)
    write_hyp(f"{out_dir}/hypotheses.txt", res["hyps"])
    write_top(f"{out_dir}/top_hypotheses.csv", res["hyps"], cfg["top"])
    with open(f"{out_dir}/recognized.txt", "w", encoding="utf-8") as file:
        file.write(res["recog"])

def main():
    cfg = get_data()
    main_res = run_case(cfg["size"], "output", "main.png", cfg)
    exp_res = run_case(cfg["exp_sz"], "output/experiment", "experiment.png", cfg)
    save_case(main_res, "output", cfg)
    exp_rows = []
    for num, seg in enumerate(exp_res["segs"], 1):
        exp_rows.append(feat_row(num, "", seg))
    write_csv("output/experiment/segment_features.csv", exp_rows)
    write_hyp("output/experiment/hypotheses.txt", exp_res["hyps"])
    write_top("output/experiment/top_hypotheses.csv", exp_res["hyps"], cfg["top"])
    with open("output/experiment/recognized.txt", "w", encoding="utf-8") as file:
        file.write(exp_res["recog"])
    print(f"main: {len(main_res['segs'])} segments, {main_res['errs']} errors, {main_res['acc']:.2f}%")
    print(f"exp: {len(exp_res['segs'])} segments, {exp_res['errs']} errors, {exp_res['acc']:.2f}%")

if __name__ == "__main__":
    main()