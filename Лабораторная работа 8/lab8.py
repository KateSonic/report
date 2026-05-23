import csv
from PIL import Image, ImageDraw, ImageFilter, ImageFont

def open_img(name):
    img = Image.open(f"input/{name}")
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        back = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        back.alpha_composite(rgba)
        return back.convert("RGB")
    return img.convert("RGB")

def to_gray(img):
    return img.convert("L")

def lin_cont(gray):
    coef = 1.35
    table = []
    for val in range(256):
        new_val = round(128 + coef * (val - 128))
        table.append(max(0, min(255, new_val)))
    return gray.point(table)

def quant(gray, lev):
    scale = lev - 1
    table = [round(val * scale / 255) for val in range(256)]
    return gray.point(table)

def ngtdm(gray):
    dist = 1
    lev = 32
    q_img = quant(gray, lev)
    kern = ImageFilter.Kernel(
        (3, 3),
        (1 / 8, 1 / 8, 1 / 8, 1 / 8, 0, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
        scale=1,
        offset=0
    )
    avg_img = q_img.filter(kern)

    q_crop = q_img.crop((dist, dist, gray.width - dist, gray.height - dist))
    a_crop = avg_img.crop((dist, dist, gray.width - dist, gray.height - dist))

    cnts = [0] * lev
    sums = [0.0] * lev
    q_data = q_crop.tobytes()
    a_data = a_crop.tobytes()

    for lvl, avg in zip(q_data, a_data):
        cnts[lvl] += 1
        sums[lvl] += abs(lvl - avg)

    total = len(q_data)
    probs = [cnt / total if total else 0.0 for cnt in cnts]
    used = [idx for idx, cnt in enumerate(cnts) if cnt > 0]
    u_cnt = len(used)

    w_diff = sum(probs[i] * sums[i] for i in used)
    cos = 1.0 / w_diff if w_diff else 0.0

    if u_cnt > 1 and total:
        t_dist = 0.0
        for i in used:
            for j in used:
                t_dist += probs[i] * probs[j] * ((i - j) ** 2)
        con = t_dist * sum(sums) / (total * u_cnt * (u_cnt - 1))
    else:
        con = 0.0

    denom = 0.0
    for i in used:
        for j in used:
            denom += abs(i * probs[i] - j * probs[j])
    bus = w_diff / denom if denom else 0
    return {
        "cnts": cnts,
        "sums": sums,
        "probs": probs,
        "cos": cos,
        "con": con,
        "bus": bus
    }

def get_font(size=14):
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)

def draw_hist(gray, title, path):
    hist = gray.histogram()
    img_w = 760
    img_h = 360
    left = 58
    top = 42
    right = 22
    bottom = 44
    plot_w = img_w - left - right
    plot_h = img_h - top - bottom
    max_val = max(hist) or 1

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    font1 = get_font(15)
    font2 = get_font(11)

    draw.text((left, 14), title, fill=(0, 0, 0), font=font1)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(35, 35, 35))

    for tick in range(5):
        val = round(max_val * tick / 4)
        y = top + plot_h - round(val / max_val * plot_h)
        draw.line((left, y, left + plot_w, y), fill=(228, 228, 228))
        draw.text((8, y - 6), str(val), fill=(0, 0, 0), font=font2)

    bar_w = plot_w / 256
    for idx, val in enumerate(hist):
        x0 = left + int(idx * bar_w)
        x1 = left + max(1, int((idx + 1) * bar_w))
        y0 = top + plot_h - round(val / max_val * plot_h)
        draw.rectangle((x0, y0, x1, top + plot_h), fill=(55, 105, 150))

    for lab in (0, 64, 128, 192, 255):
        x = left + round(lab / 255 * plot_w)
        draw.line((x, top + plot_h, x, top + plot_h + 4), fill=(0, 0, 0))
        draw.text((x - 10, top + plot_h + 10), str(lab), fill=(0, 0, 0), font=font2)

    img.save(path)

def draw_ng(res, title, path):
    vals = res["sums"]
    img_w = 760
    img_h = 360
    left = 58
    top = 42
    right = 22
    bottom = 44
    plot_w = img_w - left - right
    plot_h = img_h - top - bottom
    max_val = max(vals) or 1.0

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    font1 = get_font(15)
    font2 = get_font(11)

    draw.text((left, 14), title, fill=(0, 0, 0), font=font1)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(35, 35, 35))

    for tick in range(5):
        val = max_val * tick / 4
        y = top + plot_h - round(val / max_val * plot_h)
        draw.line((left, y, left + plot_w, y), fill=(228, 228, 228))
        draw.text((8, y - 6), f"{val:.0f}", fill=(0, 0, 0), font=font2)

    bar_w = plot_w / len(vals)
    for idx, val in enumerate(vals):
        x0 = left + int(idx * bar_w)
        x1 = left + max(1, int((idx + 1) * bar_w)) - 1
        y0 = top + plot_h - round(val / max_val * plot_h)
        draw.rectangle((x0, y0, x1, top + plot_h), fill=(80, 130, 75))

    for lab in (0, 8, 16, 24, 31):
        x = left + round(lab / (len(vals) - 1) * plot_w)
        draw.line((x, top + plot_h, x, top + plot_h + 4), fill=(0, 0, 0))
        draw.text((x - 8, top + plot_h + 10), str(lab), fill=(0, 0, 0), font=font2)

    img.save(path)

def save_ng(res, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["level", "n_i", "p_i", "s_i"])
        rows = zip(res["cnts"], res["probs"], res["sums"])
        for lvl, (cnt, prob, val) in enumerate(rows):
            writer.writerow([lvl, cnt, f"{prob:.10f}", f"{val:.6f}"])

def save_feat(res_all):
    with open("output/features.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "state", "COS", "CON", "BUS"])
        for item in res_all:
            for state, res in (("before", item["before"]), ("after", item["after"])):
                writer.writerow(
                    [
                        item["name"],
                        state,
                        f"{res['cos']:.10f}",
                        f"{res['con']:.10f}",
                        f"{res['bus']:.10f}",
                    ]
                )

def save_imgs(name, src, gray, cont, before, after, out_dir):
    stem = name.rsplit(".", 1)[0]
    src_path = f"{out_dir}/{stem}_source.png"
    gray_bmp = f"{out_dir}/{stem}_gray.bmp"
    gray_png = f"{out_dir}/{stem}_gray.png"
    cont_bmp = f"{out_dir}/{stem}_contrasted_gray.bmp"
    cont_png = f"{out_dir}/{stem}_contrasted_gray.png"
    hst_bef = f"{out_dir}/{stem}_hist_before.png"
    hst_aft = f"{out_dir}/{stem}_hist_after.png"
    ng_bef = f"{out_dir}/{stem}_ngtdm_before.png"
    ng_aft = f"{out_dir}/{stem}_ngtdm_after.png"
    csv_bef = f"{out_dir}/{stem}_ngtdm_before.csv"
    csv_aft = f"{out_dir}/{stem}_ngtdm_after.csv"

    src.save(src_path)
    gray.save(gray_bmp)
    gray.save(gray_png)
    cont.save(cont_bmp)
    cont.save(cont_png)

    draw_hist(gray, f"{name}: brightness histogram before contrast", hst_bef)
    draw_hist(cont, f"{name}: brightness histogram after contrast", hst_aft)
    draw_ng(before, f"{name}: NGTDM before contrast", ng_bef)
    draw_ng(after, f"{name}: NGTDM after contrast", ng_aft)
    save_ng(before, csv_bef)
    save_ng(after, csv_aft)

def proc_img(name):
    src = open_img(name)
    gray = to_gray(src)
    cont = lin_cont(gray)
    before = ngtdm(gray)
    after = ngtdm(cont)
    save_imgs(name, src, gray, cont, before, after, "output")
    return {
        "name": name,
        "before": before,
        "after": after
    }

def main():
    in_files = ["1.png", "2.png"]
    res_all = []
    for name in in_files:
        res_all.append(proc_img(name))
    save_feat(res_all)

if __name__ == "__main__":
    main()
