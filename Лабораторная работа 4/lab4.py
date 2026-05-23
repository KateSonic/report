from PIL import Image

def dil_black(gray, se):
    width, height = gray.size
    src = gray.tobytes()
    res = bytearray(width * height)
    radius = len(se) // 2

    offs = []
    for mask_y, row in enumerate(se):
        for mask_x, val in enumerate(row):
            if val:
                offs.append((mask_x - radius, mask_y - radius))

    for y in range(height):
        row_off = y * width
        for x in range(width):
            vals = []
            for dx, dy in offs:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    vals.append(src[ny * width + nx])
            res[row_off + x] = min(vals)

    return Image.frombytes("L", gray.size, bytes(res))

def sub_imgs(img1, img2):
    bytes1 = img1.tobytes()
    bytes2 = img2.tobytes()
    res = bytearray()
    for val1, val2 in zip(bytes1, bytes2):
        res.append(max(0, val1 - val2))
    return Image.frombytes("L", img1.size, bytes(res))

def norm_255(img):
    src = img.tobytes()
    min_val = min(src)
    max_val = max(src)

    if max_val == min_val:
        return Image.new("L", img.size, 0)

    scale = 255 / (max_val - min_val)
    res = bytearray()
    for val in src:
        res.append(round((val - min_val) * scale))

    return Image.frombytes("L", img.size, bytes(res))

def otsu(img):
    hist = img.histogram()
    total = img.size[0] * img.size[1]
    sum_all = sum(val * cnt for val, cnt in enumerate(hist))
    sum_b = 0
    w_b = 0
    best_var = -1
    best_thr = 0

    for thr, cnt in enumerate(hist):
        w_b += cnt
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += thr * cnt
        mean_b = sum_b / w_b
        mean_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2

        if var > best_var:
            best_var = var
            best_thr = thr

    return best_thr

def bin_img(img, thr):
    src = img.tobytes()
    res = bytearray()
    for val in src:
        res.append(255 if val >= thr else 0)
    return Image.frombytes("L", img.size, bytes(res))

def save_pair(img, stem, out_dir):
    png = f"{out_dir}/{stem}.png"
    bmp = f"{out_dir}/{stem}.bmp"
    img.save(png)
    img.save(bmp)

def save_imgs(src_nm, src, gray, out_dir):
    se = ((1, 1, 1),(1, 1, 1),(1, 1, 1))
    usr_thr = None
    stem = src_nm.rsplit("_", 1)[0]
    dil = dil_black(gray, se)
    diff = sub_imgs(gray, dil)
    grad = norm_255(diff)
    thr = usr_thr if usr_thr is not None else otsu(grad)
    bin_g = bin_img(grad, thr)

    save_pair(gray, f"{stem}_gray", out_dir)
    save_pair(dil, f"{stem}_dilated_black", out_dir)
    save_pair(grad, f"{stem}_g", out_dir)
    save_pair(bin_g, f"{stem}_g_binary", out_dir)

def proc_img(name, out_dir):
    src = Image.open(f"input/{name}").convert("RGBA")
    gray = src.convert("L")
    save_imgs(name, src, gray, out_dir)

def main():
    proc_img("1_source.png", "output")

if __name__ == "__main__":
    main()
