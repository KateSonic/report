from PIL import Image

def open_gray(path):
    return Image.open(path).convert("L")

def dilate(gray, se):
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
            max_val = 0
            for dx, dy in offs:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    val = src[ny * width + nx]
                    if val > max_val:
                        max_val = val
            res[row_off + x] = max_val

    return Image.frombytes("L", gray.size, bytes(res))

def diff_imgs(img1, img2):
    bytes1 = img1.tobytes()
    bytes2 = img2.tobytes()
    res = bytearray()
    for val1, val2 in zip(bytes1, bytes2):
        res.append(abs(val1 - val2))
    return Image.frombytes("L", img1.size, bytes(res))

def save_imgs(src_name, gray, out_dir):
    stem = src_name.rsplit("_", 1)[0]
    se = ((0, 1, 0), (1, 1, 1), (0, 1, 0))
    dil_img = dilate(gray, se)
    diff = diff_imgs(gray, dil_img)

    dil_bmp = f"{out_dir}/{stem}_dilated.bmp"
    dil_png = f"{out_dir}/{stem}_dilated.png"
    diff_bmp = f"{out_dir}/{stem}_diff.bmp"
    diff_png = f"{out_dir}/{stem}_diff.png"

    dil_img.save(dil_bmp)
    dil_img.save(dil_png)
    diff.save(diff_bmp)
    diff.save(diff_png)

def proc_img(filename):
    gray = open_gray(f"input/{filename}")
    save_imgs(filename, gray, "output")

def main():
    in_files = ["1_source.png", "2_source.png"]
    for filename in in_files:
        proc_img(filename)

if __name__ == "__main__":
    main()
