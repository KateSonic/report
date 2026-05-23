from PIL import Image

def open_rgb(path):
    image = Image.open(path)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def to_gray(image):
    grayscale = bytearray()
    rgb_bytes = image.tobytes()
    for i in range(0, len(rgb_bytes), 3):
        r = rgb_bytes[i]
        g = rgb_bytes[i + 1]
        b = rgb_bytes[i + 2]
        value = (299 * r + 587 * g + 114 * b + 500) // 1000
        grayscale.append(value)
    return Image.frombytes("L", image.size, bytes(grayscale))


def br_bin(gray_img, win_size, sens):
    width, height = gray_img.size
    radius = win_size // 2
    gray_bytes = gray_img.tobytes()
    binary = bytearray()

    col_sums = [0] * width
    y_top = 0
    y_bot = min(height - 1, radius)
    for y in range(y_top, y_bot + 1):
        row_pos = y * width
        for x in range(width):
            col_sums[x] += gray_bytes[row_pos + x]

    for y in range(height):
        if y > 0:
            old_top = max(0, y - radius - 1)
            new_top = max(0, y - radius)
            old_bot = min(height - 1, y + radius - 1)
            new_bot = min(height - 1, y + radius)

            if new_top > old_top:
                row_pos = old_top * width
                for x in range(width):
                    col_sums[x] -= gray_bytes[row_pos + x]

            if new_bot > old_bot:
                row_pos = new_bot * width
                for x in range(width):
                    col_sums[x] += gray_bytes[row_pos + x]

        y_min = max(0, y - radius)
        y_max = min(height - 1, y + radius)
        win_h = y_max - y_min + 1

        window_sum = 0
        x_right = min(width - 1, radius)
        for x in range(x_right + 1):
            window_sum += col_sums[x]

        for x in range(width):
            if x > 0:
                old_left = max(0, x - radius - 1)
                new_left = max(0, x - radius)
                old_right = min(width - 1, x + radius - 1)
                new_right = min(width - 1, x + radius)

                if new_left > old_left:
                    window_sum -= col_sums[old_left]
                if new_right > old_right:
                    window_sum += col_sums[new_right]

            x_min = max(0, x - radius)
            x_max = min(width - 1, x + radius)
            area = (x_max - x_min + 1) * win_h
            pixel = gray_bytes[y * width + x]
            th_mul = 100 - sens
            is_dark = pixel * area * 100 <= window_sum * th_mul
            binary.append(0 if is_dark else 255)

    return Image.frombytes("L", gray_img.size, bytes(binary))


def save_imgs(src_name, rgb_image, out_dir):
    stem = src_name.rsplit("_", 1)[0]
    gray = to_gray(rgb_image)
    binary = br_bin(gray,5,15)

    gray_bmp = f"{out_dir}/{stem}_gray.bmp"
    gray_png = f"{out_dir}/{stem}_gray.png"
    bin_bmp = f"{out_dir}/{stem}_bradley_roth.bmp"
    bin_png = f"{out_dir}/{stem}_bradley_roth.png"

    gray.save(gray_bmp)
    gray.save(gray_png)
    binary.save(bin_bmp)
    binary.save(bin_png)


def main():
    in_files = ["1_source.png", "2_source.png", "3_source.png", "4_source.png"]

    for src_name in in_files:
        rgb_image = open_rgb(f"input/{src_name}")
        save_imgs(src_name, rgb_image, "output")


if __name__ == "__main__":
    main()
