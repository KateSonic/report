import math

from PIL import Image


def clamp(value, low=0, high=255):
    return max(low, min(high, int(round(value))))

def open_rgb(path):
    return Image.open(path).convert("RGB")

def save_rgb(image, output_dir):
    width, height = image.size
    source = image.load()

    channels = {
        "r_component.png": lambda r, g, b: (r, 0, 0),
        "g_component.png": lambda r, g, b: (0, g, 0),
        "b_component.png": lambda r, g, b: (0, 0, b),
    }

    for filename, transform in channels.items():
        result = Image.new("RGB", image.size)
        pixels = result.load()
        for y in range(height):
            for x in range(width):
                pixels[x, y] = transform(*source[x, y])
        result.save(f"{output_dir}/{filename}")

def rgb_to_hsi(r, g, b):
    rn = r / 255.0
    gn = g / 255.0
    bn = b / 255.0
    intensity = (rn + gn + bn) / 3.0
    minimum = min(rn, gn, bn)
    saturation = 0.0 if intensity == 0 else 1.0 - minimum / intensity

    numerator = 0.5 * ((rn - gn) + (rn - bn))
    denom = math.sqrt((rn - gn) ** 2 + (rn - bn) * (gn - bn))
    if denom == 0:
        hue = 0.0
    else:
        theta = math.acos(max(-1.0, min(1.0, numerator / denom)))
        hue = theta if bn <= gn else 2 * math.pi - theta

    return hue, saturation, intensity

def hsi_to_rgb(hue, saturation, intensity):
    hue = hue % (2 * math.pi)

    if saturation == 0:
        value = clamp(255 * intensity)
        return value, value, value

    if hue < 2 * math.pi / 3:
        b = intensity * (1 - saturation)
        r = intensity * (1 + saturation * math.cos(hue) / math.cos(math.pi / 3 - hue))
        g = 3 * intensity - (r + b)
    elif hue < 4 * math.pi / 3:
        hue -= 2 * math.pi / 3
        r = intensity * (1 - saturation)
        g = intensity * (1 + saturation * math.cos(hue) / math.cos(math.pi / 3 - hue))
        b = 3 * intensity - (r + g)
    else:
        hue -= 4 * math.pi / 3
        g = intensity * (1 - saturation)
        b = intensity * (1 + saturation * math.cos(hue) / math.cos(math.pi / 3 - hue))
        r = 3 * intensity - (g + b)

    return clamp(255 * r), clamp(255 * g), clamp(255 * b)

def save_hsi(image, output_dir):
    width, height = image.size
    source = image.load()
    int_img = Image.new("RGB", image.size)
    inv_img = Image.new("RGB", image.size)
    int_pix = int_img.load()
    inv_pix = inv_img.load()

    for y in range(height):
        for x in range(width):
            hue, saturation, intensity = rgb_to_hsi(*source[x, y])
            value = clamp(255 * intensity)
            int_pix[x, y] = (value, value, value)
            inv_pix[x, y] = hsi_to_rgb(hue, saturation, 1.0 - intensity)

    int_img.save(f"{output_dir}/hsi_intensity.png")
    inv_img.save(f"{output_dir}/hsi_inverted_intensity.png")

def resize_img(image, new_width, new_height):
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)

def enlarge(image, factor):
    width, height = image.size
    return resize_img(image, width * factor, height * factor)

def decimate(image, factor):
    width, height = image.size
    new_width = max(1, width // factor)
    new_height = max(1, height // factor)
    return image.resize((new_width, new_height), Image.Resampling.BOX)

def res_one(image, numerator, denom):
    width, height = image.size
    new_width = max(1, width * numerator // denom)
    new_height = max(1, height * numerator // denom)
    return resize_img(image, new_width, new_height)

def save_res(image, output_dir, m, n):
    enlarged = enlarge(image, m)
    reduced = decimate(image, n)
    two_pass = decimate(enlarged, n)
    one_pass = res_one(image, m, n)

    enlarged.save(f"output/resample_enlarge_M{m}.png")
    reduced.save(f"output/resample_decimate_N{n}.png")
    two_pass.save(f"output/resample_two_pass_M{m}_N{n}.png")
    one_pass.save(f"output/resample_one_pass_M{m}_N{n}.png")

def main():
    source = open_rgb("input/port.png")
    save_rgb(source, "output")
    save_hsi(source, "output")
    save_res(source, "output", 3, 4)

if __name__ == "__main__":
    main()
