import csv
import math
import struct
import wave
from PIL import Image

def read_wav(path):
    with wave.open(path, "rb") as wav:
        chans = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        count = wav.getnframes()
        data = wav.readframes(count)
    vals = struct.unpack(f"<{count}h", data)
    return rate, [val / 32768.0 for val in vals]

def write_wav(path, rate, vals):
    vals = [max(-1.0, min(1.0, val)) for val in vals]
    data = struct.pack(f"<{len(vals)}h", *(int(val * 32767) for val in vals))
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data)

def dsamp(vals, src, dst=8000):
    step = max(1, round(src / dst))
    return vals[::step]

def rms_arr(vals, size, hop):
    res = []
    for beg in range(0, len(vals) - size + 1, hop):
        part = vals[beg:beg + size]
        res.append(math.sqrt(sum(val * val for val in part) / size))
    return res

def trim(vals, rate, rel=0.06):
    size = int(0.02 * rate)
    hop = int(0.01 * rate)
    rms = rms_arr(vals, size, hop)
    if not rms:
        return vals

    lim = max(rms) * rel
    act = [idx for idx, val in enumerate(rms) if val > lim]
    if not act:
        return vals

    pad = int(0.05 * rate)
    beg = max(0, act[0] * hop - pad)
    end = min(len(vals), act[-1] * hop + size + pad)
    return vals[beg:end]

def segm(vals, rate):
    size = int(0.02 * rate)
    hop = int(0.01 * rate)
    rms = rms_arr(vals, size, hop)
    lim = max(rms) * 0.08

    raw = []
    beg = None
    for idx, val in enumerate(rms):
        is_act = val > lim
        if is_act and beg is None:
            beg = idx
        if (not is_act or idx == len(rms) - 1) and beg is not None:
            end = idx if not is_act else idx + 1
            raw.append([beg * hop, min(len(vals), end * hop + size)])
            beg = None

    segs = []
    min_gap = int(0.18 * rate)
    min_len = int(0.08 * rate)
    for beg, end in raw:
        if segs and beg - segs[-1][1] < min_gap:
            segs[-1][1] = end
        elif end - beg > min_len:
            segs.append([beg, end])

    pad = int(0.05 * rate)
    min_w = int(0.12 * rate)
    return [
        (max(0, beg - pad), min(len(vals), end + pad))
        for beg, end in segs
        if end - beg > min_w
    ]

def fft(vals):
    data = [complex(val, 0.0) for val in vals]
    size = len(data)
    rev = 0
    for idx in range(1, size):
        bit = size >> 1
        while rev & bit:
            rev ^= bit
            bit >>= 1
        rev ^= bit
        if idx < rev:
            data[idx], data[rev] = data[rev], data[idx]

    block = 2
    while block <= size:
        root = complex(math.cos(-2 * math.pi / block), math.sin(-2 * math.pi / block))
        half = block // 2
        for off in range(0, size, block):
            fac = 1 + 0j
            for idx in range(off, off + half):
                even = data[idx]
                odd = data[idx + half] * fac
                data[idx] = even + odd
                data[idx + half] = even - odd
                fac *= root
        block *= 2
    return data

def feat(vals, rate):
    vals = trim(vals, rate)
    size = 256
    hop = 80
    win = [0.54 - 0.46 * math.cos(2 * math.pi * idx / (size - 1)) for idx in range(size)]
    if len(vals) < size:
        vals = vals + [0.0] * (size - len(vals))

    res = []
    for beg in range(0, len(vals) - size + 1, hop):
        part = vals[beg:beg + size]
        enrg = sum(val * val for val in part) / size
        if enrg < 1e-6:
            continue
        spec = fft([part[idx] * win[idx] for idx in range(size)])
        row = []
        for low in range(3, 120, 3):
            powr = sum(abs(spec[idx]) ** 2 for idx in range(low, low + 3))
            row.append(math.log(powr + 1e-12))

        avg = sum(row) / len(row)
        dev = math.sqrt(sum((val - avg) ** 2 for val in row) / len(row)) + 1e-9
        res.append([(val - avg) / dev for val in row])
    return res

def fdist(left, right):
    return sum((one - two) ** 2 for one, two in zip(left, right)) / len(left)

def dtw(left, right):
    inf = 10 ** 9
    wide = len(right)
    prev = [inf] * (wide + 1)
    prev[0] = 0.0
    for lrow in left:
        curr = [inf] * (wide + 1)
        for idx in range(1, wide + 1):
            cost = fdist(lrow, right[idx - 1])
            curr[idx] = cost + min(prev[idx], curr[idx - 1], prev[idx - 1])
        prev = curr
    return prev[wide] / (len(left) + len(right))

def recog(vals, rate, temps):
    segs = segm(vals, rate)
    rows = []
    for num, (beg, end) in enumerate(segs, 1):
        ftrs = feat(vals[beg:end], rate)
        scores = sorted((dtw(ftrs, temp), sym) for sym, temp in temps.items())
        best, sym = scores[0]
        sec = scores[1][0]
        conf = (sec - best) / sec if sec else 0.0
        rows.append({"num": num, "beg": beg, "end": end, "sym": sym, "best": best, "sec": sec, "conf": conf})
    return rows


def lev(left, right):
    prev = list(range(len(right) + 1))
    for ridx, lchr in enumerate(left, 1):
        curr = [ridx]
        for cidx, rchr in enumerate(right, 1):
            curr.append(min(prev[cidx] + 1, curr[cidx - 1] + 1, prev[cidx - 1] + (lchr != rchr)))
        prev = curr
    return prev[-1]

def cmap(val):
    val = max(0.0, min(1.0, val))
    red = int(255 * min(1.0, 2.2 * val))
    green = int(255 * max(0.0, min(1.0, 2.2 * val - 0.45)))
    blue = int(255 * max(0.0, min(1.0, 2.0 * val - 1.0)))
    return blue, green, red

def save_bmp(path, pixs):
    high = len(pixs)
    wide = len(pixs[0])
    data = bytearray()
    for row in pixs:
        for blue, green, red in row:
            data.extend((red, green, blue))
    img = Image.frombytes("RGB", (wide, high), bytes(data))
    img.save(path)

def save_spec(path, vals, rate):
    dst = 12000
    vals = dsamp(vals, rate, dst)
    size = 1024
    hop = 512
    high = 320
    win = [0.5 - 0.5 * math.cos(2 * math.pi * idx / (size - 1)) for idx in range(size)]
    frames = []
    for beg in range(0, len(vals) - size + 1, hop):
        part = vals[beg:beg + size]
        spec = fft([part[idx] * win[idx] for idx in range(size)])
        powrs = [math.log10(abs(val) ** 2 + 1e-12) for val in spec[:size // 2]]
        frames.append(powrs)
    min_v = min(min(row) for row in frames)
    max_v = max(max(row) for row in frames)
    log_min = math.log10(80)
    log_max = math.log10(6000)
    pixs = []
    for yy in range(high):
        freq = 10 ** (log_min + (high - 1 - yy) * (log_max - log_min) / (high - 1))
        bin_idx = min(size // 2 - 1, round(freq * size / dst))
        row = []
        for frame in frames:
            val = (frame[bin_idx] - min_v) / (max_v - min_v + 1e-12)
            row.append(cmap(val ** 0.65))
        pixs.append(row)
    save_bmp(path, pixs)

def save_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["index", "start_s", "end_s", "duration_s", "symbol", "best_score", "second_score", "confidence"])
        for row in rows:
            beg = int(row["beg"])
            end = int(row["end"])
            writer.writerow([
                row["num"],
                f"{beg / 8000:.3f}",
                f"{end / 8000:.3f}",
                f"{(end - beg) / 8000:.3f}",
                row["sym"],
                f"{float(row['best']):.6f}",
                f"{float(row['sec']):.6f}",
                f"{float(row['conf']):.3f}",
            ])

def save_segs(rows, rate, vals):
    for row in rows:
        beg = round(int(row["beg"]) * rate / 8000)
        end = round(int(row["end"]) * rate / 8000)
        name = f"output/segments/{int(row['num']):02d}_{row['sym']}.wav"
        write_wav(name, rate, vals[beg:end])

def main():
    expect = "+79852018533"
    chars = ["+"] + [str(num) for num in range(10)]
    files = ["+.wav"] + [f"{num}.wav" for num in range(10)]
    temps = {}
    for sym, name in zip(chars, files):
        rate, vals = read_wav(f"input/{name}")
        temps[sym] = feat(dsamp(vals, rate), 8000)
    rate, vals = read_wav("input/number.wav")
    vals_8k = dsamp(vals, rate)
    rows = recog(vals_8k, 8000, temps)
    res = "".join(str(row["sym"]) for row in rows)
    errs = lev(expect, res)
    rel = sum(float(row["conf"]) for row in rows) / len(rows)
    save_spec("output/number_spectrogram.bmp", vals, rate)
    save_csv("output/segments.csv", rows)
    save_segs(rows, rate, vals)
    print(f"Expected number: {expect}")
    print(f"Recognized number: {res}")
    print(f"Errors: {errs}")
    print(f"Reliability: {round(rel, 3)}")

if __name__ == "__main__":
    main()
