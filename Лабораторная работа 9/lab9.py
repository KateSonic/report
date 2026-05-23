import csv
import math
import struct
import subprocess
import wave

def run_ff(cmd):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *cmd], check=True)

def read_wav(name):
    with wave.open(name, "rb") as wav:
        chans = getattr(wav, "getnchannels")()
        width = getattr(wav, "getsampwidth")()
        rate = getattr(wav, "getframerate")()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    vals = []
    if width == 1:
        scale = 128.0
        for i in range(0, len(raw), chans):
            frame = [(raw[i + ch] - 128) / scale for ch in range(chans)]
            vals.append(sum(frame) / chans)
    elif width == 2:
        scale = 32768.0
        step = chans * width
        for i in range(0, len(raw), step):
            frame = []
            for ch in range(chans):
                pos = i + ch * width
                val = getattr(struct, "unpack_from")("<h", raw, pos)[0]
                frame.append(val / scale)
            vals.append(sum(frame) / chans)
    elif width == 3:
        scale = 8388608.0
        step = chans * width
        for i in range(0, len(raw), step):
            frame = []
            for ch in range(chans):
                pos = i + ch * width
                part = raw[pos : pos + width]
                val = int.from_bytes(part, "little", signed=False)
                if val & 0x800000:
                    val -= 0x1000000
                frame.append(val / scale)
            vals.append(sum(frame) / chans)
    elif width == 4:
        scale = 2147483648.0
        step = chans * width
        for i in range(0, len(raw), step):
            frame = []
            for ch in range(chans):
                pos = i + ch * width
                val = getattr(struct, "unpack_from")("<i", raw, pos)[0]
                frame.append(val / scale)
            vals.append(sum(frame) / chans)

    meta = {
        "chans": chans,
        "bits": width * 8,
        "frames": frames,
    }
    return rate, vals, meta

def read_raw(name, rate):
    raw = getattr(subprocess, "check_output")(["ffmpeg", "-v", "error", "-i",
            name, "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"])
    vals = []
    for pos in range(0, len(raw), 2):
        val = getattr(struct, "unpack_from")("<h", raw, pos)[0]
        vals.append(val / 32768.0)
    return vals

def rms(vals):
    if not vals:
        return 0.0
    return math.sqrt(sum(val * val for val in vals) / len(vals))

def rms_blk(vals, rate, dt):
    size = max(1, round(rate * dt))
    blocks = []
    for start in range(0, len(vals), size):
        block = vals[start : start + size]
        if len(block) < size // 2:
            continue
        blocks.append((start / rate, rms(block)))
    return blocks

def hann(num, size):
    if size <= 1:
        return 1.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * num / (size - 1))

def goert(vals, rate, freq):
    size = len(vals)
    if size == 0:
        return 0.0
    omega = 2.0 * math.pi * freq / rate
    coeff = 2.0 * math.cos(omega)
    prev = 0.0
    prev2 = 0.0
    for i, val in enumerate(vals):
        cur = val * hann(i, size) + coeff * prev - prev2
        prev2 = prev
        prev = cur
    power = prev2 * prev2 + prev * prev - coeff * prev * prev2
    return power / size

def max_eng(vals, rate):
    dt = 0.1
    df = 50
    maxf = 5000
    topn = 15
    size = round(rate * dt)
    points = []
    freqs = list(range(df, maxf + 1, df))

    for start in range(0, len(vals), size):
        block = vals[start : start + size]
        if len(block) < size // 2:
            continue
        t0 = start / rate
        for cfreq in freqs:
            power = 0.0
            for offs in (-20, -10, 0, 10, 20):
                freq = cfreq + offs
                if freq > 0:
                    power += goert(block, rate, freq)
            points.append(
                {
                    "t0": t0,
                    "t1": min((start + size) / rate, len(vals) / rate),
                    "f0": cfreq - df / 2,
                    "f1": cfreq + df / 2,
                    "fc": cfreq,
                    "energy": power,
                }
            )
    return sorted(points, key=lambda item: item["energy"], reverse=True)[:topn]

def csv_eng(pts, out_dir, stem):
    csv_path = f"{out_dir}/{stem}_energy.csv"
    names = ["t0", "t1", "f0", "f1", "fc", "energy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        getattr(writer, "writeheader")()
        writer.writerows(pts)
    return csv_path

def dbfs(val):
    if val <= 0:
        return float("-inf")
    return 20.0 * math.log10(val)

def noise(vals, rate):
    blocks = rms_blk(vals, rate, 0.1)
    quiet, ns_rms = min(blocks, key=lambda item: item[1])
    all_rms = rms(vals)
    sig_rms = math.sqrt(max(all_rms * all_rms - ns_rms * ns_rms, 0.0))
    snr = 20.0 * math.log10(sig_rms / ns_rms) if ns_rms > 0 and sig_rms > 0 else float("inf")
    return {"quiet": quiet,
        "ns_rms": ns_rms,
        "all_rms": all_rms,
        "snr": snr}

def txt_res(vals, den, rate, meta, pts):
    src_ns = noise(vals, rate)
    den_ns = noise(den, rate)
    dur = len(vals) / rate

    print(f"Длительность: {round(dur, 3)} с")
    print(f"Частота дискретизации: {rate} Гц")
    print(f"Каналов: {meta['chans']}")
    print(f"Разрядность: {meta['bits']}")
    print(f"Самый тихий интервал начинается с {round(src_ns['quiet'], 2)} s")
    print(f"RMS шума до фильтрации: {round(src_ns['ns_rms'], 6)} ({round(dbfs(src_ns['ns_rms']), 2)} dBFS)")
    print(f"RMS всей записи до фильтрации: {round(src_ns['all_rms'], 6)} ({round(dbfs(src_ns['all_rms']), 2)} dBFS)")
    print(f"Оценка SNR до фильтрации: {round(src_ns['snr'], 2)} dB")
    print(f"RMS шума после фильтрации: {round(den_ns['ns_rms'], 6)} ({round(dbfs(den_ns['ns_rms']), 2)} dBFS)")
    print(f"RMS всей записи после фильтрации: {round(den_ns['all_rms'], 6)} ({round(dbfs(den_ns['all_rms']), 2)} dBFS)")
    print(f"Оценка SNR после фильтрации: {round(den_ns['snr'], 2)} dB")
    print("Максимумы энергии, dt=0.1 с, df=50 Гц:")
    for num, pt in enumerate(pts, start=1):
        print(
            f"{num}. t={round(pt['t0'])}-{round(pt['t1'])} с, "
            f"f={round(pt['f0'])}-{round(pt['f1'])} Гц, "
            f"энергия={pt['energy']:.6e}"
        )

def save_wav(name, vals, rate, meta, out_dir):
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    den_wav = f"{out_dir}/{stem}_denoised.wav"
    src_png = f"{out_dir}/{stem}_source.png"
    den_png = f"{out_dir}/{stem}_denoised.png"
    run_ff(["-i", name, "-filter_complex",
        "showspectrumpic=s=1600x900:legend=1:win_func=hann:scale=log:fscale=log:color=viridis",
        "-frames:v", "1", "-update", "1", src_png])
    run_ff(["-i", name, "-af", "afftdn=nr=18:nf=-35", "-c:a", "pcm_s24le", den_wav])
    run_ff(["-i", den_wav, "-filter_complex",
        "showspectrumpic=s=1600x900:legend=1:win_func=hann:scale=log:fscale=log:color=viridis",
        "-frames:v", "1", "-update", "1", den_png])
    den = read_raw(den_wav, rate)
    pts = max_eng(vals, rate)
    csv_path = csv_eng(pts, out_dir, stem)
    txt_res(vals, den, rate, meta, pts)
    return den_wav, src_png, den_png, csv_path

def proc_wav(name):
    in_file, out_dir = f"input/{name}", "output"
    rate, vals, meta = read_wav(in_file)
    return save_wav(in_file, vals, rate, meta, out_dir)

def main():
    name = "1.wav"
    out = proc_wav(name)

if __name__ == "__main__":
    main()
