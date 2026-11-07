import os, re, subprocess, wave, sys
import fitz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_video import FF, say
D = os.path.dirname(os.path.abspath(__file__)); S = os.path.dirname(D)
from pptx import Presentation
deck = Presentation(os.path.join(D, "team-opal-pitch-3min.pptx"))
notes = [re.sub(r"^\d+ s\.\s*", "", s.notes_slide.notes_text_frame.text.strip()) for s in deck.slides]
pdf = fitz.open(os.path.join(D, "team-opal-pitch-3min.pdf"))
imgs = []
for i in range(pdf.page_count):
    p = os.path.join(D, f"d3_{i}.png"); pdf[i].get_pixmap(dpi=144).save(p); imgs.append(p)
plan = [(imgs[i], notes[i]) for i in range(5)]
plan += [(os.path.join(S, "demo_page.png"), "Here is the model running live. Forty-five Shelley Street, a one-bedroom on level four: twenty-four neighbouring flats, a small view adjustment, eight hundred dollars a week."),
         (os.path.join(S, "demo_page_f7.png"), "Move to level seven. The floor now sees more water than its neighbours, so the same flat type is priced higher, and every step is shown.")]
plan += [(imgs[i], notes[i]) for i in range(5, len(imgs))]
parts, total = [], 0.0
for i, (img, text) in enumerate(plan):
    wav = os.path.join(D, f"d3_{i}.wav"); dur = say(text, wav) + 0.3; total += dur
    seg = os.path.join(D, f"d3_{i}.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", "-loop", "1", "-i", img, "-i", wav, "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white", "-r", "30",
                    "-c:a", "aac", "-b:a", "128k", "-af", "apad=pad_dur=0.3", "-t", f"{dur:.2f}", seg], check=True)
    parts.append(seg); print(f"part {i}: {dur:.1f}s", flush=True)
lst = os.path.join(D, "d3_list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
out = os.path.join(D, "demo.mp4")
subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
print(f"demo.mp4 total {total:.1f}s, {os.path.getsize(out)/1e6:.1f} MB")
