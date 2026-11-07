"""Narrated slide videos: slide PNGs (from the PowerPoint PDF) + Windows SAPI voice -> mp4 via ffmpeg.
Usage: python make_video.py pitch|demo"""
import json, os, subprocess, sys, wave
import fitz

D = os.path.dirname(os.path.abspath(__file__))
FF = r"C:\Users\c931u\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
PDF = os.path.join(D, "team-opal-pitch.pdf")

PITCH = [  # (slide number, narration)
    (1, "Would you let one flat vote ten times? Team Opal prices every City of Sydney flat from its own past rent, and from its neighbours in the same building."),
    (2, "In the same building, with the same bedrooms, asking rents differ by up to five hundred dollars a week. Suburb averages cannot see that."),
    (3, "On held-out June and July adverts, government bond data misses by one hundred dollars. We miss by forty-five, with seventy-seven percent within one hundred dollars."),
    (7, "And government LiDAR showed us it is the view, not the floor: a water view adds about seventeen percent."),
    (11, "A rent you can check, flat by flat."),
]
DEMO = [
    (1, "Team Opal, task four: what will this flat rent for. We estimate weekly asking rent for all sixty-five thousand, seven hundred and seventy-seven City of Sydney flats, with and without a car space."),
    (2, "The problem. At forty-five Shelley Street, one-bedroom flats on floors two and three ask seven hundred to eight hundred and seventy-five dollars. Floors six and seven, which see the harbour, ask up to twelve hundred. Suburb averages cannot see this, but each flat's own past rent and its neighbours' rents can."),
    (3, "The result, on two thousand eight hundred and ninety-eight adverts from June and July that the model never saw. Government bond data by postcode and bedrooms misses by one hundred dollars. Suburb medians, ninety-seven. Adding the flat's own past rent, fifty-six. Our full method, forty-five dollars, with seventy-seven percent within one hundred dollars and an R squared of zero point eight eight."),
    (4, "The method has four steps, all rebuilt from adverts before each cut-off: a repeat-rent index from pairs of adverts of the same flat; the flat's own history, only when the bedrooms match and the advert is less than three years old; same-building comparables where each neighbour gets one vote; and the LiDAR floor-view difference, snapped to the five-dollar asking grid."),
    (5, "Every feature is rebuilt for each validation window. Dropping stale history cut the error from fifty-three to forty-six fifty. One-vote building comparables cut the error for flats without history from ninety-nine to sixty. Snapping to the price grid took it to forty-five."),
    (6, "The public data. Two NSW government laser surveys, five point six gigabytes, give the height of four thousand three hundred and ninety-nine buildings and a line-of-sight test from every floor, in thirty-six directions, out to one and a half kilometres."),
    (7, "Inside the same building, with the same bedrooms, a floor that sees water in every direction asks about seventeen percent more. Ten extra floors on their own add about one percent. It is the view, not the floor."),
    (8, "We are honest about what did not work. Building-level layers such as heat, canopy, flight paths and building age made errors worse, because the neighbours' rents already contain them. We kept only the floor-view difference."),
    (9, "Three findings: old evidence is worse than no evidence; the same address is not always the same flat; and asking rents live on a price grid."),
    (10, "Validation is always by date, never random. Rules were chosen on May, evaluated once on June and July, and the final model is trained on everything up to the thirty-first of July."),
    (11, "The code, the report and the predictions are in our public repository. Thank you."),
]


def say(text, wav):
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.SelectVoice('Microsoft Zira Desktop'); $s.Rate = 1; $s.SetOutputToWaveFile('%s'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()" % wav)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text, text=True, check=True)
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate()


def main(kind):
    script = PITCH if kind == "pitch" else DEMO
    doc = fitz.open(PDF)
    parts = []
    total = 0.0
    for i, (n, text) in enumerate(script):
        png = os.path.join(D, f"v_{kind}_{i}.png")
        doc[n - 1].get_pixmap(dpi=144).save(png)
        wav = os.path.join(D, f"v_{kind}_{i}.wav")
        dur = say(text, wav) + 0.35
        total += dur
        seg = os.path.join(D, f"v_{kind}_{i}.mp4")
        subprocess.run([FF, "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-i", wav, "-c:v", "libx264", "-tune", "stillimage",
                        "-pix_fmt", "yuv420p", "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white",
                        "-r", "30", "-c:a", "aac", "-b:a", "128k", "-af", "apad=pad_dur=0.35", "-t", f"{dur:.2f}", seg], check=True)
        parts.append(seg)
        print(f"slide {n}: {dur:.1f}s")
    lst = os.path.join(D, f"v_{kind}_list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    out = os.path.join(D, f"{kind}.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    print(f"{out}: {total:.1f}s, {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
