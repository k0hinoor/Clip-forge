"""Generate the media fixture used by the end-to-end tests.

CLIPFORGE is tested against real media on purpose - the end-to-end suite runs
ffmpeg, silence detection and the renderer for real. This script manufactures
that media locally, so nobody has to commit a video or download one.

What it produces (``--out`` names the video, everything else lands next to it):

    source.mp4      1280x720 @ 30 fps, H.264 + AAC, testsrc2 picture
    audio/voice.wav the scripted soundtrack (speech when a TTS engine exists)
    manifest.json   the ground truth: every segment's text, speaker and timing

The manifest drives ``backend/tests/conftest.py``, which turns it into an SRT
("the user already owns captions") and asserts against the real pipeline.

Speech is synthesised with whatever is available: ``espeak-ng``/``espeak`` if it
is on PATH, otherwise ``mespeak`` through node. When neither exists the segments
are rendered as syllable-modulated formant tones instead of speech - the audio
still has real energy where the script speaks and real silence between turns, so
silence removal and diarisation are exercised honestly. In that case the script
text still drives the analysis through the caption file, and the manifest records
``"speech": "synthetic-tones"`` so no result is ever mistaken for a transcription.

Usage::

    python tools/make_test_media.py --out /tmp/fix/source.mp4
    CLIPFORGE_TEST_MEDIA=/tmp/fix/source.mp4 pytest backend/tests -m e2e
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

RATE = 44_100
SAMPLE_WIDTH = 2


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


@dataclass
class Line:
    speaker: str
    text: str
    pause: float = 0.45


# Two hosts, three stories, one argument, one lesson, plus the sponsor/admin
# block a real episode would contain - the analyser has to reject that part.
SCRIPT: list[Line] = [
    Line("host_a", "Nobody tells you this when you start a company, and it cost me four years."),
    Line("host_b", "Okay, that sounds like a story worth hearing. What actually happened?"),
    Line("host_a", "I raised money, hired twelve people, and then watched the whole thing collapse in six weeks."),
    Line("host_b", "Six weeks. That is brutal. What broke first?"),
    Line("host_a", "The pricing. We charged ten dollars a month for something that cost us forty to run.", 0.6),
    Line("host_b", "So every customer made the hole deeper."),
    Line("host_a", "Exactly. Growth was the problem, not the solution. That is the part nobody says out loud."),
    Line("host_b", "And everyone thinks success is going to solve their problems.", 0.5),
    Line("host_a", "It does not. It just gives you bigger, more expensive problems, faster.", 0.6),
    Line("host_b", "Let us talk about the discipline side, because I think that is where people actually fail."),
    Line("host_a", "I never learned to prioritise. I learned to be busy. Those are completely different skills."),
    Line("host_b", "Busy is comfortable. Prioritising means admitting most of your ideas do not matter."),
    Line("host_a", "Right. And I could not admit that, because the ideas were mine.", 0.5),
    Line("host_b", "What changed? What was the moment it clicked?"),
    Line("host_a", "A customer emailed me and said, your product is not worth what you charge, and you know it.", 0.6),
    Line("host_b", "Ouch. That had to sting."),
    Line("host_a", "It saved the company. We tripled the price, lost half the customers, and finally made money."),
    Line("host_b", "So the lesson is charge more? That feels too simple."),
    Line("host_a", "The lesson is charge what it is worth, and stop hiding behind being cheap.", 0.6),
    Line("host_b", "I want to push back on that, because cheap is how you get your first hundred users."),
    Line("host_a", "Cheap gets you users who leave. Value gets you users who stay and bring their friends."),
    Line("host_b", "That is fair. But if you have no money, price is the only lever you control."),
    Line("host_a", "Price is a signal. If you charge nothing, you are telling people it is worth nothing."),
    Line("host_b", "Okay, I will take that. What would you do differently tomorrow if you started again?"),
    Line("host_a", "I would talk to ten customers before writing a single line of code. Ten. That is it."),
    Line("host_b", "Ten conversations versus four years of building. That math is not even close.", 0.6),
    Line("host_a", "And I would charge from day one, even if it was five dollars, so the feedback is real."),
    Line("host_b", "Free users lie. I have said that for years and people still argue with me.", 0.5),
    Line("host_a", "They lie politely, which is worse than lying.", 0.6),
    Line("host_b", "Let us do a quick break, and then I want to talk about the hiring mistake.", 0.7),
    # ---- sponsor / admin block: must never become a clip -------------------
    Line("host_a", "This episode is brought to you by our sponsorship slots, which are still open.", 0.5),
    Line("host_b", "And the newsletter is moving to tuesday, so update your calendars, everyone.", 0.5),
    Line("host_a", "We also want to thank the twelve people who filled in the listener survey last month.", 0.5),
    Line("host_b", "Back to the show. You were about to tell me about the hiring mistake.", 0.6),
    # ---- back to real content ---------------------------------------------
    Line("host_a", "I hired for credentials instead of evidence, and it broke the team culture.", 0.5),
    Line("host_b", "Give me an example, because 'culture' is a word people hide behind."),
    Line("host_a", "I hired a director who had never shipped anything, and he reorganised the team in week two.", 0.6),
    Line("host_b", "Week two. Before he understood what anyone did."),
    Line("host_a", "He renamed three teams, cancelled the roadmap, and quit after five months.", 0.6),
    Line("host_b", "And you paid a senior salary for that."),
    Line("host_a", "We paid a senior salary and lost two engineers who were actually brilliant."),
    Line("host_b", "That is the real cost. Not the salary. The people who leave because of it.", 0.6),
    Line("host_a", "Losing them cost more than the entire mis-hire. I think about it constantly.", 0.6),
    Line("host_b", "So what is the rule? Never hire senior people?"),
    Line("host_a", "Never hire anyone whose work you cannot inspect. That is the rule I use now.", 0.5),
    Line("host_b", "Show me what you built. Show me what you broke. Show me what you learned."),
    Line("host_a", "And ask them what they would refuse to do. That answer tells you everything."),
    Line("host_b", "I like that question. Most people have never been asked it.", 0.5),
    Line("host_a", "The people who answered it honestly are still on the team today.", 0.5),
    Line("host_b", "Let us talk about money, because that is the thing everybody whispers about."),
    Line("host_a", "I was broke for two years and told nobody. I kept posting about momentum.", 0.6),
    Line("host_b", "The gap between the story and the bank account is enormous, and nobody photographs it."),
    Line("host_a", "I had two hundred dollars and a very confident website.", 0.6),
    Line("host_b", "That is the most honest sentence I have heard on this show."),
    Line("host_a", "Here is the uncomfortable part. I almost quit on a tuesday, over a spreadsheet error.", 0.6),
    Line("host_b", "A spreadsheet error?"),
    Line("host_a", "A formula was wrong. I thought we had three months of runway and we had five weeks.", 0.6),
    Line("host_b", "So the panic was fake, but the deadline was real.", 0.5),
    Line("host_a", "The panic was fake and it almost ended the company. Emotions do not check the numbers.", 0.6),
    Line("host_b", "That is a genuinely useful thing to say out loud."),
    Line("host_a", "Check your numbers before you check your feelings. That is the whole lesson.", 0.6),
    Line("host_b", "Okay, final question, and I ask this everyone. What do you know now that you wish you knew then?"),
    Line("host_a", "That slow progress that compounds beats fast progress that collapses. Every single time.", 0.7),
    Line("host_b", "Say that again for the people in the back.", 0.5),
    Line("host_a", "Slow progress that compounds beats fast progress that collapses. Every single time.", 0.7),
    Line("host_b", "That is the episode. Thank you for listening, and we will see you next week.", 0.6),
]


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def _synth_espeak(text: str, path: Path, voice: str, speed: int) -> bool:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        return False
    command = [binary, "-v", voice, "-s", str(speed), "-w", str(path), text]
    return subprocess.run(command, capture_output=True).returncode == 0


def _synth_mespeak(text: str, path: Path, *, speed: int, pitch: int, module: Path | None) -> bool:
    """mespeak (npm) renders to WAV through node - used when espeak is absent."""
    node = shutil.which("node")
    if not node:
        return False
    runner = REPO / "tools" / "_mespeak_render.js"
    if not runner.exists():
        return False
    command = [node, str(runner), text, str(path), str(speed), str(pitch)]
    return subprocess.run(command, capture_output=True, cwd=str(module or REPO)).returncode == 0


def _formant_tone(text: str, *, duration: float, base: float, seed: int) -> list[float]:
    """Syllable-modulated tone standing in for speech when no TTS is available.

    Real energy where the line speaks, a real pause after it: enough for silence
    detection and speaker clustering to do genuine work.
    """
    generator = random.Random(seed)
    total = int(duration * RATE)
    samples: list[float] = []
    syllables = max(1, len(text.split()))
    syllable_seconds = duration / syllables
    for index in range(total):
        position = index / RATE
        syllable = int(position / max(syllable_seconds, 0.05))
        phase = (position - syllable * syllable_seconds) / max(syllable_seconds, 0.05)
        envelope = math.sin(math.pi * min(max(phase, 0.0), 1.0)) ** 0.6
        jitter = 1.0 + 0.04 * generator.random()
        value = (
            0.55 * math.sin(2 * math.pi * base * jitter * position)
            + 0.28 * math.sin(2 * math.pi * base * 2.0 * position)
            + 0.12 * math.sin(2 * math.pi * base * 3.0 * position)
        )
        samples.append(0.42 * envelope * value)
    return samples


def _write_wav(path: Path, samples: list[float], *, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(RATE)
        frames = bytearray()
        for value in samples:
            clipped = max(-1.0, min(1.0, value))
            pcm = struct.pack("<h", int(clipped * 32_000))
            for _ in range(channels):
                frames += pcm
        handle.writeframes(bytes(frames))


def build_track(script: list[Line], *, engine: str, workdir: Path, voice_map: dict[str, tuple[str, int]]) -> tuple[Path, list[dict]]:
    """Render every line and return (wav path, segments with real timings)."""
    workdir.mkdir(parents=True, exist_ok=True)
    segments: list[dict] = []
    timeline: list[dict] = []
    cursor = 0.6  # short head so the first words are not at t=0

    for index, line in enumerate(script):
        voice, speed = voice_map.get(line.speaker, ("en", 165))
        segment_path = workdir / f"line_{index:03d}.wav"
        rendered = False
        if engine in {"auto", "espeak"}:
            rendered = _synth_espeak(line.text, segment_path, voice, speed)
        if not rendered and engine in {"auto", "mespeak"}:
            rendered = _synth_mespeak(line.text, segment_path, speed=speed, pitch=60 if line.speaker == "host_a" else 74, module=None)
        if rendered and segment_path.exists():
            with wave.open(str(segment_path), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
                samples = [value[0] / 32768.0 for value in struct.iter_unpack("<h", frames)]
                if rate != RATE:  # resample by linear interpolation so offsets match
                    step = rate / RATE
                    samples = [samples[min(int(position * step), len(samples) - 1)] for position in range(int(len(samples) / step))]
        else:
            duration = max(1.2, len(line.text) / 15.5)  # ~15.5 characters per second
            samples = _formant_tone(line.text, duration=duration, base=118.0 if line.speaker == "host_a" else 196.0, seed=index)
            engine_used = "synthetic-tones"
        start = cursor
        timeline.append({"start": start, "samples": samples})
        duration = len(samples) / RATE
        segments.append(
            {
                "index": index,
                "speaker": line.speaker,
                "start": round(start, 3),
                "end": round(start + duration, 3),
                "duration": round(duration, 3),
                "text": line.text,
            }
        )
        cursor = start + duration + line.pause

    total = int((cursor + 0.4) * RATE)
    track = [0.0] * total
    for entry in timeline:
        offset = int(entry["start"] * RATE)
        for position, value in enumerate(entry["samples"]):
            if offset + position < total:
                track[offset + position] += value
    peak = max((abs(value) for value in track), default=1.0)
    if peak > 0.9:
        track = [value * (0.9 / peak) for value in track]

    destination = workdir / "voice.wav"
    _write_wav(destination, track)
    return destination, segments


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #


def build_video(wav: Path, out: Path, *, duration: float, width: int, height: int, fps: int) -> None:
    from clipforge.media.ffmpeg import ffmpeg_bin
    from clipforge.media.runner import run_command

    out.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration:.2f}",
            "-i",
            str(wav),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "27",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-shortest",
            str(out),
        ],
        timeout=1800,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the CLIPFORGE end-to-end test fixture.")
    parser.add_argument("--out", default="/tmp/fix/source.mp4", help="where to write the video")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tts", choices=("auto", "espeak", "mespeak", "none"), default="auto")
    parser.add_argument("--keep-lines", action="store_true", help="keep the per-line wav files next to the output")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    workdir = out.parent / "audio"
    voice_map = {"host_a": ("en-us", 172), "host_b": ("en-gb", 165)}

    print(f"rendering {len(SCRIPT)} lines -> {workdir}")
    wav, segments = build_track(SCRIPT, engine=args.tts, workdir=workdir, voice_map=voice_map)
    duration = segments[-1]["end"] + 0.5

    # Which engine actually produced the audio is recorded, never guessed.
    with wave.open(str(wav), "rb") as handle:
        voiced = any(abs(value[0]) > 900 for value in struct.iter_unpack("<h", handle.readframes(min(handle.getnframes(), RATE * 60))))
    speech = "tts" if (shutil.which("espeak-ng") or shutil.which("espeak")) and args.tts != "none" else "synthetic-tones"
    manifest = {
        "speech": speech if voiced else "synthetic-tones",
        "duration": round(duration, 3),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "language": "en",
        "speakers": sorted({line.speaker for line in SCRIPT}),
        "segments": segments,
    }
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"muxing {duration / 60:.1f} minutes of video -> {out}")
    build_video(wav, out, duration=duration, width=args.width, height=args.height, fps=args.fps)

    if not args.keep_lines:
        for leftover in workdir.glob("line_*.wav"):
            leftover.unlink()
    size_mb = out.stat().st_size / 1e6
    print(f"done: {out} ({size_mb:.1f} MB, {len(segments)} segments, audio={manifest['speech']})")
    print(f"      manifest: {out.parent / 'manifest.json'}")
    print(f"      run the suite with CLIPFORGE_TEST_MEDIA={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
