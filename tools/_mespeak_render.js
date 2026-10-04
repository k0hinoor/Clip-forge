// Minimal text-to-speech bridge used by tools/make_test_media.py.
//
// espeak-ng is the preferred engine. When it is not installed (Windows, most
// CI images) this script renders the same line through the pure-JavaScript
// meSpeak synthesiser instead:
//
//     npm install mespeak
//     node tools/_mespeak_render.js "hello world" out.wav 165 60
//
// It writes a mono 16-bit WAV and exits non-zero when meSpeak is unavailable,
// which is what the Python caller checks before falling back to tones.

const fs = require("fs");
const path = require("path");

const [, , text, outPath, speedArg, pitchArg] = process.argv;
if (!text || !outPath) {
  console.error("usage: node tools/_mespeak_render.js <text> <out.wav> [speed] [pitch]");
  process.exit(2);
}

let meSpeak;
for (const candidate of ["mespeak", path.join(process.cwd(), "node_modules", "mespeak", "src", "mespeak.js")]) {
  try {
    meSpeak = require(candidate);
    break;
  } catch (error) {
    /* try the next candidate */
  }
}
if (!meSpeak) {
  console.error("mespeak is not installed (npm install mespeak)");
  process.exit(3);
}

// en-rp ships with meSpeak; en-gb does not.
meSpeak.loadConfig(require("mespeak/src/mespeak_config.json"));
meSpeak.loadVoice(require("mespeak/voices/en/en-rp.json"));

const wav = meSpeak.speak(text, {
  rawdata: "arraybuffer",
  speed: Number(speedArg) || 165,
  pitch: Number(pitchArg) || 60,
  amplitude: 120,
  voice: "en/en-rp",
});

fs.writeFileSync(outPath, Buffer.from(wav));
