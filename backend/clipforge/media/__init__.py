"""Media layer: process runner, ffmpeg probing/encoding, framing, timelines, captions.

FFmpeg is the only hard requirement and every command is built as an argv list -
no shell strings, no user supplied filter fragments.
"""
