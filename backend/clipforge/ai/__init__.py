"""Local AI layer: speech recognition, diarization, language ID, segmentation, scoring.

Every module here is import-safe on a machine with none of the heavy packages
installed - the optional dependencies are imported inside the functions that
need them and report a friendly error through :mod:`clipforge.errors`.
"""
