"""The offline spectrogram must equal the streamed STFT columns."""

from __future__ import annotations

import logging

import numpy as np

from characterization_cases import pipeline_config, seeded_default_rng
from iot_radar.dsp.spectral import compute_spectrogram, get_window
from iot_radar.acquisition.sources import open_source
from iot_radar.pipeline import streaming_frame_generator


def test_offline_spectrogram_equals_stream() -> None:
    cfg = pipeline_config()
    chunks: list[np.ndarray] = []
    logging.disable(logging.WARNING)
    with seeded_default_rng():
        frames = streaming_frame_generator(cfg, open_source(cfg, simulation=True), decimated_iq_chunks=chunks)
        streamed = [frame["spectre_colonne"] for _, frame in zip(range(5), frames)]
        frames.close()
    logging.disable(logging.NOTSET)

    n_fft = cfg["spectrogramme"]["n_fft"]
    hop = int(n_fft * (1.0 - cfg["spectrogramme"]["overlap"]))
    offline = compute_spectrogram(
        np.concatenate(chunks), f_s=1000.0,
        window=get_window("hann", n_fft), hop=hop,
        skip_frames=cfg["spectrogramme"]["skip_warmup"],
    )
    assert offline.shape[0] >= 5
    np.testing.assert_array_equal(offline[:5], np.array(streamed))


def test_offline_spectrogram_too_short_signal() -> None:
    out = compute_spectrogram(np.zeros(10, np.complex64), 1000.0, get_window("hann", 16), hop=4)
    assert out.shape == (0, 16)
