"""The offline spectrogram must equal the columns computed one at a time."""

from __future__ import annotations

import numpy as np

from iot_radar.dsp.spectral import compute_single_column, compute_spectrogram, get_window


def test_offline_spectrogram_equals_column_by_column_processing() -> None:
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(5000) + 1j * rng.standard_normal(5000)).astype(np.complex64)
    n_fft, hop, skip = 1024, 102, 2
    window = get_window("hann", n_fft)

    # Streaming framing: a buffer that drops `hop` samples after each column.
    streamed, buffer, frame_number = [], iq.copy(), 0
    while len(buffer) >= n_fft:
        frame_number += 1
        if frame_number > skip:
            streamed.append(compute_single_column(buffer[:n_fft].copy(), 1000.0, window).power_db)
        buffer = buffer[hop:]

    offline = compute_spectrogram(iq, f_s_hz=1000.0, window=window, hop=hop, skip_frames=skip)
    np.testing.assert_array_equal(offline, np.array(streamed))


def test_offline_spectrogram_too_short_signal() -> None:
    out = compute_spectrogram(np.zeros(10, np.complex64), 1000.0, get_window("hann", 16), hop=4)
    assert out.shape == (0, 16)


def test_zero_padded_column() -> None:
    column = compute_single_column(np.ones(64, np.complex64), 20.0, get_window("hann", 64), n_fft=256)
    assert column.power_db.shape == column.f_hz.shape == (256,)
    assert column.f_hz[np.argmax(column.power_db)] == 0.0
