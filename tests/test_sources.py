"""IQ sources: block format, continuity, PlutoSDR configuration (fake device)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from iot_radar.acquisition.pluto import RX_OVERFLOW_BIT, RX_STATUS_REGISTER
from iot_radar.acquisition.sources import CWSimulationSource, PlutoSource


def _simulation(seed: int | None = 7, **scene) -> CWSimulationSource:
    parameters = dict(
        f_c=3.5e9, f_s=100e3, buffer_size=1024, f_offset=244.140625,
        breath_rate_hz=0.3, breath_amplitude_mm=10, seed=seed,
    )
    parameters.update(scene)
    return CWSimulationSource(**parameters)


def test_simulation_blocks_are_contiguous() -> None:
    source = _simulation()
    first, second = source.read_block(), source.read_block()
    assert source.sample_rate_hz == 100e3 and source.n_channels == 1
    assert first.samples.shape == (1, 1024) and first.samples.dtype == np.complex64
    assert (first.sample_start, second.sample_start) == (0, 1024)
    assert not first.overflow and not second.overflow
    assert second.host_time_s >= first.host_time_s


def test_simulation_delivers_adc_integers() -> None:
    samples = _simulation().read_block().samples
    assert np.array_equal(samples.real, np.round(samples.real))
    assert np.array_equal(samples.imag, np.round(samples.imag))
    loud = _simulation(static_clutter_amplitude=5000.0).read_block().samples
    assert loud.real.max() <= 2047 and loud.real.min() >= -2048


def test_simulated_echoes_sit_at_the_tx_frequency() -> None:
    """Bug B2: leakage and static clutter are copies of the TX tone."""
    f_offset = 244.140625
    source = _simulation(
        f_offset=f_offset, presence=False, receiver_dc_amplitude=400.0,
        static_clutter_amplitude=800.0, snr_db=40.0,
    )
    x = np.concatenate([source.read_block().samples[0] for _ in range(8)])
    spectrum = np.abs(np.fft.fft(x)) / x.size
    f_hz = np.fft.fftfreq(x.size, d=1.0 / 100e3)
    assert spectrum[np.argmin(np.abs(f_hz - f_offset))] == pytest.approx(800.0, rel=1e-3)
    assert spectrum[0] == pytest.approx(400.0, rel=1e-3)


def test_empty_scene_has_no_breathing_echo() -> None:
    quiet = dict(snr_db=200.0, static_clutter_amplitude=0.0, receiver_dc_amplitude=0.0, target_amplitude=500.0)
    present = _simulation(**quiet).read_block().samples
    empty = _simulation(presence=False, **quiet).read_block().samples
    assert np.abs(present).mean() == pytest.approx(500.0, rel=1e-2)
    assert np.abs(empty).max() == 0.0


def test_realtime_simulation_is_paced() -> None:
    """Bug B5: the simulation must not run faster than the hardware."""
    import time

    source = _simulation(realtime=True)  # 1024 samples at 100 kHz = 10.24 ms per block
    start = time.monotonic()
    for _ in range(5):
        source.read_block()
    assert time.monotonic() - start >= 0.9 * 5 * 1024 / 100e3


def test_simulation_seed_makes_it_reproducible() -> None:
    a, b = _simulation(seed=3), _simulation(seed=3)
    np.testing.assert_array_equal(a.read_block().samples, b.read_block().samples)


class _FakeRxAdc:
    def __init__(self) -> None:
        self.status = 0
        self.writes: list[tuple[int, int]] = []

    def reg_read(self, address: int) -> int:
        assert address == RX_STATUS_REGISTER
        return self.status

    def reg_write(self, address: int, value: int) -> None:
        self.writes.append((address, value))
        self.status &= ~value


class _FakePluto:
    """Records attribute assignments in order, like a pyadi-iio device."""

    instances: list["_FakePluto"] = []

    def __init__(self, uri: str) -> None:
        object.__setattr__(self, "settings", [("uri", uri)])
        object.__setattr__(self, "transmitted", None)
        object.__setattr__(self, "destroyed", False)
        object.__setattr__(self, "_rxadc", _FakeRxAdc())
        _FakePluto.instances.append(self)

    def __setattr__(self, name, value) -> None:
        self.settings.append((name, value))
        object.__setattr__(self, name, value)

    def tx(self, buffer) -> None:
        object.__setattr__(self, "transmitted", buffer)

    def rx(self) -> np.ndarray:
        return (np.arange(8) + 1j).astype(np.complex128)

    def tx_destroy_buffer(self) -> None:
        object.__setattr__(self, "destroyed", True)


@pytest.fixture
def fake_adi(monkeypatch) -> type:
    module = types.ModuleType("adi")
    module.Pluto = _FakePluto
    monkeypatch.setitem(sys.modules, "adi", module)
    _FakePluto.instances.clear()
    return _FakePluto


def test_pluto_configuration_sequence(fake_adi) -> None:
    tx = np.ones(16, dtype=np.complex64)
    source = PlutoSource("ip:192.168.2.1", 3.5e9, 2e6, 45, -20, 8, tx)
    sdr = fake_adi.instances[0]
    assert sdr.settings == [
        ("uri", "ip:192.168.2.1"),
        ("sample_rate", 2_000_000),
        ("rx_lo", 3_500_000_000),
        ("tx_lo", 3_500_000_000),
        ("rx_rf_bandwidth", 2_000_000),
        ("tx_rf_bandwidth", 2_000_000),
        ("rx_buffer_size", 8),
        ("gain_control_mode_chan0", "manual"),
        ("rx_hardwaregain_chan0", 45),
        ("tx_hardwaregain_chan0", -20),
        ("tx_cyclic_buffer", True),
    ]
    assert sdr.transmitted is tx
    assert sdr._rxadc.writes == [(RX_STATUS_REGISTER, 0x6)]  # sticky bits cleared at start

    first = source.read_block()
    sdr._rxadc.status = RX_OVERFLOW_BIT  # the FPGA dropped samples
    second = source.read_block()
    third = source.read_block()
    assert first.samples.shape == (1, 8) and first.samples.dtype == np.complex64
    assert [b.sample_start for b in (first, second, third)] == [0, 8, 16]
    assert [b.overflow for b in (first, second, third)] == [False, True, False]

    source.close()
    assert sdr.destroyed


def test_pluto_without_status_register(fake_adi) -> None:
    class _NoRegister:
        def reg_read(self, address):
            raise OSError("not supported")

        def reg_write(self, address, value):
            raise OSError("not supported")

    source = PlutoSource("usb:", 3.5e9, 2e6, 45, -20, 8, np.ones(16, np.complex64))
    object.__setattr__(fake_adi.instances[0], "_rxadc", _NoRegister())
    source._overflow_check_available = True
    assert source.read_block().overflow is False
    assert source._overflow_check_available is False
