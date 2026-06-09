"""Parse BCI2000 header metadata not exposed by mne.io.read_raw_bci2k."""

from pathlib import Path

import numpy as np
from mne.io.bci2k.bci2k import (
    _decode_bci2k_states,
    _parse_bci2k_header,
    _read_bci2k_data,
)


def parse_channel_names(path: Path) -> list[str]:
    """Parse ChannelNames list parameter from the BCI2000 header."""
    info = _parse_bci2k_header(path)
    n_channels = info["n_channels"]

    with open(path, "rb") as f:
        header_text = f.read(info["header_len"]).decode("utf-8", errors="replace")

    for line in header_text.splitlines():
        if "ChannelNames=" not in line:
            continue
        _, value = line.split("ChannelNames=", 1)
        tokens = value.split()
        count = int(tokens[0])
        names = tokens[1 : 1 + count]
        if len(names) != n_channels:
            raise ValueError(
                f"Expected {n_channels} channel names, got {len(names)} from header"
            )
        return names

    raise ValueError(f"ChannelNames not found in BCI2000 header: {path}")


def decode_states(path: Path) -> dict[str, np.ndarray]:
    """Decode all BCI2000 state vector fields (e.g. DigitalInput1)."""
    info = _parse_bci2k_header(path)
    _, state_bytes = _read_bci2k_data(path, info)
    return _decode_bci2k_states(state_bytes, info["state_defs"])


def digital_input_to_events(
    digital_input: np.ndarray, sfreq: float, event_id: int = 1
) -> np.ndarray:
    """Convert a binary digital input trace to an MNE events array (rising edges)."""
    rising = np.where((digital_input[1:] == 1) & (digital_input[:-1] == 0))[0] + 1
    events = np.column_stack([rising, np.zeros(len(rising), dtype=int), np.full(len(rising), event_id)])
    return events
