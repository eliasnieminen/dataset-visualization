def to_samples(time_in_ms: float, sr: int):
    return int((time_in_ms / 1000) * sr)
