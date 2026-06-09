def normalize_frames(frames: list[list[float]], target: int) -> list[list[float]]:
    n = len(frames)
    if n == 0 or n == target:
        return frames
    if n > target:
        indices = [round((n - 1) * i / (target - 1)) for i in range(target)]
        return [frames[i] for i in indices]
    last = frames[-1]
    return frames + [last.copy() for _ in range(target - n)]
