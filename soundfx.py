import os


def load_first_sfx(app, base_names: list[str]):
    for base in base_names:
        candidates = [base]
        if "." not in base.split("/")[-1]:
            candidates.extend([f"{base}.wav", f"{base}.ogg", f"{base}.mp3"])
        for path in candidates:
            if not os.path.exists(path):
                continue
            sfx = app.audio3d.loadSfx(path) if hasattr(app, "audio3d") else app.loader.loadSfx(path)
            if sfx:
                return sfx
    return None


def play_sound(sound, volume: float, play_rate: float) -> None:
    if not sound:
        return
    sound.stop()
    sound.setVolume(max(0.0, min(1.0, volume)))
    sound.setPlayRate(max(0.5, min(2.4, play_rate)))
    sound.play()
