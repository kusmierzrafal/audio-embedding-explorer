import json
import random
from pathlib import Path
from typing import Dict, List, Set

######################### CONFIG
### RANDOM CONFIG

SEED = 42
random.seed(SEED)

### GENERATOR CONFIG
CONFIG = {
    "music": {
        "input": Path("assets/pseudo_captioning/atoms/music_level_atoms.json"),
        "output": Path("assets/pseudo_captioning/captions/music_level_captions.json"),
        "probability": 0.4,
    },
    "sound": {
        "input": Path("assets/pseudo_captioning/atoms/sound_level_atoms.json"),
        "output": Path("assets/pseudo_captioning/captions/sound_level_captions.json"),
        "probability": 0.4,
    },
}

######################### HELPERS


def load_atoms(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_captions(
    captions: List[str],
    atoms_meta: Dict,
    level: str,
    source_file: str,
    path: Path,
) -> None:
    output = {
        "meta": {
            "level": level,
            "source_version": atoms_meta.get("version", "x"),
            "generated_from": source_file,
            "count": len(captions),
        },
        "captions": captions,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


######################### GENERATORS


def generate_music_captions(atoms: Dict, probability: float) -> List[str]:
    instruments = atoms["instruments"]
    roles = atoms["roles"]
    descriptors = atoms["descriptors"]

    captions: Set[str] = set()

    # instrument + role
    for instrument in instruments:
        for role in roles:
            captions.add(f"{instrument} {role}")

    # descriptor + instrument
    for descriptor in descriptors:
        for instrument in instruments:
            captions.add(f"{descriptor} {instrument}")

    # descriptor + instrument + role (sampled)
    for descriptor in descriptors:
        for instrument in instruments:
            for role in roles:
                if random.random() < probability:
                    captions.add(f"{descriptor} {instrument} {role}")

    return sorted(captions)


def generate_sound_captions(atoms: Dict, probability: float) -> List[str]:
    sound_types = atoms["sound_types"]
    materials = atoms["materials"]
    envelopes = atoms["envelopes"]
    space = atoms["space"]

    captions: Set[str] = set()

    # material + sound_type
    for material in materials:
        for sound in sound_types:
            captions.add(f"{material} {sound}")

    # envelope + sound_type
    for envelope in envelopes:
        for sound in sound_types:
            captions.add(f"{envelope} {sound}")

    # space + sound_type
    for s in space:
        for sound in sound_types:
            captions.add(f"{s} {sound}")

    # material + envelope + sound_type
    for material in materials:
        for envelope in envelopes:
            for sound in sound_types:
                if random.random() < probability:
                    captions.add(f"{material} {envelope} {sound}")

    # space + envelope + sound_type
    for s in space:
        for envelope in envelopes:
            for sound in sound_types:
                if random.random() < probability:
                    captions.add(f"{s} {envelope} {sound}")

    return sorted(captions)


######################### MAIN


def main() -> None:
    # MUSIC
    music_atoms = load_atoms(CONFIG["music"]["input"])
    music_captions = generate_music_captions(
        music_atoms,
        CONFIG["music"]["descriptor_probability"],
    )
    save_captions(
        captions=music_captions,
        atoms_meta=music_atoms.get("meta", {}),
        level="music",
        source_file=CONFIG["music"]["input"].name,
        path=CONFIG["music"]["output"],
    )

    print(
        f"Generated {len(music_captions)} music captions "
        f"& saved to {CONFIG['music']['output'].resolve()}"
    )

    # SOUND
    sound_atoms = load_atoms(CONFIG["sound"]["input"])
    sound_captions = generate_sound_captions(
        sound_atoms,
        CONFIG["sound"]["combined_probability"],
    )
    save_captions(
        captions=sound_captions,
        atoms_meta=sound_atoms.get("meta", {}),
        level="sound",
        source_file=CONFIG["sound"]["input"].name,
        path=CONFIG["sound"]["output"],
    )

    print(
        f"Generated {len(sound_captions)} sound captions "
        f"& Saved to {CONFIG['sound']['output'].resolve()}"
    )


if __name__ == "__main__":
    main()
