"""Conservative League of Legends D/F summoner-spell identity probe.

Input: the 900x132 HUD ability-row crops already generated in
results/hud_temporal/LOL_001/<scene>/hud_*_ability_row.png.
Optional: scan original videos/LOL_001.mp4 every N seconds to find spells
when the icon is COLORFUL and not covered by a cooldown number.

Uses Riot Data Dragon spell icons (no Riot API key / no Gemini key).
Returns candidates and visible HUD states. Does NOT invent casts, exact
cooldowns or force a spell identity when the icon is masked/too ambiguous.

Example:
    python OpenCV/summoner_spell_probe.py --video videos/LOL_001.mp4 --every 8
    python OpenCV/summoner_spell_probe.py --skip-download

Output: results/summoner_spells/LOL_001/{summoner_spell_probe.json,
         summoner_spell_frames.csv,spell_review_contact_sheet.jpg}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF_W, REF_H = 900, 132
# Coordinates calibrated on the 1282x832 LOL_001 HUD (4x row image).
SLOTS = {"D": (654, 754, 14, 111), "F": (768, 866, 14, 111)}
VIDEO_HUD_BOX = (0, 708, 225, 741)  # x0,y0,x1,y1 for 1282x832 video
CDN = "https://ddragon.leagueoflegends.com"
USER_AGENT = "LoL-VOD-HUD-prototype/0.1"


def read_url_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=14) as response:
        return json.loads(response.read().decode("utf-8"))


def download_binary(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=14) as response:
        return response.read()


def normalize_row(img):
    h, w = img.shape[:2]
    if (w, h) != (REF_W, REF_H):
        if abs(w / max(h, 1) - REF_W / REF_H) > 0.17:
            return None
        img = cv2.resize(img, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
    return img


def crop_slot(row, slot):
    x0, x1, y0, y1 = SLOTS[slot]
    # Trim black HUD borders and icon gutter; center contains game artwork.
    im = row[y0:y1, x0:x1]
    if im.size == 0:
        return None
    return cv2.resize(im[9:-7, 9:-10], (64, 64), interpolation=cv2.INTER_AREA)


def classify_visibility(icon):
    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    bright = float(np.mean(hsv[:, :, 2]))
    colorful = float(np.mean((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 65)))
    center = hsv[7:54, 5:59]
    white = float(np.mean((center[:, :, 1] < 60) & (center[:, :, 2] > 155)))
    # The overlay text often turns a spell grayscale. No name inference then.
    if sat > 77 and bright > 70 and colorful > 0.43 and white < 0.16:
        state = "colored_icon_candidate"
    elif white >= 0.070 and sat < 78:
        state = "timer_overlay_candidate"
    else:
        state = "dark_or_unclear"
    return dict(visual_state=state, saturation=round(sat, 2),
                brightness=round(bright, 2), white_overlay_fraction=round(white, 4),
                colorful_fraction=round(colorful, 4))


def collect_scene_rows(base: Path, only: str | None):
    if not base.exists():
        return []
    folders = [base / only] if only else sorted(x for x in base.iterdir() if x.is_dir())
    records = []
    for scene in folders:
        if not scene.is_dir():
            continue
        times = {}
        timeline = scene / "hud_timeline.json"
        if timeline.exists():
            try:
                data = json.loads(timeline.read_text(encoding="utf-8"))
                frames = data.get("frames", []) if isinstance(data, dict) else data
                for frame in frames:
                    key = (frame.get("ability_row_image") or frame.get("ability_row_path")
                           or frame.get("files", {}).get("ability_row"))
                    if key:
                        times[Path(key).name] = frame.get("source_timecode", "")
            except (ValueError, TypeError, OSError):
                pass
        for f in sorted(scene.glob("hud_*_ability_row.png")):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            row = normalize_row(img)
            if row is None:
                continue
            records.append((f"scene:{scene.name}", times.get(f.name, ""), f.name, row))
    return records


def collect_video_rows(video_path: Path, step_sec: float, profile: dict | None = None):
    if not video_path.is_file():
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise RuntimeError("Video FPS/frame count unavailable")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if profile is None:
        cap.release()
        raise ValueError('Full-video spell scanning requires --hud-profile; never guess HUD coordinates.')
    rw, rh = profile['reference_size']
    if abs(w/h - rw/rh) > 0.06:
        cap.release()
        raise ValueError('Video/HUD profile aspect ratio mismatch; calibrate new profile.')
    ref_rect = profile['ability_row_roi_ref']
    x0, y0, x1, y1 = [round(z*scale) for z, scale in zip(ref_rect,[w/rw,h/rh,w/rw,h/rh])]

    out = []
    for i, sec in enumerate(np.arange(0, total / fps, step_sec)):
        frame_idx = int(round(float(sec) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            continue
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        row = cv2.resize(crop, (REF_W, REF_H), interpolation=cv2.INTER_CUBIC)
        ms = int(round(float(sec) * 1000))
        tc = f"{ms//3600000:02d}:{(ms//60000)%60:02d}:{(ms//1000)%60:02d}.{ms%1000:03d}"
        out.append(("video:full", tc, f"video_{i:04d}.jpg", row))
    cap.release()
    return out


def load_references(cache: Path, patch: str, allow_download: bool):
    cache.mkdir(parents=True, exist_ok=True)
    # Explicit user-specified patch is preferred when the recorded game patch is known.
    if not patch:
        ver_file = cache / "patch_version.txt"
        if allow_download:
            try:
                versions = read_url_json(f"{CDN}/api/versions.json")
                patch = str(versions[0])
                ver_file.write_text(patch, encoding="utf-8")
            except (OSError, ValueError, KeyError, IndexError):
                if ver_file.exists():
                    patch = ver_file.read_text(encoding="utf-8").strip()
        elif ver_file.exists():
            patch = ver_file.read_text(encoding="utf-8").strip()
    if not patch:
        return "", [], "Cannot determine patch (network/cache unavailable)"
    spell_file = cache / f"summoner_{patch}.json"
    if not spell_file.exists() and allow_download:
        try:
            spell_file.write_text(json.dumps(read_url_json(
                f"{CDN}/cdn/{patch}/data/ko_KR/summoner.json"), ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError):
            pass
    if not spell_file.exists():
        return patch, [], "Summoner metadata missing; try again with internet or --patch <game version>"
    try:
        entries = json.loads(spell_file.read_text(encoding="utf-8")).get("data", {})
    except (ValueError, OSError):
        return patch, [], "Summoner metadata JSON invalid"
    out = []
    for spell_id, info in entries.items():
        # Most modes are named 'CLASSIC' etc, not map id 11. Include all but
        # exclude entries whose icon cannot be downloaded/loaded.
        imagefile = info.get("image", {}).get("full")
        if not imagefile:
            continue
        local = cache / patch / "spell" / imagefile
        if not local.exists() and allow_download:
            try:
                raw = download_binary(f"{CDN}/cdn/{patch}/img/spell/{imagefile}")
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(raw)
            except OSError:
                continue
        if not local.exists():
            continue
        template = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if template is None:
            continue
        template = cv2.resize(template[2:-2, 2:-2], (64, 64), interpolation=cv2.INTER_AREA)
        out.append(dict(spell_id=spell_id, name=info.get("name", spell_id),
                        template=template, image_file=imagefile))
    return patch, out, "" if out else "No spell icons cached; connection may be unavailable"


def cosine_score(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def visual_similarity(image, template):
    # Identity comparison only on colorful, unmasked HUD crops.
    # Blur handles differences caused by game video scaling/compression.
    im = cv2.GaussianBlur(image, (5, 5), 0)
    ref = cv2.GaussianBlur(template, (5, 5), 0)
    hsv1 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    hist_sim = 1.0 - float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    # Chromatic structure + grayscale edge maps. Only relative ranking,
    # not a calibrated probability of being the correct spell.
    g1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    edge1 = cv2.Laplacian(g1, cv2.CV_32F)
    edge2 = cv2.Laplacian(g2, cv2.CV_32F)
    edge_score = (cosine_score(edge1, edge2) + 1) / 2
    lab1 = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = float(np.mean(np.abs(lab1 - lab2)))
    pixel_score = max(0.0, 1.0 - diff / 105.0)
    return round(0.40 * hist_sim + 0.25 * edge_score + 0.35 * pixel_score, 4)


def build_sheet(examples, dest: Path):
    # Static review image: most colorful frames + best candidate names.
    tiles = []
    for row in examples:
        icon = row["icon"]
        patch = cv2.resize(icon, (128, 128), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((164, 280, 3), np.uint8) + 35
        canvas[24:152, 8:136] = patch
        cv2.putText(canvas, row["slot"] + " " + row.get("timecode", ""),
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, row["state"][:23], (145, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, row.get("top_name", "NOT EVALUATED")[:20], (145, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(canvas)
    if not tiles:
        return
    cols = 3
    gridrows = math.ceil(len(tiles) / cols)
    dst = np.zeros((gridrows * 164, cols * 280, 3), np.uint8) + 245
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        dst[r*164:(r+1)*164, c*280:(c+1)*280] = tile
    cv2.imwrite(str(dest), dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--scene-root", type=Path, default=None, help="Per-video pipeline HUD scene folder")
    p.add_argument("--outdir", type=Path, default=None, help="Per-video spell output directory")
    p.add_argument("--hud-profile", type=Path, default=None, help="Calibrated HUD JSON used for full-video sample extraction")
    p.add_argument("--only", default=None, help="Only this scene directory")
    p.add_argument("--video", default="", help="Optional full MP4 for ready spell frames")
    p.add_argument("--every", type=float, default=8, help="Video sampling seconds")
    p.add_argument("--patch", default="", help="Recorded game's LoL patch, if known")
    p.add_argument("--skip-download", action="store_true", help="Use cached templates only")
    p.add_argument("--top", type=int, default=10, help="Number of colorful frames per slot used")
    args = p.parse_args()
    if args.every < 0.5:
        p.error("--every must be at least 0.5 seconds")
    scene_dir = args.scene_root if args.scene_root is not None else ROOT / 'results' / 'pipeline' / args.game / 'hud_temporal'
    dest = args.outdir if args.outdir is not None else ROOT / 'results' / 'pipeline' / args.game / 'summoner_spells'
    profile = json.loads(args.hud_profile.read_text(encoding='utf-8-sig')) if args.hud_profile else None
    if args.video and profile is None:
        p.error('--video requires --hud-profile so the HUD coordinates are explicit')
    dest.mkdir(parents=True, exist_ok=True)
    scene_rows = collect_scene_rows(scene_dir, args.only)
    video_path = Path(args.video)
    if args.video and not video_path.is_absolute():
        video_path = ROOT / video_path
    video_rows = collect_video_rows(video_path, args.every, profile) if args.video else []
    all_rows = scene_rows + video_rows
    if not all_rows:
        raise FileNotFoundError(f"No rows found at {scene_dir}. Run the HUD sampler first")
    patch, refs, error = load_references(dest / "reference_cache", args.patch, not args.skip_download)
    samples = []
    for origin, tc, filename, row in all_rows:
        for slot in ("D", "F"):
            icon = crop_slot(row, slot)
            if icon is None:
                continue
            vis = classify_visibility(icon)
            samples.append(dict(origin=origin, timecode=tc, filename=filename,
                                slot=slot, icon=icon, **vis))
    # Pick distinct temporal frames; prefer bright, colorful samples not covered
    # by a large white timer. A slot that stays masked remains unknown.
    summary = {}
    contact_samples = []
    for slot in ("D", "F"):
        rows = [x for x in samples if x["slot"] == slot]
        clear = [x for x in rows if x["visual_state"] == "colored_icon_candidate"]
        clear.sort(key=lambda x: x["saturation"] - 180*x["white_overlay_fraction"], reverse=True)
        chosen = clear[:max(1, args.top)]
        names = []
        frames = []
        for x in chosen:
            scored = sorted(((visual_similarity(x["icon"], r["template"]), r["name"], r["spell_id"])
                              for r in refs), reverse=True)
            best = scored[0] if scored else None
            second = scored[1] if len(scored) > 1 else None
            margin = round(best[0] - second[0], 4) if best and second else None
            # Relative template scores are NOT calibrated confidence probabilities.
            frames.append(dict(origin=x["origin"], timecode=x["timecode"],
                               source_image=x["filename"], visual_state=x["visual_state"],
                               top_candidates=[dict(name=name, template_similarity=score,
                                                    id=spell_id) for score, name, spell_id in scored[:3]],
                               margin_to_second=margin))
            if best:
                names.append(best[1])
            if len(contact_samples) < 20:
                contact_samples.append(dict(icon=x["icon"], slot=slot,
                                            timecode=x["timecode"],
                                            state=x["visual_state"],
                                            top_name=best[1] if best else "NO REFERENCE"))
        freq = Counter(names)
        winner, winner_count = freq.most_common(1)[0] if freq else (None, 0)
        if not chosen and rows:
            # Include masked examples in the review image even though their names
            # remain unknown: these explain why the slot was not recognized.
            for x in rows[:2]:
                contact_samples.append(dict(icon=x["icon"], slot=slot,
                                            timecode=x["timecode"],
                                            state=x["visual_state"],
                                            top_name="UNKNOWN - MASKED"))
        summary[slot] = dict(visual_states=dict(Counter(x["visual_state"] for x in rows)),
                             inspected_samples=len(rows),
                             colorful_unmasked_samples=len(clear),
                             matching_samples=len(chosen),
                             tentative_name=winner,
                             template_vote_support=round(winner_count / len(names), 3) if names else None,
                             status="human_review_required" if winner else "unknown_no_unmasked_icon_or_reference",
                             candidates=frames,
                             warning="Name is NOT verified. Visual match is a candidate; "
                                     "patch/art mismatch and active cooldown overlays may bias matching.")
    output = dict(version="summoner-spell-probe-v1", video_id=args.game,
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  patch_reference=patch or None, reference_spell_count=len(refs),
                  reference_error=error, scene_row_samples=len(scene_rows),
                  full_video_row_samples=len(video_rows),
                  slots=summary,
                  constraints=["Names require manual verification before coaching.",
                               "Spell name and cooldown are independent fields.",
                               "Dark/icon-grayscale is not evidence of spell absence.",
                               "Timer overlay is a visual candidate, NOT precise cooldown seconds.",
                               "Identity may change within a game due to game mechanics.",
                               "HUD coordinates calibrated only for 1282x832 LOL_001; "
                               "calibrate on other resolutions."])
    json_path = dest / "summoner_spell_probe.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    csvpath = dest / "summoner_spell_frames.csv"
    with csvpath.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["origin", "timecode", "file", "slot", "visual_state",
                         "saturation", "white_overlay_fraction", "colorful_fraction"])
        for x in samples:
            writer.writerow([x["origin"], x["timecode"], x["filename"], x["slot"],
                             x["visual_state"], x["saturation"],
                             x["white_overlay_fraction"], x["colorful_fraction"]])
    build_sheet(contact_samples, dest / "spell_review_contact_sheet.jpg")
    print(f"HUD images {len(scene_rows)}, full-video samples {len(video_rows)}")
    print(f"Spell icon references {len(refs)}; patch {patch or 'unknown'}")
    if error:
        print("Reference note:", error)
    for slot in ("D", "F"):
        x = summary[slot]
        print(f"{slot}: clear {x['colorful_unmasked_samples']}, masked/dark "
              f"{x['inspected_samples'] - x['colorful_unmasked_samples']}; "
              f"name candidate={x['tentative_name'] or 'unknown'} [NOT VERIFIED]")
    print("Output:", json_path)


if __name__ == "__main__":
    main()
