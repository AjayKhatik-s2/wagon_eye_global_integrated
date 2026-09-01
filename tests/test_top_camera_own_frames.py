"""A top camera with no load/damage evidence publishes its OWN frames.

Reproduces the observed failure on batch 20260724_081227: a clean train (zero
damage) where the fused load frame belonged to RIGHT_UP_TOP, so LEFT_UP_TOP
published an EMPTY gallery and showed no images in the dashboard -- while its own
four frames per wagon sat in S3, referenced only by the global report.
"""

from __future__ import annotations

import os

import pytest

from core import constants as C
from delivery import inspection_json as IJ

ANGLE = {C.CAMERA_LEFT_UP_TOP: "left_top", C.CAMERA_RIGHT_UP_TOP: "right_top"}


def _url_for_factory(evidence_root):
    """Mirror dashboard_ingest's url maker: nested layout, existence-checked."""
    def _url_for(*, gw_id, feature, camera, filename):
        for rel in (os.path.join(gw_id, feature, camera, filename),
                    os.path.join(gw_id, feature, filename)):
            if os.path.isfile(os.path.join(evidence_root, rel)):
                return "https://bucket.s3.ap-south-1.amazonaws.com/%s" \
                       % rel.replace("\\", "/")
        return None
    return _url_for


def _write(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\xff\xd8\xff")


@pytest.fixture
def evidence(tmp_path):
    """GW_1 as the real run produced it: fused load owned by RIGHT_UP_TOP, no
    damage, and materialized positional frames for BOTH top cameras."""
    root = tmp_path / "evidence"
    _write(str(root / "GW_1" / "load" / "best_frame.jpg"))
    (root / "GW_1" / "load" / "metadata.json").write_text(
        '{"source_camera": "RIGHT_UP_TOP"}')
    for angle, frames in (("right_top", (358, 372, 388, 402)),
                          ("left_top", (430, 445, 460, 475))):
        for n in frames:
            _write(str(root / "GW_1" / "wagon_frames" / angle
                       / ("w1_frame_%06d.jpg" % n)))
    return str(root)


def test_left_up_top_published_nothing_before_the_fallback(evidence):
    """The bug: LEFT_UP_TOP's gallery from load/damage alone is EMPTY."""
    url_for = _url_for_factory(evidence)
    side = "left"
    templates = tuple(t.format(cam=C.CAMERA_LEFT_UP_TOP) for t in IJ._TOP_GALLERY)
    found = [t for t in templates
             if url_for(gw_id="GW_1", feature=t.split("/", 1)[0],
                        camera=C.CAMERA_LEFT_UP_TOP,
                        filename=t.split("/", 1)[1])]
    assert found == [], "fixture no longer reproduces the empty-gallery case"


def test_left_up_top_now_publishes_its_own_four_frames(evidence):
    frames = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_LEFT_UP_TOP,
                              IJ.FLAVOUR_TOP, _url_for_factory(evidence))
    assert len(frames) == 4
    assert [f["position"] for f in frames] == list(IJ.POSITION_NAMES)
    for f in frames:
        assert "/wagon_frames/left_top/" in f["s3_url"], f["s3_url"]


def test_each_top_camera_gets_only_its_own_angle(evidence):
    """The whole point: no borrowing. The two top cameras shoot the same roof."""
    url_for = _url_for_factory(evidence)
    left = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_LEFT_UP_TOP,
                            IJ.FLAVOUR_TOP, url_for)
    right = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_RIGHT_UP_TOP,
                             IJ.FLAVOUR_TOP, url_for)
    lset = {f["s3_url"] for f in left}
    rset = {f["s3_url"] for f in right}
    assert lset and rset
    assert not (lset & rset), "a frame was published by BOTH top cameras"
    assert all("/left_top/" in u for u in lset)
    # RIGHT_UP_TOP owns the fused load frame, so it keeps its existing gallery
    # rather than falling back -- the fallback fires only on an empty one.
    assert any("load/best_frame.jpg" in u for u in rset)


def test_a_camera_that_already_has_evidence_is_unchanged(evidence):
    """RIGHT_UP_TOP published load evidence before, and must still publish it."""
    right = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_RIGHT_UP_TOP,
                             IJ.FLAVOUR_TOP, _url_for_factory(evidence))
    assert right, "RIGHT_UP_TOP lost the gallery it used to have"
    assert "load/best_frame.jpg" in right[0]["s3_url"]


def test_no_urls_when_nothing_was_materialized(tmp_path):
    """No directory -> no URLs. A frame is never fabricated."""
    root = tmp_path / "evidence"
    (root / "GW_1").mkdir(parents=True)
    frames = IJ._wagon_frames(str(root), "GW_1", C.CAMERA_LEFT_UP_TOP,
                              IJ.FLAVOUR_TOP, _url_for_factory(str(root)))
    assert frames == []


def test_positions_follow_frame_number_not_listing_order(evidence):
    """`start` must be the wagon's start, not whatever os.listdir returned."""
    frames = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_LEFT_UP_TOP,
                              IJ.FLAVOUR_TOP, _url_for_factory(evidence))
    nums = [int(f["s3_url"].rsplit("_", 1)[-1].split(".")[0]) for f in frames]
    assert nums == sorted(nums), "positions are out of frame order: %s" % nums
    assert nums == [430, 445, 460, 475]


# -----------------------------------------------------------------------------
# Padding a PARTIAL top gallery
# -----------------------------------------------------------------------------

def test_right_up_top_is_padded_to_four(evidence):
    """RIGHT_UP_TOP published ONE frame; both top panels must show four.

    Without padding the dashboard shows 4 images for LEFT_UP_TOP and 1 for
    RIGHT_UP_TOP, which reads as one camera being broken. Both have four of their
    own frames per wagon.
    """
    frames = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_RIGHT_UP_TOP,
                              IJ.FLAVOUR_TOP, _url_for_factory(evidence))
    assert len(frames) == 4
    assert [f["position"] for f in frames] == list(IJ.POSITION_NAMES)


def test_the_curated_load_frame_stays_first(evidence):
    """The fused load close-up is chosen for judging load state.

    Padding must APPEND, never displace it -- an evenly spaced sample is not a
    substitute for the frame the load processor picked.
    """
    frames = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_RIGHT_UP_TOP,
                              IJ.FLAVOUR_TOP, _url_for_factory(evidence))
    assert "load/best_frame.jpg" in frames[0]["s3_url"]
    for f in frames[1:]:
        assert "/wagon_frames/right_top/" in f["s3_url"]


def test_padding_never_duplicates_a_frame(evidence):
    frames = IJ._wagon_frames(evidence, "GW_1", C.CAMERA_RIGHT_UP_TOP,
                              IJ.FLAVOUR_TOP, _url_for_factory(evidence))
    urls = [f["s3_url"] for f in frames]
    assert len(urls) == len(set(urls))


def test_padding_keeps_the_two_top_cameras_disjoint(evidence):
    """Still no borrowing: they photograph the same roof from opposite sides."""
    url_for = _url_for_factory(evidence)
    left = {f["s3_url"] for f in IJ._wagon_frames(
        evidence, "GW_1", C.CAMERA_LEFT_UP_TOP, IJ.FLAVOUR_TOP, url_for)}
    right = {f["s3_url"] for f in IJ._wagon_frames(
        evidence, "GW_1", C.CAMERA_RIGHT_UP_TOP, IJ.FLAVOUR_TOP, url_for)}
    assert len(left) == len(right) == 4
    assert not (left & right)


def test_side_cameras_are_not_padded(tmp_path):
    """Side galleries are what the dashboard has always shown -- leave them.

    A side camera with two door frames keeps two, even though positional frames
    exist for it: nothing about the side panels is broken, and changing them
    would be an unrequested visual change to a working view.
    """
    root = tmp_path / "evidence"
    for name in ("left_best.jpg", "left_crop.jpg"):
        _write(str(root / "GW_1" / "door" / name))
    for n in (350, 364, 378, 392):
        _write(str(root / "GW_1" / "wagon_frames" / "left_up"
                   / ("w1_frame_%06d.jpg" % n)))
    frames = IJ._wagon_frames(str(root), "GW_1", C.CAMERA_LEFT_UP,
                              IJ.FLAVOUR_SIDE, _url_for_factory(str(root)))
    assert len(frames) == 2
    assert all("/door/" in f["s3_url"] for f in frames)


# -----------------------------------------------------------------------------
# An EMPTY gallery is filled for ANY camera, side included
# -----------------------------------------------------------------------------

def test_a_side_camera_with_no_door_evidence_is_filled(tmp_path):
    """RIGHT_UP had wagons with ZERO frames -- those panels render blank.

    Gating the fill on the top flavour left them empty. A wagon with no frames is
    strictly worse than one showing the camera's own footage, and that footage
    exists, so the EMPTY case applies to every camera. This is distinct from
    padding a PARTIAL side gallery, which is deliberately not done.
    """
    root = tmp_path / "evidence"
    (root / "GW_1").mkdir(parents=True)
    for n in (272, 285, 297, 310):
        _write(str(root / "GW_1" / "wagon_frames" / "right_up"
                   / ("w1_frame_%06d.jpg" % n)))
    frames = IJ._wagon_frames(str(root), "GW_1", C.CAMERA_RIGHT_UP,
                              IJ.FLAVOUR_SIDE, _url_for_factory(str(root)))
    assert len(frames) == 4, "an empty side gallery was left empty"
    assert all("/wagon_frames/right_up/" in f["s3_url"] for f in frames)


def test_a_partial_side_gallery_is_still_not_padded(tmp_path):
    """The other half of the rule: 2 door frames stay 2, not 4."""
    root = tmp_path / "evidence"
    for name in ("right_best.jpg", "right_crop.jpg"):
        _write(str(root / "GW_1" / "door" / name))
    for n in (272, 285, 297, 310):
        _write(str(root / "GW_1" / "wagon_frames" / "right_up"
                   / ("w1_frame_%06d.jpg" % n)))
    frames = IJ._wagon_frames(str(root), "GW_1", C.CAMERA_RIGHT_UP,
                              IJ.FLAVOUR_SIDE, _url_for_factory(str(root)))
    assert len(frames) == 2
    assert all("/door/" in f["s3_url"] for f in frames)


def test_no_camera_ever_publishes_an_empty_gallery_when_frames_exist(tmp_path):
    """The invariant, across all four cameras."""
    root = tmp_path / "evidence"
    (root / "GW_1").mkdir(parents=True)
    angles = {C.CAMERA_RIGHT_UP: "right_up", C.CAMERA_LEFT_UP: "left_up",
              C.CAMERA_RIGHT_UP_TOP: "right_top", C.CAMERA_LEFT_UP_TOP: "left_top"}
    for angle in angles.values():
        for n in (100, 110, 120, 130):
            _write(str(root / "GW_1" / "wagon_frames" / angle
                       / ("w1_frame_%06d.jpg" % n)))
    url_for = _url_for_factory(str(root))
    for cam, angle in angles.items():
        frames = IJ._wagon_frames(str(root), "GW_1", cam,
                                  IJ.flavour_for(cam), url_for)
        assert frames, "%s published nothing" % cam
        assert all("/%s/" % angle in f["s3_url"] for f in frames), cam
