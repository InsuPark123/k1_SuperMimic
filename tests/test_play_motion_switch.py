from types import SimpleNamespace

import torch
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET

from scripts.play import MotionViewer


def test_keyboard_switches_and_wraps_clips():
    motion = SimpleNamespace(
        library=SimpleNamespace(clips=[SimpleNamespace(name="walk", start=0), SimpleNamespace(name="dance", start=100)]),
        forced_frame=torch.tensor([-1]),
    )
    viewer = MotionViewer.__new__(MotionViewer)
    viewer.motion = motion
    viewer.clip_index = 0
    viewer.request_action = lambda name, delta: viewer._handle_custom_action(ViewerAction.CUSTOM, delta)
    resets = []
    viewer.reset_environment = lambda: resets.append(int(motion.forced_frame[0]))

    viewer.on_key(KEY_RIGHT_BRACKET)
    viewer.on_key(KEY_RIGHT_BRACKET)
    viewer.on_key(KEY_LEFT_BRACKET)
    viewer.on_key(65)

    assert resets == [100, 0, 100]
    assert viewer.clip_index == 1
