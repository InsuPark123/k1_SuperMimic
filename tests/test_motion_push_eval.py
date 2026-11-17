from scripts.eval_motion_push import make_trials


def test_push_trials_are_reproducible_and_paired_across_strengths():
    trials = make_trials(['motion'], (0.2, 0.5), 3, (3.0, 12.0), 4)
    assert trials == make_trials(['motion'], (0.2, 0.5), 3, (3.0, 12.0), 4)
    assert len(trials) == 7
    assert trials[0]['speed'] == 0.0
    for i in range(3):
        a, b = trials[1 + i], trials[4 + i]
        assert a['angle'] == b['angle'] and a['push_time'] == b['push_time']
        assert 3.0 <= a['push_time'] <= 12.0
