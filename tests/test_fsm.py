import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fsm import RoadStateFSM, INIT

C = ["dry", "wet", "snow", "gravel", "damaged"]


def test_initial_state():
    assert RoadStateFSM(C, 3).state.current == INIT


def test_needs_k_consecutive_frames():
    f = RoadStateFSM(C, 3)
    assert f.run(["dry", "dry"]) == [INIT, INIT]
    assert f.step("dry") == "dry"


def test_noise_is_filtered():
    f = RoadStateFSM(C, 3)
    f.run(["dry"] * 3)
    out = f.run(["wet", "dry", "wet", "dry", "snow", "dry"])
    assert set(out) == {"dry"}


def test_real_change_accepted():
    f = RoadStateFSM(C, 3)
    f.run(["dry"] * 3)
    assert f.run(["wet"] * 3)[-1] == "wet"


def test_unknown_ignored_and_does_not_break_streak():
    f = RoadStateFSM(C, 3)
    f.run(["dry"] * 3)
    assert f.run(["wet", "unk", "wet", "unk", "wet"])[-1] == "wet"


def test_bad_symbol():
    try:
        RoadStateFSM(C, 3).step("lava")
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
