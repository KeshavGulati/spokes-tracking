from Spoke import *
from Frame import *
from collections import defaultdict


class AssignerUninitializedError(Exception):
    """Assigner has not been initialized!"""


class DuplicateSpokesError(Exception):
    """There are duplicate spokes in this frame!"""


class TrackAssigner:
    curr: int | None
    spoke_list: list[Spoke]
    frame_map: defaultdict[int, list[int]]

    def __init__(self):
        self.curr = 1
        self.spoke_list = []
        self.frame_map = defaultdict(list)

    def set_spoke_track(self, id: int, spoke: Spoke, frame: Frame) -> int:
        assert frame.time is not None

        if id in self.frame_map[frame.get_time()]:
            raise DuplicateSpokesError()

        if self.curr is None:
            raise AssignerUninitializedError()

        track_id = self.curr
        spoke.set_track_id(track_id)
        self.spoke_list.append(spoke)
        self.frame_map[frame.get_time()].append(id)
        self.curr += 1
        return track_id

    def get_spoke_with_track(self, track_id: int) -> int | None:
        for spoke in self.spoke_list:
            if spoke.track_id == f"t{track_id}":
                return spoke.get_id()

        return None
