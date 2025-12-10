import time
from collections import defaultdict, Counter
import math

def iou(box1, box2):
    x1,y1,x2,y2 = box1
    x3,y3,x4,y4 = box2

    xi1 = max(x1,x3)
    yi1 = max(y1,y3)
    xi2 = min(x2,x4)
    yi2 = min(y2,y4)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h

    area1 = (x2-x1)*(y2-y1)
    area2 = (x4-x3)*(y4-y3)
    union = area1 + area2 - inter + 1e-6

    return inter / union


class PlateTrack:
    def __init__(self, track_id, bbox, plate_text, conf):
        self.id = track_id
        self.bbox = bbox
        self.history = []   # list of (plate_text, conf, timestamp)
        self.last_update = time.time()
        self.add_observation(plate_text, conf)

    def add_observation(self, plate_text, conf):
        self.history.append((plate_text, conf, time.time()))
        self.last_update = time.time()

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.last_update = time.time()

    def best_plate(self, min_samples=5):
        """
        Majority vote among cleaned plate texts.
        min_samples: only confirm if we have enough readings.
        """
        if len(self.history) < min_samples:
            return None

        texts = [h[0] for h in self.history if h[0]]
        if not texts:
            return None

        counts = Counter(texts)
        best, count = counts.most_common(1)[0]

        # require majority & some stability
        if count >= max(3, len(texts) // 2):
            return best
        return None


class PlateTracker:
    def __init__(self, iou_thresh=0.3, track_ttl=2.0):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.track_ttl = track_ttl  # seconds to keep inactive tracks

    def update(self, detections):
        """
        detections: list of dicts from detect_plates_in_frame()
        Returns list of confirmed plates:
        [
          {
            "track_id": int,
            "plate": str,
            "bbox": (x1,y1,x2,y2)
          }
        ]
        """
        now = time.time()
        confirmed = []

        # First, expire old tracks
        to_delete = []
        for tid, tr in self.tracks.items():
            if now - tr.last_update > self.track_ttl:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Associate detections with tracks using IoU
        for det in detections:
            bbox = det["bbox"]
            plate = det["cleaned_text"]
            conf = det["confidence"]

            # find best matching track
            best_tid = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                i = iou(bbox, tr.bbox)
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_iou > self.iou_thresh:
                # update existing track
                tr = self.tracks[best_tid]
                tr.update_bbox(bbox)
                if plate:  # only add non-empty
                    tr.add_observation(plate, conf)
                best_plate = tr.best_plate()
                if best_plate:
                    confirmed.append({
                        "track_id": best_tid,
                        "plate": best_plate,
                        "bbox": tr.bbox
                    })
            else:
                # create new track
                tid = self.next_id
                self.next_id += 1
                tr = PlateTrack(tid, bbox, plate, conf)
                self.tracks[tid] = tr

        return confirmed
