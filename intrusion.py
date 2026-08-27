class IntrusionDetector:
    def __init__(self, line_pos, frame_width, frame_height, orientation='vertical'):
        """
        orientation: 'vertical' (line goes top-to-bottom, checks x-crossing)
                     'horizontal' (line goes left-to-right, checks y-crossing)
        line_pos: x-coordinate if vertical, y-coordinate if horizontal
        """
        self.line_pos = line_pos
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.orientation = orientation
        self.prev_positions = {}   # track_id -> last known position (x or y)
        self.alerted_ids = set()   # track_ids that already triggered an alert

    def check(self, detections):
        """
        Input: list of detections from detector.py (each has 'id' and 'centroid')
        Output: list of NEW intrusion events this frame
        """
        events = []

        for d in detections:
            obj_id = d['id']
            cx, cy = d['centroid']
            current_pos = cx if self.orientation == 'vertical' else cy

            prev_pos = self.prev_positions.get(obj_id)

            if prev_pos is not None:
                crossed = (prev_pos < self.line_pos <= current_pos) or \
                          (prev_pos > self.line_pos >= current_pos)

                if crossed and obj_id not in self.alerted_ids:
                    events.append({
                        'id': obj_id,
                        'class': d['class'],
                        'centroid': (cx, cy)
                    })
                    self.alerted_ids.add(obj_id)

            self.prev_positions[obj_id] = current_pos

        return events

    def draw_line(self, frame):
        """Draws the virtual fence line on the frame for visualization."""
        import cv2
        if self.orientation == 'vertical':
            cv2.line(frame, (self.line_pos, 0), (self.line_pos, self.frame_height),
                      (0, 0, 255), 2)
        else:
            cv2.line(frame, (0, self.line_pos), (self.frame_width, self.line_pos),
                      (0, 0, 255), 2)
        return frame
