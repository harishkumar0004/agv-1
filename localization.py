class Localization:
    def __init__(self, world):
        self.world = world
        self.reset()

    def reset(self):
        self.landmark = None
        self.position = None
        self.helper = None
        self.tag = None

        self.heading = None
        self.lateral = None
        self.forward = None

        self.raw_lateral = None
        self.center_lateral_offset = 0.0
        self.correction_tag = None
        self.visible_tags = []

    def update(self, detections):
        self.reset()

        if len(detections) == 0:
            return

        valid_candidates = []

        for detection in detections:
            result = self.world.find_landmark(detection.tag_id)

            if result is None:
                print(f"Unknown tag: {detection.tag_id}")
                continue

            valid_candidates.append(
                {
                    "detection": detection,
                    "result": result,
                    "landmark": result["landmark"],
                    "landmark_id": result["id"],
                    "position": result["position"],
                }
            )

            self.visible_tags.append(int(detection.tag_id))

        if not valid_candidates:
            return

        selected = self.select_detection(valid_candidates)

        detection = selected["detection"]
        result = selected["result"]

        self.tag = detection.tag_id
        self.correction_tag = detection.tag_id

        self.landmark = result["landmark"]
        self.position = result["position"]

        self.heading = detection.heading

        self.raw_lateral = detection.lateral

        self.center_lateral_offset = self.get_lateral_offset_to_center(
            self.position
        )

        if detection.lateral is None:
            self.lateral = None
        else:
            self.lateral = (
                detection.lateral +
                self.center_lateral_offset
            )

        self.forward = detection.forward

    def select_detection(self, candidates):
        """
        Choose which detected tag should be used for correction.

        Rule:
        1. If center tag is visible, use center tag.
        2. Otherwise use the biggest visible helper tag.

        The selected tag heading is used directly.
        The selected tag lateral is shifted to the center-tag path.
        """

        center_candidates = [
            candidate for candidate in candidates
            if candidate["position"] == "center"
        ]

        if center_candidates:
            return self.best_by_area(center_candidates)

        return self.best_by_area(candidates)

    def best_by_area(self, candidates):
        best_candidate = candidates[0]
        best_area = self.get_detection_area(
            best_candidate["detection"]
        )

        for candidate in candidates[1:]:
            area = self.get_detection_area(
                candidate["detection"]
            )

            if area > best_area:
                best_area = area
                best_candidate = candidate

        return best_candidate

    def get_detection_area(self, detection):
        corners = detection.corners

        if corners is None or len(corners) != 4:
            return 0.0

        x0, y0 = corners[0]
        x1, y1 = corners[1]
        x2, y2 = corners[2]
        x3, y3 = corners[3]

        area = 0.5 * abs(
            x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0
            - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0
        )

        return area

    def get_lateral_offset_to_center(self, position):
        """
        Convert helper-tag lateral measurement into center-tag lateral.

        Your testbed.json uses helper_spacing_m = 0.015.

        Based on your example:
            tag 264 raw lateral = 0.009 m
            tag 264 is south_east
            center correction offset should be 0.015 m
            corrected lateral = 0.024 m

        So for your current camera/sign convention:
            east column helpers  -> 0.015
            west column helpers  -> -0.015
            center column helpers ->  0.000
        """

        spacing = self.world.data["grid"]["helper_spacing_m"]

        if position in (
            "east",
            "north_east",
            "south_east",
        ):
            return spacing

        if position in (
            "west",
            "north_west",
            "south_west",
        ):
            return -spacing

        return 0.0

    def valid(self):
        return (
            self.landmark is not None
            and self.heading is not None
            and self.lateral is not None
        )