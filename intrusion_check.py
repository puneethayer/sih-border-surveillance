import cv2
import numpy as np


def is_inside_zone(centroid, polygon):
    """
    Check whether an object's centroid is inside
    the virtual restricted zone.
    """

    polygon = np.array(polygon, dtype=np.int32)

    result = cv2.pointPolygonTest(
        polygon,
        centroid,
        False
    )

    return result >= 0


def check_line_crossing(previous_centroid, current_centroid, line_y):
    """
    Check whether an object crossed a horizontal
    virtual line between two frames.
    """

    previous_y = previous_centroid[1]
    current_y = current_centroid[1]

    crossed_down = (
        previous_y < line_y and
        current_y >= line_y
    )

    crossed_up = (
        previous_y > line_y and
        current_y <= line_y
    )

    return crossed_down or crossed_up


def check_intrusion(previous_objects, current_objects, polygon):
    """
    Check tracked objects for new intrusions.

    previous_objects and current_objects are lists
    containing tracked object dictionaries.
    """

    intrusion_events = []

    # Convert previous frame list into a dictionary
    # using object ID as the key.
    previous_by_id = {
        obj["id"]: obj
        for obj in previous_objects
    }

    # Check every object in the current frame
    for current_object in current_objects:

        object_id = current_object["id"]
        current_centroid = current_object["centroid"]

        # Object must have existed in previous frame
        if object_id not in previous_by_id:
            continue

        previous_object = previous_by_id[object_id]

        previous_centroid = previous_object["centroid"]

        # Check previous position
        was_inside = is_inside_zone(
            previous_centroid,
            polygon
        )

        # Check current position
        is_inside = is_inside_zone(
            current_centroid,
            polygon
        )

        # Intrusion = outside -> inside
        if not was_inside and is_inside:

            intrusion_events.append({
                "object_id": object_id,
                "category": current_object.get("category"),
                "event": "intrusion",
                "centroid": current_centroid
            })

    return intrusion_events