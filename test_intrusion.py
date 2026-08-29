from intrusion_check import check_intrusion


# Virtual restricted zone
polygon = [
    (1000, 400),
    (1600, 400),
    (1600, 900),
    (1000, 900)
]


# Previous frame
previous_objects = [
    {
        "id": 12,
        "class": "person",
        "category": "person",
        "confidence": 0.59,
        "bbox": (1175, 242, 1253, 395),
        "centroid": (1214, 318)
    },

    {
        "id": 15,
        "class": "person",
        "category": "person",
        "confidence": 0.51,
        "bbox": (628, 329, 689, 466),
        "centroid": (658, 398)
    },

    {
        "id": 11,
        "class": "truck",
        "category": "vehicle",
        "confidence": 0.60,
        "bbox": (709, 46, 941, 323),
        "centroid": (825, 184)
    }
]


# Current frame
current_objects = [
    {
        "id": 12,
        "class": "person",
        "category": "person",
        "confidence": 0.48,
        "bbox": (1166, 432, 1243, 548),
        "centroid": (1205, 490)
    },

    {
        "id": 15,
        "class": "person",
        "category": "person",
        "confidence": 0.66,
        "bbox": (514, 505, 589, 684),
        "centroid": (552, 595)
    },

    {
        "id": 11,
        "class": "truck",
        "category": "vehicle",
        "confidence": 0.60,
        "bbox": (771, 69, 995, 362),
        "centroid": (883, 215)
    }
]


events = check_intrusion(
    previous_objects,
    current_objects,
    polygon
)


print("Intrusion events:")

for event in events:
    print(event)