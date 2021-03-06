action_sets = [
    [
        [0.0, 0.3, 0.0],  # throttle
        [0.0, 0.1, 0.0],  # throttle
        [0.0, 0.0, 0.5],  # break
        [0.0, 0.0, 0.2],  # break
        [-1.0, 0.0, 0.05],  # left
        [-1.0, 0.0, 0.05],  # left
        [-1.0, 0.0, 0.05],  # left
        [1.0, 0.0, 0.05],  # right
        [1.0, 0.0, 0.05],  # right
        [1.0, 0.0, 0.05],  # right
    ],
    [
        [0.0, 0.8, 0.0],  # throttle
        [0.0, 0.0, 0.6],  # break
        [-0.8, 0.0, 0.0],  # left
        [-0.5, 0.0, 0.0],  # left
        [-0.2, 0.0, 0.0],  # left
        [0.8, 0.0, 0.0],  # right
        [0.5, 0.0, 0.0],  # right
        [0.2, 0.0, 0.0],  # right
    ],
    [
        [0.0, 0.8, 0.0],  # throttle
        [0.0, 0.0, 0.6],  # break
        [-0.8, 0.0, 0.0],  # left
        [0.8, 0.0, 0.0],  # right
    ],
    [
        [0.0, 0.0, 0.0],  # no action
        [0.0, 0.8, 0.0],  # throttle
        [0.0, 0.0, 0.6],  # break
        [-0.8, 0.0, 0.0],  # left
        [0.8, 0.0, 0.0],  # right
    ],
    [
        [0.0, 0.3, 0.0],  # throttle
        [0.0, 0.1, 0.0],  # throttle
        [0.0, 0.0, 0.0],  # throttle
        [0.0, 0.0, 0.7],  # break
        [0.0, 0.0, 0.5],  # break
        [0.0, 0.0, 0.2],  # break
        [-1.0, 0.0, 0.05],  # left
        [-0.5, 0.0, 0.05],  # left
        [-0.2, 0.0, 0.05],  # left
        [1.0, 0.0, 0.05],  # right
        [0.5, 0.0, 0.05],  # right
        [0.2, 0.0, 0.05],  # right
    ]
]


def get_action(set_num):
    if set_num >= len(action_sets):
        assert "Wrong available set num. It should go from 0 to {}".format(len(action_sets) - 1)
        return None
    return action_sets[set_num]
