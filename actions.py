action_sets = [
    [
        [0.0, 0.0, 0.0],  # no action
        [0.0, 0.8, 0.0],  # throttle
        [0.0, 0.3, 0.0],  # throttle
        [0.0, 0.0, 0.6],  # brake
        [0.0, 0.0, 0.2],  # brake
        [-0.9, 0.0, 0.0],  # left
        [-0.5, 0.0, 0.0],  # left
        [-0.2, 0.0, 0.0],  # left
        [0.9, 0.0, 0.0],  # right
        [0.5, 0.0, 0.0],  # right
        [0.2, 0.0, 0.0],  # right
    ],
    [
        [0.0, 0.0, 0.0],  # no action
        [0.0, 0.8, 0.0],  # throttle
        [0.0, 0.0, 0.6],  # brake
        [-0.9, 0.0, 0.0],  # left
        [-0.5, 0.0, 0.0],  # left
        [-0.2, 0.0, 0.0],  # left
        [0.9, 0.0, 0.0],  # right
        [0.5, 0.0, 0.0],  # right
        [0.2, 0.0, 0.0],  # right
    ],
    [
        [0.0, 0.0, 0.0],  # no action
        [0.0, 0.8, 0.0],  # throttle
        [0.0, 0.0, 0.6],  # brake
        [-0.9, 0.0, 0.0],  # left
        [0.9, 0.0, 0.0],  # right
    ],
    [
        [0.0, 0.0, 0.0],  # no action
        [0.0, 0.9, 0.0],  # throttle high
        [0.0, 0.6, 0.0],  # throttle medium-high
        [0.0, 0.4, 0.0],  # throttle medium-low
        [0.0, 0.2, 0.0],  # throttle low
        [0.0, 0.0, 0.9],  # brake high
        [0.0, 0.0, 0.6],  # brake medium-high
        [0.0, 0.0, 0.4],  # brake medium-low
        [0.0, 0.0, 0.2],  # brake low
        [-0.9, 0.9, 0.0],  # left high, throttle high
        [-0.9, 0.6, 0.0],  # left high, throttle medium-high
        [-0.9, 0.4, 0.0],  # left high, throttle medium-low
        [-0.9, 0.2, 0.0],  # left high, throttle low
        [-0.9, 0.0, 0.9],  # left high, brake high
        [-0.9, 0.0, 0.6],  # left high, brake medium-high
        [-0.9, 0.0, 0.4],  # left high, brake medium-low
        [-0.9, 0.0, 0.2],  # left high, brake low
        [-0.9, 0.0, 0.0],  # left high, no throttle
        [-0.6, 0.9, 0.0],  # left medium-high, throttle high
        [-0.6, 0.6, 0.0],  # left medium-high, throttle medium-high
        [-0.6, 0.4, 0.0],  # left medium-high, throttle medium-low
        [-0.6, 0.2, 0.0],  # left medium-high, throttle low
        [-0.6, 0.0, 0.9],  # left medium-high, brake high
        [-0.6, 0.0, 0.6],  # left medium-high, brake medium-high
        [-0.6, 0.0, 0.4],  # left medium-high, brake medium-low
        [-0.6, 0.0, 0.2],  # left medium-high, brake low
        [-0.6, 0.0, 0.0],  # left medium-high, no throttle
        [-0.4, 0.9, 0.0],  # left medium-low, throttle high
        [-0.4, 0.6, 0.0],  # left medium-low, throttle medium-high
        [-0.4, 0.4, 0.0],  # left medium-low, throttle medium-low
        [-0.4, 0.2, 0.0],  # left medium-low, throttle low
        [-0.4, 0.0, 0.9],  # left medium-low, brake high
        [-0.4, 0.0, 0.6],  # left medium-low, brake medium-high
        [-0.4, 0.0, 0.4],  # left medium-low, brake medium-low
        [-0.4, 0.0, 0.2],  # left medium-low, brake low
        [-0.4, 0.0, 0.0],  # left medium-low, no throttle
        [-0.2, 0.9, 0.0],  # left low, throttle high
        [-0.2, 0.6, 0.0],  # left low, throttle medium-high
        [-0.2, 0.4, 0.0],  # left low, throttle medium-low
        [-0.2, 0.2, 0.0],  # left low, throttle low
        [-0.2, 0.0, 0.9],  # left low, brake high
        [-0.2, 0.0, 0.6],  # left low, brake medium-high
        [-0.2, 0.0, 0.4],  # left low, brake medium-low
        [-0.2, 0.0, 0.2],  # left low, brake low
        [-0.2, 0.0, 0.0],  # left low, no throttle
        [0.9, 0.9, 0.0],  # right high, throttle high
        [0.9, 0.6, 0.0],  # right high, throttle medium-high
        [0.9, 0.4, 0.0],  # right high, throttle medium-low
        [0.9, 0.2, 0.0],  # right high, throttle low
        [0.9, 0.0, 0.9],  # right high, brake high
        [0.9, 0.0, 0.6],  # right high, brake medium-high
        [0.9, 0.0, 0.4],  # right high, brake medium-low
        [0.9, 0.0, 0.2],  # right high, brake low
        [0.9, 0.0, 0.0],  # right high, no throttle
        [0.6, 0.9, 0.0],  # right medium-high, throttle high
        [0.6, 0.6, 0.0],  # right medium-high, throttle medium-high
        [0.6, 0.4, 0.0],  # right medium-high, throttle medium-low
        [0.6, 0.2, 0.0],  # right medium-high, throttle low
        [0.6, 0.0, 0.9],  # right medium-high, brake high
        [0.6, 0.0, 0.6],  # right medium-high, brake medium-high
        [0.6, 0.0, 0.4],  # right medium-high, brake medium-low
        [0.6, 0.0, 0.2],  # right medium-high, brake low
        [0.6, 0.0, 0.0],  # right medium-high, no throttle
        [0.4, 0.9, 0.0],  # right medium-low, throttle high
        [0.4, 0.6, 0.0],  # right medium-low, throttle medium-high
        [0.4, 0.4, 0.0],  # right medium-low, throttle medium-low
        [0.4, 0.2, 0.0],  # right medium-low, throttle low
        [0.4, 0.0, 0.9],  # right medium-low, brake high
        [0.4, 0.0, 0.6],  # right medium-low, brake medium-high
        [0.4, 0.0, 0.4],  # right medium-low, brake medium-low
        [0.4, 0.0, 0.2],  # right medium-low, brake low
        [0.4, 0.0, 0.0],  # right medium-low, no throttle
        [0.2, 0.9, 0.0],  # right low, throttle high
        [0.2, 0.6, 0.0],  # right low, throttle medium-high
        [0.2, 0.4, 0.0],  # right low, throttle medium-low
        [0.2, 0.2, 0.0],  # right low, throttle low
        [0.2, 0.0, 0.9],  # right low, brake high
        [0.2, 0.0, 0.6],  # right low, brake medium-high
        [0.2, 0.0, 0.4],  # right low, brake medium-low
        [0.2, 0.0, 0.2],  # right low, brake low
        [0.2, 0.0, 0.0],  # right low, no throttle
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
