class Actions:

    available_actions = [
        [0.0, 0.7, 0.0],  # throttle
        [0.0, 0.5, 0.0],  # throttle
        [0.0, 0.2, 0.0],  # throttle
        [0.0, 0.0, 0.7],  # break
        [0.0, 0.0, 0.5],  # break
        [0.0, 0.0, 0.2],  # break
        [-0.8, 0.1, 0.0],  # left
        [-0.5, 0.1, 0.0],  # left
        [-0.2, 0.1, 0.0],  # left
        [0.8, 0.1, 0.0],  # right
        [0.5, 0.1, 0.0],  # right
        [0.2, 0.1, 0.0],  # right
    ]

    def __class_getitem__(cls, item):
        if item > len(cls.available_actions) - 1:
            print('Nobody is driving! Action not found: {0}'.format(item))
            return cls.available_actions[0]
        else:
            return cls.available_actions[item]

    def __getitem__(cls, item):
        if item > len(cls.available_actions) - 1:
            print('Nobody is driving! Action not found: {0}'.format(item))
            return cls.available_actions[0]
        else:
            return cls.available_actions[item]