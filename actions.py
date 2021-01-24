class Actions:

    available_actions = [
        [0.0, 0.5, 0.0],  # Need for speed throttle
        [0.0, 0.2, 0.0],  # Grandpa throttle
        [0.0, 0.0, 0.6],  # Hard break
        [0.0, 0.2, 0.6],  # Soft break
        [-0.8, 0.0, 0.0],  # Hard left
        [-0.3, 0.0, 0.0],  # Soft left
        [0.8, 0.0, 0.0],  # Hard right
        [0.3, 0.0, 0.0],  # Soft right
    ]

    def __class_getitem__(cls, item):
        if item > len(cls.available_actions) - 1:
            print('Nobody is driving! Action not found: {0}'.format(item))
            return cls.available_actions[0]
        else:
            return cls.available_actions[item]
