import unittest
from actions import Actions


class ActionsTest(unittest.TestCase):
    def test_getitem(self):
        # It returns the expected action
        self.assertEqual(Actions[0], Actions.available_actions[0])

        # It returns the first action when the index is out of bounds
        self.assertEqual(Actions[20], Actions[0])


if __name__ == '__main__':
    unittest.main()
