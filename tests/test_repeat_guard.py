import logging
import unittest

from lecturelens.core.repeat_guard import Guards, RepeatGuard


class TestRepeatGuard(unittest.TestCase):
    def test_the_fault_is_said_the_first_time(self) -> None:
        guard = RepeatGuard("the thing")
        self.assertTrue(guard.saw(ValueError("first")))
        self.assertFalse(guard.saw(ValueError("second")))

    def test_the_fault_is_said_again_now_and_then(self) -> None:
        guard = RepeatGuard("the thing", every_at=10)
        said = [guard.saw(ValueError("x")) for _ in range(30)]
        self.assertTrue(said[0])
        self.assertTrue(said[9])
        self.assertTrue(said[19])
        self.assertTrue(said[29])
        self.assertEqual(sum(said), 4)

    def test_how_many_times_it_happened_is_kept(self) -> None:
        guard = RepeatGuard("the thing")
        for _ in range(7):
            guard.saw(ValueError("x"))
        self.assertEqual(guard.said(), 7)

    def test_a_fault_that_is_over_is_said_again(self) -> None:
        guard = RepeatGuard("the thing")
        guard.saw(ValueError("x"))
        guard.reset()
        self.assertTrue(guard.saw(ValueError("x")))

    def test_the_message_says_how_often_it_happened(self) -> None:
        records: list[str] = []

        class Collecting(logging.Logger):
            def warning(self, message: str, *args: object, **kwargs: object) -> None:
                records.append(message % args)

        guard = RepeatGuard("the thing", Collecting("test"))
        guard.saw(ValueError("the reason"))
        self.assertIn("the thing", records[0])
        self.assertIn("1 times now", records[0])
        self.assertIn("the reason", records[0])


class TestGuards(unittest.TestCase):
    def test_a_guard_is_made_once_and_kept(self) -> None:
        guards = Guards()
        first = guards.guard("the thing")
        second = guards.guard("the thing")
        self.assertIs(first, second)

    def test_the_count_of_each_fault_is_kept(self) -> None:
        guards = Guards()
        guards.saw("one", ValueError("x"))
        guards.saw("one", ValueError("x"))
        guards.saw("two", ValueError("x"))
        self.assertEqual(guards.counts(), {"one": 2, "two": 1})

    def test_a_fault_that_is_over_is_forgotten(self) -> None:
        guards = Guards()
        guards.saw("one", ValueError("x"))
        guards.reset("one")
        self.assertEqual(guards.said("one"), 0)


if __name__ == "__main__":
    unittest.main()
