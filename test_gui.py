"""Integration checks against real Tk widgets; skipped without a display.

These tests exercise callbacks and the event loop, not a mocked GUI toolkit.
On Linux CI they can run under Xvfb; desktop users can run unittest normally.
"""
import time
import unittest

try:
    import tkinter as tk
except ImportError:
    tk = None

from algorithms import SearchAlgorithm, SearchStatus
from engine import Environment
from infrastructure import Location


@unittest.skipIf(tk is None, 'Tkinter is not installed')
class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f'A desktop display is unavailable: {error}')
        from gui import TravelPlannerApp
        self.callback_errors = []
        self.root.report_callback_exception = lambda kind, value, trace: self.callback_errors.append(value)
        self.app = TravelPlannerApp(self.root)
        self.app._replace_environment(Environment(4, 2, Location(0, 0), Location(3, 0)))
        self.root.update()

    def tearDown(self):
        if hasattr(self, 'app'):
            self.app.close()
            self.assertEqual(self.callback_errors, [])
        elif hasattr(self, 'root'):
            self.root.destroy()

    def wait_until(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.005)
        self.root.update()
        self.assertTrue(predicate(), 'Tk callback did not complete before timeout')
        self.assertEqual(self.callback_errors, [])

    def search(self):
        self.app.find_route()
        self.wait_until(lambda: self.app.search_iterator is None)
        return self.app.result

    def test_find_route_shows_metrics_without_moving(self):
        self.assertEqual(self.app.algorithm_choice.get(), SearchAlgorithm.BFS.value)
        result = self.search()
        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(len(self.app.route_actions), 3)
        self.assertEqual(self.app.env.agent_location, Location(0, 0))
        self.assertEqual(self.app.env.steps, 0)
        self.assertEqual(self.app.metric_vars['cost'].get(), '3')
        self.assertTrue(self.app.explored)

    def test_controls_fit_the_minimum_window(self):
        self.root.geometry('1020x790')
        self.root.update()
        right = self.root.winfo_rootx() + self.root.winfo_width()
        bottom = self.root.winfo_rooty() + self.root.winfo_height()
        pending = list(self.root.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            if widget.winfo_class() in ('TButton', 'TEntry', 'TCombobox', 'TSpinbox'):
                self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), right)
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
        for widget in (self.app.status_label, self.app.message_label):
            self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), right)
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
            self.assertGreaterEqual(widget.winfo_rooty(), self.root.winfo_rooty())
        self.assertLessEqual(self.app.canvas.winfo_rooty() + self.app.canvas.winfo_height(),
                             self.app.status_label.winfo_rooty())

    def test_step_then_auto_run_reaches_destination(self):
        # Step must plan automatically, then execute exactly one action.
        self.app.step()
        self.wait_until(lambda: self.app.search_iterator is None)
        self.assertEqual(self.app.env.steps, 1)
        self.assertEqual(len(self.app.route_actions), 2)
        self.app.auto_run()
        self.wait_until(lambda: self.app.env.agent_location == self.app.env.goal)
        self.assertEqual(self.app.env.steps, 3)
        self.assertFalse(self.app.running)
        self.assertEqual(self.app.status.get(), 'Destination reached')

    def test_depth_cutoff_remains_distinct_from_failure(self):
        self.app.algorithm_choice.set(SearchAlgorithm.DLS.value)
        self.app.depth_choice.set('2')
        result = self.search()
        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(self.app.status.get(), 'Depth cutoff')
        self.assertEqual(self.app.env.steps, 0)
        self.assertEqual(self.app.route_actions, [])

    def test_invalid_depth_and_seed_do_not_modify_environment(self):
        env = self.app.env
        self.app.depth_choice.set('-1')
        self.app.find_route()
        self.assertIsNone(self.app.search_iterator)
        self.assertEqual(self.app.status.get(), 'Check the depth limit')
        self.app.seed_choice.set('invalid')
        self.app.load_seed()
        self.assertIs(self.app.env, env)
        self.assertEqual(self.app.status.get(), 'Check the seed')

    def test_cancel_discards_incomplete_search(self):
        self.app.find_route()
        self.assertIsNotNone(self.app.search_iterator)
        self.app.pause()
        self.root.update()
        self.assertIsNone(self.app.search_iterator)
        self.assertIsNone(self.app.search_job)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.env.steps, 0)
        self.assertEqual(self.app.status.get(), 'Search cancelled')

    def test_changing_strategy_cancels_old_search(self):
        self.app.find_route()
        self.app.algorithm_choice.set(SearchAlgorithm.DFS.value)
        self.app.algorithm_menu.event_generate('<<ComboboxSelected>>')
        self.root.update()
        self.assertIsNone(self.app.search_iterator)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.route_actions, [])
        self.assertEqual(self.app.status.get(), 'Strategy updated')

    def test_editing_obstacle_invalidates_plan(self):
        self.search()
        left, top, size = self.app.grid_geometry
        self.app.canvas.event_generate('<Button-1>', x=int(left + 1.5 * size),
                                       y=int(top + 0.5 * size))
        self.root.update()
        self.assertIn(Location(1, 0), self.app.env.obstacles)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.route_actions, [])
        result = self.search()
        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(len(result.actions), 5)

    def test_reset_cancels_motion_and_preserves_obstacles(self):
        self.app.env.set_obstacle(Location(1, 0), True)
        self.search()
        self.app.step()
        self.app.auto_run()
        self.app.reset_agent()
        self.root.update()
        self.assertEqual(self.app.env.agent_location, self.app.env.start)
        self.assertEqual(self.app.env.steps, 0)
        self.assertIn(Location(1, 0), self.app.env.obstacles)
        self.assertFalse(self.app.running)
        self.assertIsNone(self.app.move_job)
        self.assertIsNone(self.app.result)

    def test_inspection_and_seed_replay(self):
        self.app.inspect_successors()
        self.assertEqual(self.app.successor_states, {Location(0, 1), Location(1, 0)})
        self.assertEqual(self.app.env.steps, 0)
        self.app.seed_choice.set('42')
        self.app.load_seed()
        obstacles = set(self.app.env.obstacles)
        self.app.env.set_obstacle(Location(0, 0), Location(0, 0) not in obstacles)
        self.app.load_seed()
        self.assertEqual(self.app.env.obstacles, obstacles)
        self.assertEqual(self.app.active_seed, 42)
        self.assertEqual(self.app.env.steps, 0)


if __name__ == '__main__':
    unittest.main()
