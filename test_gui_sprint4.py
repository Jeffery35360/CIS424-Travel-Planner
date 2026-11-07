"""Real-Tk checks for team planning, joint movement, and map configuration."""
import time
import unittest

try:
    import tkinter as tk
except ImportError:
    tk = None

from algorithms import SearchAlgorithm, SearchStatus
from infrastructure import Location
from multiagent import AgentSpec, MultiAgentEnvironment


@unittest.skipIf(tk is None, 'Tkinter is not installed')
class Sprint4GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f'A desktop display is unavailable: {error}')
        from gui import TravelPlannerApp
        self.errors = []
        self.root.report_callback_exception = lambda kind, value, trace: self.errors.append(value)
        self.app = TravelPlannerApp(self.root)
        self.app.mode_checkbox.invoke()
        self.app._replace_environment(MultiAgentEnvironment(3, 3, (
            AgentSpec('A1', Location(0, 1), Location(2, 1)),
            AgentSpec('A2', Location(1, 0), Location(1, 2)))))
        self.root.update()

    def tearDown(self):
        if hasattr(self, 'app'):
            self.app.close()
            self.assertEqual(self.errors, [])

    def wait_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)
        self.root.update()
        self.assertTrue(predicate(), 'A GUI operation did not finish before timeout')
        self.assertEqual(self.errors, [])

    def search(self):
        self.app.find_route()
        self.wait_until(lambda: self.app.search_iterator is None)
        return self.app.result

    def click(self, loc):
        left, top, size = self.app.grid_geometry
        self.app.canvas.event_generate('<Button-1>', x=int(left + (loc.x + .5) * size),
                                       y=int(top + (loc.y + .5) * size))
        self.root.update()

    def test_coordinated_plan_contains_wait_and_executes_joint_steps(self):
        result = self.search()
        self.assertTrue(result.found)
        self.assertEqual(self.app.env.steps, 0)
        self.assertEqual(result.makespan, 3)
        self.assertEqual(result.total_cost, 5)
        self.assertEqual(result.paths['A2'][0], result.paths['A2'][1])
        self.app.step()
        self.assertEqual(self.app.env.steps, 1)
        self.assertEqual(self.app.env.positions, {'A1': Location(1, 1), 'A2': Location(1, 0)})
        self.app.step()
        self.app.step()
        self.assertTrue(self.app.env.all_reached)
        self.assertEqual(self.app.status.get(), 'All destinations reached')
        self.assertEqual(len(set(self.app.env.positions.values())), 2)
        self.assertEqual(len(self.app.canvas.find_withtag('team_agent')), 2)

    def test_auto_plan_runs_team_and_reset_restores_all_starts(self):
        self.app.auto_run()
        self.wait_until(lambda: self.app.env.all_reached)
        self.assertEqual(self.app.env.goals_found, 2)
        self.assertFalse(self.app.running)
        self.app.reset_agent()
        self.assertEqual(self.app.env.positions, {'A1': Location(0, 1), 'A2': Location(1, 0)})
        self.assertEqual(self.app.env.steps, 0)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.multi_paths, {})

    def test_selected_agent_and_aggregate_metrics(self):
        result = self.search()
        self.assertEqual(self.app.metric_vars['length'].get(), '3')
        self.assertEqual(self.app.metric_vars['cost'].get(), '5')
        self.app.metric_scope.set('A1')
        self.app.scope_menu.event_generate('<<ComboboxSelected>>')
        self.root.update()
        self.assertEqual(self.app.metric_vars['length'].get(), '2')
        self.assertEqual(self.app.metric_vars['expanded'].get(), str(result.results['A1'].metrics.expanded))
        self.app.metric_scope.set('A2')
        self.app._scope_changed()
        self.assertEqual(self.app.metric_vars['cost'].get(), '3')
        self.assertEqual(self.app.env.steps, 0)

    def test_seed_parameters_replay_and_invalid_values_preserve_map(self):
        self.app.width_choice.set('9')
        self.app.height_choice.set('7')
        self.app.agent_count_choice.set('3')
        self.app.density_choice.set('0.35')
        self.app.seed_choice.set('123')
        self.app.load_seed()
        env = self.app.env
        self.assertEqual((env.width, env.height, len(env.agents)), (9, 7, 3))
        fingerprint = (env.agents, set(env.obstacles))
        self.app.env.set_obstacle(Location(0, 0), Location(0, 0) not in env.obstacles)
        self.app.load_seed()
        self.assertEqual((self.app.env.agents, self.app.env.obstacles), fingerprint)
        env = self.app.env
        self.app.density_choice.set('nan')
        self.app.new_seed_map()
        self.assertIs(self.app.env, env)
        self.assertEqual(self.app.status.get(), 'Check the map parameters')

    def test_mode_switch_cancels_search_and_restores_single_agent(self):
        self.assertEqual(self.app.algorithm_choice.get(), SearchAlgorithm.ASTAR.value)
        self.app.find_route()
        self.app.mode_checkbox.invoke()
        self.root.update()
        self.assertFalse(self.app.multi_mode.get())
        self.assertIsNone(self.app.search_iterator)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.algorithm_choice.get(), SearchAlgorithm.BFS.value)
        self.assertEqual(self.app.env.agent_location, self.app.env.start)

    def test_obstacle_edits_clear_team_plan_and_protect_all_endpoints(self):
        self.search()
        for loc in (Location(0, 1), Location(2, 1), Location(1, 0), Location(1, 2)):
            self.click(loc)
            self.assertNotIn(loc, self.app.env.obstacles)
            self.assertIsNotNone(self.app.result)
        self.click(Location(0, 0))
        self.assertIn(Location(0, 0), self.app.env.obstacles)
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.multi_paths, {})

    def test_horizon_input_and_cutoff_do_not_execute_partial_plan(self):
        self.app.horizon_choice.set('invalid')
        self.app.find_route()
        self.assertIsNone(self.app.search_iterator)
        self.assertEqual(self.app.status.get(), 'Check the planning horizon')
        self.app.horizon_choice.set('1')
        result = self.search()
        self.assertFalse(result.found)
        self.app.step()
        self.assertEqual(self.app.env.steps, 0)
        self.assertEqual(self.app.env.positions['A1'], Location(0, 1))

    def test_cancel_and_configuration_changes_discard_team_search(self):
        self.app.find_route()
        self.app.pause()
        self.root.update()
        self.assertIsNone(self.app.search_iterator)
        self.assertIsNone(self.app.result)
        self.app.find_route()
        self.app.depth_choice.set('10')
        self.root.update()
        self.assertIsNone(self.app.search_iterator)
        self.assertEqual(self.app.env.steps, 0)

    def test_multi_controls_and_footer_fit_minimum_window(self):
        self.root.geometry('1020x790')
        self.root.update()
        for tab in self.app.sidebar.tabs():
            self.app.sidebar.select(tab)
            self.root.update()
            right = self.root.winfo_rootx() + self.root.winfo_width()
            bottom = self.root.winfo_rooty() + self.root.winfo_height()
            pending = list(self.root.winfo_children())
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if widget.winfo_ismapped() and widget.winfo_class() in ('TButton', 'TEntry', 'TCombobox', 'TSpinbox', 'TLabel'):
                    self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), right)
                    self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
            self.assertLessEqual(self.app.message_label.winfo_rooty() + self.app.message_label.winfo_height(), bottom)


if __name__ == '__main__':
    unittest.main()
