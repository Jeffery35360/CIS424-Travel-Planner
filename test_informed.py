"""Sprint 4 informed-search correctness, using independent distance oracles."""
from collections import deque
import random
import unittest

from algorithms import AStarSearch, GreedyBestSearch, SearchStatus
from engine import Environment
from infrastructure import Action, Location, Node, TransitionModel


def grid_distance(width, height, start, goal, obstacles):
    """Coordinate-only BFS oracle; deliberately avoids search/model helpers."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        state, distance = queue.popleft()
        if state == goal:
            return distance
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = Location(state.x + dx, state.y + dy)
            if (0 <= nxt.x < width and 0 <= nxt.y < height and
                    nxt not in obstacles and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


class InformedSearchTests(unittest.TestCase):
    def test_astar_matches_independent_shortest_distance_on_seeded_maps(self):
        start, goal = Location(0, 0), Location(5, 4)
        for seed in range(30):
            rng = random.Random(seed)
            obstacles = {Location(x, y) for y in range(5) for x in range(6)
                         if Location(x, y) not in (start, goal) and rng.random() < .28}
            expected = grid_distance(6, 5, start, goal, obstacles)
            result = AStarSearch().search(TransitionModel(6, 5, goal, obstacles), start)
            with self.subTest(seed=seed):
                if expected is None:
                    self.assertEqual(result.status, SearchStatus.FAILURE)
                    self.assertIsNone(result.path_cost)
                else:
                    self.assertEqual(result.status, SearchStatus.SUCCESS)
                    self.assertEqual(result.path_cost, expected)
                    self.assertEqual(len(result.path), expected)

    def test_manhattan_is_admissible_and_consistent_on_unit_grid(self):
        goal = Location(4, 4)
        obstacles = {Location(1, 2), Location(3, 1), Location(3, 4),
                     Location(4, 0), Location(4, 2)}
        for solver in (AStarSearch(), GreedyBestSearch()):
            with self.subTest(solver=type(solver).__name__):
                self.assertEqual(solver.heuristic(goal, goal), 0)
                for y in range(5):
                    for x in range(5):
                        state = Location(x, y)
                        if state in obstacles:
                            continue
                        h = solver.heuristic(state, goal)
                        self.assertGreaterEqual(h, 0)
                        distance = grid_distance(5, 5, state, goal, obstacles)
                        if distance is not None:
                            self.assertLessEqual(h, distance)
                        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                            nxt = Location(x + dx, y + dy)
                            if 0 <= nxt.x < 5 and 0 <= nxt.y < 5 and nxt not in obstacles:
                                self.assertLessEqual(h, 1 + solver.heuristic(nxt, goal))

    def test_astar_uses_g_plus_h_while_greedy_uses_h_only(self):
        node = Node(Location(0, 0)).child(Action.DOWN, Location(0, 1), 4)
        node = node.child(Action.RIGHT, Location(1, 1), 3)
        goal = Location(4, 4)
        self.assertEqual(AStarSearch().priority(node, goal), 13)
        self.assertEqual(GreedyBestSearch().priority(node, goal), 6)

    def test_greedy_can_take_longer_route_than_astar(self):
        # Frozen fixture discovered using seed 10, then expressed explicitly so
        # the regression does not depend on a map generator implementation.
        start, goal = Location(0, 0), Location(4, 4)
        obstacles = {Location(1, 2), Location(3, 1), Location(3, 4),
                     Location(4, 0), Location(4, 2)}
        model = TransitionModel(5, 5, goal, obstacles)
        optimal = AStarSearch().search(model, start)
        greedy = GreedyBestSearch().search(model, start)
        self.assertEqual(optimal.status, SearchStatus.SUCCESS)
        self.assertEqual(greedy.status, SearchStatus.SUCCESS)
        self.assertEqual(optimal.path_cost, 8)
        self.assertEqual(greedy.path_cost, 10)
        for result in (optimal, greedy):
            cursor = start
            for action, target in zip(result.actions, result.path):
                self.assertEqual(model.result(cursor, action), target)
                cursor = target
            self.assertEqual(cursor, goal)

    def test_informed_searches_handle_terminal_and_unreachable_states(self):
        start, goal = Location(0, 0), Location(2, 2)
        trapped = TransitionModel(3, 3, goal, {Location(0, 1), Location(1, 0)})
        for solver in (AStarSearch(), GreedyBestSearch()):
            with self.subTest(solver=type(solver).__name__):
                failure = solver.search(trapped, start)
                self.assertEqual(failure.status, SearchStatus.FAILURE)
                self.assertEqual(failure.path, [])
                terminal = solver.search(TransitionModel(1, 1, start), start)
                self.assertEqual(terminal.status, SearchStatus.SUCCESS)
                self.assertEqual(terminal.path_cost, 0)
                self.assertEqual(terminal.actions, [])
                self.assertEqual(terminal.metrics.expanded, 0)

    def test_events_match_synchronous_result_without_moving_live_agent(self):
        env = Environment(5, 4, Location(0, 0), Location(4, 3),
                          {Location(1, 0), Location(1, 1), Location(3, 2)})
        before = (env.agent_location, env.steps, frozenset(env.obstacles))
        for solver in (AStarSearch(), GreedyBestSearch()):
            with self.subTest(solver=type(solver).__name__):
                expected = solver.search(env.transition_model(), env.start)
                iterator = solver.steps(env.transition_model(), env.start)
                events = []
                while True:
                    try:
                        events.append(next(iterator))
                    except StopIteration as completion:
                        actual = completion.value
                        break
                self.assertEqual(actual.path, expected.path)
                self.assertEqual(actual.actions, expected.actions)
                self.assertEqual(actual.status, expected.status)
                self.assertEqual(len(events), actual.metrics.examined)
                self.assertEqual(sum(event.expanded for event in events), actual.metrics.expanded)
                self.assertEqual((env.agent_location, env.steps, frozenset(env.obstacles)), before)

    def test_astar_reopens_state_for_admissible_inconsistent_heuristic(self):
        # A is expanded first at cost 3. B subsequently improves A to cost 2;
        # forbidding reopening would return the suboptimal goal cost 6.
        start, a, b, goal = [Location(i, 0) for i in range(4)]

        class GraphModel:
            def __init__(self):
                self.goal = goal
                self.expanded = []

            def is_goal(self, state):
                return state == self.goal

            def expand(self, node):
                self.expanded.append(node.state)
                edges = {start: ((a, 3), (b, 1)), b: ((a, 1),), a: ((goal, 3),)}
                return tuple(node.child(Action.RIGHT, nxt, cost)
                             for nxt, cost in edges.get(node.state, ()))

        class InconsistentAStar(AStarSearch):
            def heuristic(self, state, destination):
                return {start: 0, a: 0, b: 4, goal: 0}[state]

        model = GraphModel()
        result = InconsistentAStar().search(model, start)
        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.path, [b, a, goal])
        self.assertEqual(result.path_cost, 5)
        self.assertEqual(model.expanded.count(a), 2)


if __name__ == '__main__':
    unittest.main()
