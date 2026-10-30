"""Independent algorithm contracts and Sprint 1/2 integration regressions."""
from collections import deque
import random
import unittest

from algorithms import (BreadthFirstSearch, DepthFirstSearch, DepthLimitedSearch,
                        SearchStatus, UniformCostSearch, UniformedCostSearch,
                        make_search)
from infrastructure import Action, Location, Node, TransitionModel
from engine import Environment, GameEngine, SearchAlgorithm, TravelAgent


def shortest_grid_distance(width, height, start, goal, obstacles):
    """Independent coordinate-only oracle; does not use model successors."""
    frontier = deque([(start, 0)])
    seen = {start}
    while frontier:
        position, distance = frontier.popleft()
        if position == goal:
            return distance
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = Location(position.x + dx, position.y + dy)
            if (0 <= nxt.x < width and 0 <= nxt.y < height and
                    nxt not in obstacles and nxt not in seen):
                seen.add(nxt)
                frontier.append((nxt, distance + 1))
    return None


class GraphModel(TransitionModel):
    """Directed weighted fixtures sharing the real model.expand/Node contract.

    Successors are explicit graph edges rather than geometric unit moves. This
    makes a least-cost route observably different from a fewest-edge route.
    """
    def __init__(self, edges, goal):
        locations = {goal, *edges}
        locations.update(nxt for choices in edges.values() for nxt, _ in choices)
        super().__init__(max(p.x for p in locations) + 1, 1, goal)
        object.__setattr__(self, 'edges', edges)
        object.__setattr__(self, 'expansion_calls', [])

    def successors(self, state):
        self._require_state(state)
        if state == self.goal:
            return ()
        return tuple((Action.RIGHT, target, cost)
                     for target, cost in self.edges.get(state, ()))

    def expand(self, node):
        self.expansion_calls.append(node)
        return super().expand(node)


class RequiredAlgorithmTests(unittest.TestCase):
    def assert_replayable(self, env, result):
        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(len(result.path), len(result.actions))
        self.assertEqual(result.goal_node.state_path(), [env.start] + result.path)
        self.assertEqual(result.goal_node.solution(), result.actions)
        self.assertEqual(result.goal_node.path_cost, len(result.path))
        self.assertEqual(result.goal_node.depth, len(result.path))
        for action, expected in zip(result.actions, result.path):
            self.assertTrue(env.execute(action).moved)
            self.assertEqual(env.agent_location, expected)
        self.assertEqual(env.agent_location, env.goal)

    def test_required_classes_produce_replayable_paths(self):
        for solver in (DepthFirstSearch(), BreadthFirstSearch(),
                       DepthLimitedSearch(depth_limit=12), UniformCostSearch()):
            with self.subTest(solver=type(solver).__name__):
                env = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
                before = (env.agent_location, env.steps)
                result = solver.search(env.transition_model(), env.start)
                self.assertEqual((env.agent_location, env.steps), before)
                self.assert_replayable(env, result)

    def test_bfs_and_ucs_match_independent_oracle_on_random_maps(self):
        rng = random.Random(31337)
        start, goal = Location(0, 0), Location(4, 3)
        for case in range(20):
            obstacles = {Location(x, y) for y in range(4) for x in range(5)
                         if Location(x, y) not in (start, goal) and rng.random() < 0.30}
            expected = shortest_grid_distance(5, 4, start, goal, obstacles)
            for solver in (BreadthFirstSearch(), UniformCostSearch()):
                with self.subTest(case=case, solver=type(solver).__name__):
                    model = TransitionModel(5, 4, goal, obstacles)
                    result = solver.search(model, start)
                    if expected is None:
                        self.assertEqual(result.status, SearchStatus.FAILURE)
                        self.assertIsNone(result.goal_node)
                        self.assertEqual(result.path, [])
                    else:
                        self.assertEqual(result.status, SearchStatus.SUCCESS)
                        self.assertEqual(len(result.path), expected)
                        self.assertEqual(result.goal_node.path_cost, expected)

    def test_dfs_explores_first_branch_and_bfs_finds_fewer_edges(self):
        s, a, b, c, g = [Location(x, 0) for x in range(5)]
        graph = {s: [(a, 1), (b, 1)], a: [(c, 1)], c: [(g, 1)], b: [(g, 1)]}
        dfs = DepthFirstSearch().search(GraphModel(graph, g), s)
        bfs = BreadthFirstSearch().search(GraphModel(graph, g), s)
        self.assertEqual(dfs.path, [a, c, g])
        self.assertEqual(bfs.path, [b, g])

    def test_ucs_finds_cheaper_route_with_more_edges(self):
        s, a, b, g = [Location(x, 0) for x in range(4)]
        graph = {s: [(g, 9), (a, 1)], a: [(b, 1)], b: [(g, 1)]}
        result = UniformCostSearch().search(GraphModel(graph, g), s)
        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.path, [a, b, g])
        self.assertEqual(result.goal_node.path_cost, 3)
        # BFS correctly prefers one edge here; UCS must not return a goal when
        # it is merely generated with an expensive path cost.
        bfs = BreadthFirstSearch().search(GraphModel(graph, g), s)
        self.assertEqual(bfs.path, [g])
        self.assertEqual(bfs.goal_node.path_cost, 9)

    def test_ucs_improves_a_frontier_state_and_skips_stale_entries(self):
        s, a, b, x, g = [Location(i, 0) for i in range(5)]
        graph = {s: [(a, 1), (b, 2)], a: [(x, 8)],
                 b: [(x, 1)], x: [(g, 20)]}
        model = GraphModel(graph, g)
        result = UniformCostSearch().search(model, s)
        self.assertEqual(result.path, [b, x, g])
        self.assertEqual(result.goal_node.path_cost, 23)
        self.assertEqual(sum(n.state == x for n in model.expansion_calls), 1)
        self.assertEqual(result.metrics.expanded, len(model.expansion_calls))

    def test_all_searches_handle_start_at_goal_without_expansion(self):
        start = Location(0, 0)
        for solver in (DepthFirstSearch(), BreadthFirstSearch(),
                       DepthLimitedSearch(depth_limit=0), UniformCostSearch()):
            with self.subTest(solver=type(solver).__name__):
                result = solver.search(TransitionModel(1, 1, start), start)
                self.assertEqual(result.status, SearchStatus.SUCCESS)
                self.assertEqual(result.path, [])
                self.assertEqual(result.actions, [])
                self.assertEqual(result.goal_node.path_cost, 0)
                self.assertEqual(result.metrics.expanded, 0)
                self.assertEqual(result.metrics.examined, 1)
                self.assertEqual(result.metrics.generated, 1)

    def test_exhausted_search_reports_failure(self):
        start, goal = Location(0, 0), Location(2, 2)
        model = TransitionModel(3, 3, goal, {Location(0, 1), Location(1, 0)})
        for solver in (DepthFirstSearch(), BreadthFirstSearch(),
                       DepthLimitedSearch(depth_limit=8), UniformCostSearch()):
            with self.subTest(solver=type(solver).__name__):
                result = solver.search(model, start)
                self.assertEqual(result.status, SearchStatus.FAILURE)
                self.assertIsNone(result.goal_node)
                self.assertEqual(result.path, [])
                self.assertEqual(result.actions, [])

    def test_factory_and_rubric_spelling_alias(self):
        expected = {SearchAlgorithm.DFS: DepthFirstSearch,
                    SearchAlgorithm.BFS: BreadthFirstSearch,
                    SearchAlgorithm.DLS: DepthLimitedSearch,
                    SearchAlgorithm.UCS: UniformCostSearch}
        for choice, cls in expected.items():
            with self.subTest(choice=choice):
                self.assertIsInstance(make_search(choice, depth_limit=7), cls)
        self.assertIs(UniformedCostSearch, UniformCostSearch)
        with self.assertRaises(ValueError):
            make_search('not an algorithm')


class DepthLimitedSearchTests(unittest.TestCase):
    def test_goal_exactly_at_limit_succeeds_and_below_limit_cuts_off(self):
        start, goal = Location(0, 0), Location(4, 0)
        model = TransitionModel(5, 1, goal)
        cutoff = DepthLimitedSearch(depth_limit=3).search(model, start)
        self.assertEqual(cutoff.status, SearchStatus.CUTOFF)
        self.assertIsNone(cutoff.goal_node)
        success = DepthLimitedSearch(depth_limit=4).search(model, start)
        self.assertEqual(success.status, SearchStatus.SUCCESS)
        self.assertEqual(len(success.path), 4)

    def test_zero_limit_is_cutoff_for_non_goal(self):
        start, goal = Location(0, 0), Location(1, 0)
        result = DepthLimitedSearch(depth_limit=0).search(TransitionModel(2, 1, goal), start)
        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(result.explored, frozenset({start}))
        self.assertEqual(result.metrics.expanded, 0)
        self.assertEqual(result.metrics.generated, 1)

    def test_shallower_revisit_keeps_remaining_depth_allowance(self):
        s, a, c, x, b, g = [Location(i, 0) for i in range(6)]
        graph = {s: [(a, 1), (b, 1)], a: [(c, 1)], c: [(x, 1)],
                 b: [(x, 1)], x: [(g, 1)]}
        result = DepthLimitedSearch(depth_limit=3).search(GraphModel(graph, g), s)
        self.assertEqual(result.status, SearchStatus.SUCCESS)
        self.assertEqual(result.path, [b, x, g])
        self.assertEqual(result.goal_node.depth, 3)

    def test_cycles_do_not_repeat_ancestors_in_solution(self):
        s, a, b, g = [Location(i, 0) for i in range(4)]
        graph = {s: [(a, 1)], a: [(s, 1), (b, 1)],
                 b: [(a, 1), (g, 1)]}
        result = DepthLimitedSearch(depth_limit=6).search(GraphModel(graph, g), s)
        self.assertEqual(result.path, [a, b, g])
        states = result.goal_node.state_path()
        self.assertEqual(len(states), len(set(states)))

    def test_negative_limit_rejected(self):
        with self.assertRaises(ValueError):
            DepthLimitedSearch(depth_limit=-1)


class SearchMetricsAndIterationTests(unittest.TestCase):
    def test_metrics_match_instrumented_expansion_calls(self):
        s, a, g = [Location(i, 0) for i in range(3)]
        # The back edge is generated and then pruned by graph-search duplicate
        # detection. Generated count includes this child and the root.
        model = GraphModel({s: [(a, 1)], a: [(s, 1), (g, 1)]}, g)
        result = BreadthFirstSearch().search(model, s)
        self.assertEqual(result.metrics.examined, 3)
        self.assertEqual(result.metrics.expanded, 2)
        self.assertEqual(result.metrics.generated, 4)
        self.assertEqual(result.metrics.frontier_peak, 1)
        self.assertEqual(result.explored, frozenset({s, a, g}))
        self.assertEqual(result.expanded_states, frozenset({s, a}))
        self.assertEqual([n.state for n in model.expansion_calls], [s, a])
        self.assertGreaterEqual(result.metrics.elapsed_ms, 0)
        self.assertGreaterEqual(result.metrics.cpu_ms, 0)

    def test_dls_boundary_nodes_are_examined_but_not_expanded(self):
        s, a, g = [Location(i, 0) for i in range(3)]
        model = GraphModel({s: [(a, 1)], a: [(g, 1)]}, g)
        result = DepthLimitedSearch(depth_limit=1).search(model, s)
        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(result.metrics.examined, 2)
        self.assertEqual(result.metrics.expanded, 1)
        self.assertEqual([n.state for n in model.expansion_calls], [s])
        self.assertEqual(result.expanded_states, frozenset({s}))

    def test_step_iterator_matches_full_search(self):
        start, goal = Location(0, 0), Location(3, 0)
        for solver in (DepthFirstSearch(), BreadthFirstSearch(),
                       DepthLimitedSearch(depth_limit=10), UniformCostSearch()):
            with self.subTest(solver=type(solver).__name__):
                model = TransitionModel(4, 3, goal, {Location(1, 0)})
                expected = solver.search(model, start)
                iterator = solver.steps(model, start)
                events = []
                while True:
                    try:
                        events.append(next(iterator))
                    except StopIteration as completion:
                        result = completion.value
                        break
                self.assertEqual(result.status, expected.status)
                self.assertEqual(result.path, expected.path)
                self.assertEqual(result.actions, expected.actions)
                self.assertEqual(result.metrics.expanded, expected.metrics.expanded)
                self.assertEqual(result.metrics.examined, expected.metrics.examined)
                self.assertEqual(result.metrics.generated, expected.metrics.generated)
                self.assertEqual(len(events), result.metrics.examined)
                self.assertEqual(sum(bool(e.expanded) for e in events), result.metrics.expanded)
                examined_counts = [event.metrics.examined for event in events]
                self.assertEqual(examined_counts, sorted(examined_counts))
                self.assertTrue(all(isinstance(event.node, Node) for event in events))

    def test_resource_limit_stops_search_without_claiming_no_route(self):
        start, goal = Location(0, 0), Location(4, 0)
        model = TransitionModel(5, 1, goal)
        for solver in (DepthFirstSearch(), BreadthFirstSearch(),
                       DepthLimitedSearch(depth_limit=6), UniformCostSearch()):
            with self.subTest(solver=type(solver).__name__):
                result = solver.search(model, start, max_nodes=1)
                self.assertEqual(result.status, SearchStatus.RESOURCE_LIMIT)
                self.assertEqual(result.metrics.examined, 1)
                self.assertIsNone(result.goal_node)
                self.assertEqual(result.path, [])

    def test_nonpositive_resource_limit_rejected(self):
        start, goal = Location(0, 0), Location(1, 0)
        for invalid in (0, -1):
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                BreadthFirstSearch().search(TransitionModel(2, 1, goal), start,
                                           max_nodes=invalid)


class GameLoopStatusTests(unittest.TestCase):
    def test_depth_cutoff_message_is_not_unreachable(self):
        env = Environment(5, 1, Location(0, 0), Location(4, 0))
        agent = TravelAgent(SearchAlgorithm.DLS, depth_limit=3)
        step = GameEngine(env, agent).tick()
        self.assertFalse(step.moved)
        self.assertEqual(agent.last_result.status, SearchStatus.CUTOFF)
        self.assertIn('Depth limit', step.message)
        self.assertNotIn('No reachable route', step.message)
        self.assertEqual(env.steps, 0)
        agent.set_depth_limit(4)
        self.assertIsNone(agent.last_result)
        self.assertTrue(GameEngine(env, agent).tick().moved)

    def test_resource_limit_message_and_failed_plan_cache(self):
        env = Environment(5, 1, Location(0, 0), Location(4, 0))
        agent = TravelAgent(SearchAlgorithm.BFS, max_nodes=1)
        game = GameEngine(env, agent)
        step = game.tick()
        self.assertFalse(step.moved)
        self.assertIn('incomplete', step.message)
        self.assertEqual(agent.last_result.status, SearchStatus.RESOURCE_LIMIT)
        cached = agent.last_result
        self.assertFalse(game.tick().moved)
        self.assertIs(agent.last_result, cached)
        agent.max_nodes = 10
        self.assertTrue(game.tick().moved)
        self.assertEqual(agent.last_result.status, SearchStatus.SUCCESS)


if __name__ == '__main__':
    unittest.main()
