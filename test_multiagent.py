"""Simultaneous execution, timed reservations, and deterministic seed contracts."""
from collections import deque
import random
import unittest

from algorithms import SearchAlgorithm, SearchStatus
from infrastructure import Location
from multiagent import (AgentSpec, CooperativePlanner, MultiAgentEnvironment,
                        generate_multiagent_environment, new_multiagent_environment)


def execute_plan(test, environment, plan):
    """Replay every joint tick through the independent execution validator."""
    test.assertTrue(plan.found, plan.message)
    for agent in environment.agents:
        test.assertEqual(plan.paths[agent.agent_id][0], environment.positions[agent.agent_id])
        test.assertEqual(plan.paths[agent.agent_id][-1], agent.goal)
    for time in range(1, plan.makespan + 1):
        proposed = {agent.agent_id: plan.paths[agent.agent_id][
                        min(time, len(plan.paths[agent.agent_id]) - 1)]
                    for agent in environment.agents}
        previous = dict(environment.positions)
        # These direct checks do not rely on the planner's reservations.
        test.assertEqual(len(set(proposed.values())), len(proposed))
        ids = tuple(proposed)
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                test.assertFalse(proposed[first] == previous[second] and
                                 proposed[second] == previous[first])
        environment.execute_joint(proposed)
    test.assertTrue(environment.all_reached)
    test.assertEqual(environment.steps, plan.makespan)


def individually_reachable(environment, agent):
    frontier = deque([agent.start])
    seen = {agent.start}
    while frontier:
        current = frontier.popleft()
        if current == agent.goal:
            return True
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = Location(current.x + dx, current.y + dy)
            if (0 <= nxt.x < environment.width and 0 <= nxt.y < environment.height and
                    nxt not in environment.obstacles and nxt not in seen):
                seen.add(nxt)
                frontier.append(nxt)
    return False


class JointEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.agents = (AgentSpec('A1', Location(0, 0), Location(2, 0)),
                       AgentSpec('A2', Location(2, 1), Location(0, 1)))
        self.env = MultiAgentEnvironment(3, 2, self.agents)

    def test_simultaneous_moves_waits_and_reset(self):
        self.env.execute_joint({'A1': Location(1, 0), 'A2': Location(2, 1)})
        self.assertEqual(self.env.steps, 1)
        self.assertEqual(self.env.move_counts, {'A1': 1, 'A2': 0})
        self.assertEqual(self.env.wait_counts, {'A1': 0, 'A2': 1})
        result = self.env.execute_joint({'A1': Location(2, 0), 'A2': Location(1, 1)})
        self.assertEqual(result.goals_found, 1)
        self.assertFalse(result.all_reached)
        self.env.reset()
        self.assertEqual(self.env.positions, {a.agent_id: a.start for a in self.agents})
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env.move_counts, {'A1': 0, 'A2': 0})
        self.assertEqual(self.env.wait_counts, {'A1': 0, 'A2': 0})

    def test_invalid_joint_proposals_are_atomic(self):
        proposals = [
            {'A1': Location(1, 0)},  # missing agent
            {'A1': Location(1, 0), 'A2': Location(2, 1), 'unknown': Location(0, 1)},
            {'A1': Location(1, 1), 'A2': Location(2, 1)},  # diagonal
            {'A1': Location(2, 0), 'A2': Location(2, 1)},  # teleport
            {'A1': Location(-1, 0), 'A2': Location(2, 1)},
            {'A1': None, 'A2': Location(2, 1)},
        ]
        before = (dict(self.env.positions), self.env.steps,
                  dict(self.env.move_counts), dict(self.env.wait_counts))
        for proposal in proposals:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                self.env.execute_joint(proposal)
            self.assertEqual(before, (self.env.positions, self.env.steps,
                                      self.env.move_counts, self.env.wait_counts))

    def test_vertex_conflict_and_head_on_swap_rejected(self):
        env = MultiAgentEnvironment(3, 2, (
            AgentSpec('A1', Location(0, 0), Location(2, 1)),
            AgentSpec('A2', Location(2, 0), Location(0, 1))))
        with self.assertRaises(ValueError):
            env.execute_joint({'A1': Location(1, 0), 'A2': Location(1, 0)})
        env.execute_joint({'A1': Location(1, 0), 'A2': Location(2, 0)})
        before = dict(env.positions)
        with self.assertRaises(ValueError):
            env.execute_joint({'A1': Location(2, 0), 'A2': Location(1, 0)})
        self.assertEqual(env.positions, before)
        self.assertEqual(env.steps, 1)

    def test_moving_into_vacated_cell_is_allowed(self):
        env = MultiAgentEnvironment(4, 2, (
            AgentSpec('A1', Location(0, 0), Location(3, 0)),
            AgentSpec('A2', Location(1, 0), Location(3, 1))))
        env.execute_joint({'A1': Location(1, 0), 'A2': Location(2, 0)})
        self.assertEqual(env.positions['A1'], Location(1, 0))
        self.assertEqual(env.positions['A2'], Location(2, 0))

    def test_completed_agents_cannot_leave_or_be_blocked(self):
        self.env.execute_joint({'A1': Location(1, 0), 'A2': Location(1, 1)})
        self.env.execute_joint({'A1': Location(2, 0), 'A2': Location(0, 1)})
        self.assertTrue(self.env.all_reached)
        with self.assertRaises(ValueError):
            self.env.execute_joint({'A1': Location(1, 0), 'A2': Location(0, 1)})
        before = self.env.steps
        self.env.execute_joint(dict(self.env.positions))
        self.assertEqual(self.env.steps, before)
        for agent in self.agents:
            self.env.set_obstacle(agent.start, True)
            self.env.set_obstacle(agent.goal, True)
        self.assertFalse(self.env.obstacles)

    def test_obstacles_protect_endpoints_and_current_positions(self):
        self.env.execute_joint({'A1': Location(1, 0), 'A2': Location(2, 1)})
        self.env.set_obstacle(Location(1, 0), True)
        self.env.set_obstacle(Location(0, 0), True)
        self.env.set_obstacle(Location(2, 0), True)
        self.assertEqual(self.env.obstacles, set())
        self.env.set_obstacle(Location(1, 1), True)
        self.assertIn(Location(1, 1), self.env.obstacles)
        with self.assertRaises(ValueError):
            self.env.execute_joint({'A1': Location(1, 0), 'A2': Location(1, 1)})
        self.env.set_obstacle(Location(1, 1), False)
        self.assertFalse(self.env.obstacles)

    def test_constructor_rejects_duplicate_ids_starts_and_goals(self):
        first = AgentSpec('A1', Location(0, 0), Location(2, 0))
        for other in (AgentSpec('A1', Location(1, 0), Location(2, 1)),
                      AgentSpec('A2', Location(0, 0), Location(2, 1)),
                      AgentSpec('A2', Location(1, 0), Location(2, 0))):
            with self.subTest(other=other), self.assertRaises(ValueError):
                MultiAgentEnvironment(3, 2, (first, other))
        with self.assertRaises(ValueError):
            MultiAgentEnvironment(3, 2, ())
        with self.assertRaises(ValueError):
            MultiAgentEnvironment(3, 2, (first,), {first.goal})
        with self.assertRaises(ValueError):
            MultiAgentEnvironment(0, 2, (first,))
        with self.assertRaises(ValueError):
            AgentSpec('', Location(0, 0), Location(1, 0))


class CooperativePlanningTests(unittest.TestCase):
    def crossing(self):
        free = {Location(x, 1) for x in range(3)} | {Location(1, y) for y in range(3)}
        blocked = {Location(x, y) for x in range(3) for y in range(3)} - free
        return MultiAgentEnvironment(3, 3, (
            AgentSpec('A1', Location(0, 1), Location(2, 1)),
            AgentSpec('A2', Location(1, 0), Location(1, 2))), blocked)

    def test_crossing_is_coordinated_with_an_explicit_wait(self):
        env = self.crossing()
        plan = CooperativePlanner(horizon=8).plan(env)
        self.assertEqual(plan.status, SearchStatus.SUCCESS)
        self.assertEqual(plan.makespan, 3)
        self.assertEqual(plan.total_cost, 5)
        self.assertTrue(any(a == b for path in plan.paths.values()
                            for a, b in zip(path, path[1:])))
        execute_plan(self, env, plan)
        self.assertEqual(sum(env.wait_counts.values()), 1)

    def test_adjacent_swap_uses_a_detour_instead_of_edge_swap(self):
        env = MultiAgentEnvironment(2, 2, (
            AgentSpec('A1', Location(0, 0), Location(1, 0)),
            AgentSpec('A2', Location(1, 0), Location(0, 0))))
        result = CooperativePlanner(horizon=8).plan(env)
        self.assertEqual(result.makespan, 3)
        execute_plan(self, env, result)

    def test_failed_priority_order_retries_and_finds_a_joint_schedule(self):
        # A1's shortest route ends at the corridor bottleneck, trapping A2.
        # Prioritizing A2 instead lets A1 use the side pocket, then continue.
        free = {Location(x, 0) for x in range(4)} | {Location(1, 1)}
        blocked = {Location(x, y) for x in range(4) for y in range(2)} - free
        env = MultiAgentEnvironment(4, 2, (
            AgentSpec('A1', Location(0, 0), Location(2, 0)),
            AgentSpec('A2', Location(3, 0), Location(0, 0))), blocked)
        result = CooperativePlanner(horizon=8).plan(env)
        self.assertTrue(result.found, result.message)
        self.assertGreater(result.attempts, 1)
        self.assertEqual(result.priority_order, ('A2', 'A1'))
        self.assertIn(Location(1, 1), result.paths['A1'])
        execute_plan(self, env, result)

    def test_all_six_algorithms_replay_on_fixed_three_agent_map(self):
        for algorithm in SearchAlgorithm:
            with self.subTest(algorithm=algorithm):
                env = new_multiagent_environment()
                result = CooperativePlanner(algorithm, depth_limit=20, horizon=30).plan(env)
                execute_plan(self, env, result)
                self.assertEqual(env.goals_found, 3)

    def test_completed_agent_is_reserved_before_higher_priority_mover(self):
        env = MultiAgentEnvironment(3, 2, (
            AgentSpec('A1', Location(0, 1), Location(2, 1)),
            AgentSpec('A2', Location(1, 1), Location(1, 1))))
        result = CooperativePlanner(horizon=10).plan(env)
        self.assertEqual(result.paths['A2'], [Location(1, 1)])
        self.assertNotIn(Location(1, 1), result.paths['A1'])
        self.assertEqual(len(result.paths['A1']) - 1, 4)
        execute_plan(self, env, result)

    def test_goal_is_not_entered_before_a_later_reserved_visit(self):
        # A1 crosses A2's goal at t=2; A2 could enter it at t=1, but cannot
        # subsequently leave a reached goal. A2 must delay arrival instead.
        env = MultiAgentEnvironment(4, 2, (
            AgentSpec('A1', Location(0, 0), Location(3, 0)),
            AgentSpec('A2', Location(2, 1), Location(2, 0))))
        result = CooperativePlanner(horizon=8).plan(env)
        self.assertTrue(result.found, result.message)
        self.assertNotIn(Location(2, 0), result.paths['A2'][:3])
        execute_plan(self, env, result)

    def test_planning_and_incremental_events_do_not_mutate_environment(self):
        env = self.crossing()
        before = (dict(env.positions), env.steps, set(env.obstacles), dict(env.move_counts))
        planner = CooperativePlanner(horizon=8)
        events = []
        iterator = planner.steps(env)
        while True:
            try:
                events.append(next(iterator))
            except StopIteration as completed:
                result = completed.value
                break
        self.assertEqual(before, (env.positions, env.steps, env.obstacles, env.move_counts))
        self.assertEqual(len(events), result.metrics.examined)
        self.assertEqual(sum(event.expanded for event in events), result.metrics.expanded)
        self.assertTrue(all(event.agent_id in env.positions for event in events))
        expected = planner.plan(env)
        self.assertEqual(result.paths, expected.paths)
        self.assertEqual(result.metrics.expanded, expected.metrics.expanded)
        self.assertGreaterEqual(result.metrics.cpu_ms, 0)
        self.assertGreaterEqual(result.metrics.elapsed_ms, 0)

    def test_incomplete_bounds_do_not_claim_joint_impossibility(self):
        env = MultiAgentEnvironment(2, 1, (
            AgentSpec('A1', Location(0, 0), Location(1, 0)),
            AgentSpec('A2', Location(1, 0), Location(0, 0))))
        result = CooperativePlanner(horizon=5).plan(env)
        self.assertFalse(result.found)
        self.assertEqual(result.status, SearchStatus.CUTOFF)
        self.assertEqual(result.paths, {})
        self.assertIn('does not prove', result.message)
        self.assertEqual(env.steps, 0)

    def test_horizon_and_dls_limits_are_inclusive(self):
        env = MultiAgentEnvironment(3, 1, (
            AgentSpec('A1', Location(0, 0), Location(2, 0)),))
        self.assertEqual(CooperativePlanner(horizon=1).plan(env).status, SearchStatus.CUTOFF)
        self.assertTrue(CooperativePlanner(horizon=2).plan(env).found)
        self.assertEqual(CooperativePlanner(SearchAlgorithm.DLS, depth_limit=1,
                         horizon=10).plan(env).status, SearchStatus.CUTOFF)
        self.assertTrue(CooperativePlanner(SearchAlgorithm.DLS, depth_limit=2,
                        horizon=10).plan(env).found)

    def test_zero_horizon_and_already_complete_world(self):
        env = MultiAgentEnvironment(2, 1, (
            AgentSpec('A1', Location(0, 0), Location(1, 0)),))
        self.assertEqual(CooperativePlanner(horizon=0).plan(env).status, SearchStatus.CUTOFF)
        complete = MultiAgentEnvironment(1, 1, (
            AgentSpec('A1', Location(0, 0), Location(0, 0)),))
        result = CooperativePlanner(horizon=0).plan(complete)
        self.assertTrue(result.found)
        self.assertEqual(result.paths, {'A1': [Location(0, 0)]})
        self.assertEqual((result.makespan, result.total_cost), (0, 0))
        self.assertEqual(result.metrics.expanded, 0)

    def test_global_node_budget_and_invalid_planner_parameters(self):
        result = CooperativePlanner(horizon=20, max_nodes=1).plan(self.crossing())
        self.assertEqual(result.status, SearchStatus.RESOURCE_LIMIT)
        self.assertEqual(result.metrics.examined, 1)
        self.assertEqual(result.paths, {})
        for options in ({'horizon': -1}, {'depth_limit': -1}, {'max_nodes': 0},
                        {'horizon': True}, {'max_nodes': 1.5}, {'algorithm': 'unknown'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CooperativePlanner(**options)

    def test_replanning_starts_at_current_positions(self):
        env = self.crossing()
        first = CooperativePlanner(horizon=8).plan(env)
        env.execute_joint({a.agent_id: first.paths[a.agent_id][1] for a in env.agents})
        before = dict(env.positions)
        result = CooperativePlanner(horizon=8).plan(env)
        self.assertTrue(result.found)
        self.assertEqual({key: path[0] for key, path in result.paths.items()}, before)
        for time in range(1, result.makespan + 1):
            env.execute_joint({key: path[min(time, len(path) - 1)]
                               for key, path in result.paths.items()})
        self.assertTrue(env.all_reached)


class MultiAgentSeedTests(unittest.TestCase):
    def test_seed_reproducibility_variation_and_rng_isolation(self):
        before = random.getstate()
        first = generate_multiagent_environment(42)
        self.assertEqual(before, random.getstate())
        second = generate_multiagent_environment(42)
        other = generate_multiagent_environment(43)
        self.assertEqual(first.agents, second.agents)
        self.assertEqual(first.obstacles, second.obstacles)
        self.assertNotEqual(first.agents, other.agents)
        endpoints = [p for a in first.agents for p in (a.start, a.goal)]
        self.assertEqual(len(set(endpoints)), 2 * len(first.agents))
        self.assertFalse(set(endpoints) & first.obstacles)

    def test_seeded_pairs_individually_reachable_even_at_full_density(self):
        for density in (0.0, 0.35, 1.0):
            for seed in range(10):
                with self.subTest(seed=seed, density=density):
                    env = generate_multiagent_environment(seed, 7, 6, 4, density)
                    for agent in env.agents:
                        self.assertTrue(individually_reachable(env, agent))

    def test_successful_seeded_schedules_execute_without_collisions(self):
        # These open maps have room to coordinate. The generator separately
        # guarantees only individual reachability, not universal MAPF success.
        for seed in range(10):
            with self.subTest(seed=seed):
                env = generate_multiagent_environment(seed, 8, 7, 3, 0.0)
                result = CooperativePlanner(horizon=60).plan(env)
                execute_plan(self, env, result)

    def test_invalid_seed_generator_parameters(self):
        for options in ({'agent_count': 0}, {'agent_count': 3, 'width': 2, 'height': 2},
                        {'width': 1}, {'height': 1}, {'obstacle_density': -0.1},
                        {'obstacle_density': 1.1}, {'obstacle_density': float('nan')},
                        {'agent_count': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate_multiagent_environment(42, **options)
        with self.assertRaises(ValueError):
            generate_multiagent_environment('not an integer')


if __name__ == '__main__':
    unittest.main()
