import unittest
from engine import Action, Environment, GameEngine, Location, TravelAgent, SearchAlgorithm, generate_seeded_environment


class EngineTests(unittest.TestCase):
    def test_all_algorithms_reach_goal(self):
        for algorithm in SearchAlgorithm:
            with self.subTest(algorithm=algorithm.value):
                env = generate_seeded_environment(42)
                game = GameEngine(env, TravelAgent(algorithm, depth_limit=30))
                for _ in range(env.width * env.height):
                    result = game.tick()
                    if result.reached_goal or not result.moved:
                        break
                self.assertTrue(result.reached_goal, algorithm.value)

    def test_bfs_ucs_astar_find_shortest_route(self):
        for algorithm in (SearchAlgorithm.BFS, SearchAlgorithm.UCS, SearchAlgorithm.ASTAR):
            with self.subTest(algorithm=algorithm.value):
                env = Environment(5, 4, Location(0, 0), Location(4, 0),
                                  {Location(1, 0), Location(2, 0)})
                route = TravelAgent(algorithm).plan(env.perceive())
                self.assertEqual(len(route), 6)
                self.assertEqual(route[-1], env.goal)

    def test_depth_limit_enforced(self):
        env = Environment(5, 1, Location(0, 0), Location(4, 0))
        self.assertEqual(TravelAgent(SearchAlgorithm.DLS, depth_limit=3).plan(env.perceive()), [])
        self.assertEqual(len(TravelAgent(SearchAlgorithm.DLS, depth_limit=4).plan(env.perceive())), 4)

    def test_switching_algorithms_invalidates_route(self):
        env = generate_seeded_environment(92)
        agent = TravelAgent(SearchAlgorithm.ASTAR)
        agent.plan(env.perceive())
        agent.set_algorithm(SearchAlgorithm.DFS)
        self.assertEqual(agent.last_path, [])
        self.assertEqual(agent.algorithm, SearchAlgorithm.DFS)
        self.assertEqual(agent.plan(env.perceive())[-1], env.goal)

    def test_route_cache_allows_dfs_progress(self):
        env = generate_seeded_environment(92)
        agent = TravelAgent(SearchAlgorithm.DFS)
        game = GameEngine(env, agent)
        initial = agent.plan(env.perceive())
        for _ in initial:
            result = game.tick()
            self.assertTrue(result.moved)
        self.assertTrue(result.reached_goal)
        self.assertEqual(env.steps, len(initial))

    def test_obstacle_and_boundary_are_impassable(self):
        e = Environment(3, 3, Location(0, 0), Location(2, 2), {Location(1, 0)})
        self.assertFalse(e.execute(Action.RIGHT).moved)
        self.assertFalse(e.execute(Action.UP).moved)
        self.assertEqual(e.agent_location, Location(0, 0))
        self.assertEqual(e.steps, 0)

    def test_agent_reaches_goal_on_shortest_path(self):
        e = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
        game = GameEngine(e)
        for _ in range(20):
            result = game.tick()
            if result.reached_goal:
                break
        self.assertTrue(result.reached_goal)
        self.assertEqual(e.agent_location, Location(3, 0))
        self.assertEqual(e.steps, 5)

    def test_no_route(self):
        e = Environment(3, 3, Location(0, 0), Location(2, 2),
                        {Location(0, 1), Location(1, 0)})
        result = GameEngine(e).tick()
        self.assertFalse(result.moved)
        self.assertIn('No reachable route', result.message)

    def test_goal_is_terminal(self):
        e = Environment(2, 2, Location(0, 0), Location(0, 0))
        self.assertTrue(GameEngine(e).tick().reached_goal)
        self.assertEqual(e.steps, 0)

    def test_percept_updates_and_map_edits(self):
        e = Environment(3, 2, Location(0, 0), Location(2, 0))
        e.set_obstacle(Location(1, 0), True)
        p = e.perceive()
        self.assertIn(Location(1, 0), p.obstacles)
        self.assertNotIn((Action.RIGHT, Location(1, 0)), p.traversable_neighbors)
        e.set_obstacle(Location(1, 0), False)
        self.assertIn((Action.RIGHT, Location(1, 0)), e.perceive().traversable_neighbors)

    def test_reset(self):
        e = Environment(3, 3, Location(0, 0), Location(2, 2))
        e.execute(Action.DOWN)
        e.reset()
        self.assertEqual(e.agent_location, Location(0, 0))
        self.assertEqual(e.steps, 0)

    def test_seed_reproducibility(self):
        a = generate_seeded_environment(12345)
        b = generate_seeded_environment(12345)
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.start, b.start)
        self.assertEqual(a.goal, b.goal)

    def test_seed_variation(self):
        a = generate_seeded_environment(1)
        b = generate_seeded_environment(2)
        self.assertNotEqual(a.obstacles, b.obstacles)

    def test_seeded_maps_reachable(self):
        for seed in range(100):
            env = generate_seeded_environment(seed)
            game = GameEngine(env)
            for _ in range(env.width * env.height):
                result = game.tick()
                if result.reached_goal:
                    break
                self.assertTrue(result.moved, f'Seed {seed} has no route')
            self.assertTrue(result.reached_goal, f'Seed {seed} did not reach goal')

    def test_invalid_density(self):
        with self.assertRaises(ValueError):
            generate_seeded_environment(0, obstacle_density=1.2)

    def test_invalid_setup(self):
        with self.assertRaises(ValueError):
            Environment(2, 2, Location(-1, 0), Location(1, 1))
        with self.assertRaises(ValueError):
            Environment(2, 2, Location(0, 0), Location(1, 1), {Location(1, 1)})


if __name__ == '__main__':
    unittest.main()
