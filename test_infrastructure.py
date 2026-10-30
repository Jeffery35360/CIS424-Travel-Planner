"""Sprint 2 contracts and integration with the original game engine."""
import unittest
from dataclasses import FrozenInstanceError

from engine import (Action, Environment, GameEngine, Location, Node,
                    SearchAlgorithm, TransitionModel, TravelAgent)


class NodeTests(unittest.TestCase):
    def test_root_and_path_reconstruction(self):
        root = Node(Location(0, 0))
        first = root.child(Action.DOWN, Location(0, 1))
        goal = first.child(Action.RIGHT, Location(1, 1))
        self.assertIsNone(root.parent)
        self.assertIsNone(root.action)
        self.assertEqual((root.depth, root.path_cost, root.solution()), (0, 0, []))
        self.assertEqual(goal.path(), [root, first, goal])
        self.assertEqual(goal.state_path(), [Location(0, 0), Location(0, 1), Location(1, 1)])
        self.assertEqual(goal.solution(), [Action.DOWN, Action.RIGHT])
        self.assertEqual((goal.depth, goal.path_cost), (2, 2))
        self.assertIs(goal.parent, first)

    def test_same_state_can_have_distinct_search_nodes(self):
        root = Node(Location(0, 0))
        down = root.child(Action.DOWN, Location(0, 1))
        revisited = down.child(Action.UP, Location(0, 0))
        self.assertEqual(root.state, revisited.state)
        self.assertNotEqual(root, revisited)
        self.assertEqual((revisited.depth, revisited.path_cost), (2, 2))
        self.assertEqual(len({root.state, revisited.state}), 1)

    def test_immutable_nodes_and_invalid_metadata(self):
        root = Node(Location(0, 0))
        with self.assertRaises(FrozenInstanceError):
            root.depth = 4
        with self.assertRaises(ValueError):
            Node(root.state, action=Action.UP)
        with self.assertRaises(ValueError):
            Node(root.state, path_cost=1)
        with self.assertRaises(ValueError):
            Node(Location(0, 1), parent=root)
        with self.assertRaises(ValueError):
            root.child(Action.DOWN, Location(0, 1), -1)


class TransitionModelTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
        self.model = self.env.transition_model()

    def test_corner_obstacle_and_action_order(self):
        self.assertEqual(self.model.actions(Location(0, 0)), (Action.DOWN,))
        self.assertEqual(self.model.successors(Location(0, 0)),
                         ((Action.DOWN, Location(0, 1), 1),))
        self.assertEqual(self.model.actions(Location(2, 1)), tuple(Action))

    def test_result_and_cost_do_not_change_live_environment(self):
        before = (self.env.agent_location, self.env.steps, set(self.env.obstacles))
        target = self.model.result(Location(0, 0), Action.DOWN)
        self.assertEqual(target, Location(0, 1))
        self.assertEqual(self.model.step_cost(Location(0, 0), Action.DOWN, target), 1)
        self.assertEqual(before, (self.env.agent_location, self.env.steps, self.env.obstacles))
        with self.assertRaises(ValueError):
            self.model.step_cost(Location(0, 0), Action.DOWN, Location(2, 2))

    def test_invalid_actions_are_rejected(self):
        for action in (Action.UP, Action.LEFT, Action.RIGHT, 'DOWN', None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.model.result(Location(0, 0), action)

    def test_invalid_states_are_rejected(self):
        for state in (Location(-1, 0), Location(4, 0), Location(1, 0)):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    self.model.actions(state)
                with self.assertRaises(ValueError):
                    self.model.expand(Node(state))

    def test_expansion_links_children_at_arbitrary_state(self):
        root = Node(Location(0, 0))
        node = self.model.expand(root)[0]
        children = self.model.expand(node)
        self.assertEqual([child.action for child in children],
                         [Action.UP, Action.DOWN, Action.RIGHT])
        for child in children:
            self.assertIs(child.parent, node)
            self.assertEqual((child.depth, child.path_cost), (2, 2))
            self.assertEqual(self.model.result(node.state, child.action), child.state)
        self.assertEqual((self.env.agent_location, self.env.steps), (root.state, 0))

    def test_goal_and_trapped_state_have_no_successors(self):
        goal = self.model.goal
        self.assertTrue(self.model.is_goal(goal))
        self.assertEqual(self.model.actions(goal), ())
        self.assertEqual(self.model.expand(Node(goal)), ())
        with self.assertRaises(ValueError):
            self.model.result(goal, Action.DOWN)
        trapped = TransitionModel(3, 3, Location(2, 2), {Location(0, 1), Location(1, 0)})
        self.assertFalse(trapped.is_goal(Location(0, 0)))
        self.assertEqual(trapped.expand(Node(Location(0, 0))), ())

    def test_snapshot_and_percept_construction(self):
        from_percept = TransitionModel.from_percept(self.env.perceive())
        self.assertEqual(self.model, from_percept)
        self.env.set_obstacle(Location(0, 1), True)
        self.assertEqual(self.model.actions(Location(0, 0)), (Action.DOWN,))
        self.assertEqual(self.env.transition_model().actions(Location(0, 0)), ())
        obstacles = {Location(1, 0)}
        model = TransitionModel(4, 3, Location(3, 0), obstacles)
        obstacles.clear()
        self.assertIn(Location(1, 0), model.obstacles)

    def test_invalid_map_configuration(self):
        for args in ((0, 3, Location(0, 0)), (3, 3, Location(3, 0)),
                     (3, 3, Location(2, 2), {Location(-1, 0)}),
                     (3, 3, Location(2, 2), {Location(2, 2)})):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TransitionModel(*args)


class InfrastructureIntegrationTests(unittest.TestCase):
    def test_every_cell_and_action_agree_with_execution(self):
        # Independent expected-movement calculation checks both implementations,
        # including terminal states, edges, and obstacle collisions.
        width, height, goal = 4, 3, Location(3, 0)
        obstacles = {Location(1, 0), Location(2, 2)}
        for y in range(height):
            for x in range(width):
                start = Location(x, y)
                if start in obstacles:
                    continue
                for action in Action:
                    with self.subTest(start=start, action=action):
                        env = Environment(width, height, start, goal, obstacles)
                        model = env.transition_model()
                        dx, dy = action.value
                        target = Location(x + dx, y + dy)
                        legal = (start != goal and 0 <= target.x < width and
                                 0 <= target.y < height and target not in obstacles)
                        self.assertEqual(action in model.actions(start), legal)
                        if legal:
                            self.assertEqual(model.result(start, action), target)
                        else:
                            with self.assertRaises(ValueError):
                                model.result(start, action)
                        result = env.execute(action)
                        self.assertEqual(result.moved, legal)
                        self.assertEqual(env.agent_location, target if legal else start)
                        self.assertEqual(env.steps, int(legal))

    def test_all_existing_planners_produce_replayable_node_chains(self):
        for algorithm in SearchAlgorithm:
            with self.subTest(algorithm=algorithm):
                env = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
                agent = TravelAgent(algorithm, depth_limit=12)
                route = agent.plan(env.perceive())
                goal = agent.last_goal_node
                self.assertIsNotNone(goal)
                self.assertEqual(goal.state_path(), [env.start] + route)
                self.assertEqual(goal.state, env.goal)
                self.assertEqual((goal.depth, goal.path_cost), (len(route), len(route)))
                for action, position in zip(goal.solution(), route):
                    self.assertTrue(env.execute(action).moved)
                    self.assertEqual(env.agent_location, position)
                self.assertEqual(env.agent_location, env.goal)

    def test_map_edits_replan_and_replace_the_goal_node(self):
        env = Environment(4, 2, Location(0, 0), Location(3, 0))
        agent = TravelAgent(SearchAlgorithm.BFS)
        self.assertEqual(len(agent.plan(env.perceive())), 3)
        old_goal = agent.last_goal_node
        env.set_obstacle(Location(1, 0), True)
        route = agent.plan(env.perceive())
        self.assertEqual(len(route), 5)
        self.assertNotIn(Location(1, 0), route)
        self.assertIsNot(agent.last_goal_node, old_goal)
        self.assertEqual(agent.last_goal_node.path_cost, 5)
        env.set_obstacle(Location(0, 1), True)
        self.assertEqual(agent.plan(env.perceive()), [])
        self.assertIsNone(agent.last_goal_node)
        self.assertFalse(GameEngine(env, agent).tick().moved)

    def test_start_at_goal_retains_zero_length_solution(self):
        env = Environment(1, 1, Location(0, 0), Location(0, 0))
        agent = TravelAgent()
        self.assertEqual(agent.plan(env.perceive()), [])
        self.assertEqual(agent.last_goal_node.path_cost, 0)
        self.assertEqual(agent.last_goal_node.solution(), [])
        self.assertEqual(env.perceive().traversable_neighbors, ())
        self.assertTrue(GameEngine(env, agent).tick().reached_goal)


if __name__ == '__main__':
    unittest.main()
