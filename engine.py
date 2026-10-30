"""Sprint 3 travel planner: run python engine.py for the desktop GUI.

Headless: --demo, --infrastructure-demo, or --algorithms-demo.
Python 3.10+; only the standard library is required.
"""
from __future__ import annotations
import argparse
import random
import secrets
from typing import Optional
from infrastructure import Action, Location, Node, Percept, StepResult, TransitionModel
from algorithms import SearchAlgorithm, SearchStatus, SearchResult, make_search


class Environment:
    """Grid world with impassable obstacles, one traveler, and one goal."""

    def __init__(self, width: int, height: int, start: Location,
                 goal: Location, obstacles: set[Location] | None = None):
        if width <= 0 or height <= 0:
            raise ValueError('Grid dimensions must be positive.')
        self.width, self.height = width, height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or ())
        for p in [start, goal, *self.obstacles]:
            if not self.in_bounds(p):
                raise ValueError(f'Location is outside the grid: {p}')
        if start in self.obstacles or goal in self.obstacles:
            raise ValueError('Start and destination cannot contain obstacles.')
        self.agent_location = start
        self.steps = 0

    def in_bounds(self, loc: Location) -> bool:
        return 0 <= loc.x < self.width and 0 <= loc.y < self.height

    def can_enter(self, loc: Location) -> bool:
        return self.in_bounds(loc) and loc not in self.obstacles

    def neighbors(self, loc: Location) -> tuple[tuple[Action, Location], ...]:
        return tuple((action, state) for action, state, _ in
                     self.transition_model().successors(loc))

    def transition_model(self) -> TransitionModel:
        """Snapshot the current map so a planning pass cannot mutate it."""
        return TransitionModel.from_environment(self)

    def perceive(self) -> Percept:
        return Percept(self.agent_location, self.goal,
                       self.neighbors(self.agent_location),
                       frozenset(self.obstacles), self.width, self.height,
                       self.agent_location == self.goal)

    def execute(self, action: Action) -> StepResult:
        if self.agent_location == self.goal:
            return StepResult(action, False, 'Destination already reached.', True)
        try:
            next_location = self.transition_model().result(self.agent_location, action)
        except ValueError:
            return StepResult(action, False, 'Blocked: obstacle or grid boundary.', False)
        self.agent_location = next_location
        self.steps += 1
        finished = self.agent_location == self.goal
        return StepResult(action, True,
                          'Destination reached!' if finished else 'Traveler moved.', finished)

    def reset(self) -> None:
        self.agent_location, self.steps = self.start, 0

    def set_obstacle(self, loc: Location, blocked: bool) -> None:
        if not self.in_bounds(loc) or loc in (self.start, self.goal):
            return
        if blocked:
            if loc == self.agent_location:
                return
            self.obstacles.add(loc)
        else:
            self.obstacles.discard(loc)


class TravelAgent:
    """Perceive/plan/act adapter over the independent Sprint 3 strategies.

    Successful routes are cached while the map, strategy and depth limit stay
    unchanged. Node metadata and metrics belong to the last planning pass.
    """
    def __init__(self, algorithm: SearchAlgorithm | str = SearchAlgorithm.BFS,
                 depth_limit: int = 20, max_nodes: int = 100_000):
        self.algorithm = SearchAlgorithm(algorithm)
        # Validate even when another strategy is initially selected.
        from algorithms import DepthLimitedSearch
        DepthLimitedSearch(depth_limit)
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
            raise ValueError('max_nodes must be a positive integer.')
        self.depth_limit = depth_limit
        self.max_nodes = max_nodes
        self.invalidate()

    def set_algorithm(self, algorithm: SearchAlgorithm | str) -> None:
        self.algorithm = SearchAlgorithm(algorithm)
        self.invalidate()

    def set_depth_limit(self, depth_limit: int) -> None:
        from algorithms import DepthLimitedSearch
        DepthLimitedSearch(depth_limit)
        self.depth_limit = depth_limit
        self.invalidate()

    def invalidate(self) -> None:
        self.last_path: list[Location] = []
        self.last_explored: set[Location] = set()
        self.last_goal_node: Node | None = None
        self.last_result: SearchResult | None = None
        self._cached_route: list[Location] = []
        self._cached_map: tuple | None = None
        self._cached_start: Location | None = None

    @staticmethod
    def heuristic(a: Location, b: Location) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _search(self, percept: Percept) -> tuple[list[Location], set[Location]]:
        self.last_result = make_search(self.algorithm, self.depth_limit).search(
            TransitionModel.from_percept(percept), percept.current, self.max_nodes)
        self.last_goal_node = self.last_result.goal_node
        return self.last_result.path, set(self.last_result.explored)

    def plan(self, percept: Percept) -> list[Location]:
        signature = (percept.destination, percept.obstacles, percept.width,
                     percept.height, self.algorithm, self.depth_limit, self.max_nodes)
        if self._cached_map == signature:
            if percept.current in self._cached_route:
                index = self._cached_route.index(percept.current)
                self.last_path = self._cached_route[index + 1:]
                return list(self.last_path)
            if self._cached_start == percept.current and self.last_result is not None:
                return list(self.last_path)
        route, explored = self._search(percept)
        self._cached_map = signature
        self._cached_start = percept.current
        self._cached_route = [percept.current] + route if self.last_result.found else []
        self.last_path, self.last_explored = route, explored
        return list(route)

    def decide(self, percept: Percept) -> Optional[Action]:
        path = self.plan(percept)
        if not path:
            return None
        return next((action for action, loc in percept.traversable_neighbors
                     if loc == path[0]), None)


class GameEngine:
    """Connects the Sprint 1 environment to the updated planning agent."""
    def __init__(self, environment: Environment, agent: TravelAgent | None = None):
        self.environment = environment
        self.agent = agent or TravelAgent()

    def tick(self) -> StepResult:
        percept = self.environment.perceive()
        if percept.reached_goal:
            return StepResult(None, False, 'Destination already reached.', True)
        action = self.agent.decide(percept)
        if action is None:
            result = self.agent.last_result
            if result and result.status == SearchStatus.CUTOFF:
                message = f'Depth limit {self.agent.depth_limit} reached; increase the limit.'
            elif result and result.status == SearchStatus.RESOURCE_LIMIT:
                message = 'Search node limit reached; search is incomplete.'
            else:
                message = 'No reachable route to destination.'
            return StepResult(None, False, message, False)
        return self.environment.execute(action)


DEFAULT_OBSTACLES = {
    Location(3, y) for y in range(0, 7) if y != 5
} | {
    Location(7, y) for y in range(2, 9) if y != 3
} | {
    Location(x, 7) for x in range(0, 3)
}



def generate_seeded_environment(seed: int, width: int = 12, height: int = 10,
                                obstacle_density: float = 0.25) -> Environment:
    """Create a repeatable random, always-solvable map for the given seed.

    Uses a dedicated RNG rather than global random state. A randomly carved
    start-to-goal corridor is reserved before placing other obstacles.
    """
    if width < 2 or height < 2:
        raise ValueError('Seeded maps require dimensions of at least 2 by 2.')
    if not 0 <= obstacle_density <= 1:
        raise ValueError('Obstacle density must be between 0 and 1.')
    rng = random.Random(seed)
    start, goal = Location(1, 1), Location(width - 2, height - 2)
    if start == goal:
        start, goal = Location(0, 0), Location(width - 1, height - 1)

    # Reserve a winding, traversable corridor. Randomly choose whether to
    # advance horizontally or vertically until the goal is reached.
    cursor = start
    reserved = {start, goal}
    while cursor != goal:
        directions = []
        if cursor.x != goal.x:
            directions.append((1 if goal.x > cursor.x else -1, 0))
        if cursor.y != goal.y:
            directions.append((0, 1 if goal.y > cursor.y else -1))
        dx, dy = rng.choice(directions)
        cursor = Location(cursor.x + dx, cursor.y + dy)
        reserved.add(cursor)

    obstacles = {Location(x, y) for y in range(height) for x in range(width)
                 if Location(x, y) not in reserved and rng.random() < obstacle_density}
    return Environment(width, height, start, goal, obstacles)


def random_seed() -> int:
    """Get a fresh seed without disturbing the seeded map generator's RNG."""
    return secrets.randbelow(1_000_000_000)


def new_engine() -> GameEngine:
    return GameEngine(Environment(12, 10, Location(1, 1), Location(10, 8),
                                  DEFAULT_OBSTACLES))


def console_demo(algorithm: SearchAlgorithm = SearchAlgorithm.BFS,
                 depth_limit: int = 20) -> None:
    engine = new_engine()
    engine.agent = TravelAgent(algorithm, depth_limit)
    env = engine.environment
    print('Single-Agent Travel Planner: grid demonstration')
    print(f'Start={env.start}, destination={env.goal}')
    print(f'Algorithm={algorithm.value}, DLS limit={depth_limit}')
    for _ in range(env.width * env.height):
        result = engine.tick()
        print(f'Step {env.steps:2}: position={env.agent_location} | {result.message}')
        if not result.moved or result.reached_goal:
            break
    else:
        raise RuntimeError('Exceeded maximum number of demonstration steps.')
    if engine.agent.last_result is not None:
        search = engine.agent.last_result
        print(f'Search status={search.status.value}, '
              f'expanded={search.metrics.expanded}, '
              f'path cost={search.path_cost}')


def infrastructure_demo() -> None:
    """Deterministic Sprint 2 evidence: prediction, expansion and execution."""
    env = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
    model = env.transition_model()
    root = Node(env.agent_location)
    print('Sprint 2: Node and TransitionModel demonstration')
    print('Grid: 4 x 3 | start=(0, 0) | goal=(3, 0) | obstacle=(1, 0)')
    print(f'Root: state={root.state}, parent={root.parent}, '
          f'action={root.action}, depth={root.depth}, cost={root.path_cost}')
    print('\nPredict each action at the top-left corner:')
    for action in Action:
        try:
            print(f'  {action.name}: {model.result(root.state, action)}')
        except ValueError:
            print(f'  {action.name}: not applicable (boundary or obstacle)')
    print('\nExpand root (legal successor nodes only):')
    for child in model.expand(root):
        print(f'  action={child.action.name}, state={child.state}, '
              f'parent={child.parent.state}, depth={child.depth}, cost={child.path_cost}')
    print(f'Live environment unchanged: position={env.agent_location}, steps={env.steps}')

    # The successor function accepts hypothetical states, independent of the
    # live agent's position. This interior state has all four legal actions.
    interior = Location(2, 1)
    print(f'\nActions at hypothetical state {interior}: '
          + ', '.join(action.name for action in model.actions(interior)))
    trapped = TransitionModel(3, 3, Location(2, 2),
                              frozenset({Location(0, 1), Location(1, 0)}))
    print(f'Trapped start successor count: {len(trapped.expand(Node(Location(0, 0))))}')

    agent = TravelAgent(SearchAlgorithm.BFS)
    route = agent.plan(env.perceive())
    goal_node = agent.last_goal_node
    if goal_node is None:
        raise RuntimeError('Expected a route in the demonstration map.')
    print('\nExisting planner now uses Node objects and TransitionModel.expand:')
    print('  Algorithm: Breadth First Search')
    print('  Actions: ' + ' -> '.join(action.name for action in goal_node.solution()))
    print('  States: ' + ' -> '.join(f'({p.x}, {p.y})' for p in goal_node.state_path()))
    print(f'  Goal node depth={goal_node.depth}, path cost={goal_node.path_cost}')
    print(f'  Future positions returned by plan: {len(route)}')
    print('Execute the reconstructed actions through the Sprint 1 environment:')
    for action in goal_node.solution():
        result = env.execute(action)
        print(f'  Step {env.steps}: {action.name} -> {env.agent_location} | {result.message}')
    print(f'Terminal goal successor count: {len(model.expand(goal_node))}')


SPRINT3_ALGORITHMS = (SearchAlgorithm.DFS, SearchAlgorithm.BFS,
                      SearchAlgorithm.DLS, SearchAlgorithm.UCS)


def compare_algorithms(environment: Environment, depth_limit: int = 20
                       ) -> list[tuple[SearchAlgorithm, SearchResult]]:
    """Run the four required strategies on one snapshot without moving the agent."""
    model = environment.transition_model()
    return [(algorithm, make_search(algorithm, depth_limit).search(
        model, environment.agent_location)) for algorithm in SPRINT3_ALGORITHMS]


def algorithms_demo(depth_limit: int = 20) -> None:
    env = new_engine().environment
    print('Sprint 3: four required search algorithms')
    print(f'Same original 12 x 10 map, start={env.start}, goal={env.goal}')
    print(f'DLS depth limit={depth_limit}; all moves cost 1; search does not move the agent.')
    print('Expanded = actual successor expansions; generated includes the root and pruned children.')
    for algorithm, result in compare_algorithms(env, depth_limit):
        metrics = result.metrics
        print(f'\n{algorithm.value}: {result.status.value}')
        print(f'  moves={len(result.path) if result.found else "-"}, cost={result.path_cost}, '
              f'examined={metrics.examined}, expanded={metrics.expanded}, '
              f'generated={metrics.generated}, peak frontier={metrics.frontier_peak}')
        print(f'  active search={metrics.elapsed_ms:.3f} ms; thread CPU={metrics.cpu_ms:.3f} ms')
        if result.found:
            print('  actions: ' + ' -> '.join(action.name for action in result.actions))
            # Replay through the original movement API to verify the solution.
            replay = Environment(env.width, env.height, env.agent_location, env.goal, env.obstacles)
            for action in result.actions:
                if not replay.execute(action).moved:
                    raise RuntimeError('A reconstructed action was not executable.')
            print(f'  replay: goal reached={replay.agent_location == replay.goal}, steps={replay.steps}')

    corridor = Environment(5, 1, Location(0, 0), Location(4, 0))
    print('\nDepth-limit edge cases (5-cell corridor):')
    for limit in (0, 3, 4):
        result = make_search(SearchAlgorithm.DLS, limit).search(corridor.transition_model(), corridor.start)
        print(f'  limit={limit}: {result.status.value}; expanded={result.metrics.expanded}; cost={result.path_cost}')
    blocked = Environment(3, 3, Location(0, 0), Location(2, 2), {Location(0, 1), Location(1, 0)})
    failure = make_search(SearchAlgorithm.BFS).search(blocked.transition_model(), blocked.start)
    print(f'Trapped start: {failure.status.value} (exhausted search, not a depth cutoff).')
    print(f'Original live world unchanged: position={env.agent_location}, steps={env.steps}')


def launch_gui() -> None:
    try:
        from gui import launch_gui as open_window
    except ImportError as error:
        if error.name in ('tkinter', '_tkinter'):
            raise SystemExit('Tkinter is required for the GUI. Try: python engine.py --algorithms-demo') from error
        raise
    open_window()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Execute one route without a display')
    modes.add_argument('--infrastructure-demo', action='store_true', help='Demonstrate Sprint 2 nodes and transitions')
    modes.add_argument('--algorithms-demo', action='store_true', help='Compare all four Sprint 3 algorithms')
    choices = {'dfs': SearchAlgorithm.DFS, 'bfs': SearchAlgorithm.BFS,
               'dls': SearchAlgorithm.DLS, 'ucs': SearchAlgorithm.UCS,
               'astar': SearchAlgorithm.ASTAR, 'greedy': SearchAlgorithm.GREEDY}
    parser.add_argument('--algorithm', choices=choices, default='bfs', help='Strategy for --demo (default: bfs)')
    parser.add_argument('--depth-limit', type=int, default=20, help='DLS limit for headless demos (default: 20)')
    args = parser.parse_args()
    if args.depth_limit < 0:
        parser.error('--depth-limit must be nonnegative')
    if args.infrastructure_demo:
        infrastructure_demo()
    elif args.algorithms_demo:
        algorithms_demo(args.depth_limit)
    elif args.demo:
        console_demo(choices[args.algorithm], args.depth_limit)
    else:
        launch_gui()


if __name__ == '__main__':
    main()

