"""Coordinated multi-agent grid execution and bounded cooperative search.

Agents plan in a deterministic priority order in (location, time) space. Earlier
paths reserve vertices, directed edges and their destination for all later
times. A lower-priority agent may wait or detour, but cannot share a cell or
exchange an edge head-on. Failed orders are retried with deterministic priority
rotations. This is a practical, incomplete MAPF planner, not a proof of joint
solvability or a globally optimal multi-agent algorithm.

The original four-action enum stays unchanged. Repeated consecutive positions
in a timed path represent WAIT, with unit time/cost just like a movement.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import isfinite
import random
from time import perf_counter, thread_time
from typing import Generator

from algorithms import SearchAlgorithm, SearchMetrics, SearchStatus
from infrastructure import Action, Location


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}.')


def _location(value: Location) -> None:
    if (not isinstance(value, Location) or isinstance(value.x, bool) or
            isinstance(value.y, bool) or not isinstance(value.x, int) or
            not isinstance(value.y, int)):
        raise ValueError('Agent and obstacle coordinates must be integer Locations.')


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    start: Location
    goal: Location

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError('Each agent needs a nonempty string ID.')
        _location(self.start)
        _location(self.goal)


@dataclass(frozen=True)
class JointStepResult:
    moved_agents: tuple[str, ...]
    waiting_agents: tuple[str, ...]
    goals_found: int
    all_reached: bool
    message: str


class MultiAgentEnvironment:
    """One simultaneous clock; completed agents continue occupying their goals."""
    def __init__(self, width: int, height: int, agents: tuple[AgentSpec, ...],
                 obstacles: set[Location] | None = None):
        _integer(width, 'Width', 1)
        _integer(height, 'Height', 1)
        self.width, self.height = width, height
        self.agents = tuple(agents)
        if not self.agents or any(not isinstance(a, AgentSpec) for a in self.agents):
            raise ValueError('Supply at least one AgentSpec.')
        ids = [a.agent_id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError('Agent IDs must be unique.')
        for label, values in (('starts', [a.start for a in self.agents]),
                              ('goals', [a.goal for a in self.agents])):
            if len(set(values)) != len(values):
                raise ValueError(f'Agent {label} must be unique.')
        self.obstacles = set(obstacles or ())
        self._endpoints = {p for a in self.agents for p in (a.start, a.goal)}
        for state in self.obstacles | self._endpoints:
            _location(state)
            if not self.in_bounds(state):
                raise ValueError(f'Location is outside the grid: {state}')
        if self.obstacles & self._endpoints:
            raise ValueError('Agent starts and goals cannot contain obstacles.')
        self._specs = {a.agent_id: a for a in self.agents}
        self.reset()

    def in_bounds(self, location: Location) -> bool:
        return 0 <= location.x < self.width and 0 <= location.y < self.height

    def can_enter(self, location: Location) -> bool:
        return self.in_bounds(location) and location not in self.obstacles

    @property
    def goals_found(self) -> int:
        return sum(self.positions[a.agent_id] == a.goal for a in self.agents)

    @property
    def all_reached(self) -> bool:
        return self.goals_found == len(self.agents)

    def reset(self) -> None:
        self.positions = {a.agent_id: a.start for a in self.agents}
        self.steps = 0
        self.move_counts = {a.agent_id: 0 for a in self.agents}
        self.wait_counts = {a.agent_id: 0 for a in self.agents}

    def set_obstacle(self, location: Location, blocked: bool) -> None:
        """Protect every start, goal, and occupied cell, including after motion."""
        _location(location)
        if (not self.in_bounds(location) or location in self._endpoints or
                location in self.positions.values()):
            return
        if blocked:
            self.obstacles.add(location)
        else:
            self.obstacles.discard(location)

    def execute_joint(self, next_positions: dict[str, Location]) -> JointStepResult:
        """Validate all proposals before committing any of them.

        Every agent must be included. A cardinal one-cell move or WAIT is legal;
        vertex conflicts and two-agent edge swaps are rejected atomically.
        Moving into a cell vacated by another agent is permitted unless the two
        exchange positions. An agent that reaches its goal cannot leave it.
        """
        if not isinstance(next_positions, dict) or set(next_positions) != set(self._specs):
            raise ValueError('A joint proposal must include exactly every agent ID.')
        proposed = dict(next_positions)
        for agent_id, target in proposed.items():
            _location(target)
            current = self.positions[agent_id]
            if not self.can_enter(target):
                raise ValueError('A proposed position is blocked or outside the grid.')
            if abs(target.x - current.x) + abs(target.y - current.y) > 1:
                raise ValueError('Each agent may move one cardinal cell or wait.')
            if current == self._specs[agent_id].goal and target != current:
                raise ValueError('A completed agent must remain at its destination.')
        if len(set(proposed.values())) != len(proposed):
            raise ValueError('Vertex conflict: agents cannot share a cell.')
        ids = tuple(proposed)
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                if (proposed[first] == self.positions[second] and
                        proposed[second] == self.positions[first]):
                    raise ValueError('Edge conflict: agents cannot swap cells head-on.')
        moved = tuple(a.agent_id for a in self.agents
                      if proposed[a.agent_id] != self.positions[a.agent_id])
        waiting = tuple(a.agent_id for a in self.agents if a.agent_id not in moved)
        if self.all_reached:
            return JointStepResult((), waiting, self.goals_found, True,
                                   'All destinations already reached.')
        previously_completed = {a.agent_id for a in self.agents
                                if self.positions[a.agent_id] == a.goal}
        self.positions = proposed
        self.steps += 1
        for agent_id in moved:
            self.move_counts[agent_id] += 1
        for agent_id in waiting:
            if agent_id not in previously_completed:
                self.wait_counts[agent_id] += 1
        return JointStepResult(moved, waiting, self.goals_found, self.all_reached,
                               'All destinations reached!' if self.all_reached
                               else 'Joint tick executed.')


@dataclass(frozen=True)
class AgentPlan:
    status: SearchStatus
    path: list[Location]
    metrics: SearchMetrics

    @property
    def found(self) -> bool:
        return self.status == SearchStatus.SUCCESS

    @property
    def path_cost(self) -> int | None:
        return len(self.path) - 1 if self.found else None


@dataclass(frozen=True)
class MultiPlan:
    status: SearchStatus
    paths: dict[str, list[Location]]
    results: dict[str, AgentPlan]
    message: str
    metrics: SearchMetrics
    priority_order: tuple[str, ...] = ()
    attempts: int = 1

    @property
    def found(self) -> bool:
        return self.status == SearchStatus.SUCCESS

    @property
    def makespan(self) -> int:
        return max((len(path) - 1 for path in self.paths.values()), default=0)

    @property
    def total_cost(self) -> int:
        return sum(len(path) - 1 for path in self.paths.values())


@dataclass(frozen=True)
class MultiSearchEvent:
    agent_id: str
    state: Location
    expanded: bool
    metrics: SearchMetrics
    time: int = 0
    attempt: int = 1


@dataclass(frozen=True)
class _TimedNode:
    state: Location
    time: int
    parent: _TimedNode | None = None

    def path(self) -> list[Location]:
        reverse = []
        cursor: _TimedNode | None = self
        while cursor is not None:
            reverse.append(cursor.state)
            cursor = cursor.parent
        return list(reversed(reverse))


class _Counts:
    def __init__(self):
        self.examined = self.expanded = self.generated = self.frontier_peak = 0
        self.elapsed_ms = self.cpu_ms = 0.0

    def metrics(self) -> SearchMetrics:
        return SearchMetrics(self.examined, self.expanded, self.generated,
                             self.frontier_peak, self.elapsed_ms, self.cpu_ms)


class _Reservations:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.vertices: dict[int, set[Location]] = {}
        self.edges: set[tuple[Location, Location, int]] = set()

    def reserve(self, path: list[Location]) -> None:
        for time in range(self.horizon + 1):
            state = path[min(time, len(path) - 1)]
            self.vertices.setdefault(time, set()).add(state)
            if time:
                previous = path[min(time - 1, len(path) - 1)]
                self.edges.add((previous, state, time))

    def blocked(self, source: Location, target: Location, time: int) -> bool:
        return (target in self.vertices.get(time, ()) or
                (target, source, time) in self.edges)

    def safe_goal(self, goal: Location, arrival: int) -> bool:
        # All existing paths are stationary after their final arrival. Extending
        # their reservations through the horizon also covers all later times.
        return all(goal not in self.vertices.get(time, ())
                   for time in range(arrival, self.horizon + 1))


class CooperativePlanner:
    """Bounded prioritized searches with time/vertex/edge reservations.

    A*, greedy, BFS, UCS, DFS and DLS all operate on (Location, tick). A* uses
    tick + Manhattan distance, greedy uses Manhattan only, and waits cost 1.
    DLS applies the smaller of depth_limit and horizon. The node budget is for
    the whole planning call, including failed priority attempts. Prioritized
    planning is incomplete; unsuccessful schedules are reported as bounded
    cutoffs rather than proof that no joint solution exists.
    """
    def __init__(self, algorithm: SearchAlgorithm | str = SearchAlgorithm.ASTAR,
                 depth_limit: int = 20, horizon: int = 100,
                 max_nodes: int = 100_000):
        self.algorithm = SearchAlgorithm(algorithm)
        _integer(depth_limit, 'Depth limit')
        _integer(horizon, 'Horizon')
        _integer(max_nodes, 'Node limit', 1)
        self.depth_limit = depth_limit
        self.horizon = horizon
        self.max_nodes = max_nodes

    def plan(self, environment: MultiAgentEnvironment) -> MultiPlan:
        iterator = self.steps(environment)
        while True:
            try:
                next(iterator)
            except StopIteration as finished:
                return finished.value

    @staticmethod
    def _orders(ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        if not ids:
            return ((),)
        orders = []
        for sequence in (ids, tuple(reversed(ids))):
            for offset in range(len(ids)):
                order = sequence[offset:] + sequence[:offset]
                if order not in orders:
                    orders.append(order)
        return tuple(orders)

    def steps(self, environment: MultiAgentEnvironment
              ) -> Generator[MultiSearchEvent, None, MultiPlan]:
        # Freeze all live inputs at the first iteration. Search/animation never
        # writes the caller's positions, step counts, or obstacle set.
        world = MultiAgentEnvironment(environment.width, environment.height,
                    tuple(AgentSpec(a.agent_id, environment.positions[a.agent_id], a.goal)
                          for a in environment.agents), set(environment.obstacles))
        total = _Counts()
        completed = {a.agent_id: a for a in world.agents if a.start == a.goal}
        zero = SearchMetrics(0, 0, 0, 0, 0.0, 0.0)
        initial_results = {agent_id: AgentPlan(SearchStatus.SUCCESS, [a.start], zero)
                           for agent_id, a in completed.items()}
        ids = tuple(a.agent_id for a in world.agents if a.agent_id not in completed)
        specs = {a.agent_id: a for a in world.agents}
        orders = self._orders(ids)
        last_results = dict(initial_results)
        for attempt, order in enumerate(orders, 1):
            reservations = _Reservations(self.horizon)
            for a in completed.values():
                reservations.reserve([a.start])
            results = dict(initial_results)
            paths = {agent_id: list(result.path) for agent_id, result in results.items()}
            for agent_id in order:
                result = yield from self._agent_steps(world, specs[agent_id], reservations,
                                                     total, attempt)
                results[agent_id] = result
                if not result.found:
                    break
                paths[agent_id] = result.path
                reservations.reserve(result.path)
            else:
                return MultiPlan(SearchStatus.SUCCESS, paths, results,
                    f'Coordinated plan found in priority attempt {attempt}.',
                    total.metrics(), tuple(completed) + order, attempt)
            last_results = results
            if result.status == SearchStatus.RESOURCE_LIMIT:
                return MultiPlan(SearchStatus.RESOURCE_LIMIT, {}, last_results,
                    f'Node budget {self.max_nodes} reached; coordinated search is incomplete.',
                    total.metrics(), tuple(completed) + order, attempt)
        limit = min(self.depth_limit, self.horizon) if self.algorithm == SearchAlgorithm.DLS else self.horizon
        return MultiPlan(SearchStatus.CUTOFF, {}, last_results,
            f'No coordinated plan found within {limit} ticks and {len(orders)} priority orders. '
            'This bounded search does not prove that a joint solution is impossible.',
            total.metrics(), tuple(completed) + orders[-1], len(orders))

    def _agent_steps(self, world: MultiAgentEnvironment, agent: AgentSpec,
                     reservations: _Reservations, total: _Counts, attempt: int):
        counts = _Counts()
        iterator = self._timed_steps(world, agent, reservations, counts, total)
        try:
            while True:
                wall, cpu = perf_counter(), thread_time()
                try:
                    node, expanded = next(iterator)
                except StopIteration as finished:
                    elapsed = (perf_counter() - wall) * 1000
                    cpu_elapsed = (thread_time() - cpu) * 1000
                    counts.elapsed_ms += elapsed
                    counts.cpu_ms += cpu_elapsed
                    total.elapsed_ms += elapsed
                    total.cpu_ms += cpu_elapsed
                    status, path = finished.value
                    return AgentPlan(status, path, counts.metrics())
                elapsed = (perf_counter() - wall) * 1000
                cpu_elapsed = (thread_time() - cpu) * 1000
                counts.elapsed_ms += elapsed
                counts.cpu_ms += cpu_elapsed
                total.elapsed_ms += elapsed
                total.cpu_ms += cpu_elapsed
                yield MultiSearchEvent(agent.agent_id, node.state, expanded,
                                       total.metrics(), node.time, attempt)
        finally:
            iterator.close()

    def _timed_steps(self, world: MultiAgentEnvironment, agent: AgentSpec,
                     reservations: _Reservations, counts: _Counts, total: _Counts):
        root = _TimedNode(agent.start, 0)
        tie = count()
        limit = min(self.depth_limit, self.horizon) if self.algorithm == SearchAlgorithm.DLS else self.horizon
        use_heap = self.algorithm in (SearchAlgorithm.ASTAR, SearchAlgorithm.GREEDY,
                                      SearchAlgorithm.UCS)
        depth_first = self.algorithm in (SearchAlgorithm.DFS, SearchAlgorithm.DLS)

        def priority(node: _TimedNode) -> int:
            distance = abs(node.state.x - agent.goal.x) + abs(node.state.y - agent.goal.y)
            if self.algorithm == SearchAlgorithm.GREEDY:
                return distance
            if self.algorithm == SearchAlgorithm.UCS:
                return node.time
            return node.time + distance

        frontier = [(priority(root), next(tie), root)] if use_heap else deque([root])
        discovered = {(root.state, root.time)}
        counts.generated = counts.frontier_peak = 1
        total.generated += 1
        total.frontier_peak = max(total.frontier_peak, 1)
        cutoff = False
        while frontier:
            if total.examined >= self.max_nodes:
                return SearchStatus.RESOURCE_LIMIT, []
            if use_heap:
                _, _, node = heappop(frontier)
            else:
                node = frontier.pop() if depth_first else frontier.popleft()
            counts.examined += 1
            total.examined += 1
            if node.state == agent.goal and reservations.safe_goal(agent.goal, node.time):
                yield node, False
                return SearchStatus.SUCCESS, node.path()
            # Reaching a destination is terminal: a node that cannot hold its
            # goal safely may not leave it again to dodge a future reservation.
            if node.state == agent.goal:
                yield node, False
                continue
            if node.time >= limit:
                cutoff = True
                yield node, False
                continue
            children = []
            for target in (*[action.move(node.state) for action in Action], node.state):
                arrival = node.time + 1
                if not world.can_enter(target) or reservations.blocked(node.state, target, arrival):
                    continue
                children.append(_TimedNode(target, arrival, node))
            counts.expanded += 1
            total.expanded += 1
            counts.generated += len(children)
            total.generated += len(children)
            choices = reversed(children) if depth_first else children
            for child in choices:
                key = (child.state, child.time)
                # At a fixed tick, all paths to a timed state have identical
                # cost and remaining horizon. Unlike position-only DLS, this
                # duplicate check cannot discard a shallower timed arrival.
                if key not in discovered:
                    discovered.add(key)
                    if use_heap:
                        heappush(frontier, (priority(child), next(tie), child))
                    else:
                        frontier.append(child)
            counts.frontier_peak = max(counts.frontier_peak, len(frontier))
            total.frontier_peak = max(total.frontier_peak, len(frontier))
            yield node, True
        return (SearchStatus.CUTOFF if cutoff else SearchStatus.FAILURE), []


def generate_multiagent_environment(seed: int, width: int = 12, height: int = 10,
                                    agent_count: int = 3,
                                    obstacle_density: float = 0.25) -> MultiAgentEnvironment:
    """Repeatable random endpoints/obstacles with individual reachability.

    All 2*N endpoint cells are distinct. A random monotone corridor for each
    pair is protected before Bernoulli obstacle placement. This guarantees each
    pair is connected in the static map; it does not guarantee a joint schedule.
    Density is the probability for each unreserved cell, not an exact fraction.
    A local RNG leaves the process-wide random stream unchanged.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Seed must be an integer.')
    _integer(width, 'Width', 2)
    _integer(height, 'Height', 2)
    _integer(agent_count, 'Agent count', 1)
    if (isinstance(obstacle_density, bool) or not isinstance(obstacle_density, (int, float)) or
            not isfinite(obstacle_density) or not 0 <= obstacle_density <= 1):
        raise ValueError('Obstacle density must be between 0 and 1.')
    if agent_count * 2 > width * height:
        raise ValueError('The grid needs two distinct endpoint cells per agent.')
    rng = random.Random(seed)
    cells = [Location(x, y) for y in range(height) for x in range(width)]
    endpoints = rng.sample(cells, agent_count * 2)
    agents = tuple(AgentSpec(f'A{i + 1}', endpoints[2 * i], endpoints[2 * i + 1])
                   for i in range(agent_count))
    reserved = set(endpoints)
    for a in agents:
        cursor = a.start
        while cursor != a.goal:
            directions = []
            if cursor.x != a.goal.x:
                directions.append((1 if a.goal.x > cursor.x else -1, 0))
            if cursor.y != a.goal.y:
                directions.append((0, 1 if a.goal.y > cursor.y else -1))
            dx, dy = rng.choice(directions)
            cursor = Location(cursor.x + dx, cursor.y + dy)
            reserved.add(cursor)
    obstacles = {cell for cell in cells
                 if cell not in reserved and rng.random() < obstacle_density}
    return MultiAgentEnvironment(width, height, agents, obstacles)


def new_multiagent_environment() -> MultiAgentEnvironment:
    """Three intersecting routes; the central crossing requires coordination."""
    width = height = 9
    free = {Location(x, 4) for x in range(width)} | {
        Location(4, y) for y in range(height)} | {
        Location(x, 8) for x in range(width)}
    agents = (AgentSpec('A1', Location(0, 4), Location(8, 4)),
              AgentSpec('A2', Location(4, 0), Location(4, 8)),
              AgentSpec('A3', Location(8, 8), Location(0, 8)))
    obstacles = {Location(x, y) for y in range(height) for x in range(width)} - free
    return MultiAgentEnvironment(width, height, agents, obstacles)
