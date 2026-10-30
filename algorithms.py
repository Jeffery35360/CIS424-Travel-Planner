"""Sprint 3 search strategies over the shared Sprint 2 transition model.

Each strategy offers search() for a complete result and steps() for incremental
GUI animation. A generator's return value is the same SearchResult. Neither
entry point moves the agent. All priorities and duplicate checks use states;
Node objects retain the path that produced each frontier entry.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from time import perf_counter, thread_time
from typing import Generator

from infrastructure import Action, Location, Node, TransitionModel


class SearchAlgorithm(str, Enum):
    DFS = 'Depth First Search'
    BFS = 'Breadth First Search'
    DLS = 'Depth Limited Search'
    UCS = 'Uniform Cost Search'
    ASTAR = 'A* Search'
    GREEDY = 'Greedy Best First Search'


class SearchStatus(str, Enum):
    SUCCESS = 'Success'
    FAILURE = 'No route'
    CUTOFF = 'Depth cutoff'
    RESOURCE_LIMIT = 'Search limit'


@dataclass(frozen=True)
class SearchMetrics:
    examined: int = 0
    expanded: int = 0
    generated: int = 1
    frontier_peak: int = 1
    elapsed_ms: float = 0.0
    cpu_ms: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    goal_node: Node | None
    metrics: SearchMetrics
    explored: frozenset[Location]
    expanded_states: frozenset[Location]

    @property
    def path(self) -> list[Location]:
        """Future positions, excluding the initial location."""
        return self.goal_node.state_path()[1:] if self.goal_node else []

    @property
    def actions(self) -> list[Action]:
        return self.goal_node.solution() if self.goal_node else []

    @property
    def path_cost(self) -> int | None:
        return self.goal_node.path_cost if self.goal_node else None

    @property
    def found(self) -> bool:
        return self.status == SearchStatus.SUCCESS


@dataclass(frozen=True)
class SearchEvent:
    node: Node
    expanded: bool
    metrics: SearchMetrics


class _Run:
    """Mutable counters private to one invocation (strategies are reusable)."""
    def __init__(self, max_nodes: int):
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
            raise ValueError('max_nodes must be a positive integer.')
        self.max_nodes = max_nodes
        self.examined = self.expanded = 0
        self.generated = self.frontier_peak = 1
        self.elapsed_ms = self.cpu_ms = 0.0
        self.explored: set[Location] = set()
        self.expanded_states: set[Location] = set()

    def metrics(self) -> SearchMetrics:
        return SearchMetrics(self.examined, self.expanded, self.generated,
                             self.frontier_peak, self.elapsed_ms, self.cpu_ms)

    def examine(self, node: Node) -> None:
        self.examined += 1
        self.explored.add(node.state)

    def expand(self, model: TransitionModel, node: Node) -> tuple[Node, ...]:
        children = model.expand(node)
        self.expanded += 1
        self.generated += len(children)
        self.expanded_states.add(node.state)
        return children

    def finish(self, status: SearchStatus, goal: Node | None = None) -> SearchResult:
        return SearchResult(status, goal, self.metrics(), frozenset(self.explored),
                            frozenset(self.expanded_states))


class SearchStrategy:
    def search(self, model: TransitionModel, start: Location,
               max_nodes: int = 100_000) -> SearchResult:
        iterator = self.steps(model, start, max_nodes)
        while True:
            try:
                next(iterator)
            except StopIteration as finished:
                return finished.value

    def steps(self, model: TransitionModel, start: Location,
              max_nodes: int = 100_000) -> Generator[SearchEvent, None, SearchResult]:
        """Yield after each examined node, allowing the UI to pause/cancel.

        Timing includes only active search work, not time suspended at yield
        while the GUI draws or waits. CPU timing is for the calling thread.
        """
        run = _Run(max_nodes)
        model.is_goal(start)  # validates the starting state, including 1-cell maps
        iterator = self._steps(model, Node(start), run)
        try:
            while True:
                wall, cpu = perf_counter(), thread_time()
                try:
                    node, expanded = next(iterator)
                except StopIteration as finished:
                    run.elapsed_ms += (perf_counter() - wall) * 1000
                    run.cpu_ms += (thread_time() - cpu) * 1000
                    return replace(finished.value, metrics=run.metrics())
                run.elapsed_ms += (perf_counter() - wall) * 1000
                run.cpu_ms += (thread_time() - cpu) * 1000
                yield SearchEvent(node, expanded, run.metrics())
        finally:
            iterator.close()

    def _steps(self, model: TransitionModel, root: Node, run: _Run):
        raise NotImplementedError


class BreadthFirstSearch(SearchStrategy):
    """FIFO graph search; discover on enqueue to avoid duplicate entries."""
    def _steps(self, model: TransitionModel, root: Node, run: _Run):
        return _deque_steps(model, root, run, depth_first=False)


class DepthFirstSearch(SearchStrategy):
    """LIFO graph search with deterministic Action-order traversal."""
    def _steps(self, model: TransitionModel, root: Node, run: _Run):
        return _deque_steps(model, root, run, depth_first=True)


def _deque_steps(model: TransitionModel, root: Node, run: _Run, depth_first: bool):
    frontier = deque([root])
    discovered = {root.state}
    while frontier:
        if run.examined >= run.max_nodes:
            return run.finish(SearchStatus.RESOURCE_LIMIT)
        node = frontier.pop() if depth_first else frontier.popleft()
        run.examine(node)
        if model.is_goal(node.state):
            yield node, False
            return run.finish(SearchStatus.SUCCESS, node)
        children = run.expand(model, node)
        if depth_first:
            children = tuple(reversed(children))
        for child in children:
            if child.state not in discovered:
                discovered.add(child.state)
                frontier.append(child)
        run.frontier_peak = max(run.frontier_peak, len(frontier))
        yield node, True
    return run.finish(SearchStatus.FAILURE)


class DepthLimitedSearch(SearchStrategy):
    """Depth-first tree search with ancestor checks and an inclusive limit.

    CUTOFF means a non-goal node reached the depth boundary; it does not claim
    that a deeper solution exists. FAILURE means the frontier was exhausted
    without encountering that boundary. A goal exactly at the limit succeeds.
    """
    def __init__(self, depth_limit: int = 20):
        if isinstance(depth_limit, bool) or not isinstance(depth_limit, int) or depth_limit < 0:
            raise ValueError('Depth limit must be a nonnegative integer.')
        self.depth_limit = depth_limit

    def _steps(self, model: TransitionModel, root: Node, run: _Run):
        frontier = [root]
        cutoff = False
        while frontier:
            if run.examined >= run.max_nodes:
                return run.finish(SearchStatus.RESOURCE_LIMIT)
            node = frontier.pop()
            run.examine(node)
            if model.is_goal(node.state):
                yield node, False
                return run.finish(SearchStatus.SUCCESS, node)
            if node.depth >= self.depth_limit:
                cutoff = True
                yield node, False
                continue
            ancestors = set(node.state_path())
            children = run.expand(model, node)
            for child in reversed(children):
                if child.state not in ancestors:
                    frontier.append(child)
            run.frontier_peak = max(run.frontier_peak, len(frontier))
            yield node, True
        return run.finish(SearchStatus.CUTOFF if cutoff else SearchStatus.FAILURE)


class UniformCostSearch(SearchStrategy):
    """Minimum-g graph search (Dijkstra), with goal testing on removal."""
    def priority(self, node: Node, goal: Location) -> int:
        return node.path_cost

    def _steps(self, model: TransitionModel, root: Node, run: _Run):
        tie = count()
        frontier = [(self.priority(root, model.goal), next(tie), root)]
        best_cost = {root.state: 0}
        closed: set[Location] = set()
        while frontier:
            _, _, node = heappop(frontier)
            if node.state in closed or node.path_cost != best_cost.get(node.state):
                continue  # outdated, more expensive frontier entry
            if run.examined >= run.max_nodes:
                return run.finish(SearchStatus.RESOURCE_LIMIT)
            closed.add(node.state)
            run.examine(node)
            if model.is_goal(node.state):
                yield node, False
                return run.finish(SearchStatus.SUCCESS, node)
            for child in run.expand(model, node):
                if child.state not in closed and child.path_cost < best_cost.get(child.state, float('inf')):
                    best_cost[child.state] = child.path_cost
                    heappush(frontier, (self.priority(child, model.goal), next(tie), child))
            run.frontier_peak = max(run.frontier_peak, len(frontier))
            yield node, True
        return run.finish(SearchStatus.FAILURE)


# Match the assignment's spelling while retaining the standard algorithm name.
UniformedCostSearch = UniformCostSearch


class AStarSearch(UniformCostSearch):
    """Preserved baseline feature: g + Manhattan distance on the unit grid."""
    def priority(self, node: Node, goal: Location) -> int:
        return node.path_cost + abs(node.state.x - goal.x) + abs(node.state.y - goal.y)


class GreedyBestSearch(UniformCostSearch):
    """Preserved baseline feature: Manhattan distance only."""
    def priority(self, node: Node, goal: Location) -> int:
        return abs(node.state.x - goal.x) + abs(node.state.y - goal.y)


def make_search(algorithm: SearchAlgorithm | str, depth_limit: int = 20) -> SearchStrategy:
    chosen = SearchAlgorithm(algorithm)
    if chosen == SearchAlgorithm.DLS:
        return DepthLimitedSearch(depth_limit)
    return {SearchAlgorithm.DFS: DepthFirstSearch,
            SearchAlgorithm.BFS: BreadthFirstSearch,
            SearchAlgorithm.UCS: UniformCostSearch,
            SearchAlgorithm.ASTAR: AStarSearch,
            SearchAlgorithm.GREEDY: GreedyBestSearch}[chosen]()
