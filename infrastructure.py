"""Sprint 1 value objects and Sprint 2 immutable search infrastructure."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine import Environment


@dataclass(frozen=True, order=True)
class Location:
    """Immutable grid coordinate: x increases right, y increases down."""
    x: int
    y: int


class Action(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def move(self, origin: Location) -> Location:
        dx, dy = self.value
        return Location(origin.x + dx, origin.y + dy)


@dataclass(frozen=True)
class Percept:
    """Snapshot of what the traveler knows at the current turn."""
    current: Location
    destination: Location
    traversable_neighbors: tuple[tuple[Action, Location], ...]
    obstacles: FrozenSet[Location]
    width: int
    height: int
    reached_goal: bool


@dataclass(frozen=True)
class StepResult:
    action: Optional[Action]
    moved: bool
    message: str
    reached_goal: bool


@dataclass(frozen=True, eq=False)
class Node:
    """A search-tree record, distinct from its immutable Location state.

    Two nodes can represent the same state via different parents. Graph-search
    duplicate detection must therefore use node.state, not node identity.
    """
    state: Location
    parent: Node | None = field(default=None, repr=False)
    action: Action | None = None
    path_cost: int = 0
    depth: int = field(init=False)

    def __post_init__(self) -> None:
        if self.parent is None:
            if self.action is not None or self.path_cost != 0:
                raise ValueError('A root node has no action and zero path cost.')
            object.__setattr__(self, 'depth', 0)
        else:
            if not isinstance(self.action, Action):
                raise ValueError('A child node must record its incoming action.')
            if self.path_cost < self.parent.path_cost:
                raise ValueError('Path cost cannot decrease along a path.')
            object.__setattr__(self, 'depth', self.parent.depth + 1)

    def child(self, action: Action, state: Location, step_cost: int = 1) -> Node:
        """Link a successor to this node; legality is checked by the model."""
        if step_cost < 0:
            raise ValueError('Step cost must be nonnegative.')
        return Node(state, self, action, self.path_cost + step_cost)

    def path(self) -> list[Node]:
        """Return nodes from the root through this node, inclusive."""
        nodes = []
        cursor: Node | None = self
        while cursor is not None:
            nodes.append(cursor)
            cursor = cursor.parent
        return list(reversed(nodes))

    def state_path(self) -> list[Location]:
        return [node.state for node in self.path()]

    def solution(self) -> list[Action]:
        """Return the actions from the root to this node (no root action)."""
        return [node.action for node in self.path() if node.action is not None]


@dataclass(frozen=True)
class TransitionModel:
    """Pure, deterministic successor function for a fixed map snapshot.

    A state is a Location; map and destination are problem parameters. Every
    legal move costs 1. Goal states are terminal, matching Environment.execute.
    Predicting or expanding never moves the live agent or edits the map.
    """
    width: int
    height: int
    goal: Location
    obstacles: FrozenSet[Location] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'obstacles', frozenset(self.obstacles))
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Grid dimensions must be positive.')
        if any(not self.in_bounds(p) for p in (*self.obstacles, self.goal)):
            raise ValueError('Map locations must be inside the grid.')
        if self.goal in self.obstacles:
            raise ValueError('Destination cannot contain an obstacle.')

    @classmethod
    def from_environment(cls, environment: Environment) -> TransitionModel:
        return cls(environment.width, environment.height, environment.goal,
                   frozenset(environment.obstacles))

    @classmethod
    def from_percept(cls, percept: Percept) -> TransitionModel:
        return cls(percept.width, percept.height, percept.destination,
                   percept.obstacles)

    def in_bounds(self, state: Location) -> bool:
        return 0 <= state.x < self.width and 0 <= state.y < self.height

    def can_enter(self, state: Location) -> bool:
        return self.in_bounds(state) and state not in self.obstacles

    def _require_state(self, state: Location) -> None:
        if not self.can_enter(state):
            raise ValueError(f'State is outside the grid or blocked: {state}')

    def is_goal(self, state: Location) -> bool:
        self._require_state(state)
        return state == self.goal

    def actions(self, state: Location) -> tuple[Action, ...]:
        """Applicable actions in stable UP, DOWN, LEFT, RIGHT order."""
        self._require_state(state)
        if state == self.goal:
            return ()
        return tuple(action for action in Action
                     if self.can_enter(action.move(state)))

    def result(self, state: Location, action: Action) -> Location:
        """Predict one successor; reject an inapplicable action with ValueError."""
        if action not in self.actions(state):
            raise ValueError(f'Action {action!r} is not applicable at {state}.')
        return action.move(state)

    def step_cost(self, state: Location, action: Action,
                  next_state: Location) -> int:
        if self.result(state, action) != next_state:
            raise ValueError('Next state does not match the action result.')
        return 1

    def successors(self, state: Location) -> tuple[tuple[Action, Location, int], ...]:
        """Return only valid (action, next state, step cost) triples."""
        return tuple((action, action.move(state), 1) for action in self.actions(state))

    def expand(self, node: Node) -> tuple[Node, ...]:
        """Build successor nodes, retaining parent, action, depth and cost."""
        return tuple(node.child(action, state, cost)
                     for action, state, cost in self.successors(node.state))

