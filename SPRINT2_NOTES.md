# Sprint 2: Search Infrastructure

> Retained as the Sprint 2 design record. In the Sprint 3 package, value objects,
> Node, and TransitionModel live in `infrastructure.py`, strategy code lives in
> `algorithms.py`, and the desktop front end lives in `gui.py`. `engine.py`
> re-exports the earlier public types. The current README describes the new
> controls, defaults, metrics, test files, and validation results; earlier
> statements below about file layout, test counts, and GUI validation describe
> the Sprint 2 milestone.

## What changed

The source is still in `engine.py`, so existing imports and launch commands work. `test_engine.py` is unchanged from the supplied baseline. New tests are in `test_infrastructure.py`.

Sprint 2 adds explicit `Node` and `TransitionModel` classes, uses them in all existing planners, and makes the environment use the same transition rules for perception and execution. The GUI adds an Inspect Successors button. A separate console mode demonstrates the infrastructure without needing a graphical display.

## State versus node

A **state** describes where the agent is in a fixed search problem: `Location(x, y)`. The model holds the map, obstacles, and goal. Since moves depend only on those parameters and the current position, neither the route history nor the successful-step counter belongs in this search state.

A **node** describes one search path to a state:

| Field | Meaning |
|---|---|
| `state` | The `Location` represented by this node. |
| `parent` | The node from which this node was generated; `None` at the root. |
| `action` | The incoming action taken from the parent; `None` at the root. |
| `path_cost` | Accumulated movement cost from the root, usually written as g(n). |
| `depth` | Number of parent links from the root. |

The root has depth 0 and cost 0. Every legal move in this grid costs 1, so a generated child has its parent's depth plus 1 and cost plus 1. Cost and depth are separate fields even though their values match in the current grid.

Nodes use identity equality. For example, the initial node at (0, 0) and a later node returning to (0, 0) have equal states but are distinct nodes with different depths and parent chains. Graph-search duplicate detection uses `node.state`. DLS checks ancestors on the current path, allowing another path to revisit the same state with a different remaining depth allowance.

`node.path()` returns the root-to-node list of nodes. `node.state_path()` returns the corresponding locations, including the root. `node.solution()` returns the incoming actions in execution order, omitting the root's nonexistent action. Reconstruction walks parent links iteratively.

`Node.child(...)` manages search metadata; use `TransitionModel.expand(...)` to generate children with legal grid transitions. Constructing a node manually does not itself validate whether a move is legal on a particular map.

## Transition model API

The model is an immutable snapshot. Prediction and expansion do not change `Environment.agent_location`, `Environment.steps`, or the obstacle set. An edited environment produces a fresh model on the next perception/planning/execution pass. An existing snapshot retains its original obstacles.

| Method | Result |
|---|---|
| `TransitionModel.from_environment(env)` | Model snapshot of the live environment. |
| `TransitionModel.from_percept(percept)` | Equivalent model using only the agent's perceived data. |
| `actions(state)` | Applicable actions in UP, DOWN, LEFT, RIGHT order. |
| `result(state, action)` | One predicted next `Location`; raises `ValueError` for an inapplicable action. |
| `successors(state)` | Tuple of legal `(action, next_state, step_cost)` triples. |
| `step_cost(state, action, next_state)` | Cost 1 after verifying the supplied transition. |
| `expand(node)` | Tuple of child `Node` objects with parent links and updated metadata. |
| `is_goal(state)` | Whether a valid state is the destination. |

Coordinates outside the grid and obstacle cells are rejected as input states. Actions crossing a boundary or entering an obstacle are excluded from successors. A trapped non-goal state has no successors. The goal is terminal and also has no successors, matching the original environment's rule that movement stops after arrival. Live invalid moves return an unsuccessful `StepResult`, preserving the Sprint 1 interface.

The model is deterministic: a given state and applicable action have exactly one result. The successor function lists the possible next states across all applicable actions, rather than several stochastic outcomes for one action.

### Why there is no facing direction

Page 1 illustrates applicability with a directional agent and turning actions. The existing travel game uses absolute UP/DOWN/LEFT/RIGHT movement. This update keeps that action model: direction is not stored because it does not affect these moves. At (0, 0), UP and LEFT are invalid; DOWN and RIGHT are valid unless blocked. Supporting relative forward/turn actions later would require adding orientation to the search state and changing the action model.

## Small example

Run this from the extracted project folder:

```python
from engine import Action, Environment, Location, Node, TransitionModel

env = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
model = TransitionModel.from_environment(env)
root = Node(env.agent_location)
children = model.expand(root)

assert model.actions(root.state) == (Action.DOWN,)
child = children[0]
assert child.state == Location(0, 1)
assert child.parent is root
assert child.depth == 1 and child.path_cost == 1
assert child.solution() == [Action.DOWN]
assert env.agent_location == Location(0, 0) and env.steps == 0
```

UP and LEFT leave the grid. RIGHT enters the obstacle. Only DOWN generates a successor. Inspecting the hypothetical child leaves the live agent at its original position.

## How it connects to Sprint 1

1. `Environment.perceive()` obtains legal neighboring moves through the model.
2. `TravelAgent.plan()` builds a model from the percept when a new search is needed.
3. A search starts with `Node(percept.current)`.
4. DFS, BFS, and DLS use nodes in their stacks/queues; A*, UCS, and greedy search store nodes in their heaps. Each uses `model.expand(node)` for successor generation. DLS applies its depth cutoff before expansion.
5. On success, the agent stores `last_goal_node` and reconstructs positions with its parent chain. `plan()` still returns future locations, excluding the current position, so existing GUI and movement callers continue to work.
6. `Environment.execute(action)` validates the selected move with the same model before changing the live position and step count.

Route caching remains in place. `last_goal_node` represents the most recent successful planning pass, so its path includes positions already visited while following a cached route. `last_path` gives the remaining route. A failed new search clears `last_goal_node`; switching algorithms or explicitly invalidating the agent clears it too.

## Demonstration and submission files

From a terminal inside this folder:

```text
python engine.py --infrastructure-demo
python engine.py --demo
python -m unittest discover -v
python engine.py
```

The first command demonstrates a blocked corner, hypothetical interior moves, a trapped state, expansion metadata, a reconstructed BFS action sequence, execution of that sequence, and terminal-goal expansion. The second command runs the original 12-by-10 travel demo. The final command opens the GUI; press Inspect Successors to see the same infrastructure for the live agent's current position without moving it.

`sample_outputs/infrastructure_demo.txt`, `travel_demo.txt`, and `test_results.txt` are captured from actual runs. The automated checks cover 31 tests: the original 16 plus 15 infrastructure and integration tests. These include all four actions at every passable cell of a small map, comparing predictions and execution against independently calculated expected movement. All six existing planners are checked for replayable node chains.

The graphical interface requires Tkinter and a desktop display. Headless demos and all automated tests were executed in this environment; the desktop GUI was not run here because no display server was available.

The folder includes source, documentation, sample outputs, and `HELP_RECEIVED.txt`. Review the help statement and add any other assistance your team received before submission.
