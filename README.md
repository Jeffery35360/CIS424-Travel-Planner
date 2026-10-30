# Travel Planner - Sprint 3 Search Lab

A standalone Python desktop application integrating Sprint 1's grid world, Sprint 2's nodes and transition model, and Sprint 3's four required search algorithms and front end. A traveler navigates around obstacles to a destination on a 12-by-10 grid.

## Start here

Requires **Python 3.10 or newer**. The application uses only the Python standard library. Graphical mode needs Tkinter and a desktop display; Tkinter is available with many Python installations.

1. Extract the ZIP file.
2. Open a terminal inside the extracted `travel_agent_engine` folder.
3. Run:

```bash
python engine.py
```

Choose a strategy and click **Find Route**. Shaded cells show the search's progress, and teal cells show the resulting route. **Step** executes one movement; **Auto Run** follows the planned route. **Pause / Cancel** pauses movement or cancels an unfinished search.

On Windows, `py` may be used instead of `python`. If Tkinter is missing, use your Python installer's Tcl/Tk option or your operating system's Tkinter package. Console modes work without Tkinter.

## What Sprint 3 implements

| Required item | Implementation |
|---|---|
| DepthFirstSearch | `algorithms.py`: LIFO graph search using a stack. |
| BreadthFirstSearch | `algorithms.py`: FIFO graph search using a queue. |
| DepthLimitedSearch | `algorithms.py`: depth-first search with an editable inclusive depth limit and path-local cycle checks. |
| UniformedCostSearch | `algorithms.py`: `UniformCostSearch`, also exported as `UniformedCostSearch` to match the assignment's spelling; priority is accumulated path cost. |
| Front End Design | `gui.py`: responsive search animation, route execution, strategy descriptions, depth entry, statistics, map controls, and successor inspection. |
| Sprint 1 integration | `engine.py`: Environment, TravelAgent, GameEngine, perception, legal movement, reset, and obstacle editing. |
| Sprint 2 integration | `infrastructure.py`: Location, Action, Percept, StepResult, Node, and TransitionModel. All searches expand these nodes through the shared model. |
| Sample outputs | `sample_outputs/`: captured console/test output and desktop screenshots. |
| Help Received | `HELP_RECEIVED.txt`: disclosure of AI assistance. Add any other help your team received before submitting. |

The baseline already had search choices and seeded maps. This update separates strategies into explicit classes, introduces structured results and accurate expansion metrics, distinguishes cutoff from failure, and replaces the earlier GUI. A* and greedy search remain available as existing features. This is still a single-agent project; multiagent work is outside this sprint.

## Front-end controls

| Control | Behavior |
|---|---|
| Search strategy | Select DFS, BFS, DLS, UCS, A*, or greedy search. Changing it clears the previous plan. BFS is the default. |
| DLS depth limit | Enter a nonnegative integer. A depth is a number of moves, not an obstacle count. Changing it invalidates the previous plan. |
| Find Route | Search from the traveler's current position without moving the traveler. |
| Step | Execute one move; find a route first if needed. |
| Auto Run | Find a route if needed and follow it one move at a time. |
| Pause / Cancel | Preserve a paused movement route; cancel and discard an unfinished search while keeping its partial shading. Find Route restarts a cancelled search. |
| Reset Agent | Return to the start and clear results, retaining map edits and search settings. |
| Inspect Successors | Show legal immediate moves and highlight their target cells without moving the traveler. Inspection treats the current position as a root, so child depth and cost are both 1. |
| Click grid cell | Toggle an obstacle and invalidate the plan. Start, goal, and occupied cells are protected. Editing is disabled during search and automatic travel. |
| Original Map | Restore the original obstacle layout and starting position. |
| New Seed / Load Seed | Generate a fresh map or reproduce an integer-seeded map. Loading a seed restores its generated layout, not later manual edits. |

The grid distinguishes examined states, expanded states, the remaining route, start, destination, obstacles, and the live traveler. The orange cell indicates the current search node or an inspected successor. Metrics describe the last planning pass; the movement counter and remaining route update as the traveler moves.

The GUI advances search generators in short scheduled batches. Drawing does not run a search, and cancelling does not leave a worker thread behind.

## Console modes

```bash
python engine.py --algorithms-demo
python engine.py --demo
python engine.py --demo --algorithm dfs
python engine.py --demo --algorithm dls --depth-limit 30
python engine.py --infrastructure-demo
python -m unittest discover -v
```

`--algorithms-demo` compares the four required algorithms on the same original map and replays every successful action sequence through the environment. It also demonstrates DLS cutoff, success exactly at the limit, and a trapped start. `--demo` executes a selected strategy; choices are `dfs`, `bfs`, `dls`, `ucs`, `astar`, and `greedy`. The default depth limit is 20. Console options configure console demos; set GUI options in the window.

### Validation and sample outputs

All **61 tests passed with no skips**, including 31 retained Sprint 1/2 tests, 20 algorithm/game-loop tests, and 10 tests against real Tkinter widgets. The GUI tests cover planning, movement, cancellation, cutoff reporting, seed replay, map edits, reset, and control/status visibility at the minimum window size. Algorithm checks include an independent shortest-path oracle, a weighted UCS fixture, and a DLS shallower-revisit regression.

The GUI was rendered and inspected under a virtual desktop display. On a computer without a display, the 10 GUI tests skip automatically; console and algorithm tests remain available. The supplied `test_results.txt` records the full run with all GUI tests executed.

`sample_outputs/` contains the actual output from the four console demonstration commands plus `gui_ready.png`, `gui_bfs_route.png`, and `gui_depth_cutoff.png`. Screenshots show the real application. They are sample outputs, not assets required to run the program.

## Understanding the results

| Algorithm | Next node chosen | Shortest route on this grid? |
|---|---|---|
| DFS | Most recently added node | Not guaranteed. |
| BFS | Earliest added node | Yes, because every move costs 1. |
| DLS | Deepest first, at most the configured depth | Not guaranteed; can stop at a cutoff. |
| UCS | Lowest accumulated cost g(n) | Yes. With unit costs its results can match BFS. |

The original map produces 44 moves for DFS, 18 for BFS, 20 for DLS at limit 20, and 18 for UCS with this deterministic action order. These are demonstration results on one map, not general performance rankings. Timing varies by machine and run. DLS may examine many paths to the same state and can take much more work than BFS.

All algorithms use UP, DOWN, LEFT, RIGHT successor order. DFS and DLS reverse successor insertion to preserve that order when popping a stack. The goal test occurs when a node is selected, before expansion or a DLS depth cutoff. The model's destination is terminal; entering it completes the task.

### Search statuses

- **Success:** a goal node and executable action sequence were found. Starting at the goal succeeds with cost 0 and no movements.
- **No route:** the search exhausted its frontier without finding the goal or encountering a DLS depth boundary.
- **Depth cutoff:** DLS encountered a non-goal node at the depth limit. This does not prove the goal is unreachable or that a deeper solution exists.
- **Search limit:** the search reached its cap of 100,000 examined nodes with work remaining. This is an incomplete search, not a proof of failure. The API accepts a different positive `max_nodes` value.

### Metric definitions

| Metric | Definition |
|---|---|
| Examined | Node occurrences selected for goal/cutoff testing; excludes stale heap entries. |
| Expanded | Actual calls to `TransitionModel.expand`. Excludes the terminal goal and DLS boundary nodes. |
| Generated | Root node plus every child returned by expansion, including children later pruned as duplicates or cycles. |
| Peak frontier | Largest stored queue/stack/heap size; heap size includes stale entries until removed. Root counts as 1. |
| Time (ms) | Active search wall time; excludes suspension while the GUI draws or waits between animation batches. |
| CPU (ms) | CPU time for active search on the calling thread. |
| Plan moves / cost | Length and accumulated cost of the successful plan from the position where that search began. Unsuccessful searches have no plan cost. |

The shading uses distinct location sets. DLS's examined and expanded counters can exceed the number of shaded cells because different nodes may represent the same location along different paths. `last_explored` retains its earlier meaning of examined locations; use `last_result.metrics.expanded` for the expansion count.

## Files and reuse

```text
travel_agent_engine/
    engine.py                 Environment, agent, game loop, CLI
    infrastructure.py         State/value objects, Node, TransitionModel
    algorithms.py             Independent strategy classes, results, metrics
    gui.py                    Tkinter front end
    test_engine.py            Original Sprint 1 tests
    test_infrastructure.py    Original Sprint 2 tests
    test_algorithms.py        Sprint 3 algorithm tests
    test_gui.py               Real Tkinter controller tests (skip without display)
    README.md
    SPRINT2_NOTES.md           Retained earlier design notes with migration note
    HELP_RECEIVED.txt
    sample_outputs/
```

Existing imports such as `from engine import Node, TransitionModel, TravelAgent` still work. The classes have one shared definition in `infrastructure.py`, avoiding separate state/action types between modules.

```python
from engine import Environment, Location
from algorithms import BreadthFirstSearch

env = Environment(4, 3, Location(0, 0), Location(3, 0), {Location(1, 0)})
result = BreadthFirstSearch().search(env.transition_model(), env.agent_location)
if result.found:
    print(result.path_cost, result.metrics.expanded)
    for action in result.actions:
        env.execute(action)
```

`TravelAgent.plan()` continues to return future locations and caches a route while the map and settings remain unchanged. `last_goal_node` and `last_result` describe the most recent planning pass. They retain the full plan as movement consumes it; `last_path` is the remaining route. Failed searches are cached for the same position/map/settings too. `GameEngine.tick()` now gives different messages for cutoff, exhausted failure, and resource limit.

The map remains fully observable, with four absolute movement actions and one destination. There is no facing direction, separate pickup action, real-world map service, or LLM running inside the agent. A*/greedy are retained with a Manhattan heuristic; A*'s optimality statement applies to the standard unit-cost grid. UCS is also tested against a small weighted graph fixture, but the interactive grid still uses cost 1 per move.
