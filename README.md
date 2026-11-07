# Travel Planner - Sprint 4

A standalone CIS 424 Artificial Intelligence project integrating the earlier grid engine and search infrastructure with informed search, coordinated multiagent travel, configurable seeded maps, and an updated desktop front end.

## Run

Requires **Python 3.10+**. The program uses only the standard library; graphical mode also needs Tkinter and a desktop display. On Windows, `py` can replace `python`.

Extract the submission ZIP and open a terminal inside `travel_agent_engine`, then run:

```bash
python engine.py
```

Choose a strategy and **Find Route**, then use **Step** or **Auto Run**. Enable **Multiagent mode** to plan and move the colored team. **Pause / Cancel** pauses travel or discards an unfinished search. Map and strategy changes invalidate old plans.

For GitHub visitors, `travel_agent_engine_sprint4.zip` is the ready-to-extract submission package. The earlier Sprint 3 ZIP and repository history preserve that milestone.

## Sprint 4 requirement mapping

| Requirement | Source and evidence |
|---|---|
| GreedyBestSearch | `algorithms.py`: priority h(n). The informed demo shows a 10-move greedy route where A* finds 8. |
| AStarSearch | `algorithms.py`: priority g(n)+h(n), stale-entry checks and lower-cost reopening. Tested against an independent shortest-path oracle. |
| Updated front end | `gui.py`: single/multiagent modes, matching agent/goal colors, animated search, route playback, aggregate/per-agent metrics, and configurable map generation. |
| Multiagent implementation | `multiagent.py`: timed paths, waiting, vertex/edge reservations, permanent goal occupancy, atomic joint execution, and reset. |
| Seed generation | Reproducible obstacles and random multiagent starts/goals, with seed, width, height, density, and agent-count controls. The seed demo prints a scenario fingerprint. |
| Previous sprints | Environment, Actions, Percepts, Agent, Location, Node, TransitionModel, DFS, BFS, DLS, and UCS remain integrated. |
| Deliverables | Source, actual console outputs, tests, desktop screenshots, documentation, and Help Received statement are in this folder. |

## Console demonstrations

These commands run without a desktop display:

```bash
python engine.py --informed-demo
python engine.py --multiagent-demo
python engine.py --multiagent-demo --algorithm greedy
python engine.py --seed-demo --seed 424 --agents 3
python engine.py --multiagent-demo --seed 424 --agents 3 --horizon 100
python engine.py --demo --algorithm astar --seed 42 --width 12 --height 10 --density 0.25
python engine.py --algorithms-demo
python engine.py --infrastructure-demo
python -m unittest discover -v
```

The multiagent demo replays a complete schedule through the same collision-checking environment used by the GUI. A bounded planning attempt may fail for some user maps or limits; this is reported without claiming that every joint solution is impossible.

## Informed search

| Strategy | Priority | Guarantee on the standard unit-cost grid |
|---|---|---|
| A* | g(n) + h(n) | A successful route has minimum movement cost. |
| Greedy best-first | h(n) | A successful route is legal but may be longer. |
| BFS | Shallowest first | Shortest route because moves cost 1. |
| UCS | Minimum g(n) | Least-cost route; often matches BFS here. |
| DFS | Last-in, first-out | No shortest-route guarantee. |
| DLS | Depth-first within an inclusive limit | Can stop at a cutoff. |

Here g(n) is accumulated movement cost. The heuristic is Manhattan distance, `abs(x-goal_x) + abs(y-goal_y)`. It is admissible and consistent on a four-neighbor grid with unit moves. A* also supports reopening for an admissible inconsistent heuristic supplied by a subclass.

The informed-search sample uses a 5-by-5 grid with start (0,0), goal (4,4), and obstacles at (1,2), (3,1), (3,4), (4,0), and (4,2). A*, BFS, and UCS return cost 8; greedy returns cost 10 while expanding fewer nodes on this example. This demonstrates a route-quality tradeoff, not a general performance ranking.

Single-agent searches still use Sprint 2 Node objects and TransitionModel.expand. Parent links reconstruct the selected route; planning does not move the live agent.

## Coordinated multiagent behavior

Each agent has its own identifier, unique start, and unique goal. All agents share one map and one synchronized clock. Each tick permits a cardinal one-cell movement or a wait. An agent that reaches its destination stays there.

`execute_joint` requires proposals for every agent. It rejects the entire tick if any proposal is missing, leaves the grid, enters an obstacle, jumps too far, leaves a completed goal, shares a cell, or exchanges an edge head-on with another agent. Rejected proposals leave every position and counter unchanged. Moving into a cell vacated by another traveler is allowed when the pair does not swap.

The planner searches **position plus time**, using the selected strategy. Earlier routes reserve cells, directed edges, and final destinations; later searches avoid those reservations. Consecutive identical positions encode WAIT without changing the original four-action enum. Initially completed agents are reserved before any moving agents are planned.

Priority orders are retried deterministically using rotations and reversed rotations. The horizon bounds arrival time, DLS also applies its depth limit, and the node budget bounds work across all attempts. A returned schedule avoids vertex and head-on edge conflicts. **Prioritized planning is incomplete and not globally optimal:** another priority order or a joint-state search could find a solution that this bounded planner misses.

Plan cost is the sum of arrival times, including pre-arrival waits. Makespan is the last agent's arrival time. Completed agents' later waiting does not add to plan cost. The environment separately records joint ticks, actual movement counts, and pre-arrival wait counts.

The fixed 9-by-9 demonstration has three intersecting routes. With A* or greedy, all three reach their goals in 9 joint ticks, with summed arrival cost 25. A2 waits one tick to avoid the crossing conflict. These are illustrative results from that fixed world.

## Seed generation

Reproduction requires the **seed and all parameters**: width, height, density, and agent count. A dedicated `random.Random(seed)` instance leaves the global random stream untouched. Fresh seed selection uses `secrets`.

Multiagent starts and goals are sampled from 2*N distinct cells. Random monotone corridors are protected before obstacles are placed, so each start/goal pair is individually connected. This does not guarantee a joint schedule under every priority/horizon setting. Density is a placement probability for eligible cells, not an exact final percentage.

The earlier single-agent generator retains its fixed start/goal convention and protected corridor. Loading a seed restores its generated map, removing manual edits. Edited obstacles are not encoded in the seed.

## Desktop controls

The **Search** panel selects the algorithm, DLS depth limit, route actions, and statistics scope. The **Map setup** panel configures width, height, obstacle density, agent count, and planning horizon; the seed field and Load Seed button sit below the grid. Single-agent mode initially selects BFS for compatibility; enabling multiagent mode selects A*.

Colored travelers A1, A2, and so on have matching destinations. The team can plan from its current positions and advance one synchronized tick at a time. Aggregate statistics show total search work; selecting an agent shows its returned planning record. The movement display tracks actual progress.

Click a cell to toggle an obstacle while idle. Starts, destinations, and occupied cells are protected. **Reset Agent** resets the current world and counters while retaining obstacle edits. **Original Map** loads the fixed demonstration appropriate to the mode. **Inspect Successors** highlights applicable moves without changing positions.

Search generators advance in scheduled batches so cancellation and the UI remain responsive. Rendering does not start searches. Changing modes, settings, or maps cancels unfinished work and clears stale routes.

## Status and metric definitions

- **Success:** a goal route or complete team schedule was found.
- **No route:** a single-agent frontier was exhausted. Cooperative bounded failure is explained as an unsuccessful planning attempt rather than a global impossibility proof.
- **Depth cutoff:** a configured depth/time boundary was encountered; this does not establish that a deeper solution exists.
- **Search limit:** the node budget was reached while work remained. The default budget is 100,000 examined nodes.

| Metric | Meaning |
|---|---|
| Examined | Selected node occurrences, excluding stale priority entries. |
| Expanded | Actual successor expansions; terminal goals and boundary nodes are excluded. |
| Generated | Includes roots and children before duplicate pruning. |
| Peak frontier | Largest stored frontier; heap entries can include stale candidates. |
| Time / CPU | Active search work, excluding GUI animation delays. Cooperative timing covers the per-agent timed searches, not all reservation/retry bookkeeping. |
| Plan cost | Single-agent cost or sum of coordinated arrival costs. |
| Makespan | Time when the final agent arrives. |

Timed revisits are different search states even when they share a location. Counters can exceed the number of shaded cells. Cooperative aggregate work includes failed priority attempts; per-agent records describe the returned attempt. Timings vary by machine and are not a benchmark study.

## Files and scope

The final validation run passed **100 tests with no skips**, including the 61 earlier tests, 7 informed-search tests, 23 cooperative/seed tests, and 9 new GUI tests. GUI checks used real Tkinter widgets under a virtual display, and screenshots were inspected at the default and minimum window sizes. On a machine with no desktop display, the 19 GUI tests skip automatically; the supplied report records the full run with all of them executed.

The checks include independent shortest-path distances, heuristic consistency, state reopening, crossing waits, head-on swaps, permanent goal occupancy, successful priority retries, atomic rejection, reproducible maps, current-position replanning, cancellation, settings validation, and joint playback to every goal. Tests verify correctness for these cases; they do not establish completeness of prioritized planning.

| File | Responsibility |
|---|---|
| `engine.py` | Original world, agent, game loop, seeded single-agent generator, CLI demos. |
| `infrastructure.py` | Location, Action, Percept, StepResult, Node, TransitionModel. |
| `algorithms.py` | Six strategies, results, incremental events, metrics. |
| `multiagent.py` | Team world, collision checks, timed cooperative planner, random team scenarios. |
| `gui.py` | Integrated Tkinter interface. |
| `test_*.py` | Earlier regressions plus informed, multiagent, seed, and GUI checks. |
| `sample_outputs/` | Captured demonstrations, test report, and screenshots. |
| `HELP_RECEIVED.txt` | AI-assistance disclosure; add other help received by your team. |
| `SPRINT2_NOTES.md` | Earlier design record with a migration note. |

Existing imports from `engine` still expose the earlier public types. This is a fully observed grid with terminal destinations; no external map service or LLM runs inside the agents. Sprint 5's experiment suite and research-paper draft are outside this deliverable.
