"""Sprint 3 desktop front end; all searches advance through Tk's event loop.

Rendering never starts a search. Find Route explicitly creates a search
generator. Step/Auto Run do so when no plan is available, then execute the
returned actions through the same Environment used by the console engine.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, font as tkfont
from time import perf_counter

from algorithms import SearchStatus, make_search
from infrastructure import Location, Node
from engine import (SearchAlgorithm, generate_seeded_environment, new_engine,
                    random_seed)


class TravelPlannerApp:
    """A small, testable controller around the grid and search generators."""

    BACKGROUND = '#f1f5f7'
    INK = '#183247'
    MUTED = '#567083'
    CELL_COLORS = {
        'open': '#ffffff', 'examined': '#e1e9f4', 'expanded': '#c8d8ec',
        'path': '#8cd8cf', 'obstacle': '#253f50', 'start': '#f7d78a',
        'goal': '#a3d9b5', 'current': '#ecac62', 'successor': '#f3c48c',
    }
    DESCRIPTIONS = {
        SearchAlgorithm.DFS.value: 'Stack / deepest first. Finds a route, but it may be longer than necessary.',
        SearchAlgorithm.BFS.value: 'Queue / shallowest first. Finds a shortest route on this unit-cost grid.',
        SearchAlgorithm.DLS.value: 'Depth-first up to your limit. A cutoff is different from proving no route exists.',
        SearchAlgorithm.UCS.value: 'Priority queue / lowest path cost first. Finds a least-cost route.',
        SearchAlgorithm.ASTAR.value: 'Priority queue / cost + Manhattan distance. Retained from the earlier project.',
        SearchAlgorithm.GREEDY.value: 'Priority queue / Manhattan distance. Retained from the earlier project.',
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        families = {name.lower() for name in tkfont.families(root)}
        self.font_family = 'Segoe UI' if 'segoe ui' in families else 'Arial'
        self.env = new_engine().environment
        self.active_seed = None
        self.result = None
        self.route_actions = []
        self.route_positions = []
        self.explored = set()
        self.expanded_states = set()
        self.current_search = None
        self.successor_states = set()
        self.search_iterator = None
        self.search_job = None
        self.move_job = None
        self.running = False
        self.pending_action = None
        self.closed = False
        self.grid_geometry = (0.0, 0.0, 1.0)
        self.algorithm_choice = tk.StringVar(value=SearchAlgorithm.BFS.value)
        self.depth_choice = tk.StringVar(value='20')
        self.seed_choice = tk.StringVar(value='')
        self.map_label = tk.StringVar(value='Original map / 12 x 10')
        self.description = tk.StringVar(value=self.DESCRIPTIONS[self.algorithm_choice.get()])
        self.status = tk.StringVar(value='Ready to explore')
        self.message = tk.StringVar(value='Choose a strategy, then Find Route. Click an open cell to add an obstacle.')
        self.movement = tk.StringVar(value='0 moves taken  /  0 planned moves remaining')
        self.metric_vars = {name: tk.StringVar(value='-') for name in
                            ('expanded', 'examined', 'generated', 'frontier', 'elapsed', 'cpu', 'length', 'cost')}
        self._build()
        self.depth_choice.trace_add('write', self._configuration_changed)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after_idle(self.draw)

    def _build(self):
        root = self.root
        root.title('Travel Planner | Sprint 3 Search Lab')
        root.geometry('1120x820')
        root.minsize(1020, 790)
        root.configure(bg=self.BACKGROUND)
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('.', font=(self.font_family, 10))
        style.configure('TFrame', background=self.BACKGROUND)
        style.configure('Card.TFrame', background='white')
        style.configure('TLabel', background='white', foreground=self.INK)
        style.configure('Small.TLabel', font=(self.font_family, 9), foreground=self.MUTED)
        style.configure('Title.TLabel', font=(self.font_family, 13, 'bold'))
        style.configure('TButton', padding=(10, 7))
        style.configure('Primary.TButton', background='#147f78', foreground='white', font=(self.font_family, 10, 'bold'))
        style.map('Primary.TButton', background=[('active', '#106b65'), ('disabled', '#bdd0ce')])
        style.configure('TCombobox', padding=5)
        style.configure('TEntry', padding=5)
        header = tk.Frame(root, bg=self.INK, padx=24, pady=16)
        header.pack(fill='x')
        tk.Label(header, text='TRAVEL PLANNER', bg=self.INK, fg='#8fd9d0',
                 font=(self.font_family, 9, 'bold')).pack(anchor='w')
        row = tk.Frame(header, bg=self.INK)
        row.pack(fill='x', pady=(4, 0))
        tk.Label(row, text='Explore how a search finds its way.', bg=self.INK, fg='white',
                 font=(self.font_family, 20, 'bold')).pack(side='left')
        tk.Label(row, text='SPRINT 3  /  SEARCH LAB', bg='#2b485d', fg='#d8e8f1',
                 padx=12, pady=7, font=(self.font_family, 9, 'bold')).pack(side='right')
        body = ttk.Frame(root, padding=(20, 18, 20, 10))
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=328)
        body.rowconfigure(0, weight=1)
        grid_card = ttk.Frame(body, style='Card.TFrame', padding=16)
        grid_card.grid(row=0, column=0, sticky='nsew', padx=(0, 16))
        grid_card.rowconfigure(2, weight=1)
        grid_card.columnconfigure(0, weight=1)
        ttk.Label(grid_card, text='The environment', style='Title.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(grid_card, textvariable=self.map_label, style='Small.TLabel').grid(row=1, column=0, sticky='w', pady=(3, 8))
        self.canvas = tk.Canvas(grid_card, bg='white', width=580, height=475, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', lambda _event: self.draw())
        self.canvas.bind('<Button-1>', self.click)
        ttk.Label(grid_card, textvariable=self.movement, font=(self.font_family, 10, 'bold')).grid(row=3, column=0, sticky='w', pady=(10, 9))
        legend = tk.Frame(grid_card, bg='white')
        legend.grid(row=4, column=0, sticky='ew')
        for index, (name, color) in enumerate((('Expanded', self.CELL_COLORS['expanded']),
                                               ('Route', self.CELL_COLORS['path']),
                                               ('Obstacle', self.CELL_COLORS['obstacle']),
                                               ('Goal', self.CELL_COLORS['goal']))):
            tk.Label(legend, bg=color, width=2).grid(row=0, column=index * 2, padx=(0, 5))
            tk.Label(legend, text=name, bg='white', fg=self.MUTED, font=(self.font_family, 9)).grid(row=0, column=index * 2 + 1, padx=(0, 13))
        ttk.Label(grid_card, text='A = traveler   S = start   G = destination   Orange = current search node / inspected successor',
                  style='Small.TLabel', wraplength=570).grid(row=5, column=0, sticky='w', pady=(8, 0))

        side = ttk.Frame(body)
        side.grid(row=0, column=1, sticky='nsew')
        side.columnconfigure(0, weight=1)
        controls = ttk.Frame(side, style='Card.TFrame', padding=16)
        controls.grid(row=0, column=0, sticky='ew')
        controls.columnconfigure((0, 1), weight=1)
        ttk.Label(controls, text='Search strategy', style='Title.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 9))
        self.algorithm_menu = ttk.Combobox(controls, textvariable=self.algorithm_choice,
            state='readonly', width=28, height=6, values=[algo.value for algo in SearchAlgorithm])
        self.algorithm_menu.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.algorithm_menu.bind('<<ComboboxSelected>>', self._configuration_changed)
        ttk.Label(controls, textvariable=self.description, style='Small.TLabel', wraplength=295).grid(
            row=2, column=0, columnspan=2, sticky='w', pady=(8, 10))
        ttk.Label(controls, text='DLS depth limit', style='Small.TLabel').grid(row=3, column=0, sticky='w')
        self.depth_entry = ttk.Spinbox(controls, from_=0, to=200, width=8, textvariable=self.depth_choice)
        self.depth_entry.grid(row=3, column=1, sticky='e')
        self.find_button = ttk.Button(controls, text='Find Route', style='Primary.TButton', command=self.find_route)
        self.find_button.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(12, 8))
        self.step_button = ttk.Button(controls, text='Step', command=self.step)
        self.step_button.grid(row=5, column=0, sticky='ew', padx=(0, 4))
        self.auto_button = ttk.Button(controls, text='Auto Run', command=self.auto_run)
        self.auto_button.grid(row=5, column=1, sticky='ew', padx=(4, 0))
        ttk.Button(controls, text='Pause / Cancel', command=self.pause).grid(row=6, column=0, sticky='ew', padx=(0, 4), pady=(7, 0))
        ttk.Button(controls, text='Reset Agent', command=self.reset_agent).grid(row=6, column=1, sticky='ew', padx=(4, 0), pady=(7, 0))
        ttk.Button(controls, text='Inspect Successors', command=self.inspect_successors).grid(row=7, column=0, columnspan=2, sticky='ew', pady=(7, 0))

        metrics = ttk.Frame(side, style='Card.TFrame', padding=16)
        metrics.grid(row=1, column=0, sticky='ew', pady=(12, 0))
        metrics.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Label(metrics, text='Search results', style='Title.TLabel').grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 10))
        for index, (key, label) in enumerate((('expanded', 'Expanded'), ('examined', 'Examined'),
                                             ('generated', 'Generated'), ('frontier', 'Peak frontier'),
                                             ('elapsed', 'Time (ms)'), ('cpu', 'CPU (ms)'),
                                             ('length', 'Plan moves'), ('cost', 'Plan cost'))):
            row_index, col = 1 + 2 * (index // 4), index % 4
            ttk.Label(metrics, text=label, style='Small.TLabel').grid(row=row_index, column=col, sticky='w', padx=(0, 7))
            ttk.Label(metrics, textvariable=self.metric_vars[key], font=(self.font_family, 12, 'bold')).grid(row=row_index + 1, column=col, sticky='w', pady=(3, 8))
        map_controls = ttk.Frame(grid_card, style='Card.TFrame')
        map_controls.grid(row=6, column=0, sticky='ew', pady=(14, 0))
        map_controls.columnconfigure(2, weight=1)
        ttk.Label(map_controls, text='Map controls', style='Small.TLabel').grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 5))
        ttk.Button(map_controls, text='Original Map', command=self.original_map).grid(row=1, column=0, padx=(0, 6))
        ttk.Button(map_controls, text='New Seed', command=self.new_seed_map).grid(row=1, column=1, padx=(0, 6))
        ttk.Entry(map_controls, textvariable=self.seed_choice, width=13).grid(row=1, column=2, sticky='ew', padx=(0, 6))
        ttk.Button(map_controls, text='Load Seed', command=self.load_seed).grid(row=1, column=3)
        footer = tk.Frame(root, bg='white', padx=24, pady=13)
        footer.pack(side='bottom', fill='x', before=body)
        self.status_label = tk.Label(footer, textvariable=self.status, bg='white', fg=self.INK, font=(self.font_family, 11, 'bold'))
        self.status_label.pack(anchor='w')
        self.message_label = tk.Label(footer, textvariable=self.message, bg='white', fg=self.MUTED,
                 font=(self.font_family, 10), wraplength=1020, justify='left')
        self.message_label.pack(anchor='w', pady=(4, 0))
        root.bind('<Configure>', self._window_resized, add='+')

    def _window_resized(self, event):
        if event.widget is self.root:
            self.message_label.configure(wraplength=max(300, event.width - 48))

    def _set_status(self, status, message, color=None):
        self.status.set(status)
        self.message.set(message)
        self.status_label.configure(fg=color or self.INK)

    def _set_controls(self):
        busy = self.search_iterator is not None or self.running
        for button in (self.find_button, self.step_button, self.auto_button):
            button.configure(state='disabled' if busy else 'normal')

    def _cancel_jobs(self):
        for attribute in ('search_job', 'move_job'):
            job = getattr(self, attribute)
            if job is not None:
                self.root.after_cancel(job)
                setattr(self, attribute, None)
        if self.search_iterator is not None:
            self.search_iterator.close()
            self.search_iterator = None
        self.running = False
        self.pending_action = None

    def _invalidate(self):
        self._cancel_jobs()
        self.result = None
        self.route_actions = []
        self.route_positions = []
        self.explored.clear()
        self.expanded_states.clear()
        self.successor_states.clear()
        self.current_search = None
        for variable in self.metric_vars.values():
            variable.set('-')
        self._set_controls()

    def _configuration_changed(self, *_args):
        self._invalidate()
        self.description.set(self.DESCRIPTIONS[self.algorithm_choice.get()])
        self._set_status('Strategy updated', 'The previous plan was cleared. Find a route with the new settings.')
        self.draw()

    def _depth_limit(self):
        try:
            depth = int(self.depth_choice.get())
            if depth < 0:
                raise ValueError
            return depth
        except ValueError:
            self._set_status('Check the depth limit', 'Enter a nonnegative whole number, such as 20.', '#a65c17')
            return None

    def find_route(self):
        self._begin_search()

    def _begin_search(self, pending=None):
        depth = self._depth_limit()
        if depth is None:
            return
        self._invalidate()
        self.pending_action = pending
        solver = make_search(self.algorithm_choice.get(), depth_limit=depth)
        self.search_iterator = solver.steps(self.env.transition_model(), self.env.agent_location)
        self._set_status('Searching...', 'The shaded cells show examined states. Pause / Cancel keeps the window responsive.')
        self._set_controls()
        self.search_job = self.root.after(1, self._advance_search)

    def _advance_search(self):
        self.search_job = None
        if self.search_iterator is None or self.closed:
            return
        # Ordinary graph searches animate visibly. DLS can revisit tens of
        # thousands of path nodes, so batch more of those while still giving
        # Tk control at least once per short (8 ms) work slice.
        try:
            batch_size = 512 if self.algorithm_choice.get() == SearchAlgorithm.DLS.value else 12
            deadline = perf_counter() + 0.008
            for _ in range(batch_size):
                event = next(self.search_iterator)
                self.current_search = event.node.state
                self.explored.add(event.node.state)
                if event.expanded:
                    self.expanded_states.add(event.node.state)
                self._update_metrics(event.metrics)
                if perf_counter() >= deadline:
                    break
        except StopIteration as completed:
            self.search_iterator = None
            self._finish_search(completed.value)
            return
        except Exception as error:
            self._cancel_jobs()
            self._set_controls()
            self._set_status('Search stopped', str(error), '#a65c17')
            return
        self.draw()
        self.search_job = self.root.after(12, self._advance_search)

    def _finish_search(self, result):
        pending = self.pending_action
        self.pending_action = None
        self.result = result
        self.explored = set(result.explored)
        self.expanded_states = set(result.expanded_states)
        self.current_search = None
        self.route_actions = list(result.actions)
        self.route_positions = list(result.path)
        self._update_metrics(result.metrics)
        if result.status == SearchStatus.SUCCESS:
            self.metric_vars['length'].set(str(len(result.actions)))
            self.metric_vars['cost'].set(str(result.goal_node.path_cost))
            self._set_status('Route found', f'{len(result.actions)} moves / cost {result.goal_node.path_cost}. '
                             'Use Step or Auto Run to follow this route.', '#137568')
        elif result.status == SearchStatus.CUTOFF:
            self._set_status('Depth cutoff', f'The depth limit ({self.depth_choice.get()}) was encountered. '
                             'A deeper route may exist; increase the limit and search again.', '#a65c17')
        elif result.status == SearchStatus.RESOURCE_LIMIT:
            self._set_status('Search limit reached', 'The search stopped at its safety limit. '
                             'This does not prove that the destination is unreachable.', '#a65c17')
        else:
            self._set_status('No route', 'The search exhausted the available routes. '
                             'Remove an obstacle or choose another map.', '#aa4141')
        self._set_controls()
        self.draw()
        if result.status == SearchStatus.SUCCESS:
            if pending == 'step':
                self._execute_one()
            elif pending == 'auto':
                self.running = True
                self._set_controls()
                self.move_job = self.root.after(80, self._auto_tick)

    def _update_metrics(self, metrics):
        for name, value in (('expanded', metrics.expanded), ('examined', metrics.examined),
                            ('generated', metrics.generated), ('frontier', metrics.frontier_peak)):
            self.metric_vars[name].set(str(value))
        self.metric_vars['elapsed'].set(f'{metrics.elapsed_ms:.2f}')
        self.metric_vars['cpu'].set(f'{metrics.cpu_ms:.2f}')

    def _can_move(self):
        if self.env.agent_location == self.env.goal:
            self._set_status('Destination reached', f'The traveler arrived in {self.env.steps} moves.', '#137568')
            return False
        if self.result is not None and self.result.status != SearchStatus.SUCCESS:
            self._set_status(self.result.status.value, 'There is no executable plan. Adjust the search settings or map, then Find Route.', '#a65c17')
            return False
        return True

    def step(self):
        if self.search_iterator is not None:
            return
        self.pause(announce=False)
        if not self._can_move():
            return
        if not self.route_actions:
            self._begin_search(pending='step')
        else:
            self._execute_one()

    def _execute_one(self):
        if not self.route_actions:
            self.running = False
            self._can_move()
            self._set_controls()
            return False
        self.successor_states.clear()
        action = self.route_actions.pop(0)
        predicted = self.route_positions.pop(0)
        result = self.env.execute(action)
        if not result.moved or self.env.agent_location != predicted:
            self._invalidate()
            self._set_status('Route changed', 'The selected move is no longer valid. Find a new route.', '#a65c17')
            self.draw()
            return False
        if result.reached_goal:
            self.running = False
            self._set_status('Destination reached', f'The traveler arrived in {self.env.steps} moves. '
                             'Reset Agent to try another strategy.', '#137568')
        else:
            self._set_status('Traveling' if self.running else 'One step completed',
                             f'{action.name} to ({self.env.agent_location.x}, {self.env.agent_location.y}). '
                             f'{len(self.route_actions)} planned moves remain.')
        self._set_controls()
        self.draw()
        return result.moved and not result.reached_goal

    def auto_run(self):
        if self.search_iterator is not None or self.running or not self._can_move():
            return
        if not self.route_actions:
            self._begin_search(pending='auto')
        else:
            self.running = True
            self._set_controls()
            self.move_job = self.root.after(1, self._auto_tick)

    def _auto_tick(self):
        self.move_job = None
        if self.running and not self.closed and self._execute_one():
            self.move_job = self.root.after(210, self._auto_tick)

    def pause(self, announce=True):
        was_searching = self.search_iterator is not None
        self._cancel_jobs()
        self.current_search = None
        self._set_controls()
        if announce:
            if was_searching:
                self._set_status('Search cancelled', 'The partial exploration remains visible. Find Route starts a new search.')
            else:
                self._set_status('Paused', 'Use Step or Auto Run to continue the planned route.')
        self.draw()

    def reset_agent(self):
        self._invalidate()
        self.env.reset()
        self._set_status('Agent reset', 'The traveler is back at the start. The current map and search settings are preserved.')
        self.draw()

    def _replace_environment(self, env, seed=None):
        self._invalidate()
        self.env = env
        self.active_seed = seed
        self.map_label.set(f'{"Original map" if seed is None else "Seed " + str(seed)} / {env.width} x {env.height}')
        self._set_status('Map ready', 'Choose Find Route to search this map. Start, goal, and agent cells cannot be blocked.')
        self.draw()

    def original_map(self):
        self._replace_environment(new_engine().environment)

    def new_seed_map(self):
        seed = random_seed()
        self.seed_choice.set(str(seed))
        self._replace_environment(generate_seeded_environment(seed), seed)

    def load_seed(self):
        try:
            seed = int(self.seed_choice.get().strip())
        except ValueError:
            self._set_status('Check the seed', 'Enter a whole-number map seed.', '#a65c17')
            return
        self._replace_environment(generate_seeded_environment(seed), seed)

    def inspect_successors(self):
        self.pause(announce=False)
        children = self.env.transition_model().expand(Node(self.env.agent_location))
        self.successor_states = {child.state for child in children}
        successors = ' | '.join(f'{child.action.name} -> ({child.state.x}, {child.state.y})'
                                for child in children)
        self._set_status('Successors inspected', (successors + '. Each child: depth 1, cost 1.') if successors
                         else 'No applicable actions: the current state is terminal or trapped.')
        self.draw()

    def click(self, event):
        if self.search_iterator is not None or self.running:
            return
        left, top, size = self.grid_geometry
        loc = Location(int((event.x - left) // size), int((event.y - top) // size))
        if not self.env.in_bounds(loc) or loc in (self.env.start, self.env.goal, self.env.agent_location):
            return
        self._invalidate()
        self.env.set_obstacle(loc, loc not in self.env.obstacles)
        if self.active_seed is not None:
            self.map_label.set(f'Seed {self.active_seed} (edited) / {self.env.width} x {self.env.height}')
        self._set_status('Map edited', f'Obstacle toggled at ({loc.x}, {loc.y}). The previous plan was cleared.')
        self.draw()

    def draw(self):
        """Render stored state only: no planning, mutation, or agent movement."""
        if self.closed:
            return
        canvas, env, colors = self.canvas, self.env, self.CELL_COLORS
        canvas.delete('all')
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        size = max(1.0, min((width - 24) / env.width, (height - 24) / env.height))
        left, top = (width - env.width * size) / 2, (height - env.height * size) / 2
        self.grid_geometry = (left, top, size)
        route = set(self.route_positions)
        for y in range(env.height):
            for x in range(env.width):
                loc = Location(x, y)
                fill = colors['open']
                if loc in self.explored:
                    fill = colors['examined']
                if loc in self.expanded_states:
                    fill = colors['expanded']
                if loc in route:
                    fill = colors['path']
                if loc in self.successor_states:
                    fill = colors['successor']
                if loc == self.current_search:
                    fill = colors['current']
                if loc == env.start:
                    fill = colors['start']
                if loc == env.goal:
                    fill = colors['goal']
                if loc in env.obstacles:
                    fill = colors['obstacle']
                px, py = left + x * size, top + y * size
                canvas.create_rectangle(px, py, px + size, py + size,
                                        fill=fill, outline='#d5e0e6', width=1)
                if loc in (env.start, env.goal) and loc != env.agent_location:
                    canvas.create_text(px + size / 2, py + size / 2, text='G' if loc == env.goal else 'S',
                                       fill=self.INK, font=(self.font_family, max(9, int(size * .28)), 'bold'))
                if loc == env.agent_location:
                    pad = size * .18
                    canvas.create_oval(px + pad, py + pad, px + size - pad, py + size - pad,
                                       fill='#137f79', outline='white', width=2)
                    canvas.create_text(px + size / 2, py + size / 2, text='A', fill='white',
                                       font=(self.font_family, max(9, int(size * .25)), 'bold'))
        self.movement.set(f'{env.steps} moves taken  /  {len(self.route_actions)} planned moves remaining')

    def close(self):
        self._cancel_jobs()
        self.closed = True
        self.root.destroy()


def launch_gui():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit('A desktop display is required. Try: python engine.py --algorithms-demo') from error
    TravelPlannerApp(root)
    root.mainloop()


if __name__ == '__main__':
    launch_gui()
