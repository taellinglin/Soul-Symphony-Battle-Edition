from dataclasses import dataclass
import math
import random

from accel_math import generate_labyrinth_room_dims


@dataclass
class GenerationConfig:
    scale: float = 1.0
    layout_mode: str = "labrinth"
    snake_cell_size: int = 40
    snake_layers: int = 2
    maze_cell_size: int = 12
    maze_layers: int = 2
    maze_loop_chance: float = 0.06
    maze_vertical_link_chance: float = 0.18
    average_room_size: float = 12.0
    room_size_jitter: float = 0.18
    room_height: float = 12.0
    wall_thickness: float = 0.2
    floor_thickness: float = 0.2
    corridor_width: float = 9.6
    room_density: float = 0.5
    corridor_density: float = 0.5
    decor_density: float = 0.28
    angled_room_ratio: float = 0.12
    base_cube_unit: float = 1.0
    max_rooms: int = 72


@dataclass
class Room:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w * 0.5, self.y + self.h * 0.5


@dataclass
class Leaf:
    x: float
    y: float
    w: float
    h: float
    left: "Leaf | None" = None
    right: "Leaf | None" = None
    room: Room | None = None


class BSPGenerator:
    def __init__(self, width: int, depth: int, config: GenerationConfig, min_leaf: int = 40, max_leaf: int = 88):
        self.width = width
        self.depth = depth
        self.config = config
        self.min_leaf = min_leaf
        self.max_leaf = max_leaf

    def _cube_unit(self) -> float:
        return max(0.25, float(getattr(self.config, "base_cube_unit", 1.0)))

    def _snap(self, value: float) -> float:
        unit = self._cube_unit()
        return round(float(value) / unit) * unit

    def _snap_room(self, x: float, y: float, w: float, h: float) -> Room:
        unit = self._cube_unit()
        min_size = max(2.5, unit * 3.0)
        sw = max(min_size, self._snap(w))
        sh = max(min_size, self._snap(h))
        sx = self._snap(x)
        sy = self._snap(y)

        sx = max(0.0, min(sx, self.width - sw))
        sy = max(0.0, min(sy, self.depth - sh))
        return Room(sx, sy, sw, sh)

    def _fit_room_to_cell(self, gx: int, gy: int, size: float, room_w: float, room_h: float, pad: float) -> Room:
        corridor_pad = max(0.5, min(size * 0.22, float(getattr(self.config, "corridor_width", 6.0)) * 0.18))
        edge_pad = max(float(pad), corridor_pad)
        max_dim = max(3.0, size - edge_pad * 2.0)
        rw = min(max_dim, max(2.8, room_w))
        rh = min(max_dim, max(2.8, room_h))

        rx = gx * size + (size - rw) * 0.5
        ry = gy * size + (size - rh) * 0.5
        return self._snap_room(rx, ry, rw, rh)

    def _room_budget(self, default_value: int) -> int:
        configured = int(getattr(self.config, "max_rooms", default_value))
        return max(8, min(256, configured))

    def generate(self) -> tuple[list[Room], list[tuple[int, int]]]:
        root = Leaf(0, 0, self.width, self.depth)
        leaves = [root]

        did_split = True
        while did_split:
            did_split = False
            for leaf in list(leaves):
                if leaf.left or leaf.right:
                    continue
                split_chance = max(0.05, min(0.85, 0.12 + self.config.room_density * 0.6))
                if leaf.w > self.max_leaf or leaf.h > self.max_leaf or random.random() < split_chance:
                    if self._split(leaf):
                        leaves.append(leaf.left)
                        leaves.append(leaf.right)
                        did_split = True

        rooms: list[Room] = []
        self._create_rooms(root, rooms)

        edges = self._connect_rooms(rooms)
        return rooms, edges

    def generate_labyrinth(self, cell_size: int = 20) -> tuple[list[Room], list[tuple[int, int]]]:
        size = max(8, int(cell_size))
        cols = max(4, int(self.width // size))
        rows = max(4, int(self.depth // size))
        room_budget = self._room_budget(cols * rows)
        while cols * rows > room_budget and (cols > 4 or rows > 4):
            if cols >= rows and cols > 4:
                cols -= 1
            elif rows > 4:
                rows -= 1
            else:
                break

        pad = max(0.4, min(size * 0.16, 2.4))
        base_room_w = max(3.2, size - pad * 2.0)
        base_room_h = max(3.2, size - pad * 2.0)
        jitter = max(0.0, min(0.75, float(getattr(self.config, "room_size_jitter", 0.2))))
        precomputed_dims = generate_labyrinth_room_dims(cols, rows, base_room_w, base_room_h, jitter)

        rooms: list[Room] = []
        index_map: dict[tuple[int, int], int] = {}
        for gy in range(rows):
            for gx in range(cols):
                cell_idx = gy * cols + gx
                if precomputed_dims is not None:
                    room_w = float(precomputed_dims[0][cell_idx])
                    room_h = float(precomputed_dims[1][cell_idx])
                else:
                    size_scale = random.uniform(1.0 - jitter * 0.55, 1.0 + jitter * 0.38)
                    aspect_skew = random.uniform(-jitter * 0.48, jitter * 0.48)
                    if random.random() < (0.14 + jitter * 0.22):
                        size_scale *= random.uniform(0.78, 1.24)

                    room_w = max(2.8, base_room_w * size_scale * (1.0 + aspect_skew))
                    room_h = max(2.8, base_room_h * size_scale * (1.0 - aspect_skew))

                idx = len(rooms)
                rooms.append(self._fit_room_to_cell(gx, gy, size, room_w, room_h, pad))
                index_map[(gx, gy)] = idx

        visited: set[tuple[int, int]] = set()
        stack: list[tuple[int, int]] = [(0, 0)]
        visited.add((0, 0))
        edges: list[tuple[int, int]] = []

        def neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
            cx, cy = cell
            out: list[tuple[int, int]] = []
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < cols and 0 <= ny < rows:
                    out.append((nx, ny))
            random.shuffle(out)
            return out

        while stack:
            current = stack[-1]
            unvisited = [n for n in neighbors(current) if n not in visited]
            if not unvisited:
                stack.pop()
                continue
            nxt = random.choice(unvisited)
            a = index_map[current]
            b = index_map[nxt]
            edges.append((a, b))
            visited.add(nxt)
            stack.append(nxt)

        existing = {tuple(sorted((a, b))) for a, b in edges}
        all_neighbor_pairs: list[tuple[int, int]] = []
        for gy in range(rows):
            for gx in range(cols):
                a = index_map[(gx, gy)]
                if gx + 1 < cols:
                    b = index_map[(gx + 1, gy)]
                    all_neighbor_pairs.append((a, b))
                if gy + 1 < rows:
                    b = index_map[(gx, gy + 1)]
                    all_neighbor_pairs.append((a, b))

        loop_factor = max(0.01, min(0.2, self.config.corridor_density * 0.14))
        random.shuffle(all_neighbor_pairs)
        extras = max(1, int(len(edges) * loop_factor))
        for a, b in all_neighbor_pairs:
            key = tuple(sorted((a, b)))
            if key in existing:
                continue
            edges.append((a, b))
            existing.add(key)
            extras -= 1
            if extras <= 0:
                break

        return rooms, edges

    def generate_hex_mixed(self, cell_size: int = 24) -> tuple[list[Room], list[tuple[int, int]]]:
        size = max(12, int(cell_size))
        step_x = size * 0.9
        step_y = size * 0.78
        margin = size * 0.6

        rows = max(3, int((self.depth - margin * 2.0) // step_y))
        cols = max(3, int((self.width - margin * 2.0) // step_x))

        room_budget = self._room_budget(rows * cols)
        cells: list[tuple[int, int, float, float]] = []
        for r in range(rows):
            x_offset = (step_x * 0.5) if (r % 2 == 1) else 0.0
            for c in range(cols):
                cx = margin + c * step_x + x_offset
                cy = margin + r * step_y
                if cx < margin or cy < margin or cx > (self.width - margin) or cy > (self.depth - margin):
                    continue
                cells.append((c, r, cx, cy))

        if len(cells) > room_budget:
            random.shuffle(cells)
            cells = cells[:room_budget]

        rooms: list[Room] = []
        index_map: dict[tuple[int, int], int] = {}
        jitter = max(0.0, min(0.75, float(getattr(self.config, "room_size_jitter", 0.22))))

        for c, r, cx, cy in cells:
            shape_roll = random.random()
            if shape_roll < 0.54:
                rw = size * random.uniform(0.72, 0.96)
                rh = size * random.uniform(0.66, 0.9)
            elif shape_roll < 0.68:
                base = size * random.uniform(0.58, 0.82)
                rw = base
                rh = base
            elif shape_roll < 0.8:
                rw = size * random.uniform(1.0, 1.38)
                rh = size * random.uniform(0.45, 0.7)
            elif shape_roll < 0.92:
                rw = size * random.uniform(0.45, 0.7)
                rh = size * random.uniform(1.0, 1.38)
            else:
                rw = size * random.uniform(0.6, 0.95)
                rh = size * random.uniform(0.6, 0.95)

            rw *= random.uniform(1.0 - jitter * 0.45, 1.0 + jitter * 0.4)
            rh *= random.uniform(1.0 - jitter * 0.45, 1.0 + jitter * 0.4)

            rx = cx - rw * 0.5
            ry = cy - rh * 0.5
            room = self._snap_room(rx, ry, rw, rh)
            idx = len(rooms)
            rooms.append(room)
            index_map[(c, r)] = idx

        edges: set[tuple[int, int]] = set()
        for c, r, _, _ in cells:
            a = index_map.get((c, r))
            if a is None:
                continue

            if r % 2 == 0:
                neighbors = [
                    (c - 1, r), (c + 1, r),
                    (c, r - 1), (c - 1, r - 1),
                    (c, r + 1), (c - 1, r + 1),
                ]
            else:
                neighbors = [
                    (c - 1, r), (c + 1, r),
                    (c, r - 1), (c + 1, r - 1),
                    (c, r + 1), (c + 1, r + 1),
                ]

            for nc, nr in neighbors:
                b = index_map.get((nc, nr))
                if b is None or a == b:
                    continue
                edges.add(tuple(sorted((a, b))))

            if random.random() < 0.32:
                for nc, nr in ((c + 2, r), (c - 2, r), (c, r + 2), (c, r - 2)):
                    b = index_map.get((nc, nr))
                    if b is None or a == b:
                        continue
                    edges.add(tuple(sorted((a, b))))

        if not edges and len(rooms) > 1:
            for i in range(len(rooms) - 1):
                edges.add((i, i + 1))

        return rooms, list(edges)

    def generate_snake3d(self, cell_size: int = 24, layers: int = 4) -> tuple[list[Room], list[tuple[int, int]], dict[int, int]]:
        size = max(10, int(cell_size))
        cols = max(3, int(self.width // size))
        rows = max(3, int(self.depth // size))
        layer_count = max(2, int(layers))
        room_budget = self._room_budget(cols * rows * layer_count)

        pad = max(0.45, min(size * 0.14, 2.4))
        base_room_w = max(3.0, size - pad * 2.0)
        base_room_h = max(3.0, size - pad * 2.0)
        jitter = max(0.0, min(0.7, self.config.room_size_jitter))

        cells_2d: list[tuple[int, int]] = []
        for gy in range(rows):
            x_iter = range(cols) if (gy % 2 == 0) else range(cols - 1, -1, -1)
            for gx in x_iter:
                cells_2d.append((gx, gy))

        rooms: list[Room] = []
        room_levels: dict[int, int] = {}
        index_map: dict[tuple[int, int, int], int] = {}
        edge_set: set[tuple[int, int]] = set()

        previous_idx: int | None = None
        for gz in range(layer_count):
            layer_cells = cells_2d if (gz % 2 == 0) else list(reversed(cells_2d))
            for gx, gy in layer_cells:
                if len(rooms) >= room_budget:
                    break
                rw = max(2.8, base_room_w * random.uniform(1.0 - jitter * 0.35, 1.0 + jitter * 0.2))
                rh = max(2.8, base_room_h * random.uniform(1.0 - jitter * 0.35, 1.0 + jitter * 0.2))

                rw = min(rw, size - 0.35)
                rh = min(rh, size - 0.35)

                idx = len(rooms)
                rooms.append(self._fit_room_to_cell(gx, gy, size, rw, rh, pad))
                room_levels[idx] = gz
                index_map[(gx, gy, gz)] = idx

                if previous_idx is not None:
                    a, b = sorted((previous_idx, idx))
                    edge_set.add((a, b))
                previous_idx = idx
            if len(rooms) >= room_budget:
                break

        if self.config.corridor_density > 0.25:
            extra_ratio = max(0.05, min(0.35, self.config.corridor_density * 0.35))
            candidates: list[tuple[int, int]] = []
            for gz in range(layer_count):
                for gy in range(rows):
                    for gx in range(cols):
                        a = index_map[(gx, gy, gz)]
                        if gx + 1 < cols:
                            b = index_map[(gx + 1, gy, gz)]
                            candidates.append(tuple(sorted((a, b))))
                        if gy + 1 < rows:
                            b = index_map[(gx, gy + 1, gz)]
                            candidates.append(tuple(sorted((a, b))))
            random.shuffle(candidates)
            extras = max(1, int(len(edge_set) * extra_ratio))
            for e in candidates:
                if e in edge_set:
                    continue
                edge_set.add(e)
                extras -= 1
                if extras <= 0:
                    break

        edges = list(edge_set)
        return rooms, edges, room_levels

    def generate_maze3d(
        self,
        cell_size: int = 48,
        layers: int = 2,
        loop_chance: float = 0.05,
        vertical_link_chance: float = 0.2,
    ) -> tuple[list[Room], list[tuple[int, int]], dict[int, int]]:
        size = max(16, int(cell_size))
        cols = max(3, int(self.width // size))
        rows = max(3, int(self.depth // size))
        layer_count = max(2, int(layers))
        room_budget = self._room_budget(cols * rows * layer_count)
        while cols * rows * layer_count > room_budget and (cols > 3 or rows > 3 or layer_count > 2):
            if cols >= rows and cols >= layer_count and cols > 3:
                cols -= 1
            elif rows >= layer_count and rows > 3:
                rows -= 1
            elif layer_count > 2:
                layer_count -= 1
            else:
                break

        pad = max(0.5, min(size * 0.16, 3.0))
        base_room_w = max(3.0, size - pad * 2.0)
        base_room_h = max(3.0, size - pad * 2.0)
        jitter = max(0.0, min(0.6, self.config.room_size_jitter))

        rooms: list[Room] = []
        room_levels: dict[int, int] = {}
        index_map: dict[tuple[int, int, int], int] = {}

        for gz in range(layer_count):
            for gy in range(rows):
                for gx in range(cols):
                    rw = max(2.8, base_room_w * random.uniform(1.0 - jitter * 0.28, 1.0 + jitter * 0.18))
                    rh = max(2.8, base_room_h * random.uniform(1.0 - jitter * 0.28, 1.0 + jitter * 0.18))
                    rw = min(rw, size - 0.4)
                    rh = min(rh, size - 0.4)

                    idx = len(rooms)
                    rooms.append(self._fit_room_to_cell(gx, gy, size, rw, rh, pad))
                    room_levels[idx] = gz
                    index_map[(gx, gy, gz)] = idx

        visited: set[tuple[int, int, int]] = set()
        stack: list[tuple[int, int, int]] = [(0, 0, 0)]
        visited.add((0, 0, 0))
        edge_set: set[tuple[int, int]] = set()

        vertical_ratio = max(0.0, min(0.5, float(vertical_link_chance)))

        def neighbors(cell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
            cx, cy, cz = cell
            lateral: list[tuple[int, int, int]] = []
            vertical: list[tuple[int, int, int]] = []
            if cx + 1 < cols:
                lateral.append((cx + 1, cy, cz))
            if cx - 1 >= 0:
                lateral.append((cx - 1, cy, cz))
            if cy + 1 < rows:
                lateral.append((cx, cy + 1, cz))
            if cy - 1 >= 0:
                lateral.append((cx, cy - 1, cz))
            if cz + 1 < layer_count:
                vertical.append((cx, cy, cz + 1))
            if cz - 1 >= 0:
                vertical.append((cx, cy, cz - 1))
            random.shuffle(lateral)
            random.shuffle(vertical)
            return lateral + vertical

        while stack:
            current = stack[-1]
            unvisited = [n for n in neighbors(current) if n not in visited]
            if not unvisited:
                stack.pop()
                continue

            lateral_unvisited = [n for n in unvisited if n[2] == current[2]]
            vertical_unvisited = [n for n in unvisited if n[2] != current[2]]
            if lateral_unvisited and (not vertical_unvisited or random.random() > vertical_ratio):
                nxt = random.choice(lateral_unvisited)
            elif vertical_unvisited:
                nxt = random.choice(vertical_unvisited)
            else:
                nxt = random.choice(unvisited)

            a = index_map[current]
            b = index_map[nxt]
            edge_set.add(tuple(sorted((a, b))))
            visited.add(nxt)
            stack.append(nxt)

        candidate_edges: list[tuple[int, int]] = []
        loop_ratio = max(0.0, min(0.2, float(loop_chance)))

        for gz in range(layer_count):
            for gy in range(rows):
                for gx in range(cols):
                    a = index_map[(gx, gy, gz)]
                    if gx + 1 < cols:
                        b = index_map[(gx + 1, gy, gz)]
                        candidate_edges.append(tuple(sorted((a, b))))
                    if gy + 1 < rows:
                        b = index_map[(gx, gy + 1, gz)]
                        candidate_edges.append(tuple(sorted((a, b))))
                    if gz + 1 < layer_count and random.random() < vertical_ratio:
                        b = index_map[(gx, gy, gz + 1)]
                        candidate_edges.append(tuple(sorted((a, b))))

        random.shuffle(candidate_edges)
        extra_budget = int(len(edge_set) * loop_ratio)
        if extra_budget > 0:
            for edge in candidate_edges:
                if edge in edge_set:
                    continue
                edge_set.add(edge)
                extra_budget -= 1
                if extra_budget <= 0:
                    break

        return rooms, list(edge_set), room_levels

    def _split(self, leaf: Leaf) -> bool:
        split_h = random.random() > 0.5
        if leaf.w / leaf.h >= 1.25:
            split_h = False
        elif leaf.h / leaf.w >= 1.25:
            split_h = True

        max_split = int((leaf.h if split_h else leaf.w) - self.min_leaf)
        if max_split <= self.min_leaf:
            return False

        split = random.randint(self.min_leaf, max_split)
        if split_h:
            leaf.left = Leaf(leaf.x, leaf.y, leaf.w, split)
            leaf.right = Leaf(leaf.x, leaf.y + split, leaf.w, leaf.h - split)
        else:
            leaf.left = Leaf(leaf.x, leaf.y, split, leaf.h)
            leaf.right = Leaf(leaf.x + split, leaf.y, leaf.w - split, leaf.h)
        return True

    def _create_rooms(self, leaf: Leaf, rooms: list[Room]) -> None:
        if leaf.left or leaf.right:
            if leaf.left:
                self._create_rooms(leaf.left, rooms)
            if leaf.right:
                self._create_rooms(leaf.right, rooms)
            return

        scale = max(0.5, self.config.scale)
        margin = max(3, int(4 * scale))
        available_w = max(4, int(leaf.w - margin * 2))
        available_h = max(4, int(leaf.h - margin * 2))
        if available_w < 6 or available_h < 6:
            margin = max(1, min(margin, int(min(leaf.w, leaf.h) * 0.12)))
            available_w = max(4, int(leaf.w - margin * 2))
            available_h = max(4, int(leaf.h - margin * 2))

        avg_room = max(10, int(self.config.average_room_size * scale))
        jitter = max(0.1, min(0.8, self.config.room_size_jitter))
        min_room = max(6, int(avg_room * (1.0 - jitter)))
        min_room_w = min(min_room, available_w)
        min_room_h = min(min_room, available_h)
        if min_room_w < 4 or min_room_h < 4:
            return
        target_max = max(min_room + 1, int(avg_room * (1.0 + jitter)))
        max_w = max(min_room_w, min(available_w, target_max))
        max_h = max(min_room_h, min(available_h, target_max))
        room_w = random.randint(min_room_w, max_w)
        room_h = random.randint(min_room_h, max_h)

        max_x = int(leaf.w - room_w - margin)
        max_y = int(leaf.h - room_h - margin)
        room_x = int(leaf.x + random.randint(margin, max(margin, max_x)))
        room_y = int(leaf.y + random.randint(margin, max(margin, max_y)))

        room_x = max(int(leaf.x), min(room_x, int(leaf.x + leaf.w - room_w)))
        room_y = max(int(leaf.y), min(room_y, int(leaf.y + leaf.h - room_h)))

        leaf.room = self._snap_room(room_x, room_y, room_w, room_h)
        rooms.append(leaf.room)

    def _connect_rooms(self, rooms: list[Room]) -> list[tuple[int, int]]:
        centers = [(r.x + r.w / 2, r.y + r.h / 2) for r in rooms]
        all_edges: list[tuple[float, int, int]] = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = (dx * dx + dy * dy) ** 0.5
                all_edges.append((d, i, j))

        mst = self._kruskal(len(rooms), all_edges)

        extra_edges: list[tuple[int, int]] = []
        candidates = [(d, a, b) for (d, a, b) in all_edges if (a, b) not in mst and (b, a) not in mst]
        random.shuffle(candidates)
        extra_ratio = max(0.05, min(0.9, self.config.corridor_density))
        extra_count = max(1, int(len(mst) * extra_ratio))
        for _, a, b in candidates[:extra_count]:
            extra_edges.append((a, b))

        return mst + extra_edges

    def _kruskal(self, n: int, edges: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        result: list[tuple[int, int]] = []
        for _, a, b in sorted(edges, key=lambda e: e[0]):
            if union(a, b):
                result.append((a, b))
                if len(result) == n - 1:
                    break
        return result
