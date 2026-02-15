import math
import os
import random
import colorsys
import sys
try:
    import ctypes
except Exception:
    ctypes = None
import importlib.util
import time
import heapq
import wave
import struct
import threading
import queue
from collections import deque

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.showbase import Audio3DManager
from direct.filter.CommonFilters import CommonFilters
from direct.filter.FilterManager import FilterManager
try:
    from panda3d.ai import AICharacter, AIWorld
except Exception:
    AICharacter = None
    AIWorld = None
try:
    from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode, BulletWorld
except Exception:
    BulletBoxShape = None
    BulletRigidBodyNode = None
    BulletWorld = None
from panda3d.core import AmbientLight, BitMask32, CardMaker, ColorBlendAttrib, CullFaceAttrib, DirectionalLight, Fog, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, LightRampAttrib, LineSegs, Material, NodePath, PNMImage, Point2, PointLight, Shader, TexGenAttrib, TextNode, Texture, TextureStage, TransparencyAttrib, Vec2, Vec3, WindowProperties, loadPrcFileData

from camera import camera_orbit_position, resolve_camera_collision, rotate_around_axis, setup_camera
from ball_visuals import spawn_player_ball
from accel_math import NUMBA_AVAILABLE, NUMPY_AVAILABLE, compute_level_w_batch, compute_level_w_scalar, compute_mobius_fold_twist, find_room_index_for_pos, find_zone_index_for_pos, prepare_room_bounds, prepare_zone_bounds
from level import BSPGenerator, GenerationConfig, Room
from player import queue_jump
from soundfx import load_first_sfx, play_sound
from weapon_system import apply_attack_hits, setup_weapon_system, trigger_spin_attack, trigger_swing_attack, trigger_throw_attack, update_weapon_system
from world import create_checker_texture, create_shadow_texture


def _detect_native_resolution() -> tuple[int, int]:
    if ctypes is None:
        return 1920, 1080
    try:
        user32 = ctypes.windll.user32
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1920, 1080


def _configure_display_prc() -> None:
    width, height = _detect_native_resolution()
    os.environ.setdefault("SOULSYM_SAFE_RENDER", "0" if sys.platform.startswith("linux") else "1")
    os.environ.setdefault("SOULSYM_ENABLE_VIDEO_DISTORTION", "1")
    os.environ.setdefault("SOULSYM_VERBOSE_SCENE_STATS", "0")
    os.environ.setdefault("SOULSYM_ENABLE_NUMBA", "0")
    os.environ.setdefault("SOULSYM_WATER_EMISSIVE_LINUX", "1" if sys.platform.startswith("linux") else "0")
    os.environ.setdefault("SOULSYM_FULLSCREEN", "1")
    os.environ.setdefault("SOULSYM_VSYNC", "1")
    vsync_on = _env_flag("SOULSYM_VSYNC", True)
    fullscreen_on = _env_flag("SOULSYM_FULLSCREEN", True)
    if not fullscreen_on:
        win_w = int(max(1280, min(width, 1600)))
        win_h = int(max(720, min(height, 900)))
    if fullscreen_on:
        loadPrcFileData("", "fullscreen 0")
        loadPrcFileData("", "undecorated 1")
        loadPrcFileData("", "win-origin 0 0")
        loadPrcFileData("", f"win-size {width} {height}")
        loadPrcFileData("", "win-fixed-size 1")
    else:
        loadPrcFileData("", f"win-size {win_w} {win_h}")
        loadPrcFileData("", "fullscreen 0")
        loadPrcFileData("", "undecorated 0")
    loadPrcFileData("", f"sync-video {1 if vsync_on else 0}")
    loadPrcFileData("", "clock-mode limited")
    loadPrcFileData("", f"clock-frame-rate {60 if vsync_on else 120}")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _configure_gpu_prc() -> None:
    if sys.platform == "emscripten":
        os.environ["SOULSYM_CUDA_ACTIVE"] = "0"
        os.environ["SOULSYM_CUDA_SOURCE"] = "none"
        os.environ["SOULSYM_MULTI_GPU_ACTIVE"] = "0"
        print("[GPU] Web runtime detected; CUDA/torch checks skipped.")
        return

    safe_render = _env_flag("SOULSYM_SAFE_RENDER", True)
    loadPrcFileData("", "load-display pandagl")
    if safe_render:
        loadPrcFileData("", "support-threads 0")
    else:
        loadPrcFileData("", "support-threads 1")
        loadPrcFileData("", "threading-model Cull/Draw")

    os.environ.setdefault("SOULSYM_ENABLE_CUDA", "1")
    os.environ.setdefault("SOULSYM_ENABLE_MULTI_GPU", "1")

    allow_cuda = _env_flag("SOULSYM_ENABLE_CUDA", True)
    allow_multi_gpu = _env_flag("SOULSYM_ENABLE_MULTI_GPU", True)
    selected_devices = os.getenv("SOULSYM_CUDA_DEVICES", "").strip()

    cuda_available = False
    cuda_source = "none"
    cuda_device_count = 0

    if allow_cuda:
        try:
            if importlib.util.find_spec("torch") is not None:
                torch = importlib.import_module("torch")

                cuda_available = bool(torch.cuda.is_available())
                if cuda_available:
                    cuda_device_count = int(torch.cuda.device_count())
                    cuda_source = "torch"
        except Exception:
            cuda_available = False

        if not cuda_available:
            try:
                if importlib.util.find_spec("cupy") is not None:
                    cupy = importlib.import_module("cupy")

                    cuda_device_count = int(cupy.cuda.runtime.getDeviceCount())
                    cuda_available = cuda_device_count > 0
                    if cuda_available:
                        cuda_source = "cupy"
            except Exception:
                cuda_available = False

    if selected_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_devices
    elif allow_multi_gpu and cuda_available and cuda_device_count > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(cuda_device_count))

    os.environ["SOULSYM_CUDA_ACTIVE"] = "1" if cuda_available else "0"
    os.environ["SOULSYM_CUDA_SOURCE"] = cuda_source
    if allow_multi_gpu and cuda_available and cuda_device_count > 1:
        os.environ["SOULSYM_MULTI_GPU_ACTIVE"] = "1"
    else:
        os.environ["SOULSYM_MULTI_GPU_ACTIVE"] = "0"

    print(
        "[GPU] CUDA enabled request:",
        "on" if allow_cuda else "off",
        "| CUDA active:",
        "yes" if cuda_available else "no",
        "| source:",
        cuda_source,
        "| visible devices:",
        os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
    )
    if allow_multi_gpu and cuda_available and cuda_device_count > 1:
        print(f"[GPU] Multi-GPU visibility enabled ({cuda_device_count} devices).")
    elif allow_multi_gpu:
        print("[GPU] Multi-GPU requested but only one/no CUDA device detected.")


class SoulSymphony(ShowBase):
    def __init__(self):
        if sys.platform == "emscripten" and BulletWorld is None:
            raise RuntimeError("Web build is missing panda3d.bullet in this toolchain. Rebuild Panda WebGL with Bullet static libs or run desktop build.")
        super().__init__()

        self.disableMouse()
        self.accept("window-event", self._on_window_event)

        if _env_flag("SOULSYM_FULLSCREEN", True) and self.win is not None:
            try:
                fw, fh = _detect_native_resolution()
                props = WindowProperties()
                props.setFullscreen(False)
                props.setUndecorated(True)
                props.setOrigin(0, 0)
                props.setSize(int(fw), int(fh))
                props.setFixedSize(True)
                self.win.requestProperties(props)
            except Exception:
                pass

        self.world = self.render.attachNewNode("world")
        self.physics_world = BulletWorld()
        self.physics_world.setGravity(Vec3(0, 0, -9.62))
        self.group_level = BitMask32.bit(0)
        self.group_player = BitMask32.bit(1)
        self.mask_level_hits = self.group_player
        self.mask_player_hits = self.group_level
        self.current_gravity = Vec3(0, 0, -9.81)
        self.physics_nodes: list[BulletRigidBodyNode] = []
        self.collider_visual_map: dict[int, NodePath] = {}
        self.quad_template_root = self.render.attachNewNode("quad-templates")
        self.quad_template_root.hide()
        self.quad_templates: dict[tuple[str, float, float, float, float], NodePath] = {}
        self.occluded_visuals: set[int] = set()
        self.scene_visuals: dict[int, NodePath] = {}
        self.visual_outline_nodes: dict[int, NodePath] = {}
        self.visual_alpha_targets: dict[int, float] = {}
        self.visual_alpha_state: dict[int, float] = {}
        self.ripple_events: list[tuple[Vec3, float]] = []
        self.ripple_emit_timer = 0.0
        self.ripple_emit_interval = 0.26
        self.ripple_speed = 3.5
        self.ripple_width = 15.2
        self.ripple_max_age = 1.8
        self.ripple_alpha_strength = 0.85
        self.motion_trails: list[dict] = []
        self.motion_trail_emit_timer = 0.0
        self.performance_mode = True
        self.safe_render_mode = _env_flag("SOULSYM_SAFE_RENDER", True)
        self.enable_particles = False
        self.enable_gravity_particles = False
        self.enable_motion_trails = False
        self.enable_video_distortion = _env_flag("SOULSYM_ENABLE_VIDEO_DISTORTION", True)
        self.enable_entrance_debug_overlay = _env_flag("SOULSYM_DEBUG_ENTRANCES", True)
        self.enable_ripple_effect = True
        self.enable_dynamic_shadows = (not self.performance_mode) and (not self.safe_render_mode)
        self.enable_ball_shadow = not self.performance_mode
        self.enable_occlusion_outlines = (not self.performance_mode) and (not self.safe_render_mode)
        self.performance_player_light_enabled = False
        self.performance_player_light_cycle_speed = 2.6
        self.performance_player_light_saturation = 0.9
        self.performance_player_light_value = 1.0
        self.performance_player_light_height = 1.25
        self.physics_substeps = 2 if self.performance_mode else 4
        self.physics_fixed_timestep = (1 / 120.0) if self.performance_mode else (1 / 180.0)
        self.video_distort_strength = 0.24
        self.video_bloom_strength = 0.18
        self.video_bloom_radius = 0.9
        self.video_bloom_threshold = 0.52
        self.video_distort_compression_response = 1.25
        self.video_distort_sine_response = 0.52
        self.video_distort_dynamic_min_mul = 0.62
        self.video_distort_dynamic_max_mul = 2.75
        self.video_distort_update_timer = 0.0
        self.video_distort_update_interval = 1.0 / 30.0
        self.video_distort_overlay_np: NodePath | None = None
        self.video_distort_buffer = None
        self.video_distort_post_manager = None
        self.video_distort_ready = False
        self.video_distort_error = ""
        self.video_distort_shader = None
        self.video_distort_tex_size = (0, 0)
        self.video_distort_window_size = (0, 0)
        self.color_cycle_nodes: list[dict] = []
        self.color_cycle_time = 0.0
        self.color_cycle_update_timer = 0.0
        self.texture_layer_cycle_enabled = True
        self.texture_layer_scroll_u = 0.018
        self.texture_layer_scroll_v = 0.013
        self.texture_layer_fade_speed = 0.28
        self.texture_layer_alpha_min = 0.02
        self.texture_layer_alpha_max = 0.11
        self.texture_layer_additive_gain = 0.42
        self.single_texture_per_cube = True
        self.force_single_opaque_additive_texture = True
        self.cube_water_distort_strength = 0.185
        self.gravity_magnitude = 19.62
        self.gravity_blend_speed = 4.8
        self.gravity_tilt_degrees = 0.0
        self.gravity_rotate_speed = 0.0
        self.filters = None
        self.ball_shadow_np: NodePath | None = None
        self.ball_outline_np: NodePath | None = None
        self.audio_hyper_mix = 0.0
        self.ai_world = None
        self.ai_enabled = _env_flag("SOULSYM_USE_PANDA_AI", False) and AIWorld is not None and AICharacter is not None
        self._retained_ai_worlds: list = []
        self._retained_ai_chars: list = []

        self._setup_lights()
        self._setup_toon_rendering()

        self.box_model = self._load_first_model([
            "models/box",
            "models/box.egg",
            "models/misc/rgbCube",
        ])
        if self.box_model is None:
            self.box_model = self._create_fallback_cube_model()
        self.sphere_model = self._load_first_model([
            "models/misc/sphere",
            "models/misc/sphere.egg",
            "models/smiley",
        ])
        if self.sphere_model is None:
            self.sphere_model = self._create_fallback_sphere_model()
        self.box_model.clearTexture()
        self.box_norm_scale, self.box_norm_offset = self._compute_model_normalization(self.box_model)
        self.texture_cache: dict[str, Texture] = {}
        self._texture_prefetched = False
        self.lazy_vram_loading = True
        self.max_room_texture_variants = 8 if self.performance_mode else 32
        self.room_texture_paths: list[str] = []
        self.active_room_texture_paths: list[str] = []
        self.room_texture_vram_ratio = 0.25
        self.level_checker_tex = self._create_checker_texture(
            size=256,
            cells=8,
            color_a=(0.0, 0.0, 0.0, 1.0),
            color_b=(0.0, 0.0, 0.0, 0.0),
        )
        self.level_additive_stage = TextureStage("level-single-additive")
        self.level_additive_stage.setMode(TextureStage.MAdd)
        self.level_additive_stage.setSort(8)
        add_gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
        self.level_additive_stage.setColor((add_gain, add_gain, add_gain, 1.0))
        self.floor_fractal_tex_a = self._create_fractal_symmetry_texture(size=256, symmetry=6, seed=0.37)
        self.floor_fractal_tex_b = self._create_fractal_symmetry_texture(size=256, symmetry=9, seed=0.81)
        self.water_base_tex = self._load_water_base_texture()
        self.water_specular_tex = self._create_water_specular_texture(size=256, seed=0.41)
        self.floor_wet_shader = self._load_floor_wet_shader()
        self.water_surface_shader = None
        self.floor_shader_nodes: list[NodePath] = []
        self.water_shader_nodes: list[NodePath] = []
        self.floor_contact_uv = Vec2(0.0, 0.0)
        self.floor_contact_strength = 0.2
        self.floor_contact_decay = 3.6
        self.floor_uv_projection_scale = 30.0
        self.animate_non_water_uv = True
        self.water_overlay_min_area = 20.5
        self.water_level_offset = 0.0
        self.water_surface_raise = 0.028
        self.water_uv_repeat_scale = 32.0
        self.room_floor_uv_repeat_scale = 100.0
        self.water_wave_amplitude = 0.24
        self.water_wave_speed = 10.4
        self.water_wave_freq_x = 0.09
        self.water_wave_freq_y = 0.07
        self.water_specular_detail_repeat = 7.5
        self.water_specular_strength = 0.72
        self.water_specular_scroll_speed = 1.85
        self.water_additive_opacity_scale = 0.06
        self.water_additive_gain_scale = 0.08
        self.water_rainbow_strength = 0.12
        self.water_diffusion_strength = 0.38
        self.water_color_cycle_enabled = True
        self.water_color_cycle_speed = 5.8
        self.water_color_cycle_saturation = 0.92
        self.water_color_cycle_value = 1.0
        self.water_color_cycle_alpha = 0.16
        self.water_color_cycle_smoothness = 0.24
        self.water_emissive_linux_enabled = sys.platform.startswith("linux") and _env_flag("SOULSYM_WATER_EMISSIVE_LINUX", False)
        self.water_emissive_linux_scale = 1.35
        self.water_buoyancy_bias = 0.62
        self.water_buoyancy_strength = 2.2
        self.water_drag_planar = 0.85
        self.water_drag_vertical = 1.95
        self.water_loop_overscan = 6.0
        self.water_surfaces: list[dict] = []
        self.water_crystal_spawn_enabled = True
        self.water_crystal_spawn_interval = 0.42 if self.performance_mode else 0.24
        self.water_crystal_spawn_timer = 0.0
        self.water_crystal_max_count = 48 if self.performance_mode else 84
        self.water_crystal_fall_gravity = 14.5
        self.water_crystal_spawn_area_scale = 2.1
        self.water_crystal_spawn_height_min = 8.0
        self.water_crystal_spawn_height_max = 24.0
        self.water_crystal_stuck_duration = 0.9
        self.water_crystal_float_duration = 10.0
        self.water_crystal_fade_duration = 1.5
        self.water_crystals: list[dict] = []
        self.swap_floor_and_ceiling = False
        if not self.lazy_vram_loading:
            self._prefetch_texture_assets()
        self.room_textures = self._load_room_textures()
        room_fallback_tex = self._get_random_room_texture()
        if room_fallback_tex is not None:
            self.level_checker_tex = room_fallback_tex

        self.gen = GenerationConfig(
            scale=2.4,
            layout_mode="hexmix",
            snake_cell_size=8,
            snake_layers=1,
            maze_cell_size=32,
            maze_layers=1,
            maze_loop_chance=0.15,
            maze_vertical_link_chance=0.13,
            average_room_size=13.0,
            room_size_jitter=0.62,
            room_height=4096.0,
            corridor_width=15.5,
            room_density=0.68,
            corridor_density=0.64,
            decor_density=0.3,
            angled_room_ratio=0.2,
            max_rooms=64,
        )

        self.map_w = int(176 * self.gen.scale)
        self.map_d = int(176 * self.gen.scale)
        self.wall_h = self.gen.room_height
        self.floor_y = 0
        self.wall_t = self.gen.wall_thickness
        self.floor_t = self.gen.floor_thickness
        self.corridor_w = self.gen.corridor_width
        self.level_z_step = max(self.wall_h + 2.0, self.corridor_w + 3.0)
        self.flush_eps = 0.07

        avg_scaled = self.gen.average_room_size * self.gen.scale
        min_leaf = max(24, int(avg_scaled * 0.85))
        max_leaf = max(min_leaf + 10, int(avg_scaled * 1.45))
        generator = BSPGenerator(self.map_w, self.map_d, self.gen, min_leaf=min_leaf, max_leaf=max_leaf)
        generated_levels: dict[int, int] | None = None
        layout_mode = getattr(self.gen, "layout_mode", "bsp")
        if layout_mode == "snake3d":
            cell_size = int(max(34, min(68, getattr(self.gen, "snake_cell_size", 40))))
            layers = int(max(2, min(3, getattr(self.gen, "snake_layers", 2))))
            self.rooms, self.edges, generated_levels = generator.generate_snake3d(cell_size=cell_size, layers=layers)
        elif layout_mode == "maze3d":
            maze_cell = int(max(40, min(84, getattr(self.gen, "maze_cell_size", 48))))
            maze_layers = int(max(2, min(4, getattr(self.gen, "maze_layers", 2))))
            maze_loop = float(getattr(self.gen, "maze_loop_chance", 0.05))
            maze_vertical = float(getattr(self.gen, "maze_vertical_link_chance", 0.2))
            self.rooms, self.edges, generated_levels = generator.generate_maze3d(
                cell_size=maze_cell,
                layers=maze_layers,
                loop_chance=maze_loop,
                vertical_link_chance=maze_vertical,
            )
        elif layout_mode == "labyrinth":
            cell_size = int(max(32, min(58, avg_scaled * 1.35)))
            self.rooms, self.edges = generator.generate_labyrinth(cell_size=cell_size)
        elif layout_mode == "hexmix":
            cell_size = int(max(22, min(44, avg_scaled * 0.96)))
            self.rooms, self.edges = generator.generate_hex_mixed(cell_size=cell_size)
        else:
            self.rooms, self.edges = generator.generate()

        self.room_levels = generated_levels if generated_levels is not None else {idx: 0 for idx in range(len(self.rooms))}
        if len(self.rooms) < 2:
            fallback_cell = int(max(24, min(48, avg_scaled * 0.95)))
            if layout_mode == "hexmix":
                self.rooms, self.edges = generator.generate_hex_mixed(cell_size=fallback_cell)
            else:
                self.rooms, self.edges = generator.generate_labyrinth(cell_size=fallback_cell)
            self.room_levels = {idx: 0 for idx in range(len(self.rooms))}
        if layout_mode in ("maze3d", "snake3d", "labyrinth", "hexmix"):
            self._space_rooms_apart(min_gap=max(0.2, self.corridor_w * 0.04), iterations=3)
        else:
            self._space_rooms_apart(min_gap=max(1.4, self.corridor_w * 0.24), iterations=18)

        room_count = len(self.rooms)
        self._hyper_uv_update_interval_min = 0.0
        if room_count >= 56:
            self.gen.decor_density = min(self.gen.decor_density, 0.14)
            self.water_overlay_min_area = max(self.water_overlay_min_area, 4.5)
            self._hyper_uv_update_interval_min = 1.0 / 14.0
        elif room_count >= 40:
            self.gen.decor_density = min(self.gen.decor_density, 0.2)
            self.water_overlay_min_area = max(self.water_overlay_min_area, 3.4)

        self._reshape_edges_dungeon_style()
        self.start_room_idx = self._choose_start_room_index()
        self.room_doors: dict[int, dict[str, list[float]]] = {
            idx: {"left": [], "right": [], "bottom": [], "top": []} for idx in range(len(self.rooms))
        }
        self.room_gravity_zones: list[dict] = []
        self.active_room_zone_idx = -1
        self._setup_room_gravity_zones()
        self.room_dimension_fields: dict[int, dict] = {}
        self._setup_room_dimension_fields()
        self.room_bounds_cache = None
        self.zone_bounds_cache = None
        self.liminal_fold_nodes: list[dict] = []
        self.liminal_fold_links: dict[int, list[int]] = {}
        self.room_compression_pockets: list[dict] = []
        self._setup_room_compression_pockets()
        self.corridor_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        self.corridor_joints: list[tuple[float, float, float]] = []
        self.pending_floor_rects: list[tuple[float, float, float, float, float, tuple[float, float, float, float]]] = []
        self.pending_ceiling_rects: list[tuple[float, float, float, float, float, tuple[float, float, float, float]]] = []
        self.pending_floor_holes: list[tuple[float, float, float, float, float]] = []
        self.pending_ceiling_holes: list[tuple[float, float, float, float, float]] = []
        self.disable_floor_holes = False
        self.disable_ceiling_holes = False
        self.infinite_jumps = True
        self.infinite_level_mode = False
        self.exterior_chunk_size = 96.0
        self.exterior_chunk_z_step = max(14.0, self.level_z_step * 1.4)
        self.exterior_stream_radius_xy = 1
        self.exterior_stream_radius_z = 1
        self.exterior_stream_timer = 0.0
        self.exterior_stream_interval = 0.25
        self.exterior_chunks: dict[tuple[int, int, int], dict] = {}
        self.exterior_stream_backlog: deque[tuple[int, int, int]] = deque()
        self.exterior_stream_pending: set[tuple[int, int, int]] = set()
        self.exterior_gen_budget_tick = 1 if self.performance_mode else 2
        self.exterior_gen_budget_force = 8 if self.performance_mode else 14
        self.goal_np: NodePath | None = None
        self.goal_chunk_key: tuple[int, int, int] | None = None
        self.goal_pos = Vec3(0, 0, 0)
        self.goal_path_np: NodePath | None = None
        self.entrance_debug_np: NodePath | None = None
        self.goal_path_update_timer = 0.0
        self.goal_path_update_interval = 0.12
        self.outside_islands: list[dict] = []
        self.warp_links: list[dict] = []
        self.enable_room_fold_thread = _env_flag("SOULSYM_ROOM_FOLD_THREAD", False)
        self.room_fold_thread_ready = False
        self._room_fold_links_version = 0
        self._room_fold_links_thread_cache: list[tuple[int, float, float, float, float, float, float, float]] = []
        self._room_fold_links_lock = threading.Lock()
        self._room_fold_probe_queue: queue.Queue = queue.Queue(maxsize=1)
        self._room_fold_result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._room_fold_stop_event = threading.Event()
        self._room_fold_worker: threading.Thread | None = None
        self.allow_outside_island_warps = True
        self.mobius_twist_enabled = True
        self.mobius_loop_count = 2
        self.mobius_middle_band_ratio = 0.28
        self.mobius_twist_strength = 0.9
        self.room_landmarks: list[dict] = []
        self.phase_room_overlays: list[dict] = []
        self.landmark_model_candidates: list[str] | None = None
        self.landmark_fallback_model: str | None = None
        self.landmark_linux_axis_fix = sys.platform.startswith("linux") and _env_flag("SOULSYM_LANDMARK_LINUX_AXIS_FIX", True)
        self.landmark_linux_axis_fix_degrees = -90.0
        self.warp_cooldown = 0.0
        self.vertical_movers: list[dict] = []
        self.room_bob_amp = 0.0
        self.room_bob_speed = 0.36
        self.platform_bob_amp = 0.0
        self.platform_bob_speed = 0.72
        self.move_room_colliders = False
        self.vertical_mover_update_timer = 0.0
        self.vertical_mover_update_interval = 1.0 / 18.0 if self.performance_mode else 1.0 / 28.0
        self.vertical_mover_max_distance = 72.0
        self.platform_only_mode = False
        self.four_d_obstacle_arena_mode = True
        self.subtractive_maze_mode = False
        self.platform_course_count = 84 if self.performance_mode else 126
        self.platform_min_span = 2.8
        self.platform_max_span = 7.2
        self.platform_vertical_step = 2.6
        self.platform_overlap_padding = 1.2
        self.platform_guardrail_height = 0.55
        self.platform_guardrail_thickness = 0.16
        self.platform_mover_ratio = 0.34
        self.platform_loop_range = 18.0
        self.world_wrap_margin = 0.35
        self.platform_course_spawn_pos = Vec3(self.map_w * 0.5, self.map_d * 0.5, self.floor_y + 3.0)
        self.enable_inverted_level_echo = True
        self.inverted_level_echo_plane_z = self.floor_y + 12.0
        self.inverted_level_echo_extra_offset = 0.0
        self.inverted_level_echo_opacity = 0.46
        self.inverted_level_echo_root: NodePath | None = None
        self.arena_platform_points: list[Vec3] = []
        self.infinite_level_mode = bool(self.platform_only_mode)
        self.maze_portal_points: list[Vec3] = []
        self.fall_game_over_enabled = True
        self.fall_game_over_time = 1.35
        self.fall_game_over_min_drop = 4.8
        self.fall_game_over_speed = 6.8
        self.fall_air_timer = 0.0
        self.fall_reference_z = self.floor_y + 2.0
        self.game_over_active = False
        self.game_over_auto_restart_seconds = 10.0
        self.game_over_countdown = self.game_over_auto_restart_seconds
        self.game_over_ui: NodePath | None = None
        self.game_over_prompt_text_node: TextNode | None = None
        self.win_active = False
        self.win_ui: NodePath | None = None
        self.player_ai_enabled = True
        self.player_ai_idle_delay = 30.0
        self.player_ai_idle_timer = self.player_ai_idle_delay
        self.player_ai_target_id: int | None = None
        self.player_ai_target_w = 0.0
        self.player_ai_retarget_timer = 0.0
        self.player_ai_combo_step = 0
        self.player_ai_combo_timer = 0.0
        self.player_ai_jump_cooldown = 0.0
        self.player_ai_plane_height_tolerance = 2.6
        self.player_ai_below_seek_margin = 0.35
        self.player_ai_same_existence_w_tolerance = 1.35
        self.player_ai_engage_distance = 2.35
        self.player_ai_floor_tolerance = 2.6
        self.player_ai_lock_target_id: int | None = None
        self.player_ai_lock_on_distance = 3.2
        self.player_ai_lock_release_distance = 22.0
        self.player_ai_camera_target_pos: Vec3 | None = None
        self.player_ai_move_smooth = 7.8
        self.player_ai_target_hysteresis = 1.18
        self.player_ai_last_move_dir = Vec3(0, 1, 0)
        self.player_ai_room_path: list[int] = []
        self.player_ai_room_path_goal: int | None = None
        self.player_ai_room_path_recalc_timer = 0.0
        self.player_ai_room_path_recalc_interval = 0.28
        self.player_ai_room_waypoint_reach = 1.45
        self.player_ai_w_target_blend_rate = 3.6
        self.player_ai_w_mismatch_penalty = 0.72
        self.player_ai_fold_route_bonus = 1.15
        self.player_ai_fold_hop_weight = 0.48
        self.player_ai_jump_target_up_threshold = 0.55
        self.player_ai_jump_nav_up_threshold = 0.35
        self.player_ai_jump_probe_distance = 1.8
        self.player_ai_jump_chase_distance_mul = 1.75
        self.player_ai_jump_chase_chance_per_sec = 0.85
        self.player_ai_jump_roam_chance_per_sec = 0.22
        self.player_ai_jump_attack_up_threshold = 0.28
        self.player_ai_jump_attack_dist_mul = 1.9
        self.player_ai_black_hole_avoid_strength = 2.4
        self.player_ai_black_hole_avoid_margin = 1.18
        self.player_ai_black_hole_jump_margin = 0.42
        self.player_ai_bomb_range_mul = 1.9
        self.player_ai_bomb_desire = 0.42
        self.player_ai_bomb_cluster_radius_mul = 1.18
        self.player_ai_bomb_cluster_min = 2
        self.player_ai_missile_range_mul = 4.4
        self.player_ai_missile_desire = 0.58
        self.player_ai_missile_min_range_mul = 1.08
        self.player_ai_throw_range_mul = 3.2
        self.player_ai_throw_desire = 0.46
        self.player_ai_combat_hold_distance_mul = 1.35
        self.player_ai_combat_hold_height_tolerance = 1.25
        self.player_ai_combat_hold_move_scale = 0.2
        self.w_dimension_distance_scale = 4.0
        self.subtractive_drill_max_radius = 4

        self.pitch = 14.0
        self.heading = 0.0
        self.camera_distance = 3.4
        self.camera_min_distance = 0.22
        self.camera_collision_radius = 0.18
        self.camera_smooth_speed = 12.0
        self.camera_smoothed_pos: Vec3 | None = None
        self.camera_velocity = Vec3(0, 0, 0)
        self.camera_orbit_speed = 66.0
        self.camera_pitch_speed = 66.0
        self.camera_target_height = 0.32
        self.camera_look_above = 0.5
        self.camera_min_world_height = 0.4
        self.camera_collision_deadzone = 0.04
        self.camera_collision_lerp_speed = 4.0
        self.camera_free_lerp_speed = 4.0
        self.camera_last_safe_pos: Vec3 | None = None
        self.camera_manual_turn_hold = 0.0
        self.camera_auto_align_speed = 3.5
        self.camera_auto_align_min_speed = 0.08
        self.camera_height_offset = 2.35
        self.camera_height_base = self.camera_height_offset
        self.camera_fixed_world_z = 1.2
        self.camera_height_follow_factor = 1.5
        self.camera_follow_distance = 6.8
        self.camera_follow_distance_base = self.camera_follow_distance
        self.camera_fov_base = 108.0
        self.camera_dimension_blend_speed = 6.5
        self.space_compress_3d_strength = 0.1
        self.enable_dimensional_compression = True
        self.camera_tether_max_extra = 0.85
        self.camera_wall_rise_limit = 0.42
        self.camera_spring_stiffness = 62.0
        self.camera_spring_damping = 14.0
        self.camera_pass_through_walls = False
        self.camera_parented_to_ball = False
        self.enable_mouse_look = True
        self.mouse_look_sensitivity_x = 0.16
        self.mouse_look_sensitivity_y = 0.13
        self.mouse_look_smooth = 0.62
        self.mouse_look_invert_y = False
        self._mouse_turn_input = 0.0
        self._mouse_pitch_input = 0.0
        self._mouse_centered = False
        self.roll_force = 18.0
        self.roll_torque = 14.0
        self.max_ball_speed = 15.25
        self.ball_friction_default = 0.02
        self.ball_friction_shift = 0.05
        self.link_control_gain = 32.0
        self.link_brake_drag = 2.6
        self.compression_factor_smoothed = 1.0
        self.compression_smooth_speed = 8.0
        self.compression_pocket_radius_scale = 2.4
        self.compression_pocket_influence_power = 0.78
        self.compression_pocket_dilation_gain = 1.55
        self.compression_pocket_speed_bias = 0.24
        self.jump_impulse = 24.0
        self.jump_rise_boost = 1.22
        self.jump_float_duration = 0.56
        self.jump_float_drag = 5.8
        self.jump_float_timer = 0.0
        self.jump_queued = False
        self.max_jumps = 2
        self.jumps_used = 0
        self.grounded = False
        self.prev_grounded = False
        self.prev_ball_velocity = Vec3(0, 0, 0)
        self.last_move_dir = Vec3(0, 1, 0)
        self.camera_ball_clearance = 0.16
        self.player_w = 0.0
        self.hyper_w_speed = 2.65
        self.hyper_mouse_w_speed = 2.2
        self.hyper_turn_w_speed = 0.65
        self.hyper_force_strength = 42.0
        self.hyper_force_lift = 0.18
        self.hyper_w_limit = 7.2
        self.scroll_lift_target = 0.0
        self.scroll_lift_value = 0.0
        self.scroll_lift_step = 0.45
        self.scroll_lift_smooth = 7.0
        self.scroll_lift_decay = 1.5
        self.scroll_lift_up_force = 118.0
        self.scroll_lift_down_force = 34.0
        self.float_fall_drag = 2.2
        self.hyper_slice = 2.45
        self.hyper_falloff = 1.95
        self.hyper_height_taper_range = 1.2
        self.hyper_height_taper_strength = 1.8
        self.hyperspace_threshold = 0.2
        self.hyperspace_jump_impulse = 1.95
        self.hyperspace_bounce_gain = 0.9
        self.normal_restitution = 0.04
        self.hyperspace_restitution = 0.92
        self.hyper_enclosure_ids: set[int] = set()
        max_room_level = max(self.room_levels.values()) if self.room_levels else 0
        self.hyper_bounds_top_z = self._level_base_z(max_room_level) + self.wall_h + 4.0
        self.hyper_bounds_bottom_z = self.floor_y - 1.4
        self.room_hyper_layer_count = 4 if self.performance_mode else 8
        self.room_hyper_w_spacing = 2.6
        self.visual_w_map: dict[int, float] = {}
        self.monsters: list[dict] = []
        self.monster_max_hp = 100.0
        self.monster_giant_spawn_ratio = 0.08
        self.monster_giant_hp_mult = 6.5
        self.monster_fast_ratio = 0.22
        self.monster_fast_speed_min = 2.0
        self.monster_fast_speed_max = 3.4
        self.monster_contact_sfx_cooldown = 0.0
        self.hyper_uv_nodes: list[dict] = []
        self.dynamic_room_uv_nodes: list[dict] = []
        self.floor_contact_pulses: list[dict] = []
        self.floor_contact_emit_timer = 0.0
        self.floor_contact_emit_interval = 1.0 / 14.0
        self.star_particles: list[dict] = []
        self.hyper_uv_repeat_base = 2.15
        self.hyper_uv_repeat_min = 1.35
        self.hyper_uv_repeat_max = 4.8
        self.water_uv_active_radius = 110.0
        self.room_uv_grid_density = 0.32
        self.room_uv_grid_snap = 0.25
        self.room_uv_repeat_cache: dict[tuple[str, int, int], tuple[float, float]] = {}
        self.hyper_uv_update_timer = 0.0
        self.hyper_uv_update_interval = 1.0 / 24.0 if self.performance_mode else 1.0 / 48.0
        self.hyper_uv_update_interval = max(self.hyper_uv_update_interval, float(getattr(self, "_hyper_uv_update_interval_min", 0.0)))
        self.star_update_timer = 0.0
        self.occlusion_update_timer = 0.0
        self.occlusion_update_interval = 1.0 / 30.0
        self.transparency_update_timer = 0.0
        self.transparency_update_interval = 1.0 / 30.0
        self.scene_cull_timer = 0.0
        self.scene_cull_interval = 0.12 if self.performance_mode else 0.16
        self.scene_cull_hidden: set[int] = set()
        self.scene_cull_miss_counts: dict[int, int] = {}
        self.enable_scene_culling = True
        self.enable_wall_occlusion_culling = True
        self.scene_cull_ray_budget = 140 if self.performance_mode else 220
        self.scene_cull_occlusion_min_dist = 8.0
        self.camera_occlusion_alpha_min = 0.14
        self.camera_occlusion_alpha_range = 0.34
        self.camera_occlusion_alpha_smooth = 0.38
        self.cull_behind_camera_only = True
        self.show_portal_markers = False
        self.monster_anim_tick = 0
        self.shadow_update_timer = 0.0
        self.health_ui_update_timer = 0.0
        self.input_hud_enabled = True
        self.input_hud_buttons: dict[str, dict] = {}
        self.input_hud_mouse_state = {"mouse1": False, "mouse2": False, "mouse3": False}
        self.input_hud_r_state = False
        self.input_hud_r_pulse = 0.0
        self.holo_map_enabled = True
        self.holo_map_ui: NodePath | None = None
        self.holo_map_marker_root: NodePath | None = None
        self.holo_map_markers: list[NodePath] = []
        self.holo_map_radius_world = 36.0
        self.holo_map_update_timer = 0.0
        self.holo_map_update_interval = 1.0 / 14.0
        self.light_update_timer = 0.0
        self.light_update_interval = 1.0 / 20.0 if self.performance_mode else 1.0 / 48.0
        self.gravity_particles_update_timer = 0.0
        self.gravity_particles_update_interval = 1.0 / 30.0 if self.performance_mode else 1.0 / 45.0
        self.monster_collision_update_timer = 0.0
        self.monster_collision_update_interval = 1.0 / 24.0 if self.performance_mode else 1.0 / 48.0
        self.monster_ai_jump_enabled = True
        self.monster_ai_jump_impulse = 4.8
        self.monster_ai_jump_gravity = 12.0
        self.monster_ai_jump_cooldown_duration = 0.9

        self.level_texgen_mode = None
        self.ball_texgen_mode = None
        self.ball_tex_stage = TextureStage("ball-cube-uv")
        self.water_tex_stage = TextureStage("water-cube-uv")

        self.weapon_forward = Vec3(0, 1, 0)
        self.sword_anchor_pos: Vec3 | None = None
        self.sword_anchor_vel = Vec3(0, 0, 0)
        self.sword_reach = 1.8
        self.sword_forward_offset = 0.34
        self.sword_side_offset = 0.22
        self.sword_up_offset = 0.36
        self.sword_anchor_follow_speed = 19.0
        self.attack_mode = "idle"
        self.attack_timer = 0.0
        self.attack_cooldown = 0.0
        self.attack_hit_targets: set[int] = set()
        self.swing_duration = 0.18
        self.spin_duration = 0.32
        self.sword_upgrade_level = 0
        self.sword_damage_multiplier = 1.0
        self.sword_damage_per_pickup = 0.16
        self.sword_damage_multiplier_cap = 4.0
        self.pickup_attract_radius = 3.1
        self.pickup_attract_speed = 8.2
        self.player_attack_stat = 0
        self.player_defense_stat = 0
        self.player_dex_stat = 0
        self.player_sta_stat = 0
        self.player_int_stat = 0
        self.attack_cooldown_multiplier = 1.0
        self.sword_reach_multiplier = 1.0
        self.combat_damage_multiplier = 1.0
        self.damage_taken_multiplier = 1.0
        self.critical_hit_base_chance = 0.08
        self.critical_hit_multiplier = 2.1
        self.critical_knockback_multiplier = 2.75
        self.critical_chance_bonus_permanent = 0.0
        self.critical_chance_bonus_temp_bonus = 0.14
        self.critical_hit_chance_current = self.critical_hit_base_chance
        self.skill_buffs = {
            "haste": 0.0,
            "longblade": 0.0,
            "fury": 0.0,
            "critical": 0.0,
        }
        self.sword_pickup_serial = 0
        self.sword_powerups: list[dict] = []
        self.hyperspace_gravity_hold = False
        self.zero_g_mode = False
        self.space_boost_impulse = 2.4
        self.shift_brake_drag = 4.1
        self.player_hp_max = 100.0
        self.player_hp = self.player_hp_max
        self.monsters_total = 0
        self.monsters_slain = 0
        self.player_damage_cooldown = 0.0
        self.kill_protection_stacks = 0
        self.kill_protection_max_stacks = 24
        self.kill_protection_light = None
        self.kill_protection_light_np: NodePath | None = None
        self.kill_protection_root: NodePath | None = None
        self.kill_protection_rings: list[dict] = []
        self.floating_texts: list[dict] = []
        self.hyperbomb_active = False
        self.hyperbomb_origin = Vec3(0, 0, 0)
        self.hyperbomb_timer = 0.0
        self.hyperbomb_duration = 6.0
        self.hyperbomb_spawn_duration = 2.2
        self.hyperbomb_spawn_timer = 0.0
        self.hyperbomb_spawn_interval_fast = 0.022
        self.hyperbomb_spawn_interval_slow = 0.16
        self.hyperbomb_sphere_life = 2.6
        self.hyperbomb_scale_start_factor = 0.18
        self.hyperbomb_scale_start = 0.045
        self.hyperbomb_growth_speed = 18.5
        self.hyperbomb_growth_slowdown = 4.2
        self.hyperbomb_max_scale_factor = 8.2
        self.hyperbomb_max_scale = 64.0
        self.hyperbomb_damage_radius_factor = 1.0
        self.hyperbomb_damage_interval = 0.08
        self.hyperbomb_damage_per_tick = 36.0
        self.hyperbomb_damage_timer = 0.0
        self.hyperbomb_alpha_log_k = 9.0
        self.hyperbomb_alpha_min = 0.12
        self.hyperbomb_cooldown = 2.0
        self.hyperbomb_cooldown_duration = 1.75
        self.hyperbomb_spheres: list[dict] = []
        self.hyperbomb_audio_nodes: list[dict] = []
        self.magic_missiles: list[dict] = []
        self.magic_missile_cooldown = 0.0
        self.magic_missile_cooldown_duration = 4.5
        self.magic_missile_cast_count = 6
        self.magic_missile_life = 4.2
        self.magic_missile_speed = 25.0
        self.magic_missile_turn_rate = 7.6
        self.magic_missile_damage = 34.0
        self.magic_missile_retarget_interval = 0.12
        self.magic_missile_color_cycle_rate = 11.5
        self.magic_missile_emissive_strength = 1.0
        self.magic_missile_arc_height = 0.9
        self.magic_missile_arc_drop = 1.15
        self.magic_missile_trails: list[dict] = []
        self.magic_missile_trail_emit_interval = 0.02
        self.magic_missile_template: NodePath | None = None
        self.magic_missile_cylinder_model: NodePath | None = None
        self.magic_missile_cone_model: NodePath | None = None
        self.health_powerups: list[dict] = []
        self.black_holes: list[dict] = []
        self.black_hole_distort_intensity = 0.0
        self.black_hole_count = 14 if self.performance_mode else 28
        self.black_hole_blower_ratio = 0.46
        self.black_hole_pull_strength = 292.0
        self.black_hole_influence_radius = 15.5
        self.black_hole_roam_speed = 2.1
        self.black_hole_visual_radius = 2.05
        self.black_hole_suck_max_force = 165.0
        self.black_hole_suck_escape_soften = 0.38
        self.black_hole_monster_force_scale = 0.032
        self.black_hole_monster_warp_cooldown = 2.4
        self.black_hole_monster_spit_speed = 8.2
        self.black_hole_suck_outline_tex = None
        self.black_hole_blow_dial_tex = None
        self.black_hole_suck_sfx_path = None
        self.black_hole_blow_sfx_path = None
        self.black_hole_sfx_volume = 0.9
        self.black_hole_sfx_min_distance = 3.0
        self.black_hole_sfx_max_distance = 520.0
        self.black_hole_doppler_exaggeration = 13.5
        self.timespace_tone = None
        self.timespace_tone_target_rate = 1.0
        self.timespace_tone_current_rate = 1.0
        self.timespace_tone_target_volume = 0.0
        self.timespace_tone_current_volume = 0.0
        self.timespace_tone_rate_smooth = 5.2
        self.timespace_tone_volume_smooth = 3.8
        self.roll_time = 0.0
        self.dungeon_build_count = 0
        self.dungeon_built_once = False
        self._setup_liminal_fold_graph()
        self._rebuild_spatial_acceleration_caches()

        print(
            "[accel] numpy:",
            "on" if NUMPY_AVAILABLE else "off",
            "| numba:",
            "on" if NUMBA_AVAILABLE else "off",
        )

        self._build_dungeon()
        self._setup_outside_islands_and_warps()
        self._setup_phase_room_overlays()
        self._start_room_fold_worker_if_needed()
        self._setup_room_landmarks()
        self._spawn_player_ball()
        self._spawn_roaming_black_holes(self.black_hole_count)
        if hasattr(self, "ball_np") and self.ball_np is not None and hasattr(self, "platform_course_spawn_pos"):
            spawn = Vec3(self.platform_course_spawn_pos)
            self.ball_np.setPos(spawn)
            self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
            self.ball_body.setAngularVelocity(Vec3(0, 0, 0))
        self._setup_kill_protection_system()
        self._initialize_infinite_world_goal()
        self._setup_weapon()
        self._setup_magic_missile_visuals()
        self._spawn_hypercube_monsters(count=64)
        self._setup_monster_ai_system()
        self._setup_ball_outline()
        if self.enable_ball_shadow:
            self._setup_ball_shadow()
        self._setup_orbit_lights()
        self._setup_camera()
        if hasattr(self, "camLens") and self.camLens is not None:
            try:
                self.camLens.setFov(self.camera_fov_base)
            except Exception:
                pass
        self._setup_mouse_look()
        self._setup_video_distortion()
        self._setup_player_health_ui()
        self._setup_input_hud()
        self._setup_holographic_map_ui()
        self._setup_game_over_ui()
        self._setup_win_ui()
        if self.enable_particles:
            self._setup_star_particles()
        if self.enable_gravity_particles:
            self._setup_room_gravity_particles()
        self._spawn_health_powerups(count=8 if self.performance_mode else 16)
        self._setup_inverted_level_echo()

        self.audio3d = Audio3DManager.Audio3DManager(self.sfxManagerList[0], self.camera)
        self.audio3d.setDistanceFactor(20.0)
        self.audio3d.setDopplerFactor(20.1)
        self.audio3d.setDropOffFactor(10.0)
        if hasattr(self.audio3d, "attachListener"):
            self.audio3d.attachListener(self.camera)
        if hasattr(self.audio3d, "setListenerVelocityAuto"):
            self.audio3d.setListenerVelocityAuto()
        self._attach_black_hole_loop_sounds()

        self.keys = {k: False for k in ["w", "a", "s", "d"]}
        for key in self.keys:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])
        self.camera_keys = {k: False for k in ["arrow_left", "arrow_right", "arrow_up", "arrow_down"]}
        for key in self.camera_keys:
            self.accept(key, self._set_camera_key, [key, True])
            self.accept(f"{key}-up", self._set_camera_key, [key, False])
        self.hyper_keys = {k: False for k in ["q", "e"]}
        for key in self.hyper_keys:
            self.accept(key, self._set_hyper_key, [key, True])
            self.accept(f"{key}-up", self._set_hyper_key, [key, False])
        self.accept("space", self._queue_jump)
        self.accept("wheel_up", self._on_mouse_wheel, [1.0])
        self.accept("wheel_down", self._on_mouse_wheel, [-1.0])
        self.accept("mouse1", self._on_mouse1_pressed)
        self.accept("mouse1-up", self._on_mouse_button_released, ["mouse1"])
        self.accept("mouse2", self._on_mouse2_pressed)
        self.accept("mouse2-up", self._on_mouse_button_released, ["mouse2"])
        self.accept("mouse3", self._on_mouse3_pressed)
        self.accept("mouse3-up", self._on_mouse_button_released, ["mouse3"])
        self.accept("lshift", self._set_hyperspace_gravity_hold, [True])
        self.accept("lshift-up", self._set_hyperspace_gravity_hold, [False])
        self.accept("rshift", self._set_hyperspace_gravity_hold, [True])
        self.accept("rshift-up", self._set_hyperspace_gravity_hold, [False])
        self.accept("1", self._toggle_performance_mode)
        self.accept("r", self._on_r_pressed)
        self.accept("r-up", self._on_r_released)
        self.accept("escape", self._on_escape_pressed)

        self.sfx_roll = self._load_first_sfx([
            "qigongbell",
            "qigong_bell",
            "audio/qigongbell",
            "soundfx/qigongbell",
            "sfx/qigongbell",
        ])
        self.sfx_hit = self._load_first_sfx([
            "qigonghit",
            "qigong_hit",
            "audio/qigonghit",
            "soundfx/qigonghit",
            "sfx/qigonghit",
        ])
        self.sfx_jump = self._load_first_sfx([
            "qigongbounce",
            "quigongbounce",
            "qigong_bounce",
            "audio/qigongbounce",
            "audio/quigongbounce",
            "soundfx/qigongbounce",
            "soundfx/quigongbounce",
            "sfx/qigongbounce",
            "sfx/quigongbounce",
        ])
        self.sfx_qigong_jump = self._load_first_sfx_2d([
            "qigongjump",
            "qigong_jump",
            "audio/qigongjump",
            "audio/qigong_jump",
            "soundfx/qigongjump",
            "soundfx/qigong_jump",
            "sfx/qigongjump",
            "sfx/qigong_jump",
        ])
        self.sfx_pickup = self._load_first_sfx_2d([
            "pickuphealth",
            "pickup_health",
            "pickup",
            "healthpickup",
            "soundfx/pickuphealth",
            "sfx/pickuphealth",
        ])
        self.sfx_attack = self._load_first_sfx_2d([
            "attack",
            "soundfx/attack",
            "sfx/attack",
        ])
        self.sfx_attack_homingmissile = self._load_first_sfx_2d([
            "attackhomingmissle",
            "attack_homingmissle",
            "attackhomingmissile",
            "attack_homingmissile",
            "soundfx/attackhomingmissle",
            "soundfx/attack_homingmissle",
            "soundfx/attackhomingmissile",
            "soundfx/attack_homingmissile",
            "sfx/attackhomingmissle",
            "sfx/attack_homingmissle",
            "sfx/attackhomingmissile",
            "sfx/attack_homingmissile",
        ])
        self.sfx_attack_spin = self._load_first_sfx_2d([
            "attackspin",
            "soundfx/attackspin",
            "sfx/attackspin",
        ])
        self.sfx_monster_hit = self._load_first_sfx_2d([
            "monsterhit",
            "soundfx/monsterhit",
            "sfx/monsterhit",
        ])
        self.sfx_monster_guard = self._load_first_sfx_2d([
            "monsterguard",
            "monster_guard",
            "soundfx/monsterguard",
            "soundfx/monster_guard",
            "sfx/monsterguard",
            "sfx/monster_guard",
        ])
        self.sfx_monster_die = self._load_first_sfx_2d([
            "monsterdie",
            "soundfx/monsterdie",
            "sfx/monsterdie",
        ])
        self.sfx_critical_damage_bank = self._load_critical_damage_sfx_bank()
        self.sfx_critical_damage_last = None
        self.sfx_kill_bank = self._load_kill_sfx_bank()
        self.sfx_kill_last = None
        self.sfx_game_over_bank = self._load_game_over_sfx_bank()
        self.sfx_game_over_last = None
        self.sfx_win_bank = self._load_win_sfx_bank()
        self.sfx_win_last = None
        self.voiceover_volume_scale = 0.5
        self.sfx_attackbomb_path = self._resolve_attackbomb_path()
        self.sfx_monster_hum_path = self._resolve_monster_hum_path()
        self.black_hole_suck_sfx_path = self._resolve_black_hole_sfx_path("suck")
        self.black_hole_blow_sfx_path = self._resolve_black_hole_sfx_path("blow")
        self.sfx_monster_hum_is_idle = bool(
            self.sfx_monster_hum_path
            and "monsteridle" in os.path.basename(self.sfx_monster_hum_path).lower()
        )
        self._attach_black_hole_loop_sounds()
        self.hit_cooldown = 0.0
        self.bgm_track = None
        self.bgm_track_path = None
        self.bgm_volume = 0.3

        if self.sfx_roll:
            self.audio3d.attachSoundToObject(self.sfx_roll, self.ball_np)
            self.audio3d.setSoundMinDistance(self.sfx_roll, 1.2)
            self.audio3d.setSoundMaxDistance(self.sfx_roll, 65.0)
            if hasattr(self.audio3d, "setSoundVelocityAuto"):
                self.audio3d.setSoundVelocityAuto(self.sfx_roll)
        if self.sfx_hit:
            self.audio3d.attachSoundToObject(self.sfx_hit, self.ball_np)
            self.audio3d.setSoundMinDistance(self.sfx_hit, 1.0)
            self.audio3d.setSoundMaxDistance(self.sfx_hit, 90.0)
            if hasattr(self.audio3d, "setSoundVelocityAuto"):
                self.audio3d.setSoundVelocityAuto(self.sfx_hit)
        if self.sfx_jump:
            self.audio3d.attachSoundToObject(self.sfx_jump, self.ball_np)
            self.audio3d.setSoundMinDistance(self.sfx_jump, 1.0)
            self.audio3d.setSoundMaxDistance(self.sfx_jump, 85.0)
            if hasattr(self.audio3d, "setSoundVelocityAuto"):
                self.audio3d.setSoundVelocityAuto(self.sfx_jump)

        if self.sfx_roll:
            self.sfx_roll.setLoop(True)
            self.sfx_roll.setVolume(0.0)

        self._start_random_bgm_loop()

        self._setup_timespace_tone()

        self._attach_monster_hum_sounds()
        self._print_scene_graph_resources()

        self.taskMgr.add(self.update, "update")

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.62, 0.67, 0.74, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        self.ambient_light = ambient
        self.ambient_light_np = ambient_np

        sun = DirectionalLight("sun")
        sun.setColor((0.9, 0.97, 1.0, 1))
        if self.enable_dynamic_shadows:
            sun.setShadowCaster(True, 2048, 2048)
        sun.getLens().setNearFar(2, 260)
        sun.getLens().setFilmSize(120, 120)
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(18, -62, 0)
        self.render.setLight(sun_np)
        self.sun_light = sun
        self.sun_light_np = sun_np
        self.sun_default_pos = Vec3(sun_np.getPos())
        self.sun_default_hpr = Vec3(sun_np.getHpr())

        self.up_lamp = PointLight("up-lamp")
        self.up_lamp.setColor((0.6, 0.98, 1.0, 1))
        if self.enable_dynamic_shadows:
            self.up_lamp.setShadowCaster(True, 1024, 1024)
        self.up_lamp.setAttenuation((1.0, 0.012, 0.001))
        self.up_lamp_np = self.render.attachNewNode(self.up_lamp)
        cx = getattr(self, "map_w", 220) * 0.5
        cy = getattr(self, "map_d", 220) * 0.5
        cz = getattr(self, "wall_h", 18.5) * 0.9
        self.up_lamp_np.setPos(cx, cy, cz)
        self.render.setLight(self.up_lamp_np)

        self.lower_lamp = PointLight("lower-lamp")
        self.lower_lamp.setColor((0.05, 0.07, 0.14, 1))
        self.lower_lamp.setAttenuation((1.0, 0.04, 0.012))
        self.lower_lamp_np = self.render.attachNewNode(self.lower_lamp)
        self.lower_lamp_np.setPos(cx, cy, -0.8)
        self.render.setLight(self.lower_lamp_np)

        self._ensure_performance_player_light()

    def _remove_performance_player_light(self) -> None:
        perf_np = getattr(self, "performance_player_light_np", None)
        if perf_np is not None and not perf_np.isEmpty():
            try:
                self.render.clearLight(perf_np)
            except Exception:
                pass
            perf_np.removeNode()
        self.performance_player_light = None
        self.performance_player_light_np = None

    def _ensure_performance_player_light(self) -> None:
        should_enable = bool(getattr(self, "performance_mode", False)) and bool(getattr(self, "performance_player_light_enabled", True))
        if not should_enable:
            self._remove_performance_player_light()
            return

        perf_np = getattr(self, "performance_player_light_np", None)
        if perf_np is not None and not perf_np.isEmpty():
            return

        perf_light = PointLight("perf-player-light")
        perf_light.setColor((0.95, 0.7, 1.0, 1.0))
        perf_light.setAttenuation((1.0, 0.06, 0.008))
        perf_np = self.render.attachNewNode(perf_light)
        if hasattr(self, "ball_np") and self.ball_np is not None and not self.ball_np.isEmpty():
            perf_np.setPos(self.ball_np.getPos(self.render))
        self.render.setLight(perf_np)
        self.performance_player_light = perf_light
        self.performance_player_light_np = perf_np

    def _print_scene_graph_resources(self) -> None:
        verbose_stats = _env_flag("SOULSYM_VERBOSE_SCENE_STATS", False)
        print("\n=== Scene Graph Resource Summary ===")
        print(f"Performance mode: {self.performance_mode}")
        print(f"Dynamic shadows: {self.enable_dynamic_shadows}")
        print(f"Video distortion: {self.enable_video_distortion} (ready={self.video_distort_ready})")
        if getattr(self, "video_distort_error", ""):
            print(f"Video distortion error: {self.video_distort_error}")
        print(f"Scene culling: {self.enable_scene_culling}")
        print(f"Ball shadow: {self.enable_ball_shadow}")
        print(f"Occlusion outlines: {self.enable_occlusion_outlines}")
        print(f"Physics substeps: {self.physics_substeps} @ {self.physics_fixed_timestep:.5f}s")
        if self.win is not None:
            print(f"Window: {self.win.getXSize()}x{self.win.getYSize()}")
        print(f"Physics bodies: {len(self.physics_nodes)}")
        print(f"Scene visuals: {len(self.scene_visuals)}")
        print(f"Monster count: {len(self.monsters)}")
        print(f"Room count: {len(self.rooms)}")
        print(f"Compression pockets: {len(self.room_compression_pockets)} | dimensional compression: {self.enable_dimensional_compression}")
        if hasattr(self, "ball_np") and self.ball_np is not None and not self.ball_np.isEmpty():
            sample_cf = self._compression_factor_at(self.ball_np.getPos(), self.roll_time)
            print(f"Compression factor at spawn: {sample_cf:.3f}")
        print(f"Texture cache entries: {len(self.texture_cache)}")
        print(f"Water base texture: {getattr(self, 'water_base_tex_path', 'procedural')}")
        mode_counts: dict[str, int] = {}
        for entry in self.dynamic_room_uv_nodes:
            mode = str(entry.get("mode", "?"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        print(f"Dynamic UV nodes: {len(self.dynamic_room_uv_nodes)} {mode_counts}")
        if verbose_stats:
            print("--- render.analyze() ---")
            self.render.analyze()
            print("--- world.analyze() ---")
            self.world.analyze()
        else:
            print("Scene analyze: skipped (set SOULSYM_VERBOSE_SCENE_STATS=1 to enable detailed analyze())")
        print("=== End Scene Graph Resource Summary ===\n")

    def _resolve_monster_hum_path(self) -> str | None:
        candidates = [
            "monsteridle.wav",
            "soundfx/monsteridle.wav",
            "sfx/monsteridle.wav",
            "monsteridle.ogg",
            "soundfx/monsteridle.ogg",
            "sfx/monsteridle.ogg",
            "monsterhum.wav",
            "soundfx/monsterhum.wav",
            "sfx/monsterhum.wav",
            "monsterhum.ogg",
            "soundfx/monsterhum.ogg",
            "sfx/monsterhum.ogg",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _resolve_attackbomb_path(self) -> str | None:
        candidates = [
            "attackbomb.wav",
            "soundfx/attackbomb.wav",
            "sfx/attackbomb.wav",
            "attackbomb.ogg",
            "soundfx/attackbomb.ogg",
            "sfx/attackbomb.ogg",
            "attackbomb.mp3",
            "soundfx/attackbomb.mp3",
            "sfx/attackbomb.mp3",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _resolve_black_hole_sfx_path(self, kind: str) -> str | None:
        key = str(kind).strip().lower()
        if key == "suck":
            stems = [
                "sucker",
                "suck",
                "blackholesucker",
                "blackhole_sucker",
                "blackhole_suck",
            ]
        else:
            stems = [
                "blower",
                "blow",
                "blackholeblower",
                "blackhole_blower",
                "blackhole_blow",
            ]

        candidates: list[str] = []
        exts = ["wav", "ogg", "mp3"]
        prefixes = ["", "soundfx/", "sfx/"]
        for stem in stems:
            for prefix in prefixes:
                for ext in exts:
                    candidates.append(f"{prefix}{stem}.{ext}")

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _attach_black_hole_loop_sounds(self) -> None:
        if not hasattr(self, "audio3d") or self.audio3d is None:
            return
        if not getattr(self, "black_holes", None):
            return

        for entry in self.black_holes:
            if entry.get("loop_sfx") is not None:
                continue
            root = entry.get("root")
            if root is None or root.isEmpty():
                continue
            kind = str(entry.get("kind", "suck")).lower()
            sfx_path = self.black_hole_suck_sfx_path if kind == "suck" else self.black_hole_blow_sfx_path
            if not sfx_path:
                continue
            try:
                sound = self.audio3d.loadSfx(sfx_path)
            except Exception:
                sound = None
            if not sound:
                continue
            self.audio3d.attachSoundToObject(sound, root)
            self.audio3d.setSoundMinDistance(sound, float(getattr(self, "black_hole_sfx_min_distance", 5.0)))
            self.audio3d.setSoundMaxDistance(sound, float(getattr(self, "black_hole_sfx_max_distance", 280.0)))
            if hasattr(self.audio3d, "setSoundVelocityAuto"):
                self.audio3d.setSoundVelocityAuto(sound)
            sound.setLoop(True)
            sound.setVolume(float(getattr(self, "black_hole_sfx_volume", 0.72)))
            sound.setPlayRate(1.0)
            sound.play()
            entry["loop_sfx"] = sound

    def _distance4d_sq(self, pos_a: Vec3, w_a: float, pos_b: Vec3, w_b: float) -> float:
        d3_sq = (Vec3(pos_b) - Vec3(pos_a)).lengthSquared()
        w_scale = max(0.1, float(getattr(self, "w_dimension_distance_scale", 4.0)))
        dw = (float(w_b) - float(w_a)) * w_scale
        return d3_sq + dw * dw

    def _distance4d(self, pos_a: Vec3, w_a: float, pos_b: Vec3, w_b: float) -> float:
        return math.sqrt(max(0.0, self._distance4d_sq(pos_a, w_a, pos_b, w_b)))

    def _attach_monster_hum_sounds(self) -> None:
        if not hasattr(self, "audio3d") or not self.monsters:
            return
        if not self.sfx_monster_hum_path:
            return

        for monster in self.monsters:
            if monster.get("hum_sfx") is not None:
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            hum = self.audio3d.loadSfx(self.sfx_monster_hum_path)
            if not hum:
                continue
            self.audio3d.attachSoundToObject(hum, root)
            if bool(getattr(self, "sfx_monster_hum_is_idle", False)):
                self.audio3d.setSoundMinDistance(hum, 12.0)
                self.audio3d.setSoundMaxDistance(hum, 240.0)
            else:
                self.audio3d.setSoundMinDistance(hum, 2.6)
                self.audio3d.setSoundMaxDistance(hum, 52.0)
            if hasattr(self.audio3d, "setSoundVelocityAuto"):
                self.audio3d.setSoundVelocityAuto(hum)
            hum.setLoop(True)
            hum.setVolume(0.0)
            monster["hum_sfx"] = hum
            monster["hum_active"] = False

    def _update_monster_hum(self, monster: dict, ball_pos: Vec3) -> None:
        hum = monster.get("hum_sfx")
        if not hum:
            return
        root = monster.get("root")
        if root is None or root.isEmpty() or monster.get("dead", False):
            if monster.get("hum_active", False):
                hum.stop()
                monster["hum_active"] = False
            return

        player_w = float(getattr(self, "player_w", 0.0))
        monster_w = float(monster.get("w", 0.0))
        dist = self._distance4d(root.getPos(), monster_w, ball_pos, player_w)
        use_idle_hum = bool(getattr(self, "sfx_monster_hum_is_idle", False))
        max_dist = 220.0 if use_idle_hum else 44.0
        if dist >= max_dist or root.isStashed():
            if monster.get("hum_active", False):
                hum.stop()
                monster["hum_active"] = False
            return

        strength = 1.0 - (dist / max_dist)
        strength = max(0.0, min(1.0, strength))
        if use_idle_hum:
            target_vol = 0.095 + 0.46 * (strength ** 1.0)
        else:
            target_vol = 0.02 + 0.12 * strength
        mix = max(0.0, min(1.0, float(getattr(self, "audio_hyper_mix", 0.0))))
        target_vol = target_vol * (1.0 - 0.3 * mix) + 0.04 * mix
        if not monster.get("hum_active", False):
            hum.play()
            monster["hum_active"] = True
        hum.setVolume(max(0.0, min(0.58 if use_idle_hum else 0.2, target_vol)))
        base_rate = 0.9 + 0.06 * math.sin(self.roll_time * 1.9 + monster.get("phase", 0.0))
        hum.setPlayRate(base_rate * (1.0 - 0.28 * mix) + 1.0 * (0.28 * mix))

    def _setup_camera(self) -> None:
        setup_camera(self)
        if hasattr(self, "camLens") and self.camLens is not None:
            try:
                fov = self.camLens.getFov()
                self.camera_fov_base = float(fov[0]) if len(fov) > 0 else float(self.camera_fov_base)
            except Exception:
                pass
        if not getattr(self, "camera_parented_to_ball", False):
            try:
                self.camera.wrtReparentTo(self.render)
            except Exception:
                self.camera.reparentTo(self.render)
        if getattr(self, "camera_parented_to_ball", False) and hasattr(self, "ball_np"):
            self.camera_anchor = self.ball_np.attachNewNode("camera-anchor")
            self.camera_anchor.setCompass(self.render)
            self.camera.reparentTo(self.camera_anchor)
            gravity_up = self._get_gravity_up()
            ref_forward = Vec3(0, 1, 0)
            if abs(ref_forward.dot(gravity_up)) > 0.95:
                ref_forward = Vec3(1, 0, 0)
            ref_forward = ref_forward - gravity_up * ref_forward.dot(gravity_up)
            if ref_forward.lengthSquared() < 1e-8:
                ref_forward = Vec3(1, 0, 0)
            ref_forward.normalize()

            heading_offset = 180.0 if self._is_ceiling_mode() else 0.0
            yaw = math.radians(self.heading + heading_offset)
            orbit_planar = self._rotate_around_axis(-ref_forward, gravity_up, yaw)
            if orbit_planar.lengthSquared() < 1e-8:
                orbit_planar = Vec3(0, -1, 0)
            else:
                orbit_planar.normalize()
            cam_offset = orbit_planar * self.camera_follow_distance + gravity_up * self.camera_height_offset
            self.camera.setPos(cam_offset)
            self.camera.lookAt(self.ball_np.getPos(self.render) + gravity_up * self.camera_height_offset, gravity_up)
            self.camera_smoothed_pos = Vec3(self.camera.getPos(self.render))
        if hasattr(self, "fog") and self.fog is not None:
            self.fog.setColor(0.035, 0.045, 0.075)
            self.fog.setLinearRange(6, 40)
        if hasattr(self, "ball_np"):
            ball_pos = self.ball_np.getPos()
            self.camera.setPos(self._clamp_camera_to_current_room_bounds(self.camera.getPos(), ball_pos))

    def _setup_video_distortion(self) -> None:
        if not getattr(self, "enable_video_distortion", True):
            self.video_distort_ready = False
            return
        self._teardown_video_distortion()
        shader_vert = "graphics/shaders/viscous_distort.vert"
        shader_frag = "graphics/shaders/viscous_distort.frag"
        if not os.path.exists(shader_vert) or not os.path.exists(shader_frag):
            self.video_distort_ready = False
            self.video_distort_error = "distortion shader file missing"
            return

        last_error = "overlay setup failed"
        for attempt in range(2):
            try:
                width = max(640, self.win.getXSize())
                height = max(360, self.win.getYSize())
                self.video_distort_window_size = (int(width), int(height))
                manager = FilterManager(self.win, self.cam)
                tex = Texture("video-distort-buffer-tex")
                overlay = manager.renderSceneInto(colortex=tex)
                if overlay is None or overlay.isEmpty():
                    raise RuntimeError("overlay setup failed")
                overlay.setPos(0.0, 0.0, 0.0)
                overlay.setScale(1.0, 1.0, 1.0)
                overlay.setDepthWrite(False)
                overlay.setDepthTest(False)
                overlay.setBin("fixed", 100)

                tex_w = max(1, int(tex.getXSize()))
                tex_h = max(1, int(tex.getYSize()))
                self.video_distort_tex_size = (tex_w, tex_h)

                self.video_distort_shader = Shader.load(Shader.SL_GLSL, shader_vert, shader_frag)
                overlay.setShader(self.video_distort_shader)
                overlay.setShaderInput("tx", tex)
                overlay.setShaderInput("screen_tex", tex)
                overlay.setShaderInput("u_resolution", Vec2(float(tex_w), float(tex_h)))
                overlay.setShaderInput("u_time", 0.0)
                overlay.setShaderInput("u_speed", 0.15)
                overlay.setShaderInput("u_strength", self.video_distort_strength)
                overlay.setShaderInput("u_bloom_strength", self.video_bloom_strength)
                overlay.setShaderInput("u_bloom_radius", self.video_bloom_radius)
                overlay.setShaderInput("u_bloom_threshold", self.video_bloom_threshold)

                self.video_distort_buffer = tex
                self.video_distort_post_manager = manager
                self.video_distort_overlay_np = overlay
                self.video_distort_ready = True
                self.video_distort_error = ""
                return
            except Exception as exc:
                last_error = str(exc)
                self._teardown_video_distortion()
                if attempt == 0 and self.filters is not None and "displayregion" in last_error.lower():
                    try:
                        cleanup = getattr(self.filters, "cleanup", None)
                        if callable(cleanup):
                            cleanup()
                    except Exception:
                        pass
                    self.filters = None
                    continue
                break

        self.video_distort_ready = False
        self.video_distort_error = last_error
        if self.filters is None and not getattr(self, "safe_render_mode", False):
            self._enable_toon_filter_fallback()
            if self.filters is not None:
                self.video_distort_error = f"{last_error} | fallback: toon filter"

    def _load_floor_wet_shader(self):
        if getattr(self, "safe_render_mode", False):
            return None
        shader_vert = "graphics/shaders/floor_wet.vert"
        shader_frag = "graphics/shaders/floor_wet.frag"
        if not os.path.exists(shader_vert) or not os.path.exists(shader_frag):
            return None
        try:
            return Shader.load(Shader.SL_GLSL, shader_vert, shader_frag)
        except Exception:
            return None

    def _load_water_surface_shader(self):
        shader_vert = "graphics/shaders/water_surface.vert"
        shader_frag = "graphics/shaders/water_surface.frag"
        if not os.path.exists(shader_vert) or not os.path.exists(shader_frag):
            return None
        try:
            return Shader.load(Shader.SL_GLSL, shader_vert, shader_frag)
        except Exception:
            return None

    def _update_floor_wet_shader_inputs(self, dt: float, grounded_contact: Vec3 | None, speed: float) -> None:
        if self.floor_wet_shader is None:
            return

        if grounded_contact is not None:
            target_uv = Vec2(
                grounded_contact.x * self.floor_uv_projection_scale,
                grounded_contact.y * self.floor_uv_projection_scale,
            )
            follow = min(1.0, dt * 18.0)
            self.floor_contact_uv = self.floor_contact_uv + (target_uv - self.floor_contact_uv) * follow
            speed_norm = min(1.0, speed / max(0.001, self.max_ball_speed))
            self.floor_contact_strength = min(1.0, self.floor_contact_strength + (0.22 + speed_norm * 0.78) * dt * 7.0)
        else:
            self.floor_contact_strength = max(0.0, self.floor_contact_strength - dt * self.floor_contact_decay)

        keep: list[NodePath] = []
        for node in self.floor_shader_nodes:
            if node is None or node.isEmpty():
                continue
            node.setShaderInput("u_time", self.roll_time)
            node.setShaderInput("u_contact_uv", self.floor_contact_uv)
            node.setShaderInput("u_wake_strength", self.floor_contact_strength)
            node.setShaderInput("u_room_uv_scale", self.floor_uv_projection_scale)
            keep.append(node)
        self.floor_shader_nodes = keep

    def _update_water_surface_shader_inputs(self) -> None:
        if self.water_surface_shader is None:
            return
        keep: list[NodePath] = []
        uv_scale = max(0.2, float(getattr(self, "water_uv_repeat_scale", 1.0)))
        alpha = self._clamp(float(getattr(self, "water_color_cycle_alpha", 0.52)), 0.05, 1.0)
        rainbow_strength = self._clamp(float(getattr(self, "water_rainbow_strength", 0.28)), 0.0, 2.0)
        diffusion_strength = self._clamp(float(getattr(self, "water_diffusion_strength", 0.72)), 0.0, 2.0)
        spec_strength = self._clamp(float(getattr(self, "water_specular_strength", 0.72)), 0.0, 2.0)
        for node in self.water_shader_nodes:
            if node is None or node.isEmpty():
                continue
            node.setShaderInput("u_time", self.roll_time)
            node.setShaderInput("u_uv_scale", uv_scale)
            node.setShaderInput("u_alpha", alpha)
            node.setShaderInput("u_rainbow_strength", rainbow_strength)
            node.setShaderInput("u_diffusion_strength", diffusion_strength)
            node.setShaderInput("u_spec_strength", spec_strength)
            keep.append(node)
        self.water_shader_nodes = keep

    def _update_video_distortion(self, dt: float, speed: float) -> None:
        if not getattr(self, "enable_video_distortion", True):
            return
        if self.video_distort_overlay_np is None or self.video_distort_overlay_np.isEmpty():
            if self.video_distort_ready:
                self.video_distort_ready = False
                self.video_distort_error = "overlay missing"
            return

        self.video_distort_update_timer -= dt
        if self.video_distort_update_timer > 0.0:
            return
        self.video_distort_update_timer = self.video_distort_update_interval

        current_win_size = (int(max(1, self.win.getXSize())), int(max(1, self.win.getYSize())))
        if current_win_size != tuple(getattr(self, "video_distort_window_size", (0, 0))):
            self._setup_video_distortion()
            if self.video_distort_overlay_np is None or self.video_distort_overlay_np.isEmpty():
                return

        speed_norm = min(1.0, speed / max(0.001, self.max_ball_speed))
        local_cf = float(getattr(self, "compression_factor_smoothed", 1.0))
        compression_intensity = self._clamp((1.0 - local_cf) / 0.92, 0.0, 1.0)
        dilation_intensity = self._clamp((local_cf - 1.0) / 0.9, 0.0, 1.0)
        timespace_intensity = max(compression_intensity, dilation_intensity * 0.42)
        black_hole_intensity = self._clamp(float(getattr(self, "black_hole_distort_intensity", 0.0)), 0.0, 1.65)
        timespace_intensity = max(timespace_intensity, black_hole_intensity * 0.9)

        tone_rate = self._clamp(float(getattr(self, "timespace_tone_current_rate", 1.0)), 0.28, 3.4)
        sine_cycle = 0.5 + 0.5 * math.sin(self.roll_time * (2.2 + tone_rate * 2.9))
        sine_signed = (sine_cycle * 2.0) - 1.0

        base_mul = 1.0 + timespace_intensity * float(getattr(self, "video_distort_compression_response", 1.25))
        sine_mul = 1.0 + sine_signed * timespace_intensity * float(getattr(self, "video_distort_sine_response", 0.52))
        dynamic_mul = self._clamp(
            base_mul * sine_mul,
            float(getattr(self, "video_distort_dynamic_min_mul", 0.62)),
            float(getattr(self, "video_distort_dynamic_max_mul", 2.75)),
        )

        swim_speed = (0.4 + 0.95 * speed_norm) * (1.0 + 0.36 * timespace_intensity + black_hole_intensity * 0.55) * (0.92 + 0.24 * sine_cycle)
        pulse = 0.92 + 0.4 * (0.5 + 0.5 * math.sin(self.roll_time * (3.7 + tone_rate * 0.8)))
        strength = self.video_distort_strength * (1.05 + 0.95 * speed_norm + black_hole_intensity * 1.1) * pulse * dynamic_mul
        bloom_strength = self.video_bloom_strength * (0.8 + speed_norm * 0.9) * (0.86 + 0.44 * timespace_intensity)
        tex_w, tex_h = getattr(self, "video_distort_tex_size", (0, 0))
        if tex_w <= 0 or tex_h <= 0:
            tex_w = int(max(1, self.win.getXSize()))
            tex_h = int(max(1, self.win.getYSize()))
        self.video_distort_overlay_np.setShaderInput(
            "u_resolution",
            Vec2(float(tex_w), float(tex_h)),
        )
        self.video_distort_overlay_np.setShaderInput("u_time", self.roll_time)
        self.video_distort_overlay_np.setShaderInput("u_speed", swim_speed)
        self.video_distort_overlay_np.setShaderInput("u_strength", strength)
        self.video_distort_overlay_np.setShaderInput("u_bloom_strength", bloom_strength)
        self.video_distort_overlay_np.setShaderInput("u_bloom_radius", self.video_bloom_radius)
        self.video_distort_overlay_np.setShaderInput("u_bloom_threshold", self.video_bloom_threshold)

    def _on_window_event(self, window) -> None:
        if window is None:
            return
        if self.win is not None:
            current_win_size = (int(max(1, self.win.getXSize())), int(max(1, self.win.getYSize())))
            if getattr(self, "enable_video_distortion", True) and current_win_size != tuple(getattr(self, "video_distort_window_size", (0, 0))):
                self._setup_video_distortion()
        self._layout_hud_to_corners()

    def _layout_hud_to_corners(self) -> None:
        if self.win is not None:
            win_w = float(max(1, self.win.getXSize()))
            win_h = float(max(1, self.win.getYSize()))
            aspect = max(0.6, min(3.5, win_w / win_h))
        else:
            aspect = max(0.6, min(3.5, float(self.getAspectRatio())))

        left = -aspect
        right = aspect
        bottom = -1.0
        top = 1.0

        margin_x = max(0.05, 0.032 * aspect + 0.015)
        margin_y = 0.055

        hp_ui = getattr(self, "player_hp_ui", None)
        if hp_ui is not None and not hp_ui.isEmpty():
            hp_ui.setPos(left + margin_x, 0.0, top - (margin_y + 0.14))

        monster_ui = getattr(self, "monster_hud_ui", None)
        if monster_ui is not None and not monster_ui.isEmpty():
            monster_ui.setPos(right - (margin_x + 0.72), 0.0, top - (margin_y + 0.12))

        input_ui = getattr(self, "input_hud_ui", None)
        if input_ui is not None and not input_ui.isEmpty():
            input_ui.setPos(left + margin_x, 0.0, bottom + (margin_y + 0.09))
            try:
                min_pt, max_pt = input_ui.getTightBounds()
                if min_pt is not None and max_pt is not None:
                    hud_pad = 0.012
                    shift_x = 0.0
                    shift_z = 0.0
                    min_x_allowed = left + hud_pad
                    max_x_allowed = right - hud_pad
                    min_z_allowed = bottom + hud_pad
                    max_z_allowed = top - hud_pad
                    if min_pt.x < min_x_allowed:
                        shift_x += (min_x_allowed - min_pt.x)
                    if max_pt.x > max_x_allowed:
                        shift_x -= (max_pt.x - max_x_allowed)
                    if min_pt.z < min_z_allowed:
                        shift_z += (min_z_allowed - min_pt.z)
                    if max_pt.z > max_z_allowed:
                        shift_z -= (max_pt.z - max_z_allowed)
                    if abs(shift_x) > 1e-6 or abs(shift_z) > 1e-6:
                        cur = input_ui.getPos()
                        input_ui.setPos(cur.x + shift_x, cur.y, cur.z + shift_z)
            except Exception:
                pass

        holo_ui = getattr(self, "holo_map_ui", None)
        if holo_ui is not None and not holo_ui.isEmpty():
            holo_ui.setPos(right - (margin_x + 0.315), 0.0, bottom + (margin_y + 0.295))

    def _update_atmosphere_lights(self, t: float) -> None:
        if hasattr(self, "ambient_light"):
            amb_h = (t * 0.9) % 1.0
            ar, ag, ab = colorsys.hsv_to_rgb(amb_h, 0.33, 0.66 + 0.14 * math.sin(t * 6.2))
            self.ambient_light.setColor((ar, ag, ab, 1.0))

        if hasattr(self, "up_lamp") and hasattr(self, "up_lamp_np"):
            up_h = (t * 1.8) % 1.0
            ur, ug, ub = colorsys.hsv_to_rgb(up_h, 0.82, 1.0)
            self.up_lamp.setColor((ur, ug, ub, 1.0))
            if hasattr(self, "ball_np"):
                b = self.ball_np.getPos()
                self.up_lamp_np.setPos(b.x, b.y, b.z + 3.1)

        if hasattr(self, "lower_lamp") and hasattr(self, "lower_lamp_np"):
            if hasattr(self, "ball_np"):
                b = self.ball_np.getPos()
                self.lower_lamp_np.setPos(b.x, b.y, -0.95)
            depth = 0.04 + 0.02 * (0.5 + 0.5 * math.sin(t * 2.0))
            self.lower_lamp.setColor((depth, depth * 1.15, depth * 2.1, 1.0))
            if hasattr(self, "fog") and self.fog is not None:
                self.fog.setColor(depth * 0.8, depth * 1.0, depth * 1.7)

        if bool(getattr(self, "performance_mode", False)) and bool(getattr(self, "performance_player_light_enabled", True)):
            self._ensure_performance_player_light()
            perf_light = getattr(self, "performance_player_light", None)
            perf_np = getattr(self, "performance_player_light_np", None)
            if perf_light is not None and perf_np is not None and not perf_np.isEmpty() and hasattr(self, "ball_np"):
                perf_np.setPos(self.ball_np.getPos(self.render))
                hue = (t * float(getattr(self, "performance_player_light_cycle_speed", 2.6))) % 1.0
                sat = self._clamp(float(getattr(self, "performance_player_light_saturation", 0.9)), 0.0, 1.0)
                val = self._clamp(float(getattr(self, "performance_player_light_value", 1.0)), 0.0, 1.0)
                lr, lg, lb = colorsys.hsv_to_rgb(hue, sat, val)
                perf_light.setColor((lr, lg, lb, 1.0))
        else:
            self._remove_performance_player_light()

    def _update_hyperspace_background(self, t: float, hyperspace_amount: float) -> None:
        amount = max(0.0, min(1.0, hyperspace_amount))
        pulse = 0.9 + 0.1 * math.sin(t * 2.7)
        base = (0.02 + 0.06 * (1.0 - amount)) * pulse
        r = base * (0.35 + 0.65 * (1.0 - amount))
        g = base * (0.42 + 0.58 * (1.0 - amount))
        b = base * (0.7 + 0.3 * (1.0 - amount))
        self.setBackgroundColor(r, g, b, 1.0)
        if hasattr(self, "fog") and self.fog is not None:
            self.fog.setColor(r * 0.65, g * 0.72, b * 0.9)

    def _update_hyperspace_illumination(self, hyperspace_active: bool, hyperspace_amount: float) -> None:
        if not hasattr(self, "ball_np"):
            return

        ball = self.ball_np.getPos()
        amount = max(0.0, min(1.0, hyperspace_amount))

        if hasattr(self, "sun_light_np") and self.sun_light_np is not None:
            if hyperspace_active:
                offset = Vec3(16.0, -14.0, 24.0)
                self.sun_light_np.setPos(ball + offset)
                self.sun_light_np.lookAt(ball + Vec3(0, 0, 0.35))
            else:
                self.sun_light_np.setPos(self.sun_default_pos)
                self.sun_light_np.setHpr(self.sun_default_hpr)

        if hasattr(self, "ambient_light") and self.ambient_light is not None:
            base = 0.52 + 0.22 * amount
            self.ambient_light.setColor((base * 0.7, base * 0.76, base * 0.9, 1.0))

        if hasattr(self, "up_lamp") and self.up_lamp is not None:
            boost = 1.0 + amount * 0.55
            c = self.up_lamp.getColor()
            self.up_lamp.setColor((min(1.0, c[0] * boost), min(1.0, c[1] * boost), min(1.0, c[2] * boost), 1.0))

    def _get_gravity_up(self) -> Vec3:
        g = self.current_gravity
        if g.lengthSquared() < 1e-8:
            return Vec3(0, 0, 1)
        up = -g
        up.normalize()
        return up

    def _get_ball_floor_contact_point(self) -> Vec3 | None:
        if not hasattr(self, "ball_np"):
            return None
        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()
        ball_pos = self.ball_np.getPos()
        ray_from = ball_pos + up * (self.ball_radius * 0.22)
        ray_to = ball_pos - up * (self.ball_radius + 1.4)
        hit = self.physics_world.rayTestClosest(ray_from, ray_to)
        if not hit.hasHit():
            return None
        node = hit.getNode()
        if node is None or node == self.ball_body:
            return None
        return Vec3(hit.getHitPos())

    def _is_ceiling_mode(self) -> bool:
        return False

    def _setup_room_gravity_zones(self) -> None:
        if not self.rooms:
            return
        self.room_gravity_zones.clear()

        for room_idx, room in enumerate(self.rooms):
            direction = -1.0
            g = Vec3(0, 0, -self.gravity_magnitude)
            base_hue = 0.56
            hue = (base_hue + (room_idx * 0.031)) % 1.0
            cr, cg, cb = colorsys.hsv_to_rgb(hue, 0.78, 0.98)
            self.room_gravity_zones.append(
                {
                    "room_idx": room_idx,
                    "room": room,
                    "gravity": g,
                    "flow_speed": random.uniform(1.1, 2.6),
                    "color": (cr, cg, cb, 0.9),
                    "direction": direction,
                    "particles": [],
                    "particles_root": None,
                }
            )

    def _rebuild_spatial_acceleration_caches(self) -> None:
        self.room_bounds_cache = prepare_room_bounds(
            self.rooms,
            self.room_levels,
            self.floor_y,
            self.level_z_step,
            self.wall_h,
        )
        self.zone_bounds_cache = prepare_zone_bounds(self.room_gravity_zones)

    def _ensure_timespace_tone_asset(self) -> str | None:
        folder = os.path.join("soundfx")
        path = os.path.join(folder, "timespace_sine_v2.wav")
        try:
            if not os.path.isdir(folder):
                os.makedirs(folder, exist_ok=True)
            sample_rate = 44100
            period_samples = 400
            count = period_samples * 220
            amp = 0.2

            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                frames = bytearray()
                for i in range(count):
                    phase = (i % period_samples) / float(period_samples)
                    sample = amp * math.sin(2.0 * math.pi * phase)
                    iv = int(max(-1.0, min(1.0, sample)) * 32767.0)
                    frames += struct.pack("<h", iv)
                wav_file.writeframes(bytes(frames))
            return path
        except Exception:
            return None

    def _setup_timespace_tone(self) -> None:
        self.timespace_tone = None
        tone_path = self._ensure_timespace_tone_asset()
        if not tone_path:
            return
        try:
            tone = self.loader.loadSfx(tone_path)
            if tone is None:
                return
            tone.setLoop(True)
            tone.setVolume(0.0)
            tone.setPlayRate(1.0)
            tone.play()
            self.timespace_tone = tone
        except Exception:
            self.timespace_tone = None

    def _update_timespace_tone(self, compression_factor: float, dt: float) -> None:
        tone = self.timespace_tone
        if tone is None:
            return

        cf = max(0.08, min(1.9, float(compression_factor)))
        warp_intensity = max(0.0, min(1.0, (1.0 - cf) / 0.92))
        dilation_intensity = max(0.0, min(1.0, (cf - 1.0) / 0.9))
        extreme = max(warp_intensity, dilation_intensity)

        self.timespace_tone_target_rate = 0.34 + 3.1 * (extreme ** 2.8)
        self.timespace_tone_target_volume = 0.02 + 0.31 * (extreme ** 0.72)

        rate_alpha = min(1.0, dt * self.timespace_tone_rate_smooth)
        vol_alpha = min(1.0, dt * self.timespace_tone_volume_smooth)
        self.timespace_tone_current_rate += (self.timespace_tone_target_rate - self.timespace_tone_current_rate) * rate_alpha
        self.timespace_tone_current_volume += (self.timespace_tone_target_volume - self.timespace_tone_current_volume) * vol_alpha

        try:
            tone.setPlayRate(max(0.28, min(3.4, self.timespace_tone_current_rate)))
            tone.setVolume(max(0.0, min(0.35, self.timespace_tone_current_volume)))
            if tone.status() != tone.PLAYING:
                tone.play()
        except Exception:
            pass

    def _setup_room_compression_pockets(self) -> None:
        self.room_compression_pockets.clear()
        if not self.rooms:
            return

        arena_mode = bool(getattr(self, "four_d_obstacle_arena_mode", False))

        for room_idx, room in enumerate(self.rooms):
            room_level = self.room_levels.get(room_idx, 0)
            base_z = self._level_base_z(room_level)
            if arena_mode:
                pocket_count = random.choice([3, 4, 4])
            else:
                pocket_count = random.choice([2, 2, 3])
            for _ in range(pocket_count):
                cx = random.uniform(room.x + room.w * 0.18, room.x + room.w * 0.82)
                cy = random.uniform(room.y + room.h * 0.18, room.y + room.h * 0.82)
                if arena_mode:
                    z_low = self.floor_y + 2.0
                    z_high = self.floor_y + 15.5
                    cz = random.uniform(z_low, z_high)
                    radius = random.uniform(max(6.2, min(room.w, room.h) * 0.24), max(11.5, min(room.w, room.h) * 0.52))
                    compression = random.uniform(0.36, 0.82)
                    dilation = random.uniform(0.18, 0.58)
                    scale_x = random.uniform(0.74, 1.95)
                    scale_y = random.uniform(0.74, 1.95)
                    scale_z = random.uniform(0.48, 1.22)
                else:
                    cz = base_z + random.uniform(self.wall_h * 0.35, self.wall_h * 0.72)
                    radius = random.uniform(max(2.4, min(room.w, room.h) * 0.16), max(4.8, min(room.w, room.h) * 0.38))
                    compression = random.uniform(0.52, 0.92)
                    dilation = random.uniform(0.08, 0.44)
                    scale_x = random.uniform(0.58, 1.7)
                    scale_y = random.uniform(0.58, 1.7)
                    scale_z = random.uniform(0.58, 1.7)

                radius_scale = max(0.5, float(getattr(self, "compression_pocket_radius_scale", 2.4)))
                if arena_mode:
                    radius_scale *= 1.18
                radius *= radius_scale
                self.room_compression_pockets.append(
                    {
                        "room_idx": room_idx,
                        "center": Vec3(cx, cy, cz),
                        "radius": radius,
                        "compression": compression,
                        "dilation": dilation,
                        "scale": Vec3(scale_x, scale_y, scale_z),
                        "phase": random.uniform(0.0, math.tau),
                    }
                )

    def _setup_room_dimension_fields(self) -> None:
        self.room_dimension_fields.clear()
        if not self.rooms:
            return

        for room_idx, room in enumerate(self.rooms):
            area = max(1.0, float(room.w * room.h))
            area_norm = min(1.0, area / 360.0)
            aspect = room.w / max(0.001, room.h)
            baseline = random.uniform(0.94, 1.07)
            if random.random() < 0.22:
                baseline += random.choice([-1.0, 1.0]) * random.uniform(0.06, 0.12)

            amplitude = random.uniform(0.06, 0.18) * (0.72 + area_norm * 0.52)
            frequency = random.uniform(0.55, 1.35)
            center_bias = random.uniform(0.3, 1.0)
            edge_bias = random.uniform(0.3, 1.0)
            if aspect >= 1.22:
                persp_u = random.uniform(0.3, 0.78)
                persp_v = random.uniform(-0.52, -0.12)
            elif aspect <= 0.82:
                persp_u = random.uniform(-0.52, -0.12)
                persp_v = random.uniform(0.3, 0.78)
            else:
                persp_u = random.uniform(-0.38, 0.38)
                persp_v = random.uniform(-0.38, 0.38)

            self.room_dimension_fields[room_idx] = {
                "base": baseline,
                "amp": amplitude,
                "freq": frequency,
                "phase": random.uniform(0.0, math.tau),
                "center_bias": center_bias,
                "edge_bias": edge_bias,
                "persp_u": persp_u,
                "persp_v": persp_v,
                "aspect": aspect,
            }

    def _setup_liminal_fold_graph(self) -> None:
        self.liminal_fold_nodes.clear()
        self.liminal_fold_links.clear()
        if not self.rooms:
            return

        room_centers: list[tuple[float, float, float]] = []
        for room_idx, room in enumerate(self.rooms):
            level = self.room_levels.get(room_idx, 0)
            base_z = self._level_base_z(level)
            room_centers.append((room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + self.wall_h * 0.52))
        hyper_seed_w = compute_level_w_batch(room_centers, self.corridor_w, self.level_z_step)

        for room_idx, room in enumerate(self.rooms):
            level = self.room_levels.get(room_idx, 0)
            base_z = self._level_base_z(level)
            center = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + self.wall_h * 0.52)
            field = self.room_dimension_fields.get(room_idx, {})
            w_bias = (float(field.get("base", 1.0)) - 1.0) * self.hyper_w_limit * 1.8
            if room_idx < len(hyper_seed_w):
                w_bias += float(hyper_seed_w[room_idx]) * 0.52
            w_bias += random.uniform(-0.75, 0.75)
            w_coord = self._clamp(w_bias, -self.hyper_w_limit, self.hyper_w_limit)
            node_idx = len(self.liminal_fold_nodes)
            self.liminal_fold_nodes.append({"pos": center, "w": w_coord, "room_idx": room_idx})
            self.liminal_fold_links[node_idx] = []

        def link(a: int, b: int) -> None:
            if a == b:
                return
            self.liminal_fold_links.setdefault(a, [])
            self.liminal_fold_links.setdefault(b, [])
            if b not in self.liminal_fold_links[a]:
                self.liminal_fold_links[a].append(b)
            if a not in self.liminal_fold_links[b]:
                self.liminal_fold_links[b].append(a)

        for a, b in self.edges:
            if 0 <= a < len(self.liminal_fold_nodes) and 0 <= b < len(self.liminal_fold_nodes):
                link(a, b)

        extra_pairs = max(4, min(20, len(self.rooms) // 2))
        for _ in range(extra_pairs):
            a = random.randrange(len(self.rooms))
            b = random.randrange(len(self.rooms))
            if a == b:
                continue
            ra = self.rooms[a]
            rb = self.rooms[b]
            da = Vec2(ra.x + ra.w * 0.5, ra.y + ra.h * 0.5)
            db = Vec2(rb.x + rb.w * 0.5, rb.y + rb.h * 0.5)
            if (da - db).length() < max(18.0, self.corridor_w * 2.2):
                continue

            for room_idx in (a, b):
                room = self.rooms[room_idx]
                level = self.room_levels.get(room_idx, 0)
                base_z = self._level_base_z(level)
                ghost_pos = Vec3(
                    room.x + room.w * random.uniform(0.28, 0.72),
                    room.y + room.h * random.uniform(0.28, 0.72),
                    base_z + self.wall_h * random.uniform(0.32, 0.74),
                )
                ghost_w = self._clamp(
                    random.uniform(-self.hyper_w_limit, self.hyper_w_limit),
                    -self.hyper_w_limit,
                    self.hyper_w_limit,
                )
                ghost_idx = len(self.liminal_fold_nodes)
                self.liminal_fold_nodes.append({"pos": ghost_pos, "w": ghost_w, "room_idx": room_idx})
                self.liminal_fold_links[ghost_idx] = []
                link(room_idx, ghost_idx)

            ghost_a = len(self.liminal_fold_nodes) - 2
            ghost_b = len(self.liminal_fold_nodes) - 1
            link(ghost_a, ghost_b)

    def _nearest_liminal_fold_idx(self, pos: Vec3, w_coord: float) -> int | None:
        if not self.liminal_fold_nodes:
            return None
        best_idx = None
        best_score = float("inf")
        for idx, node in enumerate(self.liminal_fold_nodes):
            d3 = (Vec3(node["pos"]) - pos).length()
            dw = abs(float(node.get("w", 0.0)) - float(w_coord))
            score = d3 + dw * 4.0
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _next_liminal_fold_hop(self, start_idx: int, goal_idx: int) -> int | None:
        if start_idx == goal_idx:
            return start_idx
        neighbors = self.liminal_fold_links.get(start_idx, [])
        if not neighbors:
            return None

        goal = self.liminal_fold_nodes[goal_idx]
        best = None
        best_cost = float("inf")
        for nxt in neighbors:
            if not (0 <= nxt < len(self.liminal_fold_nodes)):
                continue
            node = self.liminal_fold_nodes[nxt]
            d3 = (Vec3(node["pos"]) - Vec3(goal["pos"])).length()
            dw = abs(float(node.get("w", 0.0)) - float(goal.get("w", 0.0)))
            jitter = random.uniform(0.0, 0.35)
            cost = d3 + dw * 3.0 + jitter
            if cost < best_cost:
                best_cost = cost
                best = nxt
        return best

    def _liminal_fold_hop_count(self, start_idx: int, goal_idx: int, max_depth: int = 24) -> int | None:
        if start_idx == goal_idx:
            return 0
        if not self.liminal_fold_nodes:
            return None
        if not (0 <= start_idx < len(self.liminal_fold_nodes) and 0 <= goal_idx < len(self.liminal_fold_nodes)):
            return None
        visited: set[int] = {start_idx}
        q: deque[tuple[int, int]] = deque([(start_idx, 0)])
        depth_cap = max(1, int(max_depth))
        while q:
            node_idx, depth = q.popleft()
            if depth >= depth_cap:
                continue
            for nxt in self.liminal_fold_links.get(node_idx, []):
                if nxt in visited:
                    continue
                if nxt == goal_idx:
                    return depth + 1
                visited.add(nxt)
                q.append((nxt, depth + 1))
        return None

    def _compression_factor_at(self, pos: Vec3, t: float) -> float:
        if not self.room_compression_pockets and not self.room_dimension_fields:
            return 1.0

        factor = 1.0
        room_idx = self._get_current_room_idx_for_pos(pos)
        if room_idx is not None:
            room_field = self.room_dimension_fields.get(room_idx)
            if room_field is not None and room_idx < len(self.rooms):
                room = self.rooms[room_idx]
                nx = (pos.x - (room.x + room.w * 0.5)) / max(0.001, room.w * 0.5)
                ny = (pos.y - (room.y + room.h * 0.5)) / max(0.001, room.h * 0.5)
                edge = max(abs(nx), abs(ny))
                center = max(0.0, 1.0 - edge)
                wave = math.sin(t * room_field["freq"] + room_field["phase"])
                room_base = room_field["base"] + room_field["amp"] * wave
                room_delta = room_base - 1.0
                spatial_blend = center * room_field["center_bias"] + edge * room_field["edge_bias"]
                factor *= 1.0 + room_delta * max(0.0, min(1.35, spatial_blend))

        for pocket in self.room_compression_pockets:
            center = pocket["center"]
            radius = max(0.001, float(pocket["radius"]))
            scale = pocket["scale"]
            phase = pocket["phase"]

            pulse = 0.75 + 0.35 * math.sin(t * 1.1 + phase)
            inv = 1.0 / (radius * pulse)
            dx = (pos.x - center.x) * scale.x * inv
            dy = (pos.y - center.y) * scale.y * inv
            dz = (pos.z - center.z) * scale.z * inv
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist >= 1.0:
                continue

            influence = 1.0 - dist
            influence_power = max(0.25, float(getattr(self, "compression_pocket_influence_power", 0.78)))
            influence = influence ** influence_power
            local_factor = 1.0 - (1.0 - pocket["compression"]) * influence
            dilation_gain = max(0.0, float(getattr(self, "compression_pocket_dilation_gain", 1.55)))
            speed_bias = self._clamp(float(getattr(self, "compression_pocket_speed_bias", 0.24)), 0.0, 0.95)
            sin_norm = 0.5 + 0.5 * math.sin(t * 1.35 + phase * 1.7)
            local_factor += pocket.get("dilation", 0.0) * dilation_gain * influence * (speed_bias + (1.0 - speed_bias) * sin_norm)
            factor *= max(0.55, min(1.85, local_factor))

        if hasattr(self, "ball_body"):
            speed = self.ball_body.getLinearVelocity().length()
            speed_norm = min(1.0, speed / max(0.001, self.max_ball_speed))
            travel_dilation = 1.0 + speed_norm * 0.12
            factor *= travel_dilation

        return max(0.42, min(1.9, factor))

    def _update_camera_dimension_from_compression(self, dt: float, compression_factor: float, speed: float, hyperspace_active: bool) -> None:
        if not hasattr(self, "camLens") or self.camLens is None:
            return

        compress = max(0.0, 1.0 - float(compression_factor))
        dilate = max(0.0, float(compression_factor) - 1.0)
        speed_norm = min(1.0, max(0.0, speed / max(0.001, self.max_ball_speed)))
        if hyperspace_active:
            compress = min(1.0, compress + 0.1)

        amount = min(1.0, compress * (0.65 + 0.35 * speed_norm) * (1.0 + self.space_compress_3d_strength))
        dilate_amount = min(1.0, dilate * 0.85)

        target_follow = self.camera_follow_distance_base * (1.0 - 0.22 * amount + 0.1 * dilate_amount)
        target_height = self.camera_height_base * (1.0 - 0.12 * amount + 0.08 * dilate_amount)
        target_fov = self.camera_fov_base * (1.0 - 0.14 * amount + 0.1 * dilate_amount)
        target_fov = self._clamp(target_fov, 62.0, 128.0)

        blend = min(1.0, dt * self.camera_dimension_blend_speed)
        self.camera_follow_distance += (target_follow - self.camera_follow_distance) * blend
        self.camera_height_offset += (target_height - self.camera_height_offset) * blend

        try:
            current_fov = float(self.camLens.getFov()[0])
            new_fov = current_fov + (target_fov - current_fov) * blend
            self.camLens.setFov(new_fov)
        except Exception:
            pass

    def _setup_room_gravity_particles(self) -> None:
        for zone in self.room_gravity_zones:
            room = zone["room"]
            room_idx = zone["room_idx"]
            base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            root = self.world.attachNewNode(f"gravity-field-{room_idx}")
            zone["particles_root"] = root

            area = room.w * room.h
            count = max(32, min(120, int(area * 0.24))) if self.performance_mode else max(56, min(180, int(area * 0.38)))
            particles: list[dict] = []
            color = zone["color"]
            for _ in range(count):
                node = self.box_model.copyTo(root)
                node.setPos(self.box_norm_offset)
                node.setScale(self.box_norm_scale)
                pixel_size = random.uniform(0.015, 0.036)
                node.setScale(pixel_size, pixel_size, pixel_size)
                node.setColor(color)
                node.setLightOff(1)
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setBin("transparent", 28)
                node.setDepthWrite(False)
                node.clearTexture()
                #node.setTexture(self.level_checker_tex, 1)
                self._register_color_cycle(node, color, min_speed=0.85, max_speed=1.9)

                pos = Vec3(
                    random.uniform(room.x + 0.8, room.x + room.w - 0.8),
                    random.uniform(room.y + 0.8, room.y + room.h - 0.8),
                    random.uniform(base_z + 0.5, base_z + self.wall_h - 0.75),
                )
                node.setPos(pos)
                particles.append(
                    {
                        "node": node,
                        "pos": pos,
                        "speed": random.uniform(0.75, 1.5),
                        "drift_x": random.uniform(-0.28, 0.28),
                        "drift_y": random.uniform(-0.28, 0.28),
                        "phase": random.uniform(0.0, math.tau),
                    }
                )
            zone["particles"] = particles

    def _get_zone_for_position(self, pos: Vec3) -> dict | None:
        zone_idx = find_zone_index_for_pos(pos.x, pos.y, self.zone_bounds_cache)
        if zone_idx is not None and 0 <= zone_idx < len(self.room_gravity_zones):
            return self.room_gravity_zones[zone_idx]

        for zone in self.room_gravity_zones:
            room = zone["room"]
            if room.x <= pos.x <= (room.x + room.w) and room.y <= pos.y <= (room.y + room.h):
                return zone
        return None

    def _get_current_room_idx_for_pos(self, pos: Vec3) -> int | None:
        fast_idx = find_room_index_for_pos(pos.x, pos.y, pos.z, self.room_bounds_cache)
        if fast_idx is not None:
            return fast_idx

        candidates: list[int] = []
        for idx, room in enumerate(self.rooms):
            if room.x <= pos.x <= (room.x + room.w) and room.y <= pos.y <= (room.y + room.h):
                candidates.append(idx)
        if not candidates:
            return None

        best_idx = candidates[0]
        best_dist = float("inf")
        for idx in candidates:
            level = self.room_levels.get(idx, 0)
            base_z = self._level_base_z(level)
            room_center_z = base_z + self.wall_h * 0.5
            dz = abs(pos.z - room_center_z)
            if dz < best_dist:
                best_dist = dz
                best_idx = idx
        return best_idx

    def _clamp_point_inside_room(self, room_idx: int, pos: Vec3, inset: float = 0.6) -> Vec3:
        if not (0 <= room_idx < len(self.rooms)):
            return Vec3(pos)
        room = self.rooms[room_idx]
        pad = max(0.12, float(inset))
        x = self._clamp(pos.x, room.x + pad, room.x + room.w - pad)
        y = self._clamp(pos.y, room.y + pad, room.y + room.h - pad)
        level = self.room_levels.get(room_idx, 0)
        base_z = self._level_base_z(level)
        ball_r = float(getattr(self, "ball_radius", 0.68))
        z = self._clamp(pos.z, base_z + ball_r + 0.06, base_z + self.wall_h - 0.4)
        return Vec3(x, y, z)

    def _safe_room_spawn_pos(self, room_idx: int, z_lift: float = 0.28) -> Vec3:
        ball_r = float(getattr(self, "ball_radius", 0.68))
        if not (0 <= room_idx < len(self.rooms)):
            return Vec3(self.map_w * 0.5, self.map_d * 0.5, self.floor_y + ball_r + z_lift)
        room = self.rooms[room_idx]
        base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
        center = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + ball_r + z_lift)
        return self._clamp_point_inside_room(room_idx, center, inset=0.62)

    def _room_center_pos(self, room_idx: int) -> Vec3:
        if not (0 <= room_idx < len(self.rooms)):
            return Vec3(self.map_w * 0.5, self.map_d * 0.5, self.floor_y + self.wall_h * 0.5)
        room = self.rooms[room_idx]
        base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
        return Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + 0.55)

    def _floor_anchor_z_for_pos(self, pos: Vec3) -> float:
        if bool(getattr(self, "four_d_obstacle_arena_mode", False)):
            return float(self.floor_y + float(getattr(self, "water_surface_raise", 0.0)))

        room_idx = self._get_current_room_idx_for_pos(pos)
        if room_idx is None:
            return float(self.floor_y + float(getattr(self, "ball_radius", 0.68)) + 0.06)

        base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
        return float(base_z + float(getattr(self, "ball_radius", 0.68)) + 0.06)

    def _find_room_path(self, start_idx: int, goal_idx: int) -> list[int]:
        if start_idx == goal_idx:
            return [start_idx]
        room_count = len(self.rooms)
        if not (0 <= start_idx < room_count and 0 <= goal_idx < room_count):
            return []

        adj: dict[int, list[int]] = {i: [] for i in range(room_count)}
        for a, b in self.edges:
            if 0 <= a < room_count and 0 <= b < room_count:
                adj[a].append(b)
                adj[b].append(a)

        q: deque[int] = deque([start_idx])
        prev: dict[int, int] = {start_idx: -1}
        while q:
            cur = q.popleft()
            if cur == goal_idx:
                break
            for nxt in adj.get(cur, []):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                q.append(nxt)

        if goal_idx not in prev:
            return []
        path = [goal_idx]
        while path[-1] != start_idx:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _get_alive_monster_entries(self) -> list[dict]:
        alive: list[dict] = []
        for monster in self.monsters:
            if monster.get("dead", False):
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            alive.append(monster)
        return alive

    def _choose_player_ai_target(self, alive: list[dict], ball_pos: Vec3) -> dict | None:
        if not alive:
            return None
        self.player_ai_retarget_timer -= 1.0 / 30.0
        player_w = float(getattr(self, "player_w", 0.0))
        plane_tol = max(0.15, float(getattr(self, "player_ai_plane_height_tolerance", 2.6)))
        below_margin = float(getattr(self, "player_ai_below_seek_margin", 0.35))
        w_plane_tol = max(0.05, float(getattr(self, "player_ai_same_existence_w_tolerance", 1.35)))
        engage_dist = max(0.5, float(getattr(self, "player_ai_engage_distance", 2.35)))
        floor_tol = max(0.25, float(getattr(self, "player_ai_floor_tolerance", 2.6)))
        lock_release_dist = max(engage_dist + 1.0, float(getattr(self, "player_ai_lock_release_distance", 22.0)))
        ball_floor_z = self._floor_anchor_z_for_pos(ball_pos)

        def _fold_hops_to(monster_pos: Vec3, monster_w: float) -> int | None:
            if not self.liminal_fold_nodes:
                return None
            start_fold = self._nearest_liminal_fold_idx(ball_pos, player_w)
            goal_fold = self._nearest_liminal_fold_idx(monster_pos, monster_w)
            if start_fold is None or goal_fold is None:
                return None
            return self._liminal_fold_hop_count(start_fold, goal_fold)

        def _monster_metrics(monster: dict) -> tuple[float, float, float, float, float, int | None]:
            root = monster.get("root")
            if root is None or root.isEmpty():
                return float("inf"), 0.0, float("inf"), float("inf"), float("inf"), None
            m_pos = root.getPos()
            m_w = float(monster.get("w", 0.0))
            dz = float(m_pos.z) - float(ball_pos.z)
            dw = abs(m_w - player_w)
            dist4d = self._distance4d(ball_pos, player_w, m_pos, m_w)
            floor_delta = abs(self._floor_anchor_z_for_pos(m_pos) - ball_floor_z)
            hops = _fold_hops_to(m_pos, m_w)
            return dist4d, dz, dw, m_w, floor_delta, hops

        def _find_by_id(target_id: int | None) -> dict | None:
            if target_id is None:
                return None
            for monster in alive:
                root = monster.get("root")
                if root is not None and id(root) == target_id:
                    return monster
            return None

        locked = _find_by_id(getattr(self, "player_ai_lock_target_id", None))
        if locked is not None:
            locked_dist4d, locked_dz, _, _, locked_floor_delta, _ = _monster_metrics(locked)
            locked_engaging = locked_dist4d <= engage_dist
            locked_below = locked_dz < -below_margin
            locked_allowed = (not locked_below) or locked_engaging
            if locked_dist4d <= lock_release_dist and locked_floor_delta <= floor_tol * 2.4 and locked_allowed:
                root = locked.get("root")
                self.player_ai_target_id = id(root) if root is not None else None
                self.player_ai_retarget_timer = max(self.player_ai_retarget_timer, 0.18)
                return locked
            self.player_ai_lock_target_id = None

        current = _find_by_id(self.player_ai_target_id)

        if current is not None and self.player_ai_retarget_timer > 0.0:
            current_dist4d, current_dz, _, _, current_floor_delta, _ = _monster_metrics(current)
            current_engaging = current_dist4d <= engage_dist
            current_below = current_dz < -below_margin
            if ((not current_below) and current_floor_delta <= floor_tol * 1.35) or current_engaging:
                return current

        candidates: list[dict] = []
        candidates_same_plane: list[dict] = []
        candidates_same_floor: list[dict] = []
        candidates_same_floor_plane: list[dict] = []
        candidates_same_plane_same_w: list[dict] = []
        candidates_same_floor_plane_same_w: list[dict] = []
        for monster in alive:
            dist4d, dz, dw, m_w, floor_delta, fold_hops = _monster_metrics(monster)
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            engaging = dist4d <= engage_dist
            if dz < -below_margin and not engaging:
                continue
            if floor_delta > floor_tol * 1.45 and not engaging:
                continue
            if dw > w_plane_tol * 1.7 and not engaging:
                if fold_hops is None:
                    continue
            candidates.append(monster)
            same_floor = floor_delta <= floor_tol
            if same_floor:
                candidates_same_floor.append(monster)
            if abs(dz) <= plane_tol:
                candidates_same_plane.append(monster)
                if same_floor:
                    candidates_same_floor_plane.append(monster)
                if dw <= w_plane_tol:
                    candidates_same_plane_same_w.append(monster)
                    if same_floor:
                        candidates_same_floor_plane_same_w.append(monster)

        if candidates_same_floor_plane_same_w:
            pool = candidates_same_floor_plane_same_w
        elif candidates_same_floor_plane:
            pool = candidates_same_floor_plane
        elif candidates_same_floor:
            pool = candidates_same_floor
        elif candidates_same_plane_same_w:
            pool = candidates_same_plane_same_w
        elif candidates_same_plane:
            pool = candidates_same_plane
        else:
            pool = candidates

        if not pool:
            return None

        def _score(monster: dict) -> float:
            dist4d, dz, dw, _, floor_delta, fold_hops = _monster_metrics(monster)
            w_penalty = max(0.0, float(getattr(self, "player_ai_w_mismatch_penalty", 0.72)))
            fold_hop_weight = max(0.0, float(getattr(self, "player_ai_fold_hop_weight", 0.48)))
            fold_route_bonus = max(0.0, float(getattr(self, "player_ai_fold_route_bonus", 1.15)))
            score = dist4d + abs(dz) * 0.4 + dw * w_penalty + floor_delta * 0.95
            if fold_hops is None:
                score += 3.5
            else:
                score += float(fold_hops) * fold_hop_weight
                if dw > w_plane_tol:
                    score -= fold_route_bonus
            return score

        best = min(pool, key=_score)
        if current is not None and current in pool:
            best_score = _score(best)
            current_score = _score(current)
            hysteresis = max(1.01, float(getattr(self, "player_ai_target_hysteresis", 1.18)))
            if current_score <= best_score * hysteresis:
                best = current
        best_root = best.get("root")
        self.player_ai_target_id = id(best_root) if best_root is not None else None
        self.player_ai_retarget_timer = random.uniform(0.4, 1.1)
        return best

    def _update_player_ai(self, dt: float) -> Vec3 | None:
        if not self.player_ai_enabled or self.game_over_active or self.win_active:
            self.player_ai_camera_target_pos = None
            self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
            return None
        if not hasattr(self, "ball_np"):
            self.player_ai_camera_target_pos = None
            self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
            return None

        self.player_ai_combo_timer = max(0.0, self.player_ai_combo_timer - dt)
        self.player_ai_jump_cooldown = max(0.0, self.player_ai_jump_cooldown - dt)

        ball_pos = self.ball_np.getPos()
        alive = self._get_alive_monster_entries()
        if not alive:
            self._trigger_win()
            self.player_ai_camera_target_pos = None
            self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
            return None

        player_w = float(getattr(self, "player_w", 0.0))
        target = self._choose_player_ai_target(alive, ball_pos)
        if target is None:
            self.player_ai_camera_target_pos = None
            self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
            return None
        target_root = target.get("root")
        if target_root is None or target_root.isEmpty():
            self.player_ai_camera_target_pos = None
            self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
            return None
        target_pos = target_root.getPos()
        target_w = float(target.get("w", 0.0))
        self.player_ai_camera_target_pos = Vec3(target_pos)
        engage_dist = max(0.5, float(getattr(self, "player_ai_engage_distance", 2.35)))
        lock_on_dist = max(engage_dist, float(getattr(self, "player_ai_lock_on_distance", 3.2)))
        lock_release_dist = max(lock_on_dist + 0.5, float(getattr(self, "player_ai_lock_release_distance", 22.0)))

        start_room = self._get_current_room_idx_for_pos(ball_pos)
        goal_room = self._get_current_room_idx_for_pos(target_pos)
        nav_target = Vec3(target_pos)
        w_target_goal = float(target_w)
        using_fold_nav = False
        if self.liminal_fold_nodes and abs(target_w - player_w) > 0.35:
            start_fold = self._nearest_liminal_fold_idx(ball_pos, player_w)
            goal_fold = self._nearest_liminal_fold_idx(target_pos, target_w)
            if start_fold is not None and goal_fold is not None:
                next_hop = self._next_liminal_fold_hop(start_fold, goal_fold)
                if isinstance(next_hop, int) and 0 <= next_hop < len(self.liminal_fold_nodes):
                    nav_target = Vec3(self.liminal_fold_nodes[next_hop]["pos"])
                    w_target_goal = float(self.liminal_fold_nodes[next_hop].get("w", target_w))
                    using_fold_nav = True

        if using_fold_nav:
            self.player_ai_room_path = []
            self.player_ai_room_path_goal = None
            self.player_ai_room_path_recalc_timer = 0.0
        elif start_room is not None and goal_room is not None and start_room != goal_room:
            self.player_ai_room_path_recalc_timer = max(0.0, float(getattr(self, "player_ai_room_path_recalc_timer", 0.0)) - dt)
            recalc_interval = max(0.05, float(getattr(self, "player_ai_room_path_recalc_interval", 0.28)))
            current_path = list(getattr(self, "player_ai_room_path", []))
            current_goal = getattr(self, "player_ai_room_path_goal", None)
            need_recalc = (
                current_goal != goal_room
                or self.player_ai_room_path_recalc_timer <= 0.0
                or len(current_path) < 2
                or start_room not in current_path
            )
            if need_recalc:
                current_path = self._find_room_path(start_room, goal_room)
                self.player_ai_room_path_goal = goal_room
                self.player_ai_room_path_recalc_timer = recalc_interval

            if current_path and start_room in current_path:
                start_idx = current_path.index(start_room)
                current_path = current_path[start_idx:]

            waypoint_reach = max(0.25, float(getattr(self, "player_ai_room_waypoint_reach", 1.45)))
            if len(current_path) >= 2:
                waypoint_room = current_path[1]
                waypoint_pos = self._room_center_pos(waypoint_room)
                if (waypoint_pos - ball_pos).length() <= waypoint_reach:
                    current_path = current_path[1:]
                    if len(current_path) >= 2:
                        waypoint_pos = self._room_center_pos(current_path[1])
                if len(current_path) >= 2:
                    nav_target = waypoint_pos

            self.player_ai_room_path = current_path
        else:
            self.player_ai_room_path = []
            self.player_ai_room_path_goal = None
            self.player_ai_room_path_recalc_timer = 0.0

        w_target_rate = max(0.1, float(getattr(self, "player_ai_w_target_blend_rate", 3.6)))
        current_target_w = float(getattr(self, "player_ai_target_w", player_w))
        w_target_alpha = min(1.0, dt * w_target_rate)
        self.player_ai_target_w = current_target_w + (w_target_goal - current_target_w) * w_target_alpha

        nav_delta = nav_target - ball_pos
        avoid_strength = max(0.0, float(getattr(self, "player_ai_black_hole_avoid_strength", 2.4)))
        avoid_margin = max(1.0, float(getattr(self, "player_ai_black_hole_avoid_margin", 1.18)))
        jump_margin = self._clamp(float(getattr(self, "player_ai_black_hole_jump_margin", 0.42)), 0.05, 0.9)
        escape_vec = Vec3(0, 0, 0)
        nearest_pull_ratio = 0.0
        if avoid_strength > 1e-6:
            for anomaly in getattr(self, "black_holes", []):
                if str(anomaly.get("kind", "suck")).lower() != "suck":
                    continue
                root = anomaly.get("root")
                if root is None or root.isEmpty():
                    continue
                pos = root.getPos(self.render)
                to_center = pos - ball_pos
                dist = max(1e-5, to_center.length())
                radius = max(0.8, float(anomaly.get("radius", getattr(self, "black_hole_influence_radius", 15.5))))
                outer = radius * avoid_margin
                if dist > outer:
                    continue
                away = -to_center / dist
                pull_ratio = self._clamp((outer - dist) / max(1e-5, outer), 0.0, 1.0)
                nearest_pull_ratio = max(nearest_pull_ratio, pull_ratio)
                escape_vec += away * (pull_ratio * pull_ratio)

        if escape_vec.lengthSquared() > 1e-8:
            escape_vec.normalize()
            nav_delta += escape_vec * (max(0.5, nav_delta.length()) * avoid_strength * nearest_pull_ratio)

        target_delta = target_pos - ball_pos
        move_vec = Vec3(nav_delta)
        gravity_up = self._get_gravity_up()
        target_up_delta = float(target_delta.dot(gravity_up))
        nav_up_delta = float(nav_delta.dot(gravity_up))
        move_vec = move_vec - gravity_up * move_vec.dot(gravity_up)
        if move_vec.lengthSquared() <= 1e-6:
            return None
        move_vec.normalize()
        prev_move = Vec3(getattr(self, "player_ai_last_move_dir", Vec3(0, 1, 0)))
        if prev_move.lengthSquared() > 1e-8:
            prev_move.normalize()
            smooth_speed = max(0.1, float(getattr(self, "player_ai_move_smooth", 7.8)))
            blend = min(1.0, dt * smooth_speed)
            smoothed = prev_move + (move_vec - prev_move) * blend
            if smoothed.lengthSquared() > 1e-8:
                smoothed.normalize()
                move_vec = smoothed
        self.player_ai_last_move_dir = Vec3(move_vec)

        target_dist_4d = self._distance4d(ball_pos, player_w, target_pos, target_w)
        target_root_id = id(target_root)
        if target_dist_4d <= lock_on_dist:
            self.player_ai_lock_target_id = target_root_id
        elif getattr(self, "player_ai_lock_target_id", None) == target_root_id and target_dist_4d > lock_release_dist:
            self.player_ai_lock_target_id = None

        if self.grounded and self.player_ai_jump_cooldown <= 0.0:
            target_up_threshold = max(float(getattr(self, "player_ai_jump_target_up_threshold", 0.55)), self.ball_radius * 0.8)
            nav_up_threshold = max(float(getattr(self, "player_ai_jump_nav_up_threshold", 0.35)), self.ball_radius * 0.45)
            need_jump = target_up_delta > target_up_threshold or nav_up_delta > nav_up_threshold
            if not need_jump:
                attack_up_threshold = max(0.08, float(getattr(self, "player_ai_jump_attack_up_threshold", 0.28)))
                attack_dist_mul = max(1.0, float(getattr(self, "player_ai_jump_attack_dist_mul", 1.9)))
                if target_up_delta > attack_up_threshold and target_dist_4d <= engage_dist * attack_dist_mul:
                    need_jump = True
            if not need_jump and nearest_pull_ratio >= jump_margin:
                need_jump = True
            if not need_jump:
                ray_from = ball_pos + gravity_up * (self.ball_radius * 0.16)
                probe_dist = max(1.2, float(getattr(self, "player_ai_jump_probe_distance", 1.8)))
                ray_to = ray_from + move_vec * probe_dist
                try:
                    hit = self.physics_world.rayTestClosest(ray_from, ray_to)
                    if hit.hasHit() and hit.getNode() is not None and hit.getNode() != self.ball_body:
                        need_jump = True
                except Exception:
                    pass
            if not need_jump:
                chase_mul = max(1.05, float(getattr(self, "player_ai_jump_chase_distance_mul", 1.75)))
                chase_chance_per_sec = max(0.0, float(getattr(self, "player_ai_jump_chase_chance_per_sec", 0.85)))
                if target_dist_4d > engage_dist * chase_mul:
                    jump_roll = min(1.0, chase_chance_per_sec * max(0.0, dt))
                    if random.random() < jump_roll:
                        need_jump = True
            if not need_jump:
                roam_chance_per_sec = max(0.0, float(getattr(self, "player_ai_jump_roam_chance_per_sec", 0.22)))
                moving_enough = move_vec.lengthSquared() > 0.04
                if moving_enough:
                    jump_roll = min(1.0, roam_chance_per_sec * max(0.0, dt))
                    if random.random() < jump_roll:
                        need_jump = True
            if need_jump:
                self.jump_queued = True
                self.player_ai_jump_cooldown = random.uniform(0.32, 0.55)

        combat_hold = False
        combat_hold_scale = 1.0
        if self.attack_mode == "idle" and self.attack_cooldown <= 0.0:
            bomb_range = max(self.sword_reach * 0.9, self.sword_reach * float(getattr(self, "player_ai_bomb_range_mul", 1.9)))
            bomb_desire = self._clamp(float(getattr(self, "player_ai_bomb_desire", 0.42)), 0.0, 1.0)
            missile_range = max(self.sword_reach * 1.5, self.sword_reach * float(getattr(self, "player_ai_missile_range_mul", 4.4)))
            missile_desire = self._clamp(float(getattr(self, "player_ai_missile_desire", 0.58)), 0.0, 1.0)
            throw_range = max(self.sword_reach * 0.95, self.sword_reach * float(getattr(self, "player_ai_throw_range_mul", 2.2)))
            throw_desire = self._clamp(float(getattr(self, "player_ai_throw_desire", 0.46)), 0.0, 1.0)
            missile_min_range = max(self.sword_reach * 0.85, self.sword_reach * float(getattr(self, "player_ai_missile_min_range_mul", 1.08)))

            crowd_close_count = 0
            crowd_mid_count = 0
            bomb_cluster_radius = max(self.sword_reach * 0.75, bomb_range * float(getattr(self, "player_ai_bomb_cluster_radius_mul", 1.18)))
            for monster in alive:
                root = monster.get("root")
                if root is None or root.isEmpty():
                    continue
                dist = self._distance4d(ball_pos, player_w, root.getPos(), float(monster.get("w", 0.0)))
                if dist <= bomb_cluster_radius:
                    crowd_close_count += 1
                if dist <= throw_range:
                    crowd_mid_count += 1

            cluster_min = max(2, int(getattr(self, "player_ai_bomb_cluster_min", 2)))
            close_quarters = target_dist_4d <= self.sword_reach * 0.62
            bomb_zone = target_dist_4d <= bomb_range
            missile_zone = target_dist_4d >= missile_min_range and target_dist_4d <= missile_range

            hold_dist_mul = max(0.8, float(getattr(self, "player_ai_combat_hold_distance_mul", 1.35)))
            hold_dist = max(self.sword_reach * 0.75, self.sword_reach * hold_dist_mul)
            hold_height_tol = max(0.2, float(getattr(self, "player_ai_combat_hold_height_tolerance", 1.25)))
            hold_scale = self._clamp(float(getattr(self, "player_ai_combat_hold_move_scale", 0.2)), 0.0, 1.0)
            if target_dist_4d <= hold_dist and abs(target_up_delta) <= hold_height_tol:
                combat_hold = True
                combat_hold_scale = min(combat_hold_scale, hold_scale)
            if bomb_zone and (close_quarters or crowd_close_count >= cluster_min):
                combat_hold = True
                combat_hold_scale = min(combat_hold_scale, hold_scale * 0.75)

            should_missile = (
                missile_zone
                and self.magic_missile_cooldown <= 0.0
                and target_dist_4d > self.sword_reach * 0.82
                and (target_up_delta > 0.12 or crowd_mid_count <= 1 or self.player_ai_combo_step % 5 == 0)
                and (self.player_ai_combo_step % 4 == 2 or random.random() < missile_desire)
            )
            should_bomb = (
                bomb_zone
                and self.hyperbomb_cooldown <= 0.0
                and (
                    close_quarters
                    or crowd_close_count >= cluster_min
                    or self.player_ai_combo_step % 3 == 1
                    or random.random() < bomb_desire
                )
            )

            if should_bomb:
                self._trigger_hyperbomb()
                self.player_ai_combo_step = (self.player_ai_combo_step + 1) % 8
                self.player_ai_combo_timer = random.uniform(0.2, 0.4)
            elif should_missile:
                self._trigger_magic_missile_attack()
                self.player_ai_combo_step = (self.player_ai_combo_step + 1) % 8
                self.player_ai_combo_timer = random.uniform(0.18, 0.34)
            elif target_dist_4d <= self.sword_reach * 0.95:
                if self.player_ai_combo_timer <= 0.0:
                    self.player_ai_combo_step = (self.player_ai_combo_step + 1) % 8
                should_throw = (
                    target_dist_4d >= self.sword_reach * 0.58
                    and target_dist_4d <= throw_range
                    and crowd_close_count < cluster_min
                    and (self.player_ai_combo_step in (1, 5, 7) or random.random() < throw_desire)
                )
                favor_spin = close_quarters or crowd_close_count >= cluster_min or self.player_ai_combo_step in (0, 2, 4, 6)
                if should_throw:
                    self._trigger_throw_attack()
                elif favor_spin:
                    self._trigger_spin_attack()
                else:
                    self._trigger_swing_attack()
                self.player_ai_combo_timer = random.uniform(0.12, 0.26)
            elif target_dist_4d <= throw_range and random.random() < throw_desire * 0.65:
                self._trigger_throw_attack()
                self.player_ai_combo_step = (self.player_ai_combo_step + 1) % 8
                self.player_ai_combo_timer = random.uniform(0.16, 0.3)

        if not combat_hold and target_dist_4d <= self.sword_reach * 0.95 and abs(target_up_delta) <= self.ball_radius * 2.2:
            combat_hold = True
            fallback_hold = self._clamp(float(getattr(self, "player_ai_combat_hold_move_scale", 0.2)) * 0.85, 0.0, 1.0)
            combat_hold_scale = min(combat_hold_scale, fallback_hold)
        if combat_hold:
            move_vec *= max(0.0, combat_hold_scale)
            if move_vec.lengthSquared() <= 1e-6:
                return None

        return move_vec

    def _estimate_room_level_for_z(self, z: float) -> int:
        if self.level_z_step <= 1e-6:
            return 0
        return int(round((z - self.floor_y) / self.level_z_step))

    def _clamp_camera_to_current_room_bounds(self, camera_pos: Vec3, reference_pos: Vec3) -> Vec3:
        room_idx = self._get_current_room_idx_for_pos(reference_pos)
        if room_idx is None:
            return Vec3(camera_pos)

        room = self.rooms[room_idx]
        level = self.room_levels.get(room_idx, 0)
        base_z = self._level_base_z(level)
        margin = max(0.24, self.camera_collision_radius + 0.08)
        min_z = base_z + margin
        max_z = base_z + self.wall_h - margin
        if max_z < min_z:
            mid = base_z + self.wall_h * 0.5
            min_z = mid
            max_z = mid

        clamped = Vec3(camera_pos)
        clamped.x = self._clamp(clamped.x, room.x + margin, room.x + room.w - margin)
        clamped.y = self._clamp(clamped.y, room.y + margin, room.y + room.h - margin)
        clamped.z = self._clamp(clamped.z, min_z, max_z)
        return clamped

    def _get_room_zone_gravity(self, pos: Vec3) -> Vec3:
        zone = self._get_zone_for_position(pos)
        if zone is None:
            self.active_room_zone_idx = -1
            return Vec3(0, 0, -self.gravity_magnitude)
        self.active_room_zone_idx = zone["room_idx"]
        return Vec3(zone["gravity"])

    def _update_room_gravity_particles(self, dt: float, focus_pos: Vec3) -> None:
        if not self.room_gravity_zones:
            return

        for zone in self.room_gravity_zones:
            root = zone.get("particles_root")
            if root is None or root.isEmpty():
                continue

            room = zone["room"]
            room_idx = zone["room_idx"]
            base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            center = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + self.wall_h * 0.5)
            if (center - focus_pos).lengthSquared() > (150.0 * 150.0):
                if not root.isStashed():
                    root.stash()
                continue
            if root.isStashed():
                root.unstash()

            g = Vec3(zone["gravity"])
            if g.lengthSquared() < 1e-8:
                continue
            g_dir = Vec3(g)
            g_dir.normalize()
            flow_speed = zone["flow_speed"]

            min_x = room.x + 0.8
            max_x = room.x + room.w - 0.8
            min_y = room.y + 0.8
            max_y = room.y + room.h - 0.8
            min_z = base_z + 0.45
            max_z = base_z + self.wall_h - 0.7

            for entry in zone["particles"]:
                node = entry["node"]
                if node is None or node.isEmpty():
                    continue

                pos = Vec3(entry["pos"])
                pos += g_dir * (flow_speed * entry["speed"] * dt)
                lateral = Vec3(entry["drift_x"], entry["drift_y"], 0)
                lateral *= 0.22 + 0.18 * math.sin(self.roll_time * 2.5 + entry["phase"])
                pos += lateral * dt

                if pos.x < min_x:
                    pos.x = max_x
                elif pos.x > max_x:
                    pos.x = min_x
                if pos.y < min_y:
                    pos.y = max_y
                elif pos.y > max_y:
                    pos.y = min_y
                if pos.z < min_z:
                    pos.z = max_z
                elif pos.z > max_z:
                    pos.z = min_z

                pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(self.roll_time * 4.2 + entry["phase"]))
                node.setPos(pos)
                node.setAlphaScale(0.25 + 0.65 * pulse)
                entry["pos"] = pos

    def _rotate_around_axis(self, vec: Vec3, axis: Vec3, angle_rad: float) -> Vec3:
        return rotate_around_axis(vec, axis, angle_rad)

    def _camera_orbit_position(self, target: Vec3, heading_deg: float, pitch_deg: float, dist: float) -> Vec3:
        return camera_orbit_position(target, heading_deg, pitch_deg, dist)

    def _resolve_camera_collision(self, target: Vec3, desired: Vec3) -> Vec3:
        return resolve_camera_collision(self, target, desired)

    def _resolve_camera_tight(self, target: Vec3, desired: Vec3) -> Vec3:
        ray = desired - target
        dist = ray.length()
        if dist < 1e-5:
            return Vec3(desired)

        ray_dir = ray / dist
        allowed = dist
        hit = self.physics_world.rayTestClosest(target, desired)
        if hit.hasHit():
            node = hit.getNode()
            if node is not None and node != self.ball_body:
                hit_dist = (hit.getHitPos() - target).length()
                allowed = min(allowed, max(self.camera_min_distance, hit_dist - 0.18))

        return target + ray_dir * allowed

    def _enforce_camera_above_ball(self, ball_pos: Vec3, camera_pos: Vec3, up_axis: Vec3, min_up_offset: float = 0.55) -> Vec3:
        up = Vec3(up_axis)
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        rel = camera_pos - ball_pos
        up_component = rel.dot(up)
        if up_component >= min_up_offset:
            return Vec3(camera_pos)

        planar = rel - up * up_component
        return ball_pos + planar + up * min_up_offset

    def _load_first_sfx(self, base_names: list[str]):
        return load_first_sfx(self, base_names)

    def _load_first_model(self, model_paths: list[str]):
        for path in model_paths:
            if not os.path.exists(path):
                continue
            try:
                model_np = self.loader.loadModel(path)
            except Exception:
                model_np = None
            if model_np is not None and not model_np.isEmpty():
                return model_np
        return None

    def _create_fallback_cube_model(self) -> NodePath:
        root = NodePath("fallback-cube")
        faces = [
            ((0.0, 0.5, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, -0.5, 0.0), (180.0, 0.0, 0.0)),
            ((0.5, 0.0, 0.0), (90.0, 0.0, 0.0)),
            ((-0.5, 0.0, 0.0), (-90.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.5), (0.0, -90.0, 0.0)),
            ((0.0, 0.0, -0.5), (0.0, 90.0, 0.0)),
        ]
        for idx, (pos, hpr) in enumerate(faces):
            cm = CardMaker(f"fallback-cube-face-{idx}")
            cm.setFrame(-0.5, 0.5, -0.5, 0.5)
            face_np = root.attachNewNode(cm.generate())
            face_np.setPos(*pos)
            face_np.setHpr(*hpr)
            face_np.setTwoSided(True)
        return root

    def _create_fallback_sphere_model(self) -> NodePath:
        lat_segments = 14
        lon_segments = 20
        vdata = GeomVertexData("fallback-sphere", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        uv_writer = GeomVertexWriter(vdata, "texcoord")

        for lat in range(lat_segments + 1):
            theta = math.pi * (lat / lat_segments)
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            for lon in range(lon_segments + 1):
                phi = (2.0 * math.pi) * (lon / lon_segments)
                x = sin_t * math.cos(phi)
                y = sin_t * math.sin(phi)
                z = cos_t
                v_writer.addData3(x * 0.5, y * 0.5, z * 0.5)
                n_writer.addData3(x, y, z)
                uv_writer.addData2(lon / lon_segments, 1.0 - (lat / lat_segments))

        tris = GeomTriangles(Geom.UHStatic)
        row = lon_segments + 1
        for lat in range(lat_segments):
            for lon in range(lon_segments):
                a = lat * row + lon
                b = a + 1
                c = (lat + 1) * row + lon
                d = c + 1
                tris.addVertices(a, c, b)
                tris.addVertices(b, c, d)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("fallback-sphere")
        node.addGeom(geom)
        return NodePath(node)

    def _create_fallback_cylinder_model(self, radius: float = 0.5, height: float = 1.0, segments: int = 20) -> NodePath:
        seg = max(8, int(segments))
        half_h = max(0.05, float(height) * 0.5)
        vdata = GeomVertexData("fallback-cylinder", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        uv_writer = GeomVertexWriter(vdata, "texcoord")
        tris = GeomTriangles(Geom.UHStatic)

        side_top_idx: list[int] = []
        side_bottom_idx: list[int] = []
        for i in range(seg + 1):
            t = i / seg
            ang = math.tau * t
            cx = math.cos(ang)
            cy = math.sin(ang)
            nx = cx
            ny = cy

            top_idx = v_writer.getWriteRow()
            v_writer.addData3(cx * radius, cy * radius, half_h)
            n_writer.addData3(nx, ny, 0.0)
            uv_writer.addData2(t, 1.0)
            side_top_idx.append(top_idx)

            bot_idx = v_writer.getWriteRow()
            v_writer.addData3(cx * radius, cy * radius, -half_h)
            n_writer.addData3(nx, ny, 0.0)
            uv_writer.addData2(t, 0.0)
            side_bottom_idx.append(bot_idx)

        for i in range(seg):
            a = side_top_idx[i]
            b = side_top_idx[i + 1]
            c = side_bottom_idx[i]
            d = side_bottom_idx[i + 1]
            tris.addVertices(a, c, b)
            tris.addVertices(b, c, d)

        top_center = v_writer.getWriteRow()
        v_writer.addData3(0.0, 0.0, half_h)
        n_writer.addData3(0.0, 0.0, 1.0)
        uv_writer.addData2(0.5, 0.5)
        top_ring: list[int] = []
        for i in range(seg + 1):
            t = i / seg
            ang = math.tau * t
            cx = math.cos(ang)
            cy = math.sin(ang)
            idx = v_writer.getWriteRow()
            v_writer.addData3(cx * radius, cy * radius, half_h)
            n_writer.addData3(0.0, 0.0, 1.0)
            uv_writer.addData2(0.5 + 0.5 * cx, 0.5 + 0.5 * cy)
            top_ring.append(idx)
        for i in range(seg):
            tris.addVertices(top_center, top_ring[i], top_ring[i + 1])

        bottom_center = v_writer.getWriteRow()
        v_writer.addData3(0.0, 0.0, -half_h)
        n_writer.addData3(0.0, 0.0, -1.0)
        uv_writer.addData2(0.5, 0.5)
        bottom_ring: list[int] = []
        for i in range(seg + 1):
            t = i / seg
            ang = math.tau * t
            cx = math.cos(ang)
            cy = math.sin(ang)
            idx = v_writer.getWriteRow()
            v_writer.addData3(cx * radius, cy * radius, -half_h)
            n_writer.addData3(0.0, 0.0, -1.0)
            uv_writer.addData2(0.5 + 0.5 * cx, 0.5 + 0.5 * cy)
            bottom_ring.append(idx)
        for i in range(seg):
            tris.addVertices(bottom_center, bottom_ring[i + 1], bottom_ring[i])

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("fallback-cylinder")
        node.addGeom(geom)
        return NodePath(node)

    def _create_fallback_cone_model(self, radius: float = 0.5, height: float = 1.0, segments: int = 20) -> NodePath:
        seg = max(8, int(segments))
        half_h = max(0.05, float(height) * 0.5)
        tip_z = half_h
        base_z = -half_h
        slope = max(1e-6, math.sqrt(radius * radius + height * height))
        nx_scale = float(height) / slope
        nz_base = float(radius) / slope

        vdata = GeomVertexData("fallback-cone", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        uv_writer = GeomVertexWriter(vdata, "texcoord")
        tris = GeomTriangles(Geom.UHStatic)

        tip_ring: list[int] = []
        base_ring: list[int] = []
        for i in range(seg + 1):
            t = i / seg
            ang = math.tau * t
            cx = math.cos(ang)
            cy = math.sin(ang)
            nx = cx * nx_scale
            ny = cy * nx_scale

            tip_idx = v_writer.getWriteRow()
            v_writer.addData3(0.0, 0.0, tip_z)
            n_writer.addData3(nx, ny, nz_base)
            uv_writer.addData2(t, 1.0)
            tip_ring.append(tip_idx)

            base_idx = v_writer.getWriteRow()
            v_writer.addData3(cx * radius, cy * radius, base_z)
            n_writer.addData3(nx, ny, nz_base)
            uv_writer.addData2(t, 0.0)
            base_ring.append(base_idx)

        for i in range(seg):
            tris.addVertices(tip_ring[i], base_ring[i], base_ring[i + 1])

        base_center = v_writer.getWriteRow()
        v_writer.addData3(0.0, 0.0, base_z)
        n_writer.addData3(0.0, 0.0, -1.0)
        uv_writer.addData2(0.5, 0.5)
        base_cap: list[int] = []
        for i in range(seg + 1):
            t = i / seg
            ang = math.tau * t
            cx = math.cos(ang)
            cy = math.sin(ang)
            idx = v_writer.getWriteRow()
            v_writer.addData3(cx * radius, cy * radius, base_z)
            n_writer.addData3(0.0, 0.0, -1.0)
            uv_writer.addData2(0.5 + 0.5 * cx, 0.5 + 0.5 * cy)
            base_cap.append(idx)
        for i in range(seg):
            tris.addVertices(base_center, base_cap[i + 1], base_cap[i])

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("fallback-cone")
        node.addGeom(geom)
        return NodePath(node)

    def _load_first_sfx_2d(self, base_names: list[str]):
        for base in base_names:
            candidates = [base]
            if "." not in base.split("/")[-1]:
                candidates.extend([f"{base}.wav", f"{base}.ogg", f"{base}.mp3"])
            for path in candidates:
                if not os.path.exists(path):
                    continue
                sfx = self.loader.loadSfx(path)
                if sfx:
                    return sfx
        return None

    def _load_game_over_sfx_bank(self) -> list:
        bank: list = []
        folder = os.path.join("soundfx", "gameover_")
        if not os.path.isdir(folder):
            return bank

        try:
            names = sorted(os.listdir(folder))
        except Exception:
            return bank

        for name in names:
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, name).replace("\\", "/")
            if not os.path.exists(path):
                continue
            try:
                sfx = self.loader.loadSfx(path)
            except Exception:
                sfx = None
            if sfx:
                bank.append(sfx)
        return bank

    def _play_game_over_sfx(self) -> None:
        bank = getattr(self, "sfx_game_over_bank", None)
        if not bank:
            return
        candidates = bank
        last_clip = getattr(self, "sfx_game_over_last", None)
        if last_clip is not None and len(bank) > 1:
            filtered = [clip for clip in bank if clip is not last_clip]
            if filtered:
                candidates = filtered
        try:
            clip = random.choice(candidates)
        except Exception:
            return
        self.sfx_game_over_last = clip
        vol = 0.9 * float(getattr(self, "voiceover_volume_scale", 1.0))
        self._play_sound(clip, volume=vol, play_rate=1.0)

    def _load_win_sfx_bank(self) -> list:
        bank: list = []
        folder = os.path.join("soundfx", "win_")
        if not os.path.isdir(folder):
            return bank

        try:
            names = sorted(os.listdir(folder))
        except Exception:
            return bank

        for name in names:
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, name).replace("\\", "/")
            if not os.path.exists(path):
                continue
            try:
                sfx = self.loader.loadSfx(path)
            except Exception:
                sfx = None
            if sfx:
                bank.append(sfx)
        return bank

    def _play_win_sfx(self) -> None:
        bank = getattr(self, "sfx_win_bank", None)
        if not bank:
            return
        candidates = bank
        last_clip = getattr(self, "sfx_win_last", None)
        if last_clip is not None and len(bank) > 1:
            filtered = [clip for clip in bank if clip is not last_clip]
            if filtered:
                candidates = filtered
        try:
            clip = random.choice(candidates)
        except Exception:
            return
        self.sfx_win_last = clip
        vol = 0.92 * float(getattr(self, "voiceover_volume_scale", 1.0))
        self._play_sound(clip, volume=vol, play_rate=1.0)

    def _load_kill_sfx_bank(self) -> list:
        bank: list = []
        folder = os.path.join("soundfx", "kill_")
        if not os.path.isdir(folder):
            return bank

        try:
            names = sorted(os.listdir(folder))
        except Exception:
            return bank

        for name in names:
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, name).replace("\\", "/")
            if not os.path.exists(path):
                continue
            try:
                sfx = self.loader.loadSfx(path)
            except Exception:
                sfx = None
            if sfx:
                bank.append(sfx)
        return bank

    def _play_kill_sfx(self) -> bool:
        bank = getattr(self, "sfx_kill_bank", None)
        if not bank:
            return False
        candidates = bank
        last_clip = getattr(self, "sfx_kill_last", None)
        if last_clip is not None and len(bank) > 1:
            filtered = [clip for clip in bank if clip is not last_clip]
            if filtered:
                candidates = filtered
        try:
            clip = random.choice(candidates)
        except Exception:
            return False
        self.sfx_kill_last = clip
        vol = 0.9 * float(getattr(self, "voiceover_volume_scale", 1.0))
        self._play_sound(clip, volume=vol, play_rate=1.0)
        return True

    def _load_critical_damage_sfx_bank(self) -> list:
        bank: list = []
        folder = os.path.join("soundfx", "critical_damage_")
        if not os.path.isdir(folder):
            return bank

        try:
            names = sorted(os.listdir(folder))
        except Exception:
            return bank

        for name in names:
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, name).replace("\\", "/")
            if not os.path.exists(path):
                continue
            try:
                sfx = self.loader.loadSfx(path)
            except Exception:
                sfx = None
            if sfx:
                bank.append(sfx)
        return bank

    def _play_critical_damage_sfx(self) -> bool:
        bank = getattr(self, "sfx_critical_damage_bank", None)
        if not bank:
            return False
        candidates = bank
        last_clip = getattr(self, "sfx_critical_damage_last", None)
        if last_clip is not None and len(bank) > 1:
            filtered = [clip for clip in bank if clip is not last_clip]
            if filtered:
                candidates = filtered
        try:
            clip = random.choice(candidates)
        except Exception:
            return False
        self.sfx_critical_damage_last = clip
        vol = 0.95 * float(getattr(self, "voiceover_volume_scale", 1.0))
        self._play_sound(clip, volume=vol, play_rate=1.0 + random.uniform(-0.02, 0.02))
        return True

    def _collect_bgm_tracks(self) -> list[str]:
        bgm_dir = "bgm"
        if not os.path.isdir(bgm_dir):
            return []
        exts = {".ogg", ".mp3", ".wav", ".flac", ".m4a"}
        tracks: list[str] = []
        for root, _dirs, files in os.walk(bgm_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in exts:
                    continue
                full = os.path.join(root, name)
                tracks.append(full.replace("\\", "/"))
        return tracks

    def _start_random_bgm_loop(self) -> None:
        tracks = self._collect_bgm_tracks()
        if not tracks:
            self.bgm_track = None
            self.bgm_track_path = None
            return
        random.shuffle(tracks)
        for path in tracks:
            try:
                music = self.loader.loadMusic(path)
            except Exception:
                music = None
            if not music:
                continue
            try:
                music.setLoop(True)
                music.setVolume(max(0.0, min(1.0, float(getattr(self, "bgm_volume", 0.48)))))
                music.play()
                self.bgm_track = music
                self.bgm_track_path = path
                return
            except Exception:
                continue
        self.bgm_track = None
        self.bgm_track_path = None

    def _queue_jump(self) -> None:
        if self.game_over_active:
            return
        self._play_sound(self.sfx_qigong_jump, volume=0.9, play_rate=1.0)
        queue_jump(self)

    def _on_escape_pressed(self) -> None:
        try:
            self.userExit()
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass

    def _on_mouse_wheel(self, direction: float) -> None:
        step = float(getattr(self, "scroll_lift_step", 0.45))
        self.scroll_lift_target = self._clamp(
            float(getattr(self, "scroll_lift_target", 0.0)) + float(direction) * step,
            -1.0,
            1.0,
        )

    def _play_sound(self, sound, volume: float, play_rate: float) -> None:
        mix = max(0.0, min(1.0, float(getattr(self, "audio_hyper_mix", 0.0))))
        norm_volume = volume * (1.0 - 0.32 * mix) + 0.2 * mix
        norm_rate = play_rate * (1.0 - 0.26 * mix) + 1.0 * (0.26 * mix)
        play_sound(sound, norm_volume, norm_rate)

    def _setup_monster_ai_system(self) -> None:
        self._teardown_monster_ai_system()
        if not self.ai_enabled or not self.monsters:
            self.ai_world = None
            for monster in self.monsters:
                monster["ai_char"] = None
                monster["ai_behaviors"] = None
            return
        try:
            self.ai_world = AIWorld(self.render)
        except Exception:
            self.ai_world = None
            self.ai_enabled = False
            return

        for idx, monster in enumerate(self.monsters):
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            try:
                ai_char = AICharacter(f"monster-ai-{idx}", root, 42.0, 0.08, 3.2)
                self.ai_world.addAiChar(ai_char)
                monster["ai_char"] = ai_char
                monster["ai_behaviors"] = ai_char.getAiBehaviors()
                self._apply_monster_ai_state(monster, "wandering")
            except Exception:
                monster["ai_char"] = None
                monster["ai_behaviors"] = None

    def _teardown_monster_ai_system(self) -> None:
        world = getattr(self, "ai_world", None)
        if world is None:
            return

        unresolved_chars: list = []
        for monster in self.monsters:
            ai_char = monster.get("ai_char")
            if ai_char is None:
                monster["ai_char"] = None
                monster["ai_behaviors"] = None
                continue

            removed = False
            try:
                ai_name = str(ai_char.getName())
            except Exception:
                ai_name = ""
            if ai_name:
                try:
                    world.removeAiChar(ai_name)
                    removed = True
                except Exception:
                    removed = False
            if not removed:
                try:
                    world.removeAiChar(ai_char)
                    removed = True
                except Exception:
                    removed = False

            try:
                behaviors = ai_char.getAiBehaviors()
                if behaviors is not None:
                    behaviors.removeAi("all")
            except Exception:
                pass

            if not removed:
                unresolved_chars.append(ai_char)

            monster["ai_char"] = None
            monster["ai_behaviors"] = None

        self.ai_world = None
        if unresolved_chars:
            self._retained_ai_chars.extend(unresolved_chars)
            self._retained_ai_worlds.append(world)
            if len(self._retained_ai_chars) > 128:
                self._retained_ai_chars = self._retained_ai_chars[-128:]
            if len(self._retained_ai_worlds) > 8:
                self._retained_ai_worlds = self._retained_ai_worlds[-8:]

    def _set_monster_state(self, monster: dict, new_state: str, announce: bool = False) -> None:
        old_state = monster.get("state", "wandering")
        if old_state == new_state:
            return
        monster["state"] = new_state
        monster["state_timer"] = 0.0
        auto_announce_states = {"guarding", "hunting", "attacking", "running"}
        announce_cooldown = float(monster.get("state_announce_cooldown", 0.0))
        if not announce and new_state in auto_announce_states and announce_cooldown <= 0.0:
            announce = True
            monster["state_announce_cooldown"] = 0.85

        label_np = monster.get("state_label")
        state_styles = {
            "wandering": ("WANDER", (0.65, 0.8, 1.0, 0.85)),
            "guarding": ("GUARD", (0.36, 1.0, 0.85, 0.9)),
            "hunting": ("HUNT", (1.0, 0.85, 0.3, 0.95)),
            "attacking": ("ATTACK", (1.0, 0.28, 0.28, 1.0)),
            "running": ("RUN", (1.0, 1.0, 0.35, 0.95)),
            "hit": ("HIT", (0.35, 0.75, 1.0, 1.0)),
            "dying": ("DYING", (1.0, 0.3, 0.6, 1.0)),
        }
        text, color = state_styles.get(new_state, (new_state.upper(), (0.8, 0.9, 1.0, 0.9)))

        if label_np is not None and not label_np.isEmpty():
            node = label_np.node()
            if isinstance(node, TextNode):
                node.setText(text)
                node.setTextColor(color)

        if announce:
            root = monster.get("root")
            if root is not None and not root.isEmpty():
                self._spawn_floating_text(root.getPos() + Vec3(0, 0, 1.3), text, color, scale=0.2, life=0.55)

    def _apply_monster_ai_state(self, monster: dict, state: str) -> None:
        behaviors = monster.get("ai_behaviors")
        if behaviors is None:
            return
        if monster.get("ai_state") == state:
            return
        monster["ai_state"] = state

        try:
            behaviors.removeAi("all")
        except Exception:
            pass

        try:
            if state in ("hunting", "attacking"):
                behaviors.pursue(self.ball_np)
            elif state == "running":
                behaviors.evade(self.ball_np)
            elif state == "guarding":
                behaviors.seek(self.ball_np)
            else:
                behaviors.wander(0.72, 0.45, 8.0, 2.8)
        except Exception:
            pass

    def _create_checker_texture(
        self,
        size: int = 128,
        cells: int = 8,
        color_a: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        color_b: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> Texture:
        return create_checker_texture(size=size, cells=cells, color_a=color_a, color_b=color_b)

    def _create_shadow_texture(self, size: int = 128) -> Texture:
        return create_shadow_texture(size=size)

    def _create_fractal_symmetry_texture(self, size: int = 256, symmetry: int = 6, seed: float = 0.0) -> Texture:
        img = PNMImage(size, size, 4)
        inv = 1.0 / max(1.0, float(size - 1))
        s = max(2, int(symmetry))

        for y in range(size):
            v = y * inv * 2.0 - 1.0
            for x in range(size):
                u = x * inv * 2.0 - 1.0
                r = math.sqrt(u * u + v * v)
                a = math.atan2(v, u)

                petal = 0.5 + 0.5 * math.cos(a * s + seed * 9.0)
                ring = 0.5 + 0.5 * math.cos((r * 10.5 - seed * 3.7) * math.tau)
                swirl = 0.5 + 0.5 * math.sin((u * 5.8 + v * 4.9 + r * 7.5 + seed * 2.1) * math.tau)

                fract = (petal * 0.48 + ring * 0.32 + swirl * 0.2)
                fract = max(0.0, min(1.0, fract))

                rr = 0.12 + 0.48 * fract
                gg = 0.18 + 0.56 * (0.5 + 0.5 * math.sin(fract * math.tau + seed * 4.0))
                bb = 0.22 + 0.62 * (1.0 - fract * 0.6)
                aa = 0.08 + 0.68 * fract

                img.setXelA(x, y, min(1.0, rr), min(1.0, gg), min(1.0, bb), min(1.0, aa))

        tex = Texture(f"floor-fractal-{symmetry}-{int(seed * 1000)}")
        tex.load(img)
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)
        tex.setMagfilter(Texture.FTLinear)
        tex.setAnisotropicDegree(1 if self.performance_mode else 2)
        return tex

    def _create_water_specular_texture(self, size: int = 256, seed: float = 0.0) -> Texture:
        img = PNMImage(size, size, 4)
        inv = 1.0 / max(1.0, float(size - 1))
        phase = float(seed) * math.tau

        for y in range(size):
            v = y * inv
            for x in range(size):
                u = x * inv

                n0 = 0.5 + 0.5 * math.sin((u * 78.0 + v * 61.0 + phase) * math.tau)
                n1 = 0.5 + 0.5 * math.sin((u * 143.0 - v * 117.0 + phase * 1.73) * math.tau)
                n2 = 0.5 + 0.5 * math.sin(((u + v) * 221.0 + phase * 2.41) * math.tau)
                sparkle = (n0 * 0.45 + n1 * 0.35 + n2 * 0.2)

                tight = max(0.0, (sparkle - 0.58) / 0.42)
                bright = tight ** 4.2
                medium = tight ** 2.1
                alpha = max(bright, medium * 0.36)

                r = min(1.0, 0.72 + bright * 0.36)
                g = min(1.0, 0.82 + bright * 0.28)
                b = min(1.0, 0.96 + bright * 0.18)
                img.setXelA(x, y, r, g, b, min(1.0, alpha))

        tex = Texture(f"water-spec-{int(seed * 1000)}")
        tex.load(img)
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)
        tex.setMagfilter(Texture.FTLinear)
        tex.setAnisotropicDegree(1 if self.performance_mode else 2)
        return tex

    def _get_quad_template(self, name: str, frame: tuple[float, float, float, float]) -> NodePath:
        left, right, bottom, top = frame
        key = (name, left, right, bottom, top)
        cached = self.quad_templates.get(key)
        if cached is not None and not cached.isEmpty():
            return cached

        cm = CardMaker(f"{name}-template")
        cm.setFrame(left, right, bottom, top)
        template = self.quad_template_root.attachNewNode(cm.generate())
        self.quad_templates[key] = template
        return template

    def _instance_quad(self, parent: NodePath, name: str, frame: tuple[float, float, float, float]) -> NodePath:
        return self._get_quad_template(name, frame).instanceTo(parent)

    def _create_sword_stripe_texture(self, size: int = 128, stripes: int = 12) -> Texture:
        img = PNMImage(size, size, 4)
        inv = 1.0 / max(1.0, float(size - 1))
        for y in range(size):
            v = y * inv
            for x in range(size):
                u = x * inv
                sweep = 0.5 + 0.5 * math.sin((u * 6.0 + v * 10.0) * math.tau)
                band = 0.5 + 0.5 * math.sin(v * stripes * math.tau)
                mask = 1.0 if band > 0.35 else 0.0
                r = 0.04 + 0.08 * sweep + 0.12 * mask
                g = 0.18 + 0.42 * sweep + 0.45 * mask
                b = 0.3 + 0.55 * sweep + 0.65 * mask
                a = 0.24 + 0.72 * mask
                img.setXelA(x, y, min(1.0, r), min(1.0, g), min(1.0, b), min(1.0, a))

        tex = Texture("sword-stripes")
        tex.load(img)
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)
        tex.setMagfilter(Texture.FTLinear)
        tex.setAnisotropicDegree(2)
        return tex

    def _create_radial_mirrored_rainbow_texture(self, size: int = 256) -> Texture:
        img = PNMImage(size, size, 4)
        inv = 1.0 / max(1.0, float(size - 1))
        for y in range(size):
            v = y * inv * 2.0 - 1.0
            for x in range(size):
                u = x * inv * 2.0 - 1.0
                r = math.sqrt(u * u + v * v)
                if r > 1.0:
                    img.setXelA(x, y, 0.0, 0.0, 0.0, 0.0)
                    continue

                mirrored = abs(((r * 2.8) % 2.0) - 1.0)
                hue = mirrored
                rr, gg, bb = colorsys.hsv_to_rgb(hue, 0.95, 1.0)

                ring = math.exp(-((r - 0.82) / 0.12) ** 2)
                inner = math.exp(-((r - 0.55) / 0.18) ** 2) * 0.38
                alpha = max(0.0, min(1.0, ring + inner))
                img.setXelA(x, y, rr, gg, bb, alpha)

        tex = Texture("anomaly-radial-mirrored-rainbow")
        tex.load(img)
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)
        tex.setMagfilter(Texture.FTLinear)
        tex.setAnisotropicDegree(2 if not self.performance_mode else 1)
        return tex

    def _create_angular_dial_texture(self, size: int = 256) -> Texture:
        img = PNMImage(size, size, 4)
        inv = 1.0 / max(1.0, float(size - 1))
        for y in range(size):
            v = y * inv * 2.0 - 1.0
            for x in range(size):
                u = x * inv * 2.0 - 1.0
                r = math.sqrt(u * u + v * v)
                if r > 1.0:
                    img.setXelA(x, y, 0.0, 0.0, 0.0, 0.0)
                    continue

                ang = (math.atan2(v, u) / math.tau) % 1.0
                rr, gg, bb = colorsys.hsv_to_rgb(ang, 0.9, 1.0)

                ring = math.exp(-((r - 0.82) / 0.1) ** 2)
                dial_band = math.exp(-((r - 0.62) / 0.08) ** 2) * 0.65
                pointer = max(0.0, 1.0 - abs((ang - 0.0 + 0.5) % 1.0 - 0.5) / 0.05)
                pointer *= math.exp(-((r - 0.76) / 0.22) ** 2)
                alpha = max(0.0, min(1.0, ring + dial_band + pointer * 0.95))

                img.setXelA(x, y, rr, gg, bb, alpha)

        tex = Texture("anomaly-angular-dial")
        tex.load(img)
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)
        tex.setMagfilter(Texture.FTLinear)
        tex.setAnisotropicDegree(2 if not self.performance_mode else 1)
        return tex

    def _setup_toon_rendering(self) -> None:
        if getattr(self, "safe_render_mode", False):
            self.render.setShaderOff(1)
            self.setBackgroundColor(0.03, 0.04, 0.06, 1.0)
            self.filters = None
            return

        self.render.setShaderAuto()
        self.render.setAttrib(LightRampAttrib.makeSingleThreshold(0.52, 0.5))
        self.setBackgroundColor(0.00, 0.00, 0.00, 1.0)
        if getattr(self, "enable_video_distortion", False):
            self.filters = None
            return
        self._enable_toon_filter_fallback()

    def _enable_toon_filter_fallback(self) -> None:
        try:
            self.filters = CommonFilters(self.win, self.cam)
            self.filters.setCartoonInk(separation=1.2)
        except Exception:
            self.filters = None

    def _apply_ball_cube_projection(self, node: NodePath, uv_scale: float = 2.6, uv_offset_u: float = 0.13, uv_offset_v: float = 0.37) -> None:
        if node is None or node.isEmpty():
            return
        try:
            node.clearTexGen()
        except Exception:
            node.clearTexGen()
        node.setTexScale(self.ball_tex_stage, max(0.1, float(uv_scale)), max(0.1, float(uv_scale)), 1.0)
        node.setTexOffset(self.ball_tex_stage, float(uv_offset_u) % 1.0, float(uv_offset_v) % 1.0)

    def _apply_water_cube_projection(self, node: NodePath, uv_scale: float = 2.2, uv_offset_u: float = 0.09, uv_offset_v: float = 0.23) -> None:
        if node is None or node.isEmpty():
            return
        try:
            node.clearTexGen()
        except Exception:
            node.clearTexGen()
        node.setTexScale(self.water_tex_stage, 1.0, 1.0, 1.0)
        node.setTexOffset(self.water_tex_stage, 0.0, 0.0)

    def _update_ball_texture_scroll(self, dt: float) -> None:
        ball_visual = getattr(self, "ball_visual", None)
        if ball_visual is None or ball_visual.isEmpty() or not hasattr(self, "ball_body"):
            return

        du = dt * float(getattr(self, "texture_layer_scroll_u", 0.018))
        dv = dt * float(getattr(self, "texture_layer_scroll_v", 0.013))

        accum_u = (float(getattr(self, "ball_tex_scroll_u", 0.0)) + du) % 1.0
        accum_v = (float(getattr(self, "ball_tex_scroll_v", 0.0)) + dv) % 1.0
        self.ball_tex_scroll_u = accum_u
        self.ball_tex_scroll_v = accum_v

        base_u = float(getattr(self, "ball_tex_base_u", 0.0))
        base_v = float(getattr(self, "ball_tex_base_v", 0.0))
        ball_visual.setTexOffset(self.ball_tex_stage, (base_u + accum_u) % 1.0, (base_v + accum_v) % 1.0)

    def _apply_hypercube_projection(self, node: NodePath, w_coord: float, scale_hint: float = 1.0) -> None:
        if node is None or node.isEmpty():
            return
        stage = TextureStage.getDefault()
        norm_scale = max(0.1, float(scale_hint))
        repeat = self.hyper_uv_repeat_base * (norm_scale ** (1.0 / 3.0))
        repeat = max(self.hyper_uv_repeat_min, min(self.hyper_uv_repeat_max, repeat))
        base_u = (w_coord * 0.173) % 1.0
        base_v = (w_coord * 0.261) % 1.0
        try:
            if self.level_texgen_mode is not None:
                node.setTexGen(stage, self.level_texgen_mode)
            else:
                node.clearTexGen()
        except Exception:
            node.clearTexGen()
        node.setTexScale(stage, repeat, repeat, repeat)
        node.setTexOffset(stage, base_u, base_v)
        if self.level_texgen_mode is None:
            return
        self.hyper_uv_nodes.append(
            {
                "node": node,
                "stage": stage,
                "base_u": base_u,
                "base_v": base_v,
                "w": w_coord,
                "speed": random.uniform(0.04, 0.12),
                "repeat": repeat,
            }
        )

    def _update_hyper_uv_projection(self, t: float) -> None:
        keep: list[dict] = []
        for entry in self.hyper_uv_nodes:
            node = entry["node"]
            if node is None or node.isEmpty():
                continue
            if node.isHidden() or node.isStashed():
                keep.append(entry)
                continue
            w_delta = entry["w"] - self.player_w
            u = (entry["base_u"] + t * entry["speed"] + w_delta * 0.11) % 1.0
            v = (entry["base_v"] + t * entry["speed"] * 0.7 - w_delta * 0.09) % 1.0
            node.setTexScale(entry["stage"], entry["repeat"], entry["repeat"])
            node.setTexOffset(entry["stage"], u, v)
            keep.append(entry)
        self.hyper_uv_nodes = keep

    def _update_dynamic_room_uv(self, t: float) -> None:
        keep: list[dict] = []
        ball_pos = self.ball_np.getPos() if hasattr(self, "ball_np") else None
        water_radius_sq = float(getattr(self, "water_uv_active_radius", 110.0)) ** 2
        disable_arena_water_radius_cull = bool(getattr(self, "four_d_obstacle_arena_mode", False))
        for entry in self.dynamic_room_uv_nodes:
            node = entry["node"]
            if node is None or node.isEmpty():
                continue
            if node.isHidden() or node.isStashed():
                keep.append(entry)
                continue

            stage = entry["stage"]
            mode = entry["mode"]
            base_u = entry["base_u"]
            base_v = entry["base_v"]
            base_ru = entry["base_ru"]
            base_rv = entry["base_rv"]
            speed = entry["speed"]
            phase = entry["phase"]
            w_coord = entry["w"]

            w_delta = w_coord - self.player_w
            w_wave = math.sin(t * (0.8 + speed * 0.22) + phase + w_delta * 0.45)
            room_idx = self._get_current_room_idx_for_pos(entry["center"])
            room_field = self.room_dimension_fields.get(room_idx, {}) if room_idx is not None else {}
            persp_u = float(room_field.get("persp_u", 0.0))
            persp_v = float(room_field.get("persp_v", 0.0))
            persp_mix = 0.5 + 0.5 * math.sin(t * (0.5 + speed * 0.33) + phase)

            if mode == "additive":
                ru = base_ru
                rv = base_rv
                u = (base_u + t * speed * entry["dir"] + w_delta * 0.05 + persp_u * 0.08 * persp_mix) % 1.0
                v = (base_v + t * speed * 0.82 * entry["dir"] - w_delta * 0.04 + persp_v * 0.08 * (1.0 - persp_mix)) % 1.0
                node.setTexScale(stage, max(0.02, ru), max(0.02, rv))
                node.setTexOffset(stage, u, v)
                keep.append(entry)
                continue

            if mode == "wall":
                ru = base_ru * (1.0 + 0.04 * w_wave)
                rv = base_rv * (1.0 + 0.04 * math.cos(t * 0.9 + phase))
                u = (base_u + w_delta * 0.07 + persp_u * 0.12 * persp_mix) % 1.0
                v = (base_v + t * speed * entry["dir"] + w_delta * 0.03 + persp_v * 0.12 * (1.0 - persp_mix)) % 1.0
            elif mode == "ceiling":
                scale_wave = 1.0 + 0.22 * w_wave
                ru = base_ru * scale_wave
                rv = base_rv * scale_wave
                u = (base_u + t * speed * 0.11 + persp_u * 0.09 * persp_mix) % 1.0
                v = (base_v - t * speed * 0.09 + persp_v * 0.09 * (1.0 - persp_mix)) % 1.0
            else:
                is_water = mode == "water"
                if is_water and ball_pos is not None and not disable_arena_water_radius_cull:
                    center = entry.get("center")
                    if center is not None:
                        dx = float(center.x) - float(ball_pos.x)
                        dy = float(center.y) - float(ball_pos.y)
                        if dx * dx + dy * dy > water_radius_sq:
                            keep.append(entry)
                            continue
                if is_water:
                    ru = base_ru
                    rv = base_rv
                    flow_u = 0.23
                    flow_v = 0.18
                    u = (base_u + t * speed * flow_u + w_delta * 0.08) % 1.0
                    v = (base_v + t * speed * flow_v - w_delta * 0.06) % 1.0
                else:
                    scale_wave = 1.0 + 0.24 * w_wave
                    ru = base_ru * scale_wave
                    rv = base_rv * scale_wave
                    flow_u = 0.16
                    flow_v = 0.12
                    u = (base_u + t * speed * flow_u + w_delta * 0.08 + persp_u * 0.17 * persp_mix) % 1.0
                    v = (base_v + t * speed * flow_v - w_delta * 0.06 + persp_v * 0.17 * (1.0 - persp_mix)) % 1.0

                distort = float(getattr(self, "cube_water_distort_strength", 0.085))
                wave_u = math.sin((entry["center"].x * 0.11) + t * (0.62 + speed * 0.3) + phase) * distort
                wave_v = math.cos((entry["center"].y * 0.11) - t * (0.57 + speed * 0.28) + phase * 1.13) * distort
                u = (u + wave_u) % 1.0
                v = (v + wave_v) % 1.0
                if not is_water:
                    pulse_gain = 0.45
                    pulse_scale = 1.0 + (abs(wave_u) + abs(wave_v)) * pulse_gain
                    ru = max(0.02, ru * pulse_scale)
                    rv = max(0.02, rv * pulse_scale)

                center = entry["center"]
                pulse_strength = 0.0
                pulse_u = base_u
                pulse_v = base_v
                pulse_scale = 1.0
                for pulse in self.floor_contact_pulses:
                    age = t - pulse["t0"]
                    if age < 0.0 or age > pulse["life"]:
                        continue
                    radius = pulse["speed"] * age * (1.0 if not is_water else 1.22)
                    dist = math.dist((center.x, center.y), (pulse["origin"].x, pulse["origin"].y))
                    ring_delta = abs(dist - radius)
                    band = pulse["band"] * (1.0 if not is_water else 1.35)
                    if ring_delta > band:
                        continue
                    local = 1.0 - (ring_delta / band)
                    local *= (1.0 - age / pulse["life"])
                    if local > pulse_strength:
                        pulse_strength = local
                        pulse_u = (pulse["origin"].x * 0.067 + pulse["phase"] * 0.13) % 1.0
                        pulse_v = (pulse["origin"].y * 0.073 + pulse["phase"] * 0.11) % 1.0
                        pulse_scale = 1.0 + age * 1.85

                cycle_phase = (t * (0.48 + speed * 0.4) + phase) % 1.0
                cycle_wave = 0.5 + 0.5 * math.sin(cycle_phase * math.tau)
                cycle_alpha = 0.08 + 0.92 * cycle_wave
                cycle_scale = 0.9 + 1.4 * cycle_wave
                if is_water:
                    cycle_scale = 1.1 + 1.85 * cycle_wave
                    if bool(getattr(self, "water_color_cycle_enabled", True)):
                        hue_speed = float(getattr(self, "water_color_cycle_speed", 5.8))
                        sat = self._clamp(float(getattr(self, "water_color_cycle_saturation", 0.92)), 0.0, 1.0)
                        val = self._clamp(float(getattr(self, "water_color_cycle_value", 1.0)), 0.0, 1.0)
                        alpha_base = self._clamp(float(getattr(self, "water_color_cycle_alpha", 0.34)), 0.0, 1.0)
                        smoothness = self._clamp(float(getattr(self, "water_color_cycle_smoothness", 0.24)), 0.0, 1.0)
                        hue = (t * hue_speed * 0.11 + phase * 0.17 + (center.x + center.y) * 0.0021) % 1.0
                        wr, wg, wb = colorsys.hsv_to_rgb(hue, sat, val)
                        pulse = 0.78 + 0.22 * cycle_wave
                        target_col = (
                            0.16 + wr * 0.84 * pulse,
                            0.16 + wg * 0.84 * pulse,
                            0.16 + wb * 0.84 * pulse,
                            alpha_base,
                        )
                        prev_col = entry.get("water_cycle_col")
                        if not isinstance(prev_col, tuple) or len(prev_col) != 4:
                            smoothed_col = target_col
                        else:
                            smoothed_col = (
                                prev_col[0] + (target_col[0] - prev_col[0]) * smoothness,
                                prev_col[1] + (target_col[1] - prev_col[1]) * smoothness,
                                prev_col[2] + (target_col[2] - prev_col[2]) * smoothness,
                                prev_col[3] + (target_col[3] - prev_col[3]) * smoothness,
                            )
                        entry["water_cycle_col"] = smoothed_col
                        node.setColor(smoothed_col)
                    else:
                        node.setColor(0.26, 0.46, 0.72, 0.56)

                layer_a = entry.get("layer_a")
                layer_b = entry.get("layer_b")
                layer_c = entry.get("layer_c")
                if layer_a is not None:
                    node.setTexScale(layer_a, max(0.02, base_ru * pulse_scale), max(0.02, base_rv * pulse_scale))
                    node.setTexOffset(layer_a, pulse_u, pulse_v)
                    max_a = 0.42 if not is_water else 0.58
                    gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
                    if is_water:
                        gain *= float(getattr(self, "water_additive_gain_scale", 0.62))
                        max_a *= self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                    layer_a.setColor((gain, gain, gain, max(0.0, min(max_a, pulse_strength * max_a))))
                if layer_b is not None:
                    under_scale = 0.92 + pulse_strength * (1.2 + cycle_wave * 0.55)
                    if is_water:
                        under_scale = 1.02 + pulse_strength * (1.6 + cycle_wave * 0.75)
                    node.setTexScale(layer_b, max(0.02, base_ru * under_scale), max(0.02, base_rv * under_scale))
                    node.setTexOffset(layer_b, (pulse_u + cycle_phase * 0.19) % 1.0, (pulse_v - cycle_phase * 0.14) % 1.0)
                    min_b = 0.08 if not is_water else 0.14
                    max_b = 0.38 if not is_water else 0.52
                    gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
                    if is_water:
                        gain *= float(getattr(self, "water_additive_gain_scale", 0.62))
                        op_scale = self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                        min_b *= op_scale
                        max_b *= op_scale
                    layer_b.setColor((gain, gain, gain, max(min_b, min(max_b, (1.0 - pulse_strength) * 0.34 + pulse_strength * 0.24))))
                if layer_c is not None:
                    if is_water:
                        spec_repeat = max(1.0, float(entry.get("water_spec_repeat", getattr(self, "water_specular_detail_repeat", 7.5))))
                        spec_speed = max(0.1, float(entry.get("water_spec_speed", getattr(self, "water_specular_scroll_speed", 1.85))))
                        spec_scale = 0.75 + 1.1 * cycle_wave
                        node.setTexScale(layer_c, max(0.02, base_ru * spec_repeat * spec_scale), max(0.02, base_rv * spec_repeat * spec_scale))
                        node.setTexOffset(
                            layer_c,
                            (base_u + cycle_phase * (0.42 + spec_speed * 0.12) + t * spec_speed * 0.09) % 1.0,
                            (base_v + cycle_phase * (0.38 + spec_speed * 0.1) - t * spec_speed * 0.07) % 1.0,
                        )
                    else:
                        node.setTexScale(layer_c, max(0.02, base_ru * cycle_scale), max(0.02, base_rv * cycle_scale))
                        node.setTexOffset(layer_c, (base_u + cycle_phase * 0.31) % 1.0, (base_v + cycle_phase * 0.27) % 1.0)
                    gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
                    if is_water:
                        gain *= float(getattr(self, "water_additive_gain_scale", 0.62))
                    if is_water and bool(getattr(self, "water_color_cycle_enabled", True)):
                        op_scale = self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                        smoothness = self._clamp(float(getattr(self, "water_color_cycle_smoothness", 0.24)), 0.0, 1.0)
                        spec_target = (
                            gain * (0.52 + wr * 0.95),
                            gain * (0.52 + wg * 0.95),
                            gain * (0.52 + wb * 0.95),
                            max(0.0, min(1.0, cycle_alpha * op_scale)),
                        )
                        prev_spec = entry.get("water_spec_cycle_col")
                        if not isinstance(prev_spec, tuple) or len(prev_spec) != 4:
                            spec_col = spec_target
                        else:
                            spec_col = (
                                prev_spec[0] + (spec_target[0] - prev_spec[0]) * smoothness,
                                prev_spec[1] + (spec_target[1] - prev_spec[1]) * smoothness,
                                prev_spec[2] + (spec_target[2] - prev_spec[2]) * smoothness,
                                prev_spec[3] + (spec_target[3] - prev_spec[3]) * smoothness,
                            )
                        entry["water_spec_cycle_col"] = spec_col
                        layer_c.setColor(spec_col)
                    else:
                        if is_water:
                            cycle_alpha *= self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                        layer_c.setColor((gain, gain, gain, max(0.0, min(1.0, cycle_alpha))))

            if mode in ("wall", "ceiling"):
                distort = float(getattr(self, "cube_water_distort_strength", 0.085))
                wave_u = math.sin((entry["center"].x * 0.11) + t * (0.62 + speed * 0.3) + phase) * distort
                wave_v = math.cos((entry["center"].y * 0.11) - t * (0.57 + speed * 0.28) + phase * 1.13) * distort
                u = (u + wave_u) % 1.0
                v = (v + wave_v) % 1.0
                pulse_scale = 1.0 + (abs(wave_u) + abs(wave_v)) * 0.45
                ru = max(0.02, ru * pulse_scale)
                rv = max(0.02, rv * pulse_scale)

            node.setTexScale(stage, max(0.02, ru), max(0.02, rv))
            node.setTexOffset(stage, u, v)
            keep.append(entry)
        self.dynamic_room_uv_nodes = keep

    def _register_water_surface(self, node: NodePath | None, x0: float, x1: float, y0: float, y1: float, z: float) -> None:
        if node is None or node.isEmpty():
            return
        amp = float(getattr(self, "water_wave_amplitude", 0.2))
        speed = float(getattr(self, "water_wave_speed", 1.4))
        fx = float(getattr(self, "water_wave_freq_x", 0.09))
        fy = float(getattr(self, "water_wave_freq_y", 0.07))
        self.water_surfaces.append(
            {
                "node": node,
                "x0": float(min(x0, x1)),
                "x1": float(max(x0, x1)),
                "y0": float(min(y0, y1)),
                "y1": float(max(y0, y1)),
                "base_z": float(z),
                "amp": amp,
                "speed": speed,
                "fx": fx,
                "fy": fy,
                "phase": random.uniform(0.0, math.tau),
            }
        )

    def _sample_water_height(self, pos: Vec3, t: float) -> float | None:
        best_h: float | None = None
        for entry in self.water_surfaces:
            if pos.x < entry["x0"] or pos.x > entry["x1"] or pos.y < entry["y0"] or pos.y > entry["y1"]:
                continue
            phase = pos.x * entry["fx"] + pos.y * entry["fy"] + t * entry["speed"] + entry["phase"]
            wave = math.sin(phase) * entry["amp"] + math.cos(phase * 0.63 + 0.7) * (entry["amp"] * 0.52)
            h = entry["base_z"] + wave
            if best_h is None or h > best_h:
                best_h = h
        return best_h

    def _update_water_surface_waves(self, t: float) -> None:
        keep: list[dict] = []
        for entry in self.water_surfaces:
            node = entry["node"]
            if node is None or node.isEmpty():
                continue
            cx = (entry["x0"] + entry["x1"]) * 0.5
            cy = (entry["y0"] + entry["y1"]) * 0.5
            phase = cx * entry["fx"] + cy * entry["fy"] + t * entry["speed"] + entry["phase"]
            wave = math.sin(phase) * entry["amp"] + math.cos(phase * 0.63 + 0.7) * (entry["amp"] * 0.52)
            node.setZ(entry["base_z"] + wave)
            keep.append(entry)
        self.water_surfaces = keep

    def _apply_water_buoyancy(self, dt: float) -> None:
        if not getattr(self, "four_d_obstacle_arena_mode", False):
            return
        if not self.water_surfaces:
            return
        ball_pos = self.ball_np.getPos()
        water_h = self._sample_water_height(ball_pos, self.roll_time)
        if water_h is None:
            return

        bottom_z = float(ball_pos.z) - float(self.ball_radius)
        depth = float(water_h) - bottom_z
        if depth <= 0.0:
            return

        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        try:
            mass = max(0.25, float(self.ball_body.getMass()))
        except Exception:
            mass = 1.0
        g_mag = max(0.1, float(self.current_gravity.length()))
        submerge = self._clamp(depth / max(0.05, self.ball_radius * 1.9), 0.0, 1.35)
        buoy_bias = float(getattr(self, "water_buoyancy_bias", 0.62))
        buoy_force = up * (mass * g_mag * (buoy_bias + submerge * float(getattr(self, "water_buoyancy_strength", 2.2))))
        self.ball_body.applyCentralForce(buoy_force)

        vel = self.ball_body.getLinearVelocity()
        v_up = vel.dot(up)
        v_planar = Vec3(vel) - up * v_up
        planar_drag = max(0.0, 1.0 - dt * float(getattr(self, "water_drag_planar", 0.85)) * (0.3 + submerge * 0.7))
        vertical_drag = max(0.0, 1.0 - dt * float(getattr(self, "water_drag_vertical", 1.95)) * (0.4 + submerge * 0.9))
        v_planar *= planar_drag
        v_up *= vertical_drag
        self.ball_body.setLinearVelocity(v_planar + up * v_up)

    def _apply_smart_room_uv(self, node: NodePath, pos: Vec3, scale: Vec3, surface_mode: str | None = None) -> None:
        if node is None or node.isEmpty():
            return
        sx = max(0.01, abs(scale.x))
        sy = max(0.01, abs(scale.y))
        sz = max(0.01, abs(scale.z))
        dims = [(sx, "x"), (sy, "y"), (sz, "z")]
        normal_axis = min(dims, key=lambda d: d[0])[1]

        if normal_axis == "z":
            u_extent, v_extent = sx, sy
            u_world, v_world = pos.x, pos.y
        elif normal_axis == "x":
            u_extent, v_extent = sy, sz
            u_world, v_world = pos.y, pos.z
        else:
            u_extent, v_extent = sx, sz
            u_world, v_world = pos.x, pos.z

        density = max(0.05, float(self.room_uv_grid_density))
        cache_key = (normal_axis, int(round(u_extent * 100.0)), int(round(v_extent * 100.0)))
        cached = self.room_uv_repeat_cache.get(cache_key)
        if cached is None:
            u_repeat = max(1.0, u_extent * density)
            v_repeat = max(1.0, v_extent * density)
            self.room_uv_repeat_cache[cache_key] = (u_repeat, v_repeat)
        else:
            u_repeat, v_repeat = cached

        seed = (
            int(round(pos.x * 131.0))
            ^ (int(round(pos.y * 193.0)) << 1)
            ^ (int(round(pos.z * 157.0)) << 2)
            ^ (int(round(scale.x * 149.0)) << 3)
            ^ (int(round(scale.y * 173.0)) << 4)
            ^ (int(round(scale.z * 211.0)) << 5)
        ) & 0xFFFFFFFF
        off_u = ((seed * 0.61803398875) % 1.0 + (u_world * density * 0.07)) % 1.0
        off_v = ((seed * 0.41421356237) % 1.0 + (v_world * density * 0.07)) % 1.0
        forced_mode = (surface_mode or "").strip().lower()

        stage = TextureStage.getDefault()

        if bool(getattr(self, "force_single_opaque_additive_texture", False)):
            layer_c = None
            if forced_mode in ("floor", "ceiling", "wall", "water"):
                mode = forced_mode
            elif normal_axis == "x" or normal_axis == "y":
                mode = "wall"
            else:
                mode = "ceiling" if pos.z > (self.floor_y + self.wall_h * 0.55) else "floor"

            if mode == "water":
                stage = self.water_tex_stage
            else:
                stage = TextureStage.getDefault()
            node.clearTexture()
            node.clearShader()
            node.setColor(1.0, 1.0, 1.0, 1.0)
            node.setDepthWrite(True)
            node.clearBin()
            try:
                node.setTransparency(TransparencyAttrib.MNone)
            except Exception:
                pass
            if mode in ("water", "floor"):
                node.clearTexGen()
            else:
                try:
                    node.setTexGen(stage, TexGenAttrib.MWorldPosition)
                except Exception:
                    node.clearTexGen()
            if mode == "water":
                node.setTexture(stage, self.water_base_tex, 1)
            elif mode in ("floor", "wall", "ceiling"):
                node.setTexture(stage, self._get_random_room_texture(), 1)
            else:
                node.setTexture(stage, self.level_checker_tex, 1)
            cube_repeat = max(0.08, min(0.62, (u_repeat + v_repeat) * 0.06))
            if mode == "water":
                cube_repeat = min(12.0, max(1.0, float(getattr(self, "water_uv_repeat_scale", 3.2)) * 0.6))
                self._apply_water_cube_projection(node, uv_scale=cube_repeat, uv_offset_u=off_u, uv_offset_v=off_v)
            elif mode == "floor":
                floor_repeat = max(1.0, float(getattr(self, "room_floor_uv_repeat_scale", 100.0)))
                node.setTexScale(stage, floor_repeat, floor_repeat, 1.0)
                node.setTexOffset(stage, off_u, off_v)
            else:
                node.setTexScale(stage, cube_repeat, cube_repeat, cube_repeat)
                node.setTexOffset(stage, off_u, off_v)

            if mode == "water":
                node.setColor(0.46, 0.72, 0.92, 0.14)
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setDepthWrite(False)
                node.setBin("transparent", 33)
                node.setAttrib(
                    ColorBlendAttrib.make(
                        ColorBlendAttrib.MAdd,
                        ColorBlendAttrib.OIncomingAlpha,
                        ColorBlendAttrib.OOne,
                    ),
                    1,
                )

                layer_c = TextureStage(f"water-spec-c-{len(self.dynamic_room_uv_nodes)}")
                layer_c.setMode(TextureStage.MAdd)
                layer_c.setSort(42)
                node.setTexture(layer_c, self.water_specular_tex, 42)
                try:
                    node.setTexGen(layer_c, TexGenAttrib.MWorldPosition)
                except Exception:
                    pass
                gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
                spec_strength = self._clamp(float(getattr(self, "water_specular_strength", 0.72)), 0.0, 1.0)
                layer_c.setColor((gain, gain, gain, 0.16 + spec_strength * 0.26))

            if mode == "water":
                self.dynamic_room_uv_nodes.append(
                    {
                        "node": node,
                        "stage": stage,
                        "mode": "water",
                        "base_u": off_u,
                        "base_v": off_v,
                        "base_ru": cube_repeat,
                        "base_rv": cube_repeat,
                        "center": Vec3(pos),
                        "speed": random.uniform(0.38, 0.82),
                        "phase": random.uniform(0.0, math.tau),
                        "dir": -1.0 if random.random() < 0.5 else 1.0,
                        "w": self._compute_level_w(pos),
                        "layer_a": None,
                        "layer_b": None,
                        "layer_c": layer_c,
                        "water_spec_repeat": float(getattr(self, "water_specular_detail_repeat", 7.5)),
                        "water_spec_speed": float(getattr(self, "water_specular_scroll_speed", 1.85)),
                    }
                )
            elif self.animate_non_water_uv and mode != "water":
                self.dynamic_room_uv_nodes.append(
                    {
                        "node": node,
                        "stage": stage,
                        "mode": "additive",
                        "base_u": off_u,
                        "base_v": off_v,
                        "base_ru": cube_repeat,
                        "base_rv": cube_repeat,
                        "center": Vec3(pos),
                        "speed": random.uniform(0.16, 0.34),
                        "phase": random.uniform(0.0, math.tau),
                        "dir": -1.0 if random.random() < 0.5 else 1.0,
                        "w": self._compute_level_w(pos),
                        "layer_a": None,
                        "layer_b": None,
                        "layer_c": None,
                    }
                )
            return

        node.setTexScale(stage, u_repeat, v_repeat)
        node.setTexOffset(stage, off_u, off_v)

        if forced_mode in ("floor", "ceiling", "wall", "water"):
            mode = forced_mode
        elif normal_axis == "x" or normal_axis == "y":
            mode = "wall"
        else:
            mode = "ceiling" if pos.z > (self.floor_y + self.wall_h * 0.55) else "floor"

        if mode in ("floor", "water"):
            node.setTwoSided(False)
            try:
                node.clearAttrib(CullFaceAttrib.getClassType())
            except Exception:
                pass
            if mode == "water":
                water_repeat = float(getattr(self, "water_uv_repeat_scale", 3.2))
                u_repeat = water_repeat
                v_repeat = water_repeat
            else:
                floor_repeat = max(1.0, float(getattr(self, "room_floor_uv_repeat_scale", 100.0)))
                u_repeat = floor_repeat
                v_repeat = floor_repeat
            off_u = 0.0
            off_v = 0.0
            if mode == "water":
                stage = self.water_tex_stage
                node.clearTexture()
                node.setTexture(stage, self.water_base_tex, 1)
                self._apply_water_cube_projection(node, uv_scale=max(0.1, u_repeat), uv_offset_u=off_u, uv_offset_v=off_v)
            else:
                try:
                    node.setTexGen(stage, TexGenAttrib.MWorldPosition)
                except Exception:
                    node.clearTexGen()
                node.setTexScale(stage, u_repeat, v_repeat)
                node.setTexOffset(stage, off_u, off_v)
                node.setTexture(self.floor_fractal_tex_a, 1)
            if mode == "water":
                node.setColor(0.46, 0.72, 0.92, 0.14)
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setDepthWrite(False)
                node.setBin("transparent", 33)
                node.setAttrib(
                    ColorBlendAttrib.make(
                        ColorBlendAttrib.MAdd,
                        ColorBlendAttrib.OIncomingAlpha,
                        ColorBlendAttrib.OOne,
                    ),
                    1,
                )
                if bool(getattr(self, "water_emissive_linux_enabled", False)):
                    emissive_scale = max(1.0, float(getattr(self, "water_emissive_linux_scale", 1.35)))
                    node.setLightOff(1)
                    node.setColorScale(emissive_scale, emissive_scale, emissive_scale, 1.0)
                else:
                    try:
                        node.clearLight()
                    except Exception:
                        pass
                    node.setColorScale(1.0, 1.0, 1.0, 1.0)
            else:
                node.setColor(0.9, 0.96, 1.0, 1.0)
            if self.floor_wet_shader is not None and mode == "floor":
                node.setShader(self.floor_wet_shader)
                node.setShaderInput("u_time", self.roll_time)
                node.setShaderInput("u_contact_uv", self.floor_contact_uv)
                node.setShaderInput("u_wake_strength", self.floor_contact_strength)
                node.setShaderInput("u_room_uv_scale", self.floor_uv_projection_scale)
                self.floor_shader_nodes.append(node)
        elif mode in ("wall", "ceiling"):
            cube_repeat = max(0.02, min(1.6, (u_repeat + v_repeat) * 0.12))
            try:
                node.setTexGen(stage, TexGenAttrib.MWorldPosition)
            except Exception:
                pass
            node.setTexScale(stage, cube_repeat, cube_repeat, cube_repeat)
            node.setTexOffset(stage, off_u, off_v)

        layer_a = None
        layer_b = None
        layer_c = None
        if (not getattr(self, "single_texture_per_cube", False)) and (mode == "floor" and self.animate_non_water_uv and self.floor_wet_shader is None):
            node.setTransparency(TransparencyAttrib.MAlpha)

            layer_a = TextureStage(f"floor-ripple-a-{len(self.dynamic_room_uv_nodes)}")
            layer_a.setMode(TextureStage.MModulate if mode == "water" else TextureStage.MAdd)
            layer_a.setSort(40)
            node.setTexture(layer_a, self.floor_fractal_tex_a, 40)
            try:
                node.setTexGen(layer_a, TexGenAttrib.MWorldPosition)
            except Exception:
                pass
            gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
            if mode == "water":
                gain *= float(getattr(self, "water_additive_gain_scale", 0.62))
                op_scale = self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                layer_a.setColor((gain, gain, gain, 0.08 * op_scale))
            else:
                layer_a.setColor((gain, gain, gain, 0.0))

            layer_b = TextureStage(f"floor-ripple-b-{len(self.dynamic_room_uv_nodes)}")
            layer_b.setMode(TextureStage.MModulate if mode == "water" else TextureStage.MAdd)
            layer_b.setSort(41)
            node.setTexture(layer_b, self.floor_fractal_tex_b, 41)
            try:
                node.setTexGen(layer_b, TexGenAttrib.MWorldPosition)
            except Exception:
                pass
            gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
            if mode == "water":
                gain *= float(getattr(self, "water_additive_gain_scale", 0.62))
                op_scale = self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                layer_b.setColor((gain, gain, gain, 0.14 * op_scale))
            else:
                layer_b.setColor((gain, gain, gain, 0.0))

            if mode == "water":
                layer_c = TextureStage(f"water-spec-c-{len(self.dynamic_room_uv_nodes)}")
                layer_c.setMode(TextureStage.MModulate)
                layer_c.setSort(42)
                node.setTexture(layer_c, self.water_specular_tex, 42)
                try:
                    node.setTexGen(layer_c, TexGenAttrib.MWorldPosition)
                except Exception:
                    pass
                gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
                gain *= float(getattr(self, "water_additive_gain_scale", 0.62))
                spec_strength = self._clamp(float(getattr(self, "water_specular_strength", 0.72)), 0.0, 1.0)
                op_scale = self._clamp(float(getattr(self, "water_additive_opacity_scale", 0.34)), 0.0, 1.0)
                layer_c.setColor((gain, gain, gain, (0.16 + spec_strength * 0.26) * op_scale))

        should_track_dynamic_uv = mode == "water" or self.animate_non_water_uv
        if not should_track_dynamic_uv:
            return

        self.dynamic_room_uv_nodes.append(
            {
                "node": node,
                "stage": stage,
                "mode": mode,
                "base_u": off_u,
                "base_v": off_v,
                "base_ru": u_repeat,
                "base_rv": v_repeat,
                "center": Vec3(pos),
                "speed": random.uniform(0.22, 0.62),
                "phase": random.uniform(0.0, math.tau),
                "dir": -1.0 if random.random() < 0.5 else 1.0,
                "w": self._compute_level_w(pos),
                "layer_a": layer_a,
                "layer_b": layer_b,
                "layer_c": layer_c,
                "water_spec_repeat": float(getattr(self, "water_specular_detail_repeat", 7.5)),
                "water_spec_speed": float(getattr(self, "water_specular_scroll_speed", 1.85)),
            }
        )

    def _update_floor_contact_pulses(self, dt: float, grounded_contact: Vec3 | None) -> None:
        self.floor_contact_emit_timer += dt
        if grounded_contact is not None and self.floor_contact_emit_timer >= self.floor_contact_emit_interval:
            self.floor_contact_pulses.append(
                {
                    "origin": Vec3(grounded_contact),
                    "t0": self.roll_time,
                    "life": 1.5,
                    "speed": 6.4,
                    "band": 3.2,
                    "phase": random.uniform(0.0, math.tau),
                }
            )
            self.floor_contact_emit_timer = 0.0

        self.floor_contact_pulses = [
            pulse
            for pulse in self.floor_contact_pulses
            if (self.roll_time - pulse["t0"]) <= pulse["life"]
        ]

    def _setup_star_particles(self) -> None:
        if hasattr(self, "star_root") and self.star_root is not None and not self.star_root.isEmpty():
            self.star_root.removeNode()

        self.star_root = self.render.attachNewNode("star-root")
        self.star_particles.clear()
        star_count = 64 if self.performance_mode else 128

        for _ in range(star_count):
            star = self.sphere_model.copyTo(self.star_root)
            star.setScale(0.02)
            star.setColor(0.84, 0.9, 1.0, 0.42)
            star.setLightOff(1)
            star.setShaderOff(1)
            star.setTransparency(TransparencyAttrib.MAlpha)
            star.setDepthWrite(False)
            star.setBin("transparent", 20)

            star_data = {
                "node": star,
                "x": random.uniform(-self.map_w * 0.25, self.map_w * 1.25),
                "y": random.uniform(-self.map_d * 0.25, self.map_d * 1.25),
                "z": random.uniform(0.35, self.hyper_bounds_top_z * 1.2),
                "vx": random.uniform(-0.12, 0.12),
                "vy": random.uniform(-0.12, 0.12),
                "vz": random.uniform(-0.06, 0.06),
                "phase": random.uniform(0.0, math.tau),
                "twinkle": random.uniform(1.8, 5.4),
            }
            star.setPos(star_data["x"], star_data["y"], star_data["z"])
            self.star_particles.append(star_data)

    def _update_star_particles(self, dt: float, t: float) -> None:
        if not self.star_particles:
            return
        min_x = -self.map_w * 0.3
        max_x = self.map_w * 1.3
        min_y = -self.map_d * 0.3
        max_y = self.map_d * 1.3
        min_z = self.hyper_bounds_bottom_z + 0.4
        max_z = self.hyper_bounds_top_z * 1.25
        for entry in self.star_particles:
            entry["x"] += entry["vx"] * dt
            entry["y"] += entry["vy"] * dt
            entry["z"] += entry["vz"] * dt
            if entry["x"] < min_x:
                entry["x"] = max_x
            elif entry["x"] > max_x:
                entry["x"] = min_x
            if entry["y"] < min_y:
                entry["y"] = max_y
            elif entry["y"] > max_y:
                entry["y"] = min_y
            if entry["z"] < min_z:
                entry["z"] = max_z
            elif entry["z"] > max_z:
                entry["z"] = min_z

            star = entry["node"]
            twinkle = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * entry["twinkle"] + entry["phase"]))
            star.setPos(entry["x"], entry["y"], entry["z"])
            star.setAlphaScale(twinkle)

    def _spawn_motion_trail(self, pos: Vec3, scale: float, color: tuple[float, float, float, float], life: float, vel: Vec3, use_box: bool = False) -> None:
        if not self.enable_particles:
            return
        model = self.box_model if use_box else self.sphere_model
        node = model.copyTo(self.render)
        node.setPos(pos)
        if use_box:
            node.setScale(self.box_norm_scale)
            node.setScale(scale, scale, scale)
        else:
            node.setScale(scale)
        node.setColor(color)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthWrite(False)
        node.setBin("transparent", 34)
        node.setLightOff(1)
        node.clearTexture()
        node.setTexture(self._get_random_room_texture(), 1)
        node.setTexScale(TextureStage.getDefault(), 2.0, 2.0)
        self.motion_trails.append(
            {
                "node": node,
                "age": 0.0,
                "life": max(0.05, life),
                "vel": Vec3(vel),
                "scale": scale,
                "alpha": color[3],
            }
        )

    def _update_motion_trails(self, dt: float) -> None:
        if not self.enable_particles:
            for entry in self.motion_trails:
                node = entry.get("node")
                if node is not None and not node.isEmpty():
                    node.removeNode()
            self.motion_trails.clear()
            return
        if not self.motion_trails:
            return
        keep: list[dict] = []
        for entry in self.motion_trails:
            node = entry["node"]
            if node is None or node.isEmpty():
                continue
            entry["age"] += dt
            if entry["age"] >= entry["life"]:
                node.removeNode()
                continue
            t = entry["age"] / entry["life"]
            node.setPos(node.getPos() + entry["vel"] * dt)
            node.setScale(max(0.001, entry["scale"] * (1.0 - 0.55 * t)))
            node.setAlphaScale(max(0.0, entry["alpha"] * (1.0 - t) * (1.0 - t)))
            keep.append(entry)
        self.motion_trails = keep

    def _setup_magic_missile_visuals(self) -> None:
        cylinder_model = self._load_first_model([
            "models/misc/cylinder",
            "models/misc/cylinder.egg",
            "models/cylinder",
            "models/cylinder.egg",
        ])
        if cylinder_model is None:
            cylinder_model = self._create_fallback_cylinder_model()
        cone_model = self._load_first_model([
            "models/misc/cone",
            "models/misc/cone.egg",
            "models/cone",
            "models/cone.egg",
        ])
        if cone_model is None:
            cone_model = self._create_fallback_cone_model()

        self.magic_missile_cylinder_model = cylinder_model
        self.magic_missile_cone_model = cone_model

        if self.magic_missile_template is not None and (not self.magic_missile_template.isEmpty()):
            self.magic_missile_template.removeNode()
        template = self.render.attachNewNode("magic-missile-template")
        template.hide()

        emissive_strength = max(0.0, float(getattr(self, "magic_missile_emissive_strength", 1.0)))
        body_material = Material()
        body_material.setEmission((0.45 * emissive_strength, 0.95 * emissive_strength, 1.0 * emissive_strength, 1.0))
        nose_material = Material()
        nose_material.setEmission((1.0 * emissive_strength, 1.0 * emissive_strength, 0.65 * emissive_strength, 1.0))
        flare_material = Material()
        flare_material.setEmission((0.8 * emissive_strength, 1.0 * emissive_strength, 1.0 * emissive_strength, 1.0))

        body = self.box_model.copyTo(template)
        body.setName("mm-body")
        body.clearTexture()
        body.setColor(0.25, 0.9, 1.0, 1.0)
        body.setScale(0.36, 1.05, 0.36)
        body.setPos(0.0, 0.0, 0.0)
        body.setHpr(0.0, 0.0, 0.0)
        body.clearTransparency()
        body.setLightOff(1)
        body.setShaderOff(1)
        body.setTwoSided(True)
        body.setMaterial(body_material, 1)
        body.setDepthWrite(True)
        body.setDepthTest(True)
        body.setBin("fixed", 41)
        body.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        nose = self.magic_missile_cone_model.copyTo(template)
        nose.setName("mm-nose")
        nose.clearTexture()
        nose.setColor(0.95, 1.0, 0.7, 0.98)
        nose.setScale(0.3, 0.3, 0.56)
        nose.setPos(0.0, 1.06, 0.0)
        nose.setP(90.0)
        nose.setTransparency(TransparencyAttrib.MAlpha)
        nose.setLightOff(1)
        nose.setShaderOff(1)
        nose.setMaterial(nose_material, 1)
        nose.setDepthWrite(True)
        nose.setDepthTest(True)
        nose.setBin("fixed", 42)
        nose.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        flare = self.sphere_model.copyTo(template)
        flare.setName("mm-flare")
        flare.clearTexture()
        flare.setColor(0.45, 1.0, 1.0, 0.52)
        flare.setScale(0.42, 0.42, 0.32)
        flare.setPos(0.0, -0.46, 0.0)
        flare.setTransparency(TransparencyAttrib.MAlpha)
        flare.setDepthWrite(False)
        flare.setBin("transparent", 20)
        flare.setLightOff(1)
        flare.setMaterial(flare_material, 1)
        flare.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        halo = self.sphere_model.copyTo(template)
        halo.setName("mm-halo")
        halo.clearTexture()
        halo.setColor(0.72, 1.0, 1.0, 0.65)
        halo.setScale(0.72, 0.72, 0.64)
        halo.setPos(0.0, 0.16, 0.0)
        halo.setTransparency(TransparencyAttrib.MAlpha)
        halo.setDepthWrite(False)
        halo.setDepthTest(False)
        halo.setBin("transparent", 60)
        halo.setLightOff(1)
        halo.setMaterial(flare_material, 1)
        halo.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        core = self.sphere_model.copyTo(template)
        core.setName("mm-core")
        core.clearTexture()
        core.setColor(0.95, 1.0, 1.0, 0.96)
        core.setScale(0.22, 0.22, 0.22)
        core.setPos(0.0, 0.48, 0.0)
        core.setTransparency(TransparencyAttrib.MAlpha)
        core.setDepthWrite(False)
        core.setDepthTest(False)
        core.setBin("transparent", 61)
        core.setLightOff(1)
        core.setMaterial(flare_material, 1)
        core.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        self.magic_missile_template = template

    def _on_r_pressed(self) -> None:
        self.input_hud_r_state = True
        self.input_hud_r_pulse = 1.0
        if getattr(self, "game_over_active", False) or getattr(self, "win_active", False):
            self._restart_after_game_over()
            return
        self._trigger_magic_missile_attack()

    def _on_r_released(self) -> None:
        self.input_hud_r_state = False

    def _on_mouse_button_released(self, button_name: str) -> None:
        if button_name in self.input_hud_mouse_state:
            self.input_hud_mouse_state[button_name] = False

    def _on_mouse1_pressed(self) -> None:
        self.input_hud_mouse_state["mouse1"] = True
        self._trigger_throw_attack()

    def _on_mouse2_pressed(self) -> None:
        self.input_hud_mouse_state["mouse2"] = True
        self._trigger_hyperbomb()

    def _on_mouse3_pressed(self) -> None:
        self.input_hud_mouse_state["mouse3"] = True
        self._trigger_spin_attack()

    def _trigger_magic_missile_attack(self) -> None:
        if self.magic_missile_cooldown > 0.0:
            return
        if self.magic_missile_template is None or self.magic_missile_template.isEmpty():
            self._setup_magic_missile_visuals()
        if self.magic_missile_template is None or self.magic_missile_template.isEmpty():
            return

        cast_pos = self._get_hyperbomb_spawn_pos() + Vec3(0.0, 0.0, self.ball_radius * 0.35)
        forward = Vec3(getattr(self, "last_move_dir", Vec3(0.0, 1.0, 0.0)))
        if forward.lengthSquared() < 1e-8:
            forward = self.camera.getQuat(self.render).getForward()
        forward.z *= 0.2
        if forward.lengthSquared() < 1e-8:
            forward = Vec3(0.0, 1.0, 0.0)
        else:
            forward.normalize()
        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0.0, 0.0, 1.0)
        else:
            up.normalize()
        right = up.cross(forward)
        if right.lengthSquared() < 1e-8:
            right = Vec3(1.0, 0.0, 0.0)
        else:
            right.normalize()

        targets = self._get_live_monster_targets()
        count = max(1, int(getattr(self, "magic_missile_cast_count", 6)))
        speed = max(1.0, float(getattr(self, "magic_missile_speed", 25.0)))
        for idx in range(count):
            missile_np = self.magic_missile_template.copyTo(self.render)
            missile_np.show()
            spread = (idx - (count - 1) * 0.5)
            spawn_pos = cast_pos + right * (spread * 0.55) + up * (0.16 * abs(spread))
            missile_np.setPos(spawn_pos)

            tgt = targets[idx % len(targets)] if targets else None
            target_pos = tgt.get("root").getPos() if tgt and tgt.get("root") is not None else (spawn_pos + forward * 6.0)
            dir_vec = target_pos - spawn_pos
            if dir_vec.lengthSquared() < 1e-8:
                dir_vec = Vec3(forward)
            else:
                dir_vec.normalize()
            missile_np.lookAt(spawn_pos + dir_vec, up)

            self.magic_missiles.append(
                {
                    "node": missile_np,
                    "body": missile_np.find("**/mm-body"),
                    "nose": missile_np.find("**/mm-nose"),
                    "flare": missile_np.find("**/mm-flare"),
                    "halo": missile_np.find("**/mm-halo"),
                    "core": missile_np.find("**/mm-core"),
                    "vel": dir_vec * speed,
                    "age": 0.0,
                    "life": float(getattr(self, "magic_missile_life", 4.2)),
                    "target_ref": tgt,
                    "retarget_timer": random.uniform(0.01, float(getattr(self, "magic_missile_retarget_interval", 0.12))),
                    "phase": random.uniform(0.0, math.tau),
                    "launch_up": random.uniform(0.4, 0.95),
                    "trail_timer": random.uniform(0.0, float(getattr(self, "magic_missile_trail_emit_interval", 0.02))),
                }
            )

        self.magic_missile_cooldown = float(getattr(self, "magic_missile_cooldown_duration", 4.5))
        fire_sfx = self.sfx_attack_homingmissile if self.sfx_attack_homingmissile else self.sfx_attack
        self._play_sound(fire_sfx, volume=0.7, play_rate=1.25)

    def _get_live_monster_targets(self) -> list[dict]:
        ball_pos = self.ball_np.getPos() if hasattr(self, "ball_np") and self.ball_np is not None else Vec3(0, 0, 0)
        targets: list[dict] = []
        for monster in getattr(self, "monsters", []):
            if monster.get("dead", False):
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            targets.append(monster)
        targets.sort(key=lambda m: (m.get("root").getPos() - ball_pos).lengthSquared())
        return targets

    def _find_live_monster_ref(self, target: dict | None) -> dict | None:
        if target is None:
            return None
        if target.get("dead", False):
            return None
        root = target.get("root")
        if root is None or root.isEmpty():
            return None
        return target if target in getattr(self, "monsters", []) else None

    def _clear_magic_missiles(self) -> None:
        for entry in self.magic_missiles:
            node = entry.get("node")
            if node is not None and not node.isEmpty():
                node.removeNode()
        self.magic_missiles.clear()
        for entry in self.magic_missile_trails:
            node = entry.get("node")
            if node is not None and not node.isEmpty():
                node.removeNode()
        self.magic_missile_trails.clear()

    def _spawn_magic_missile_trail(self, pos: Vec3, color: tuple[float, float, float], scale: float = 0.26, life: float = 0.22) -> None:
        node = self.sphere_model.copyTo(self.render)
        node.setPos(pos)
        node.setScale(scale)
        node.clearTexture()
        node.setColor(color[0], color[1], color[2], 0.82)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthWrite(False)
        node.setDepthTest(False)
        node.setBin("transparent", 58)
        node.setLightOff(1)
        node.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )
        self.magic_missile_trails.append(
            {
                "node": node,
                "age": 0.0,
                "life": max(0.05, life),
                "scale": max(0.05, scale),
                "alpha": 0.82,
            }
        )

    def _update_magic_missile_trails(self, dt: float) -> None:
        if not self.magic_missile_trails:
            return
        keep: list[dict] = []
        for entry in self.magic_missile_trails:
            node = entry.get("node")
            if node is None or node.isEmpty():
                continue
            entry["age"] = float(entry.get("age", 0.0)) + dt
            life = max(0.001, float(entry.get("life", 0.2)))
            t = self._clamp(entry["age"] / life, 0.0, 1.0)
            if t >= 1.0:
                node.removeNode()
                continue
            scale = float(entry.get("scale", 0.3)) * (0.84 + t * 0.62)
            node.setScale(scale)
            node.setAlphaScale(float(entry.get("alpha", 0.5)) * ((1.0 - t) ** 1.45))
            keep.append(entry)
        self.magic_missile_trails = keep

    def _update_magic_missiles(self, dt: float) -> None:
        if self.magic_missile_cooldown > 0.0:
            self.magic_missile_cooldown = max(0.0, self.magic_missile_cooldown - dt)
        if not self.magic_missiles:
            return

        keep: list[dict] = []
        speed = max(1.0, float(getattr(self, "magic_missile_speed", 25.0)))
        turn_rate = max(0.1, float(getattr(self, "magic_missile_turn_rate", 7.6)))
        color_cycle_rate = max(0.1, float(getattr(self, "magic_missile_color_cycle_rate", 11.5)))
        hit_radius = 1.15
        damage = max(1.0, float(getattr(self, "magic_missile_damage", 34.0)))
        arc_height = max(0.0, float(getattr(self, "magic_missile_arc_height", 0.9)))
        arc_drop = max(0.0, float(getattr(self, "magic_missile_arc_drop", 1.15)))
        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0.0, 0.0, 1.0)
        else:
            up.normalize()

        for entry in self.magic_missiles:
            node = entry.get("node")
            if node is None or node.isEmpty():
                continue

            entry["age"] += dt
            if entry["age"] >= float(entry.get("life", 4.2)):
                node.removeNode()
                continue

            pos = node.getPos()
            vel = Vec3(entry.get("vel", Vec3(0.0, speed, 0.0)))
            if vel.lengthSquared() < 1e-8:
                vel = Vec3(0.0, speed, 0.0)
            cur_dir = Vec3(vel)
            cur_dir.normalize()

            entry["retarget_timer"] = float(entry.get("retarget_timer", 0.0)) - dt
            target = self._find_live_monster_ref(entry.get("target_ref"))
            if target is None or entry["retarget_timer"] <= 0.0:
                best = None
                best_dist_sq = float("inf")
                for monster in self._get_live_monster_targets():
                    root = monster.get("root")
                    if root is None or root.isEmpty():
                        continue
                    dist_sq = (root.getPos() - pos).lengthSquared()
                    if dist_sq < best_dist_sq:
                        best = monster
                        best_dist_sq = dist_sq
                target = best
                entry["target_ref"] = target
                entry["retarget_timer"] = float(getattr(self, "magic_missile_retarget_interval", 0.12))

            desired_dir = Vec3(cur_dir)
            if target is not None:
                root = target.get("root")
                target_pos = root.getPos() + Vec3(0.0, 0.0, 0.28)
                to_target = target_pos - pos
                if to_target.lengthSquared() > 1e-8:
                    desired_dir = to_target
                    desired_dir.normalize()

            life = max(0.001, float(entry.get("life", 4.2)))
            age_t = self._clamp(float(entry.get("age", 0.0)) / life, 0.0, 1.0)
            arc_bias = (1.0 - age_t) * arc_height * float(entry.get("launch_up", 0.7)) - age_t * arc_drop
            desired_dir = desired_dir + up * arc_bias
            if desired_dir.lengthSquared() > 1e-8:
                desired_dir.normalize()

            steer = min(1.0, turn_rate * dt)
            new_dir = cur_dir + (desired_dir - cur_dir) * steer
            if new_dir.lengthSquared() < 1e-8:
                new_dir = Vec3(desired_dir)
            else:
                new_dir.normalize()

            wobble = 0.08 * math.sin(self.roll_time * 18.0 + float(entry.get("phase", 0.0)))
            right = up.cross(new_dir)
            if right.lengthSquared() > 1e-8:
                right.normalize()
                new_dir = new_dir + right * wobble
                if new_dir.lengthSquared() > 1e-8:
                    new_dir.normalize()

            vel = new_dir * speed
            next_pos = pos + vel * dt
            node.setPos(next_pos)
            node.lookAt(next_pos + new_dir, up)
            entry["vel"] = vel

            body_np = entry.get("body")
            nose_np = entry.get("nose")
            flare_np = entry.get("flare")
            halo_np = entry.get("halo")
            core_np = entry.get("core")
            hue_base = (self.roll_time * color_cycle_rate + float(entry.get("phase", 0.0)) * 0.37) % 1.0
            b_r, b_g, b_b = colorsys.hsv_to_rgb(hue_base, 0.92, 1.0)
            n_r, n_g, n_b = colorsys.hsv_to_rgb((hue_base + 0.12) % 1.0, 0.85, 1.0)
            f_r, f_g, f_b = colorsys.hsv_to_rgb((hue_base + 0.24) % 1.0, 0.72, 1.0)
            pulse = 0.72 + 0.28 * math.sin(self.roll_time * 27.0 + float(entry.get("phase", 0.0)) * 2.0)
            if body_np is not None and not body_np.isEmpty():
                body_np.setColor(b_r * pulse, b_g * pulse, b_b * pulse, 1.0)
            if nose_np is not None and not nose_np.isEmpty():
                nose_np.setColor(n_r, n_g, n_b, 1.0)
            if flare_np is not None and not flare_np.isEmpty():
                flare_np.setColor(f_r * 1.1, f_g * 1.1, f_b * 1.1, 0.5)
                flare_scale = 0.3 + 0.22 * pulse
                flare_np.setScale(flare_scale, flare_scale, flare_scale)
            if halo_np is not None and not halo_np.isEmpty():
                halo_np.setColor(f_r * 1.2, f_g * 1.2, f_b * 1.2, 0.62)
                halo_scale = 0.82 + 0.34 * pulse
                halo_np.setScale(halo_scale, halo_scale, halo_scale * 0.86)
            if core_np is not None and not core_np.isEmpty():
                core_np.setColor(f_r * 1.35, f_g * 1.35, f_b * 1.35, 0.94)
                core_scale = 0.24 + 0.16 * pulse
                core_np.setScale(core_scale, core_scale, core_scale)

            entry["trail_timer"] = float(entry.get("trail_timer", 0.0)) - dt
            if entry["trail_timer"] <= 0.0:
                self._spawn_magic_missile_trail(next_pos, (f_r, f_g, f_b), scale=0.26, life=0.2)
                entry["trail_timer"] += max(0.008, float(getattr(self, "magic_missile_trail_emit_interval", 0.02)))

            if target is not None:
                root = target.get("root")
                if root is not None and not root.isEmpty():
                    hit_dist = hit_radius + float(target.get("radius", 0.6))
                    if (root.getPos() - next_pos).length() <= hit_dist:
                        self._damage_monster(target, damage)
                        self._spawn_motion_trail(next_pos, scale=0.22, color=(0.42, 1.0, 0.95, 0.65), life=0.18, vel=Vec3(0, 0, 0), use_box=False)
                        node.removeNode()
                        continue

            keep.append(entry)

        self.magic_missiles = keep

    def _trigger_hyperbomb(self) -> None:
        if self.game_over_active or self.win_active:
            return
        if self.hyperbomb_cooldown > 0.0:
            return
        ball_r = max(0.05, float(getattr(self, "ball_radius", 0.68)))
        self.hyperbomb_active = True
        self.hyperbomb_timer = 0.0
        self.hyperbomb_spawn_timer = 0.0
        self.hyperbomb_scale_start = max(0.03, ball_r * float(getattr(self, "hyperbomb_scale_start_factor", 0.18)))
        self.hyperbomb_max_scale = max(self.hyperbomb_scale_start * 1.25, ball_r * float(getattr(self, "hyperbomb_max_scale_factor", 6.0)))
        self.hyperbomb_origin = self._get_hyperbomb_spawn_pos()
        self.hyperbomb_cooldown = self.hyperbomb_cooldown_duration
        self._play_hyperbomb_sfx(self.hyperbomb_origin)

    def _get_hyperbomb_spawn_pos(self) -> Vec3:
        if not hasattr(self, "ball_np") or self.ball_np is None or self.ball_np.isEmpty():
            return Vec3(0, 0, 0)
        return Vec3(self.ball_np.getPos())

    def _spawn_hyperbomb_sphere(self, life_bias: float = 0.0) -> None:
        if self.sphere_model is None:
            return
        source_pos = Vec3(self.hyperbomb_origin)
        node = self.render.attachNewNode("hyperbomb-halfdome")
        node.setPos(source_pos)
        node.setScale(self.hyperbomb_scale_start)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthWrite(True)
        node.setDepthTest(True)
        node.setBin("fixed", 120)
        node.setLightOff(1)
        node.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        dome_layers: list[NodePath] = []
        layer_count = 6
        for layer_idx in range(layer_count):
            dome_np = self.sphere_model.copyTo(node)
            dome_np.clearTexture()
            dome_np.setTransparency(TransparencyAttrib.MAlpha)
            dome_np.setDepthWrite(True)
            dome_np.setDepthTest(True)
            dome_np.setLightOff(1)
            dome_np.setShaderOff(1)
            dome_np.setTwoSided(False)
            dome_np.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
            dome_np.setAttrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.MAdd,
                    ColorBlendAttrib.OIncomingAlpha,
                    ColorBlendAttrib.OOne,
                ),
                1,
            )
            dome_radius = 0.44 + layer_idx * 0.22
            dome_height = 0.24 + layer_idx * 0.08
            dome_np.setScale(dome_radius, dome_radius, dome_height)
            dome_np.setZ(0.11 + layer_idx * 0.06)
            dome_np.setH(random.uniform(0.0, 360.0))
            dome_layers.append(dome_np)

        ring_life = max(0.25, self.hyperbomb_sphere_life + life_bias)
        phase = random.uniform(0.0, math.tau)
        hue = ((self.hyperbomb_timer * 1.9) + random.uniform(0.0, 1.0)) % 1.0
        hue_speed = random.uniform(2.8, 6.4)
        max_alpha = random.uniform(0.45, 0.78)
        self.hyperbomb_spheres.append(
            {
                "node": node,
                "age": 0.0,
                "life": ring_life,
                "origin": source_pos,
                "scale": self.hyperbomb_scale_start,
                "growth": self.hyperbomb_growth_speed * random.uniform(0.86, 1.18),
                "slowdown": self.hyperbomb_growth_slowdown * random.uniform(0.86, 1.24),
                "phase": phase,
                "hue": hue,
                "hue_speed": hue_speed,
                "alpha": max_alpha,
                "fade_in": random.uniform(0.035, 0.11),
                "layers": dome_layers,
            }
        )

    def _play_hyperbomb_sfx(self, pos: Vec3) -> None:
        if not hasattr(self, "audio3d"):
            return
        if not self.sfx_attackbomb_path:
            return
        try:
            sound = self.audio3d.loadSfx(self.sfx_attackbomb_path)
        except Exception:
            sound = None
        if not sound:
            return

        anchor = self.render.attachNewNode("hyperbomb-sfx")
        anchor.setPos(pos)
        self.audio3d.attachSoundToObject(sound, anchor)
        self.audio3d.setSoundMinDistance(sound, 2.0)
        self.audio3d.setSoundMaxDistance(sound, 280.0)
        if hasattr(self.audio3d, "setSoundVelocityAuto"):
            self.audio3d.setSoundVelocityAuto(sound)
        sound.setLoop(False)
        sound.setVolume(0.92)
        sound.setPlayRate(1.0)
        sound.play()
        self.hyperbomb_audio_nodes.append(
            {
                "node": anchor,
                "sound": sound,
                "age": 0.0,
                "life": 4.0,
            }
        )

    def _update_hyperbomb(self, dt: float) -> None:
        if self.hyperbomb_cooldown > 0.0:
            self.hyperbomb_cooldown = max(0.0, self.hyperbomb_cooldown - dt)

        if self.hyperbomb_active:
            self.hyperbomb_timer += dt
            spawn_phase = self._clamp(self.hyperbomb_timer / max(0.001, self.hyperbomb_spawn_duration), 0.0, 1.0)

            if self.hyperbomb_timer <= self.hyperbomb_spawn_duration:
                self.hyperbomb_spawn_timer -= dt
                while self.hyperbomb_spawn_timer <= 0.0:
                    interval = (
                        self.hyperbomb_spawn_interval_fast
                        + (self.hyperbomb_spawn_interval_slow - self.hyperbomb_spawn_interval_fast) * (spawn_phase * spawn_phase)
                    )
                    self._spawn_hyperbomb_sphere(life_bias=(1.0 - spawn_phase) * 0.2)
                    self.hyperbomb_spawn_timer += max(0.006, interval)

            if self.hyperbomb_timer >= self.hyperbomb_duration:
                self.hyperbomb_active = False

        if self.hyperbomb_spheres:
            keep_spheres: list[dict] = []
            for entry in self.hyperbomb_spheres:
                node = entry.get("node")
                if node is None or node.isEmpty():
                    continue

                entry["age"] += dt
                age = float(entry["age"])
                life = max(0.001, float(entry["life"]))
                if age >= life:
                    node.removeNode()
                    continue

                t = age / life
                growth_falloff = max(0.08, math.exp(-entry["slowdown"] * t))
                entry["scale"] += entry["growth"] * growth_falloff * dt
                scale = min(self.hyperbomb_max_scale, max(0.001, entry["scale"]))

                hue_speed = entry["hue_speed"] * (1.0 - 0.82 * t)
                hue = (entry["hue"] + age * max(0.18, hue_speed)) % 1.0
                saturation = self._clamp(0.92 - 0.32 * t, 0.0, 1.0)
                value = self._clamp(1.0 - 0.22 * t, 0.0, 1.0)
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

                fade_in = max(0.001, float(entry["fade_in"]))
                alpha_in = self._clamp(age / fade_in, 0.0, 1.0)
                alpha_out = (1.0 - t)
                alpha = float(entry["alpha"]) * alpha_in * (alpha_out ** 1.65)

                wobble = 1.0 + 0.05 * math.sin(self.roll_time * 4.0 + entry["phase"])
                node.setPos(entry.get("origin", self.hyperbomb_origin))
                node.setScale(scale * wobble)

                layers = entry.get("layers", [])
                if layers:
                    for layer_idx, dome_np in enumerate(layers):
                        if dome_np is None or dome_np.isEmpty():
                            continue
                        layer_hue = (hue + layer_idx * 0.11) % 1.0
                        lr, lg, lb = colorsys.hsv_to_rgb(layer_hue, saturation, value)
                        layer_alpha = self._clamp(alpha * (1.0 - layer_idx * 0.13), 0.0, 1.0)
                        dome_np.setColor(lr, lg, lb, layer_alpha)
                else:
                    node.setColor(r, g, b, self._clamp(alpha, 0.0, 1.0))
                keep_spheres.append(entry)
            self.hyperbomb_spheres = keep_spheres

        if self.hyperbomb_audio_nodes:
            keep_audio: list[dict] = []
            for entry in self.hyperbomb_audio_nodes:
                entry["age"] += dt
                node = entry.get("node")
                if node is None or node.isEmpty():
                    continue
                if entry["age"] >= entry.get("life", 4.0):
                    node.removeNode()
                    continue
                keep_audio.append(entry)
            self.hyperbomb_audio_nodes = keep_audio

        self._apply_hyperbomb_monster_effects(dt)

    def _apply_hyperbomb_monster_effects(self, dt: float) -> None:
        if not self.monsters:
            return

        active_visual = bool(self.hyperbomb_active or self.hyperbomb_spheres)
        if not active_visual:
            for monster in self.monsters:
                if monster.get("dead", False):
                    continue
                root = monster.get("root")
                if root is None or root.isEmpty():
                    continue
                root.setAlphaScale(1.0)
            self.hyperbomb_damage_timer = 0.0
            return

        radius = max(0.5, float(self.hyperbomb_max_scale) * float(getattr(self, "hyperbomb_damage_radius_factor", 1.05)))
        radius_sq = radius * radius
        alpha_k = max(0.1, float(getattr(self, "hyperbomb_alpha_log_k", 9.0)))
        alpha_min = self._clamp(float(getattr(self, "hyperbomb_alpha_min", 0.12)), 0.0, 1.0)

        self.hyperbomb_damage_timer -= dt
        do_damage = self.hyperbomb_damage_timer <= 0.0
        if do_damage:
            self.hyperbomb_damage_timer = max(0.03, float(getattr(self, "hyperbomb_damage_interval", 0.12)))

        damage = max(1.0, float(getattr(self, "hyperbomb_damage_per_tick", 22.0)))

        for monster in self.monsters:
            if monster.get("dead", False):
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue

            mpos = root.getPos()
            dx = float(mpos.x) - float(self.hyperbomb_origin.x)
            dy = float(mpos.y) - float(self.hyperbomb_origin.y)
            dz = float(mpos.z) - float(self.hyperbomb_origin.z)
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq > radius_sq:
                root.setAlphaScale(1.0)
                continue

            dist = math.sqrt(max(0.0, dist_sq))
            r = self._clamp(dist / radius, 0.0, 1.0)
            log_drop = math.log1p(alpha_k * r) / math.log1p(alpha_k)
            alpha = self._clamp(1.0 - (1.0 - alpha_min) * log_drop, alpha_min, 1.0)
            root.setAlphaScale(alpha)

            if do_damage:
                dmg_mul = self._clamp(1.0 - (r ** 1.1), 0.25, 1.0)
                self._damage_monster(monster, damage * dmg_mul)

    def _spawn_water_crystal(self) -> None:
        if not bool(getattr(self, "water_crystal_spawn_enabled", True)):
            return
        if not self.water_surfaces:
            return
        if len(self.water_crystals) >= int(getattr(self, "water_crystal_max_count", 64)):
            return

        x = random.uniform(0.0, float(self.map_w))
        y = random.uniform(0.0, float(self.map_d))
        if hasattr(self, "ball_np") and self.ball_np is not None and not self.ball_np.isEmpty():
            ball_pos = self.ball_np.getPos()
            local_spread = max(10.0, float(getattr(self, "water_crystal_spawn_area_scale", 2.1)) * 7.0)
            x = float(ball_pos.x) + random.uniform(-local_spread, local_spread)
            y = float(ball_pos.y) + random.uniform(-local_spread, local_spread)
        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        water_h = self._sample_water_height(Vec3(x, y, self.floor_y + 1.0), self.roll_time)
        if water_h is None:
            surface = random.choice(self.water_surfaces)
            x = random.uniform(float(surface["x0"]), float(surface["x1"]))
            y = random.uniform(float(surface["y0"]), float(surface["y1"]))
            water_h = self._sample_water_height(Vec3(x, y, float(surface["base_z"])), self.roll_time)
            if water_h is None:
                water_h = float(surface["base_z"])

        float_offset = random.uniform(0.02, 0.16)
        height_min = float(getattr(self, "water_crystal_spawn_height_min", 8.0))
        height_max = max(height_min + 0.1, float(getattr(self, "water_crystal_spawn_height_max", 24.0)))
        start_h = float(water_h) + random.uniform(height_min, height_max)
        pos = Vec3(x, y, start_h)

        node = self.box_model.copyTo(self.render)
        node.setPos(pos)
        sx = random.uniform(0.26, 0.56)
        sy = random.uniform(0.26, 0.56)
        sz = random.uniform(0.72, 1.45)
        node.setScale(self.box_norm_scale)
        node.setScale(sx, sy, sz)
        node.setHpr(random.uniform(0.0, 360.0), random.uniform(-26.0, 26.0), random.uniform(-24.0, 24.0))
        node.clearTexture()
        node.setLightOff(1)
        node.setShaderOff(1)

        transparent = random.random() < 0.52
        base_alpha = random.uniform(0.65, 0.9) if transparent else 1.0
        if transparent:
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthWrite(False)
            node.setBin("transparent", 42)
            node.setAttrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.MAdd,
                    ColorBlendAttrib.OIncomingAlpha,
                    ColorBlendAttrib.OOne,
                ),
                1,
            )
        else:
            node.clearTransparency()
            node.setDepthWrite(True)
            node.setBin("fixed", 0)

        hue = random.random()
        sat = random.uniform(0.75, 1.0)
        val = random.uniform(0.9, 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        node.setColor(r, g, b, base_alpha)
        mat = Material()
        mat.setAmbient((min(1.0, r * 0.95), min(1.0, g * 0.95), min(1.0, b * 0.95), 1.0))
        mat.setDiffuse((1.0, 1.0, 1.0, 1.0))
        mat.setEmission((min(3.0, r * 2.2), min(3.0, g * 2.2), min(3.0, b * 2.2), 1.0))
        node.setMaterial(mat, 1)

        self.water_crystals.append(
            {
                "node": node,
            "mat": mat,
                "pos": pos,
                "phase": random.uniform(0.0, math.tau),
                "hue": hue,
                "hue_speed": random.uniform(0.22, 0.7),
                "sat": sat,
                "val": val,
                "alpha": base_alpha,
                "transparent": transparent,
                "fall_speed": random.uniform(0.0, 1.0),
                "float_offset": float_offset,
                "state": "falling",
                "state_age": 0.0,
            }
        )

    def _update_water_crystals(self, dt: float) -> None:
        if bool(getattr(self, "water_crystal_spawn_enabled", True)) and self.water_surfaces:
            if not self.water_crystals:
                for _ in range(10):
                    self._spawn_water_crystal()
            self.water_crystal_spawn_timer -= dt
            while self.water_crystal_spawn_timer <= 0.0:
                self._spawn_water_crystal()
                self.water_crystal_spawn_timer += max(0.05, float(getattr(self, "water_crystal_spawn_interval", 0.3)))
        else:
            self.water_crystal_spawn_timer = 0.0

        if not self.water_crystals:
            return

        keep: list[dict] = []
        for entry in self.water_crystals:
            node = entry.get("node")
            if node is None or node.isEmpty():
                continue

            pos = Vec3(entry.get("pos", node.getPos()))
            entry["state_age"] = float(entry.get("state_age", 0.0)) + dt
            state = str(entry.get("state", "falling"))

            water_h = self._sample_water_height(pos, self.roll_time)
            if water_h is None:
                water_h = pos.z - 0.1
            target_h = float(water_h) + float(entry.get("float_offset", 0.05))

            if state == "falling":
                fall_speed = float(entry.get("fall_speed", 0.0)) + float(getattr(self, "water_crystal_fall_gravity", 14.5)) * dt
                entry["fall_speed"] = fall_speed
                pos.z -= fall_speed * dt
                if pos.z <= target_h:
                    pos.z = target_h
                    entry["state"] = "stuck"
                    entry["state_age"] = 0.0
            elif state == "stuck":
                pos.z += (target_h - pos.z) * min(1.0, dt * 5.5)
                if entry["state_age"] >= float(getattr(self, "water_crystal_stuck_duration", 0.9)):
                    entry["state"] = "floating"
                    entry["state_age"] = 0.0
            elif state == "floating":
                bob = 0.02 * math.sin(self.roll_time * 2.4 + float(entry.get("phase", 0.0)))
                pos.z += ((target_h + bob) - pos.z) * min(1.0, dt * 4.6)
                if entry["state_age"] >= float(getattr(self, "water_crystal_float_duration", 10.0)):
                    entry["state"] = "fading"
                    entry["state_age"] = 0.0
            elif state == "fading":
                fade_t = self._clamp(
                    entry["state_age"] / max(0.05, float(getattr(self, "water_crystal_fade_duration", 1.5))),
                    0.0,
                    1.0,
                )
                pos.z += ((target_h + 0.015) - pos.z) * min(1.0, dt * 4.0)
                if fade_t >= 1.0:
                    node.removeNode()
                    continue
            else:
                entry["state"] = "falling"
                entry["state_age"] = 0.0

            hue = (float(entry.get("hue", 0.0)) + self.roll_time * float(entry.get("hue_speed", 0.42))) % 1.0
            sat = self._clamp(float(entry.get("sat", 0.9)), 0.0, 1.0)
            val = self._clamp(float(entry.get("val", 1.0)), 0.0, 1.0)
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)

            alpha = float(entry.get("alpha", 1.0))
            if entry.get("state") == "fading":
                fade_t = self._clamp(
                    float(entry.get("state_age", 0.0)) / max(0.05, float(getattr(self, "water_crystal_fade_duration", 1.5))),
                    0.0,
                    1.0,
                )
                alpha *= (1.0 - fade_t) ** 1.6

            spin = 36.0 if bool(entry.get("transparent", False)) else 24.0
            node.setH(node.getH() + spin * dt)
            node.setPos(pos)
            node.setColor(r, g, b, self._clamp(alpha, 0.0, 1.0))
            mat = entry.get("mat")
            if mat is not None:
                try:
                    mat.setEmission((min(2.0, r * 1.55), min(2.0, g * 1.55), min(2.0, b * 1.55), 1.0))
                    mat.setAmbient((min(1.0, r * 0.95), min(1.0, g * 0.95), min(1.0, b * 0.95), 1.0))
                except Exception:
                    pass

            entry["pos"] = pos
            keep.append(entry)

        self.water_crystals = keep

    def _load_ball_texture(self) -> Texture:
        candidates: list[str] = []
        ball_dir = "graphics/ball"
        if os.path.isdir(ball_dir):
            for name in sorted(os.listdir(ball_dir)):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tga")):
                    candidates.append(f"{ball_dir}/{name}")

        if candidates:
            path = random.choice(candidates)
            tex = self._get_cached_texture(path)
            if tex:
                return tex
            random.shuffle(candidates)
            for fallback in candidates:
                tex = self._get_cached_texture(fallback)
                if tex:
                    return tex

        return self._get_random_room_texture()

    def _load_water_base_texture(self) -> Texture:
        candidates: list[str] = []
        image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tga")
        self.water_base_tex_path = "procedural"

        water_dir = "graphics/water"
        if os.path.isdir(water_dir):
            for name in sorted(os.listdir(water_dir)):
                if name.lower().endswith(image_exts):
                    candidates.append(f"{water_dir}/{name}")

        room_dir = "graphics/rooms"
        if os.path.isdir(room_dir):
            for name in sorted(os.listdir(room_dir)):
                lower = name.lower()
                if lower.endswith(image_exts) and lower.startswith("water"):
                    candidates.append(f"{room_dir}/{name}")

        random.shuffle(candidates)
        for path in candidates:
            tex = self._get_cached_texture(path)
            if tex:
                self.water_base_tex_path = path
                return tex
        return self.floor_fractal_tex_b

    def _get_ball_uv_params(self, tex: Texture | None) -> tuple[float, float, float]:
        base_scale = 2.1
        u_offset = 0.12
        v_offset = 0.21
        if tex is None:
            return base_scale, u_offset, v_offset

        w = max(1, int(tex.getXSize()))
        h = max(1, int(tex.getYSize()))
        aspect = w / max(1.0, float(h))
        aspect_bias = max(0.75, min(1.35, aspect))
        uv_scale = base_scale * (1.0 / aspect_bias) ** 0.25

        # Keep offsets stable but nudge for non-square textures to reduce seam bias.
        u_offset += max(-0.06, min(0.06, (aspect - 1.0) * 0.05))
        v_offset -= max(-0.06, min(0.06, (aspect - 1.0) * 0.05))
        return uv_scale, u_offset, v_offset

    def _load_room_textures(self) -> list[Texture]:
        textures: list[Texture] = []
        self.room_texture_paths = []
        self.active_room_texture_paths = []
        room_dir = "graphics/rooms"
        if os.path.isdir(room_dir):
            for name in sorted(os.listdir(room_dir)):
                if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tga")):
                    continue
                path = f"{room_dir}/{name}"
                self.room_texture_paths.append(path)

        if self.room_texture_paths and self.max_room_texture_variants > 0:
            self.room_texture_paths = self.room_texture_paths[: self.max_room_texture_variants]

        if self.room_texture_paths:
            target_count = max(1, int(len(self.room_texture_paths) * self.room_texture_vram_ratio))
            target_count = min(target_count, len(self.room_texture_paths))
            active = self.room_texture_paths[::2]
            if len(active) < target_count:
                remainder = [path for path in self.room_texture_paths if path not in active]
                active.extend(remainder[: target_count - len(active)])
            self.active_room_texture_paths = active[:target_count]

        if not self.lazy_vram_loading:
            preload_paths = self.active_room_texture_paths if self.active_room_texture_paths else self.room_texture_paths
            for path in preload_paths:
                tex = self._get_cached_texture(path)
                if not tex:
                    continue
                textures.append(tex)

        if not textures:
            textures.append(self.level_checker_tex)
        return textures

    def _get_random_room_texture(self) -> Texture:
        if self.lazy_vram_loading:
            active_paths = self.active_room_texture_paths if self.active_room_texture_paths else self.room_texture_paths
            if active_paths:
                path = random.choice(active_paths)
                tex = self._get_cached_texture(path)
                if tex is not None:
                    return tex
        if self.room_textures:
            return random.choice(self.room_textures)
        return self.level_checker_tex

    def _configure_texture_defaults(self, tex: Texture) -> None:
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tex.setMinfilter(Texture.FTLinearMipmapLinear)
        tex.setMagfilter(Texture.FTLinear)
        tex.setAnisotropicDegree(1 if self.performance_mode else 2)

    def _get_cached_texture(self, path: str) -> Texture | None:
        cached = self.texture_cache.get(path)
        if cached is not None:
            return cached

        try:
            tex = self.loader.loadTexture(path)
        except Exception:
            return None
        if not tex:
            return None

        self._configure_texture_defaults(tex)
        self.texture_cache[path] = tex
        return tex

    def _prefetch_texture_assets(self) -> None:
        if self.lazy_vram_loading:
            self._texture_prefetched = True
            print("[perf] Texture prefetch skipped (lazy VRAM loading enabled)")
            return
        if self._texture_prefetched:
            return

        t0 = time.perf_counter()
        loaded = 0
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tga")
        texture_dirs = ["graphics/rooms", "graphics/ball"]
        for base_dir in texture_dirs:
            if not os.path.isdir(base_dir):
                continue
            for name in sorted(os.listdir(base_dir)):
                if not name.lower().endswith(exts):
                    continue
                path = f"{base_dir}/{name}"
                if base_dir == "graphics/rooms" and self.active_room_texture_paths and path not in self.active_room_texture_paths:
                    continue
                tex = self._get_cached_texture(path)
                if tex is not None:
                    loaded += 1

        self._texture_prefetched = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[perf] Texture prefetch complete: {loaded} textures cached in {elapsed_ms:.1f} ms")

    def _spawn_player_ball(self) -> None:
        spawn_player_ball(self)

    def _setup_weapon(self) -> None:
        setup_weapon_system(self)

    def _trigger_swing_attack(self) -> None:
        if self.game_over_active:
            return
        trigger_swing_attack(self)

    def _trigger_throw_attack(self) -> None:
        if self.game_over_active:
            return
        trigger_throw_attack(self)

    def _trigger_spin_attack(self) -> None:
        if self.game_over_active:
            return
        trigger_spin_attack(self)

    def _spawn_floating_text(
        self,
        world_pos: Vec3,
        text: str,
        color: tuple[float, float, float, float],
        scale: float = 0.35,
        life: float = 0.9,
    ) -> None:
        tn = TextNode("floating-text")
        tn.setText(text)
        tn.setTextColor(color)
        tn.setAlign(TextNode.ACenter)
        np = self.render.attachNewNode(tn)
        np.setPos(world_pos)
        np.setScale(scale)
        np.setBillboardPointEye()
        np.setDepthWrite(False)
        np.setBin("transparent", 45)
        self.floating_texts.append(
            {
                "node": np,
                "age": 0.0,
                "life": max(0.2, life),
                "vel": Vec3(random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(0.7, 1.05)),
                "base_scale": scale,
            }
        )

    def _update_floating_texts(self, dt: float) -> None:
        if not self.floating_texts:
            return
        keep: list[dict] = []
        for entry in self.floating_texts:
            node = entry["node"]
            if node is None or node.isEmpty():
                continue
            entry["age"] += dt
            life = entry["life"]
            if entry["age"] >= life:
                node.removeNode()
                continue
            t = entry["age"] / life
            pos = node.getPos()
            node.setPos(pos + entry["vel"] * dt)
            entry["vel"] *= max(0.0, 1.0 - dt * 1.7)
            fade = 1.0 - t
            node.setScale(entry["base_scale"] * (1.0 + 0.35 * t))
            node.setAlphaScale(max(0.0, fade * fade))
            keep.append(entry)
        self.floating_texts = keep

    def _setup_player_health_ui(self) -> None:
        self.player_hp_ui = self.aspect2d.attachNewNode("player-hp-ui")
        self.player_hp_ui.setPos(-1.28, 0, 0.8)

        hp_bg = self._instance_quad(self.player_hp_ui, "player-hp-bg", (0.0, 0.55, -0.04, 0.04))
        hp_bg.setColor(0.05, 0.07, 0.1, 0.86)
        hp_bg.setTransparency(TransparencyAttrib.MAlpha)

        hp_glow = self._instance_quad(self.player_hp_ui, "player-hp-glow", (-0.01, 0.56, -0.05, 0.05))
        hp_glow.setColor(0.32, 0.95, 1.0, 0.24)
        hp_glow.setTransparency(TransparencyAttrib.MAlpha)
        hp_glow.setDepthWrite(False)
        hp_glow.setDepthTest(False)
        hp_glow.setBin("fixed", 102)
        hp_glow.setLightOff(1)
        hp_glow.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        fill_root = self.player_hp_ui.attachNewNode("player-hp-fill-root")
        fill_root.setPos(0.01, 0.001, 0)
        hp_fill = self._instance_quad(fill_root, "player-hp-fill", (0.0, 0.53, -0.028, 0.028))
        hp_fill.setColor(0.95, 0.22, 0.22, 0.96)
        hp_fill.setTransparency(TransparencyAttrib.MAlpha)

        label = TextNode("player-hp-label")
        label.setText("HP")
        label.setTextColor(0.95, 0.98, 1.0, 0.95)
        label_np = self.player_hp_ui.attachNewNode(label)
        label_np.setPos(0.0, 0, 0.07)
        label_np.setScale(0.05)

        self.player_hp_fill_root = fill_root
        self.player_hp_fill_np = hp_fill
        self.player_hp_bg_np = hp_bg
        self.player_hp_glow_np = hp_glow
        self.player_hp_label_text_node = label
        self.player_hp_label_np = label_np

        self.monster_hud_ui = self.aspect2d.attachNewNode("monster-hud-ui")
        self.monster_hud_ui.setPos(0.92, 0, 0.9)
        monster_label = TextNode("monster-hud-label")
        monster_label.setAlign(TextNode.ALeft)
        monster_label.setTextColor(0.95, 0.98, 1.0, 0.95)
        self.monster_hud_label_np = self.monster_hud_ui.attachNewNode(monster_label)
        self.monster_hud_label_np.setScale(0.05)
        self.monster_hud_text_node = monster_label

        self._update_player_health_ui()
        self._update_monster_hud_ui()
        self._layout_hud_to_corners()

    def _setup_input_hud(self) -> None:
        if not bool(getattr(self, "input_hud_enabled", True)):
            return
        if hasattr(self, "input_hud_ui") and self.input_hud_ui is not None and not self.input_hud_ui.isEmpty():
            self.input_hud_ui.removeNode()

        root = self.aspect2d.attachNewNode("input-hud-ui")
        root.setPos(-1.22, 0.0, -0.86)
        root.setBin("fixed", 105)
        root.setDepthTest(False)
        root.setDepthWrite(False)

        self.input_hud_buttons = {}

        def add_button(key_name: str, label_text: str, x: float, z: float, parent: NodePath | None = None, frame: tuple[float, float, float, float] | None = None, label_scale: float = 0.042) -> None:
            host = parent if parent is not None else root
            holder = host.attachNewNode(f"input-hud-{key_name}")
            holder.setPos(x, 0.0, z)

            btn_frame = frame if frame is not None else (-0.055, 0.055, -0.042, 0.042)
            glow_pad_x = 0.015
            glow_pad_y = 0.01

            bg = self._instance_quad(holder, f"input-hud-bg-{key_name}", btn_frame)
            bg.setColor(0.06, 0.08, 0.12, 0.64)
            bg.setTransparency(TransparencyAttrib.MAlpha)

            glow = self._instance_quad(
                holder,
                f"input-hud-glow-{key_name}",
                (
                    btn_frame[0] - glow_pad_x,
                    btn_frame[1] + glow_pad_x,
                    btn_frame[2] - glow_pad_y,
                    btn_frame[3] + glow_pad_y,
                ),
            )
            glow.setColor(0.28, 0.78, 1.0, 0.0)
            glow.setTransparency(TransparencyAttrib.MAlpha)
            glow.setDepthWrite(False)
            glow.setDepthTest(False)
            glow.setBin("fixed", 106)
            glow.setLightOff(1)
            glow.setAttrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.MAdd,
                    ColorBlendAttrib.OIncomingAlpha,
                    ColorBlendAttrib.OOne,
                ),
                1,
            )

            tn = TextNode(f"input-hud-label-{key_name}")
            tn.setAlign(TextNode.ACenter)
            tn.setText(label_text)
            tn.setTextColor(0.9, 0.94, 1.0, 0.92)
            label_np = holder.attachNewNode(tn)
            label_np.setPos(0.0, 0.0, -0.018)
            label_np.setScale(label_scale)

            self.input_hud_buttons[key_name] = {
                "holder": holder,
                "bg": bg,
                "glow": glow,
                "label": tn,
            }

        base_x = 0.0
        base_z = 0.0
        spacing = 0.122

        add_button("w", "W", base_x, base_z + spacing)
        add_button("a", "A", base_x - spacing, base_z)
        add_button("s", "S", base_x, base_z)
        add_button("d", "D", base_x + spacing, base_z)

        mouse_root = root.attachNewNode("input-hud-mouse")
        mouse_root.setPos(base_x + 0.62, 0.0, base_z + 0.03)

        mouse_body = self._instance_quad(mouse_root, "input-hud-mouse-body", (-0.18, 0.18, -0.14, 0.16))
        mouse_body.setColor(0.04, 0.06, 0.09, 0.72)
        mouse_body.setTransparency(TransparencyAttrib.MAlpha)

        mouse_wheel_slot = self._instance_quad(mouse_root, "input-hud-mouse-wheel-slot", (-0.03, 0.03, -0.02, 0.08))
        mouse_wheel_slot.setColor(0.02, 0.03, 0.05, 0.88)
        mouse_wheel_slot.setTransparency(TransparencyAttrib.MAlpha)

        add_button("mouse1", "L", -0.088, 0.058, parent=mouse_root, frame=(-0.072, 0.072, -0.056, 0.056), label_scale=0.034)
        add_button("mouse2", "M", 0.0, 0.0, parent=mouse_root, frame=(-0.032, 0.032, -0.04, 0.04), label_scale=0.03)
        add_button("mouse3", "R", 0.088, 0.058, parent=mouse_root, frame=(-0.072, 0.072, -0.056, 0.056), label_scale=0.034)

        self.input_hud_ui = root
        self._layout_hud_to_corners()

    def _update_input_hud(self, dt: float) -> None:
        if not bool(getattr(self, "input_hud_enabled", True)):
            return
        buttons = getattr(self, "input_hud_buttons", None)
        if not buttons:
            return

        self.input_hud_r_pulse = max(0.0, float(getattr(self, "input_hud_r_pulse", 0.0)) - dt * 2.8)
        hue = (self.roll_time * 0.28) % 1.0

        for key_name, entry in buttons.items():
            bg = entry.get("bg")
            glow = entry.get("glow")
            label = entry.get("label")
            if bg is None or bg.isEmpty() or glow is None or glow.isEmpty() or not isinstance(label, TextNode):
                continue

            pressed = False
            if key_name in self.keys:
                pressed = bool(self.keys.get(key_name, False))
            elif key_name in self.camera_keys:
                pressed = bool(self.camera_keys.get(key_name, False))
            elif key_name == "r":
                pressed = bool(getattr(self, "input_hud_r_state", False)) or float(getattr(self, "input_hud_r_pulse", 0.0)) > 0.0
            elif key_name in self.input_hud_mouse_state:
                pressed = bool(self.input_hud_mouse_state.get(key_name, False))

            if pressed:
                key_hue = (hue + 0.19 * (abs(hash(key_name)) % 7)) % 1.0
                rr, rg, rb = colorsys.hsv_to_rgb(key_hue, 0.72, 1.0)
                bg.setColor(0.14 + rr * 0.32, 0.15 + rg * 0.34, 0.18 + rb * 0.34, 0.94)
                glow.setColor(rr, rg, rb, 0.52)
                label.setTextColor(1.0, 1.0, 1.0, 1.0)
            else:
                bg.setColor(0.06, 0.08, 0.12, 0.64)
                glow.setColor(0.28, 0.78, 1.0, 0.02)
                label.setTextColor(0.9, 0.94, 1.0, 0.92)

    def _setup_holographic_map_ui(self) -> None:
        if not bool(getattr(self, "holo_map_enabled", True)):
            return
        if self.holo_map_ui is not None and not self.holo_map_ui.isEmpty():
            self.holo_map_ui.removeNode()

        root = self.aspect2d.attachNewNode("holo-map-ui")
        root.setPos(1.03, 0.0, -0.66)
        root.setBin("fixed", 110)
        root.setDepthTest(False)
        root.setDepthWrite(False)

        panel = self._instance_quad(root, "holo-map-panel", (-0.28, 0.28, -0.28, 0.28))
        panel.setColor(0.02, 0.09, 0.12, 0.42)
        panel.setTransparency(TransparencyAttrib.MAlpha)

        ring = self._instance_quad(root, "holo-map-ring", (-0.295, 0.295, -0.295, 0.295))
        ring.setColor(0.2, 0.92, 1.0, 0.24)
        ring.setTransparency(TransparencyAttrib.MAlpha)
        ring.setLightOff(1)
        ring.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )

        cross_h = self._instance_quad(root, "holo-map-cross-h", (-0.28, 0.28, -0.0025, 0.0025))
        cross_h.setColor(0.25, 0.85, 1.0, 0.22)
        cross_h.setTransparency(TransparencyAttrib.MAlpha)
        cross_v = self._instance_quad(root, "holo-map-cross-v", (-0.0025, 0.0025, -0.28, 0.28))
        cross_v.setColor(0.25, 0.85, 1.0, 0.22)
        cross_v.setTransparency(TransparencyAttrib.MAlpha)

        title = TextNode("holo-map-title")
        title.setAlign(TextNode.ACenter)
        title.setText("HOLO MAP")
        title.setTextColor(0.58, 0.98, 1.0, 0.94)
        title_np = root.attachNewNode(title)
        title_np.setPos(0.0, 0.0, 0.325)
        title_np.setScale(0.04)

        marker_root = root.attachNewNode("holo-map-markers")
        marker_root.setPos(0.0, 0.0, 0.0)

        self.holo_map_ui = root
        self.holo_map_marker_root = marker_root
        self.holo_map_markers = []
        self._layout_hud_to_corners()

    def _add_holo_map_marker(self, x: float, z: float, size: float, color: tuple[float, float, float, float]) -> None:
        if self.holo_map_marker_root is None or self.holo_map_marker_root.isEmpty():
            return
        marker = self._instance_quad(self.holo_map_marker_root, "holo-map-marker", (-size, size, -size, size))
        marker.setPos(x, 0.0, z)
        marker.setColor(*color)
        marker.setTransparency(TransparencyAttrib.MAlpha)
        marker.setDepthWrite(False)
        marker.setDepthTest(False)
        marker.setBin("fixed", 111)
        marker.setLightOff(1)
        marker.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            ),
            1,
        )
        self.holo_map_markers.append(marker)

    def _update_holographic_map_ui(self, dt: float) -> None:
        if not bool(getattr(self, "holo_map_enabled", True)):
            return
        if self.holo_map_ui is None or self.holo_map_ui.isEmpty() or self.holo_map_marker_root is None or self.holo_map_marker_root.isEmpty():
            return
        if not hasattr(self, "ball_np") or self.ball_np is None or self.ball_np.isEmpty():
            return

        panel_pulse = 0.5 + 0.5 * math.sin(self.roll_time * 2.7)
        self.holo_map_ui.setScale(1.0 + panel_pulse * 0.01)

        self.holo_map_update_timer -= dt
        if self.holo_map_update_timer > 0.0:
            return
        self.holo_map_update_timer = max(1e-3, float(getattr(self, "holo_map_update_interval", 1.0 / 14.0)))

        for marker in self.holo_map_markers:
            if marker is not None and not marker.isEmpty():
                marker.removeNode()
        self.holo_map_markers.clear()

        center = self.ball_np.getPos(self.render)
        radius = max(6.0, float(getattr(self, "holo_map_radius_world", 36.0)))
        inv_radius = 1.0 / radius
        map_scale = 0.26

        def project(world_pos: Vec3) -> tuple[float, float] | None:
            delta = Vec3(world_pos) - center
            delta.z = 0.0
            dist = delta.length()
            if dist > radius:
                return None
            return (delta.x * inv_radius * map_scale, delta.y * inv_radius * map_scale)

        self._add_holo_map_marker(0.0, 0.0, 0.014, (1.0, 1.0, 1.0, 0.95))

        for monster in getattr(self, "monsters", []):
            if monster.get("dead", False):
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            pos2 = project(root.getPos(self.render))
            if pos2 is None:
                continue
            self._add_holo_map_marker(pos2[0], pos2[1], 0.009, (1.0, 0.28, 0.36, 0.9))

        for item in getattr(self, "health_powerups", []):
            root = item.get("root")
            if root is None or root.isEmpty():
                continue
            pos2 = project(root.getPos(self.render))
            if pos2 is None:
                continue
            self._add_holo_map_marker(pos2[0], pos2[1], 0.008, (0.36, 1.0, 0.5, 0.92))

        for item in getattr(self, "sword_powerups", []):
            root = item.get("root")
            if root is None or root.isEmpty():
                continue
            pos2 = project(root.getPos(self.render))
            if pos2 is None:
                continue
            self._add_holo_map_marker(pos2[0], pos2[1], 0.008, (0.96, 0.88, 0.28, 0.92))

        for anomaly in getattr(self, "black_holes", []):
            root = anomaly.get("root")
            if root is None or root.isEmpty():
                continue
            pos2 = project(root.getPos(self.render))
            if pos2 is None:
                continue
            kind = str(anomaly.get("kind", "suck")).lower()
            if kind == "blow":
                self._add_holo_map_marker(pos2[0], pos2[1], 0.011, (0.34, 0.78, 1.0, 0.95))
            else:
                self._add_holo_map_marker(pos2[0], pos2[1], 0.011, (1.0, 0.44, 0.88, 0.95))

        for landmark in getattr(self, "room_landmarks", []):
            root = landmark.get("root")
            if root is None or root.isEmpty():
                continue
            pos2 = project(root.getPos(self.render))
            if pos2 is None:
                continue
            self._add_holo_map_marker(pos2[0], pos2[1], 0.007, (0.78, 0.98, 1.0, 0.78))

    def _update_player_health_ui(self) -> None:
        if not hasattr(self, "player_hp_fill_root"):
            return
        ratio = max(0.0, min(1.0, self.player_hp / max(1e-6, self.player_hp_max)))
        self.player_hp_fill_root.setScale(ratio, 1.0, 1.0)
        hue_shift = (self.roll_time * 0.42) % 1.0
        pulse = 0.5 + 0.5 * math.sin(self.roll_time * 7.2)
        health_hue = (0.0 + ratio * 0.33 + hue_shift * 0.08) % 1.0
        fr, fg, fb = colorsys.hsv_to_rgb(health_hue, 0.88, 1.0)

        hp_fill = getattr(self, "player_hp_fill_np", None)
        if hp_fill is not None and not hp_fill.isEmpty():
            hp_fill.setColor(fr, fg, fb, 0.88 + 0.1 * pulse)

        hp_glow = getattr(self, "player_hp_glow_np", None)
        if hp_glow is not None and not hp_glow.isEmpty():
            glow_hue = (hue_shift + 0.52) % 1.0
            gr, gg, gb = colorsys.hsv_to_rgb(glow_hue, 0.72, 1.0)
            hp_glow.setColor(gr, gg, gb, 0.16 + 0.22 * pulse)
            hp_glow.setScale(1.0 + 0.015 * pulse, 1.0, 1.0)

        label = getattr(self, "player_hp_label_text_node", None)
        if isinstance(label, TextNode):
            lr, lg, lb = colorsys.hsv_to_rgb((hue_shift + 0.12) % 1.0, 0.45, 1.0)
            label.setTextColor(lr, lg, lb, 0.96)

    def _update_monster_hud_ui(self) -> None:
        node = getattr(self, "monster_hud_text_node", None)
        if not isinstance(node, TextNode):
            return
        total = max(0, int(getattr(self, "monsters_total", 0)))
        slain = max(0, int(getattr(self, "monsters_slain", 0)))
        if total > 0:
            slain = min(slain, total)
            left = max(0, total - slain)
        else:
            alive = 0
            for monster in getattr(self, "monsters", []):
                if monster.get("dead", False):
                    continue
                root = monster.get("root")
                if root is None or root.isEmpty():
                    continue
                alive += 1
            left = alive
            slain = max(0, slain)
        node.setText(f"Slayed: {slain}\nLeft: {left}")

    def _setup_game_over_ui(self) -> None:
        if self.game_over_ui is not None and not self.game_over_ui.isEmpty():
            return
        root = self.aspect2d.attachNewNode("game-over-ui")
        root.setBin("fixed", 100)
        root.setDepthTest(False)
        root.setDepthWrite(False)

        panel = self._instance_quad(root, "game-over-bg", (-0.72, 0.72, -0.18, 0.18))
        panel.setColor(0.02, 0.03, 0.05, 0.78)
        panel.setTransparency(TransparencyAttrib.MAlpha)

        title = TextNode("game-over-title")
        title.setText("GAME OVER")
        title.setAlign(TextNode.ACenter)
        title.setTextColor(0.96, 0.2, 0.24, 0.98)
        title_np = root.attachNewNode(title)
        title_np.setPos(0.0, 0.0, 0.045)
        title_np.setScale(0.115)

        prompt = TextNode("game-over-prompt")
        prompt.setText("Press R to restart")
        prompt.setAlign(TextNode.ACenter)
        prompt.setTextColor(0.92, 0.97, 1.0, 0.92)
        prompt_np = root.attachNewNode(prompt)
        prompt_np.setPos(0.0, 0.0, -0.075)
        prompt_np.setScale(0.055)
        self.game_over_prompt_text_node = prompt
        self._refresh_game_over_prompt()

        root.hide()
        self.game_over_ui = root

    def _refresh_game_over_prompt(self) -> None:
        prompt = getattr(self, "game_over_prompt_text_node", None)
        if not isinstance(prompt, TextNode):
            return
        seconds_left = max(0, int(math.ceil(float(getattr(self, "game_over_countdown", 0.0)))))
        prompt.setText(f"Press R to restart\nAuto replay AI in {seconds_left}")

    def _update_game_over_countdown(self, dt: float) -> None:
        if not self.game_over_active:
            return
        self.game_over_countdown = max(0.0, float(self.game_over_countdown) - max(0.0, float(dt)))
        self._refresh_game_over_prompt()
        if self.game_over_countdown <= 0.0:
            self._restart_after_game_over()

    def _setup_win_ui(self) -> None:
        if self.win_ui is not None and not self.win_ui.isEmpty():
            return
        root = self.aspect2d.attachNewNode("win-ui")
        root.setBin("fixed", 101)
        root.setDepthTest(False)
        root.setDepthWrite(False)

        panel = self._instance_quad(root, "win-bg", (-0.72, 0.72, -0.18, 0.18))
        panel.setColor(0.02, 0.06, 0.04, 0.78)
        panel.setTransparency(TransparencyAttrib.MAlpha)

        title = TextNode("win-title")
        title.setText("YOU WIN")
        title.setAlign(TextNode.ACenter)
        title.setTextColor(0.38, 1.0, 0.62, 0.98)
        title_np = root.attachNewNode(title)
        title_np.setPos(0.0, 0.0, 0.045)
        title_np.setScale(0.115)

        prompt = TextNode("win-prompt")
        prompt.setText("Press R to restart")
        prompt.setAlign(TextNode.ACenter)
        prompt.setTextColor(0.92, 0.97, 1.0, 0.92)
        prompt_np = root.attachNewNode(prompt)
        prompt_np.setPos(0.0, 0.0, -0.075)
        prompt_np.setScale(0.055)

        root.hide()
        self.win_ui = root

    def _trigger_game_over(self) -> None:
        if self.game_over_active:
            return
        self.game_over_active = True
        self.win_active = False
        self._clear_magic_missiles()
        self.game_over_countdown = float(getattr(self, "game_over_auto_restart_seconds", 10.0))
        self._refresh_game_over_prompt()
        self.jump_queued = False
        if hasattr(self, "ball_body") and self.ball_body is not None:
            self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
            self.ball_body.setAngularVelocity(Vec3(0, 0, 0))
        if self.game_over_ui is not None and not self.game_over_ui.isEmpty():
            self.game_over_ui.show()
        self._play_game_over_sfx()

    def _trigger_win(self) -> None:
        if self.win_active:
            return
        self.win_active = True
        self.game_over_active = False
        self._clear_magic_missiles()
        self.jump_queued = False
        if hasattr(self, "ball_body") and self.ball_body is not None:
            self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
            self.ball_body.setAngularVelocity(Vec3(0, 0, 0))
        if self.win_ui is not None and not self.win_ui.isEmpty():
            self.win_ui.show()
        self._play_win_sfx()

    def _restart_after_game_over(self) -> None:
        if not getattr(self, "game_over_active", False) and not getattr(self, "win_active", False):
            return
        self.game_over_active = False
        self.win_active = False
        self._clear_magic_missiles()
        self.magic_missile_cooldown = 0.0
        self.game_over_countdown = float(getattr(self, "game_over_auto_restart_seconds", 10.0))
        self._refresh_game_over_prompt()

        if self.game_over_ui is not None and not self.game_over_ui.isEmpty():
            self.game_over_ui.hide()
        if self.win_ui is not None and not self.win_ui.isEmpty():
            self.win_ui.hide()

        spawn = Vec3(getattr(self, "platform_course_spawn_pos", Vec3(self.map_w * 0.5, self.map_d * 0.5, self.floor_y + 2.6)))
        if hasattr(self, "ball_np") and self.ball_np is not None and not self.ball_np.isEmpty():
            self.ball_np.setPos(spawn)
        if hasattr(self, "ball_body") and self.ball_body is not None:
            self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
            self.ball_body.setAngularVelocity(Vec3(0, 0, 0))
        self.player_hp = self.player_hp_max
        self._update_player_health_ui()
        self.fall_air_timer = 0.0
        self.fall_reference_z = float(spawn.z)
        self.player_ai_target_id = None
        self.player_ai_lock_target_id = None
        self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
        self.player_ai_retarget_timer = 0.0
        self.player_ai_combo_step = 0
        self.player_ai_combo_timer = 0.0
        self.player_ai_jump_cooldown = 0.0
        self.player_ai_room_path = []
        self.player_ai_room_path_goal = None
        self.player_ai_room_path_recalc_timer = 0.0
        self.player_ai_idle_timer = self.player_ai_idle_delay
        self.player_ai_enabled = True

        self._teardown_monster_ai_system()
        for monster in self.monsters:
            root = monster.get("root")
            if root is not None and not root.isEmpty():
                root.removeNode()
            hum = monster.get("hum_sfx")
            if hum:
                hum.stop()
        self.monsters = []
        self._spawn_hypercube_monsters(count=64)
        self._attach_monster_hum_sounds()
        self._setup_monster_ai_system()

    def _setup_kill_protection_system(self) -> None:
        if not hasattr(self, "ball_np"):
            return
        self.kill_protection_root = self.ball_np.attachNewNode("kill-protection-root")
        self.kill_protection_root.setPos(0, 0, 0)

        self.kill_protection_light = PointLight("kill-protection-light")
        self.kill_protection_light.setColor((0.0, 0.0, 0.0, 1.0))
        self.kill_protection_light.setAttenuation((1.0, 0.32, 0.08))
        self.kill_protection_light_np = self.ball_np.attachNewNode(self.kill_protection_light)
        self.kill_protection_light_np.setPos(0, 0, self.ball_radius + 0.26)
        self.render.setLight(self.kill_protection_light_np)

    def _grant_kill_protection(self) -> None:
        prev = int(self.kill_protection_stacks)
        self.kill_protection_stacks = min(self.kill_protection_max_stacks, self.kill_protection_stacks + 1)
        if self.kill_protection_stacks <= prev:
            return
        if self.kill_protection_root is None or self.kill_protection_root.isEmpty():
            return

        idx = len(self.kill_protection_rings)
        band_cycle = ("strength", "neutral", "other")
        band_kind = band_cycle[idx % len(band_cycle)]
        band_colors = {
            "strength": (1.0, 0.24, 0.24),
            "neutral": (0.24, 0.66, 1.0),
            "other": (0.24, 1.0, 0.44),
        }
        base_col = band_colors.get(band_kind, (0.24, 0.66, 1.0))
        ring = self.sphere_model.copyTo(self.kill_protection_root)
        ring.clearTexture()
        ring.setLightOff(1)
        ring.setTransparency(TransparencyAttrib.MAlpha)
        ring.setDepthWrite(False)
        ring.setBin("transparent", 38)
        ring.setScale(self.ball_radius * 1.3, self.ball_radius * 1.3, self.ball_radius * 0.08)
        ring.setColor(base_col[0], base_col[1], base_col[2], 0.42)
        ring.setPos(0, 0, 0)

        self.kill_protection_rings.append(
            {
                "node": ring,
                "band_kind": band_kind,
                "base_col": base_col,
                "phase": random.uniform(0.0, math.tau),
                "speed": random.uniform(1.8, 3.4),
                "tilt_p": random.uniform(-62.0, 62.0),
                "tilt_r": random.uniform(-62.0, 62.0),
            }
        )
        self._spawn_floating_text(self.ball_np.getPos() + Vec3(0, 0, 0.7), "GUARD +1", (0.3, 0.96, 1.0, 1.0), scale=0.24, life=0.8)

    def _consume_kill_protection(self) -> bool:
        if self.kill_protection_stacks <= 0:
            return False
        self.kill_protection_stacks -= 1
        if self.kill_protection_rings:
            ring = self.kill_protection_rings.pop()
            node = ring.get("node")
            if node is not None and not node.isEmpty():
                node.removeNode()
        self._spawn_floating_text(self.ball_np.getPos() + Vec3(0, 0, 0.75), "BLOCK", (0.38, 1.0, 1.0, 1.0), scale=0.26, life=0.7)
        self._play_sound(self.sfx_pickup, volume=0.62, play_rate=1.2)
        return True

    def _update_kill_protection_visuals(self, dt: float) -> None:
        stacks = max(0, int(self.kill_protection_stacks))
        if self.kill_protection_light is not None:
            glow = min(1.0, stacks / 8.0)
            pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(self.roll_time * 7.2))
            self.kill_protection_light.setColor((0.08 * glow * pulse, 0.58 * glow * pulse, 0.9 * glow * pulse, 1.0))

        if not self.kill_protection_rings:
            return

        max_stacks = max(1, int(getattr(self, "kill_protection_max_stacks", 24)))
        stack_ratio = self._clamp(float(stacks) / float(max_stacks), 0.0, 1.0)
        spin_boost = 1.0 + stack_ratio * 1.8
        scale_boost = stack_ratio * 0.55

        keep: list[dict] = []
        for idx, ring_entry in enumerate(self.kill_protection_rings):
            node = ring_entry.get("node")
            if node is None or node.isEmpty():
                continue
            phase = ring_entry.get("phase", 0.0)
            speed = ring_entry.get("speed", 2.0)
            tilt_p = float(ring_entry.get("tilt_p", 0.0))
            tilt_r = float(ring_entry.get("tilt_r", 0.0))
            pulse = 0.5 + 0.5 * math.sin(self.roll_time * (speed * 1.4) + phase)
            node.setPos(0, 0, 0)
            spin_deg = self.roll_time * ((56.0 + idx * 16.0) * speed * spin_boost)
            node.setHpr(spin_deg + math.degrees(phase), tilt_p, tilt_r)
            ring_scale = self.ball_radius * (1.1 + idx * 0.05 + scale_boost + pulse * 0.08)
            ring_thickness = self.ball_radius * (0.055 + idx * 0.003 + stack_ratio * 0.045)
            node.setScale(ring_scale, ring_scale, ring_thickness)
            base_col = ring_entry.get("base_col", (0.24, 0.66, 1.0))
            base_r = float(base_col[0])
            base_g = float(base_col[1])
            base_b = float(base_col[2])
            tint = 0.82 + pulse * 0.28
            node.setColor(
                min(1.0, base_r * tint),
                min(1.0, base_g * tint),
                min(1.0, base_b * tint),
                0.22 + pulse * 0.32,
            )
            keep.append(ring_entry)
        self.kill_protection_rings = keep

    def _apply_player_damage(self, amount: float, crit_chance_override: float | None = None) -> bool:
        if self.game_over_active:
            return False
        dmg = max(0.0, float(amount))
        if dmg <= 0.0:
            return False
        if self._consume_kill_protection():
            return False

        if crit_chance_override is None:
            crit_chance = self._clamp(float(getattr(self, "critical_hit_chance_current", 0.08)), 0.0, 0.95)
        else:
            crit_chance = self._clamp(float(crit_chance_override), 0.0, 0.95)
        is_critical = random.random() < crit_chance
        if is_critical:
            dmg *= max(1.2, float(getattr(self, "critical_hit_multiplier", 2.1)))

        dmg *= max(0.2, float(getattr(self, "damage_taken_multiplier", 1.0)))
        self.player_hp = max(0.0, self.player_hp - dmg)
        self._update_player_health_ui()
        if hasattr(self, "ball_np"):
            if is_critical:
                self._spawn_floating_text(
                    self.ball_np.getPos() + Vec3(0, 0, 0.72),
                    f"CRITICAL -{int(round(dmg))}",
                    (1.0, 0.14, 0.14, 1.0),
                    scale=0.34,
                    life=1.0,
                )
                self._play_critical_damage_sfx()
            else:
                self._spawn_floating_text(self.ball_np.getPos() + Vec3(0, 0, 0.62), f"HP -{int(round(dmg))}", (1.0, 0.25, 0.25, 1.0), scale=0.28, life=0.85)
        if self.player_hp <= 0.0:
            self._trigger_game_over()
        return is_critical

    def _update_combat_stat_buffs(self, dt: float) -> None:
        buffs = getattr(self, "skill_buffs", None)
        if isinstance(buffs, dict):
            for key, remain in list(buffs.items()):
                buffs[key] = max(0.0, float(remain) - dt)

        dex = max(0, int(getattr(self, "player_dex_stat", 0)))
        defense = max(0, int(getattr(self, "player_defense_stat", 0)))
        intelligence = max(0, int(getattr(self, "player_int_stat", 0)))

        haste_active = float(getattr(self, "skill_buffs", {}).get("haste", 0.0)) > 0.0
        longblade_active = float(getattr(self, "skill_buffs", {}).get("longblade", 0.0)) > 0.0
        fury_active = float(getattr(self, "skill_buffs", {}).get("fury", 0.0)) > 0.0
        critical_active = float(getattr(self, "skill_buffs", {}).get("critical", 0.0)) > 0.0

        dex_factor = 1.0 - min(0.38, dex * 0.018)
        haste_factor = 0.65 if haste_active else 1.0
        self.attack_cooldown_multiplier = max(0.35, dex_factor * haste_factor)

        int_reach = min(0.35, intelligence * 0.02)
        self.sword_reach_multiplier = 1.0 + int_reach + (0.38 if longblade_active else 0.0)

        int_damage = min(0.28, intelligence * 0.016)
        self.combat_damage_multiplier = 1.0 + int_damage + (0.35 if fury_active else 0.0)

        defense_factor = 1.0 - min(0.58, defense * 0.032)
        self.damage_taken_multiplier = max(0.3, defense_factor)

        base_crit = max(0.0, float(getattr(self, "critical_hit_base_chance", 0.08)))
        perm_crit = max(0.0, float(getattr(self, "critical_chance_bonus_permanent", 0.0)))
        temp_crit = float(getattr(self, "critical_chance_bonus_temp_bonus", 0.14)) if critical_active else 0.0
        self.critical_hit_chance_current = self._clamp(base_crit + perm_crit + temp_crit, 0.0, 0.9)

    def _apply_sword_pickup_effect(self, item: dict, ball_pos: Vec3) -> None:
        pickup_type = str(item.get("type", "sword")).lower()
        color = (0.34, 0.96, 1.0, 1.0)
        text = "POWER"

        if pickup_type == "attack":
            self.player_attack_stat += 1
            bonus = 0.09 + min(0.14, self.player_attack_stat * 0.006)
            self.sword_damage_multiplier = min(self.sword_damage_multiplier_cap, self.sword_damage_multiplier + bonus)
            text = f"ATK +1  x{self.sword_damage_multiplier:.2f}"
            color = (1.0, 0.44, 0.3, 1.0)
        elif pickup_type == "defense":
            self.player_defense_stat += 1
            text = f"DEF +1  dmg x{self.damage_taken_multiplier:.2f}"
            color = (0.42, 0.9, 1.0, 1.0)
        elif pickup_type == "dex":
            self.player_dex_stat += 1
            text = f"DEX +1  cd x{self.attack_cooldown_multiplier:.2f}"
            color = (1.0, 0.95, 0.34, 1.0)
        elif pickup_type == "sta":
            self.player_sta_stat += 1
            self.player_hp_max += 6.0
            self.player_hp = min(self.player_hp_max, self.player_hp + 8.0)
            self._update_player_health_ui()
            text = f"STA +1  HP {int(round(self.player_hp_max))}"
            color = (0.48, 1.0, 0.42, 1.0)
        elif pickup_type == "int":
            self.player_int_stat += 1
            text = f"INT +1  buffs+"
            color = (0.78, 0.55, 1.0, 1.0)
        elif pickup_type == "skill_haste":
            duration = 12.0 + min(8.0, self.player_int_stat * 0.4)
            self.skill_buffs["haste"] = max(self.skill_buffs["haste"], duration)
            text = f"SKILL: HASTE {duration:.0f}s"
            color = (1.0, 0.88, 0.38, 1.0)
        elif pickup_type == "skill_longblade":
            duration = 14.0 + min(10.0, self.player_int_stat * 0.5)
            self.skill_buffs["longblade"] = max(self.skill_buffs["longblade"], duration)
            text = f"SKILL: LONGBLADE {duration:.0f}s"
            color = (0.35, 0.95, 1.0, 1.0)
        elif pickup_type == "skill_fury":
            duration = 10.0 + min(8.0, self.player_int_stat * 0.45)
            self.skill_buffs["fury"] = max(self.skill_buffs["fury"], duration)
            text = f"SKILL: FURY {duration:.0f}s"
            color = (1.0, 0.4, 0.66, 1.0)
        elif pickup_type == "crit_core":
            add = 0.022 + min(0.02, self.player_int_stat * 0.001)
            self.critical_chance_bonus_permanent = self._clamp(
                float(getattr(self, "critical_chance_bonus_permanent", 0.0)) + add,
                0.0,
                0.5,
            )
            text = f"CRIT CHANCE +{add * 100.0:.1f}%"
            color = (1.0, 0.5, 0.22, 1.0)
        elif pickup_type == "skill_critical":
            duration = 9.0 + min(9.0, self.player_int_stat * 0.5)
            self.skill_buffs["critical"] = max(self.skill_buffs["critical"], duration)
            text = f"SKILL: CRITICAL {duration:.0f}s"
            color = (1.0, 0.28, 0.2, 1.0)
        else:
            self.sword_upgrade_level += 1
            bonus = float(self.sword_damage_per_pickup)
            self.sword_damage_multiplier = min(self.sword_damage_multiplier_cap, self.sword_damage_multiplier + bonus)
            text = f"SWORD UP x{self.sword_damage_multiplier:.2f}"

        self._update_combat_stat_buffs(0.0)
        self._spawn_floating_text(ball_pos + Vec3(0, 0, 0.6), text, color, scale=0.27, life=0.95)
        self._play_sound(self.sfx_pickup, volume=0.74, play_rate=1.15)

    def _spawn_health_powerups(self, count: int = 10) -> None:
        if not self.rooms:
            return
        self.health_powerups.clear()
        start_room_idx = int(getattr(self, "start_room_idx", 0)) if self.rooms else 0
        start_room_idx = max(0, min(len(self.rooms) - 1, start_room_idx))
        for idx in range(count):
            if idx < min(3, count):
                room_idx = start_room_idx
            else:
                room_idx = random.randrange(len(self.rooms))
            room = self.rooms[room_idx]
            room_base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            x = random.uniform(room.x + 0.9, room.x + room.w - 0.9)
            y = random.uniform(room.y + 0.9, room.y + room.h - 0.9)
            root = self.world.attachNewNode(f"health-plus-{idx}")
            root.setPos(x, y, room_base_z + 1.25)

            stem = self.box_model.copyTo(root)
            stem.setPos(self.box_norm_offset)
            stem.setScale(self.box_norm_scale)
            stem.setScale(0.14, 0.52, 0.14)
            stem.clearTexture()
            stem.setTexture(self.level_checker_tex, 1)
            stem.setLightOff(1)
            stem.setTransparency(TransparencyAttrib.MAlpha)
            stem.setColor(0.28, 1.0, 0.42, 0.98)

            arm = self.box_model.copyTo(root)
            arm.setPos(self.box_norm_offset)
            arm.setScale(self.box_norm_scale)
            arm.setScale(0.52, 0.14, 0.14)
            arm.clearTexture()
            arm.setTexture(self.level_checker_tex, 1)
            arm.setLightOff(1)
            arm.setTransparency(TransparencyAttrib.MAlpha)
            arm.setColor(0.28, 1.0, 0.42, 0.98)

            glow = self.sphere_model.copyTo(root)
            glow.setScale(0.46)
            glow.clearTexture()
            glow.setColor(0.25, 1.0, 0.38, 0.38)
            glow.setLightOff(1)
            glow.setTransparency(TransparencyAttrib.MAlpha)
            glow.setDepthWrite(False)
            glow.setDepthTest(False)
            glow.setBin("transparent", 35)

            self._register_color_cycle(stem, (0.2, 1.0, 0.4, 1.0), min_speed=0.9, max_speed=6.7)
            self._register_color_cycle(arm, (0.2, 1.0, 0.4, 1.0), min_speed=0.9, max_speed=6.7)

            self.health_powerups.append(
                {
                    "root": root,
                    "base_z": room_base_z + 1.25,
                    "phase": random.uniform(0.0, math.tau),
                    "speed": random.uniform(0.5, 1.3),
                    "vel": Vec3(random.uniform(-0.55, 0.55), random.uniform(-0.55, 0.55), 0),
                    "heal": random.choice([12, 15, 18, 20]),
                    "room_idx": room_idx,
                }
            )

    def _update_health_powerups(self, dt: float) -> None:
        if not self.health_powerups or not hasattr(self, "ball_np"):
            return
        ball_pos = self.ball_np.getPos()
        for item in self.health_powerups:
            root = item["root"]
            if root is None or root.isEmpty():
                continue

            pos = root.getPos()
            pos += item["vel"] * dt

            room_idx = int(item.get("room_idx", 0))
            room_idx = max(0, min(len(self.rooms) - 1, room_idx))
            room = self.rooms[room_idx]
            x_min = room.x + 0.85
            x_max = room.x + room.w - 0.85
            y_min = room.y + 0.85
            y_max = room.y + room.h - 0.85

            if pos.x < x_min or pos.x > x_max:
                item["vel"].x = -item["vel"].x
            if pos.y < y_min or pos.y > y_max:
                item["vel"].y = -item["vel"].y
            pos.x = max(x_min, min(x_max, pos.x))
            pos.y = max(y_min, min(y_max, pos.y))
            pos.z = item["base_z"] + 0.42 * math.sin(self.roll_time * 5.4 + item["phase"])
            can_collect = self.player_hp < self.player_hp_max - 0.01
            pos = self._apply_pickup_attraction(pos, ball_pos, dt, active=can_collect)
            root.setPos(pos)
            root.setHpr(self.roll_time * 180.0 * item["speed"], 0, 0)

            if (ball_pos - pos).length() < 0.85 and can_collect:
                heal = float(item["heal"])
                self.player_hp = min(self.player_hp_max, self.player_hp + heal)
                self._update_player_health_ui()
                self._spawn_floating_text(ball_pos + Vec3(0, 0, 0.55), f"HP +{int(round(heal))}", (0.35, 1.0, 0.35, 1.0), scale=0.28, life=0.9)
                self._play_sound(self.sfx_pickup, volume=0.72, play_rate=1.08)

                room_idx = random.randrange(len(self.rooms))
                room = self.rooms[room_idx]
                room_base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
                item["base_z"] = room_base_z + 1.25
                item["room_idx"] = room_idx
                root.setPos(
                    random.uniform(room.x + 0.9, room.x + room.w - 0.9),
                    random.uniform(room.y + 0.9, room.y + room.h - 0.9),
                    item["base_z"],
                )

    def _apply_pickup_attraction(self, pickup_pos: Vec3, ball_pos: Vec3, dt: float, active: bool = True) -> Vec3:
        if not active:
            return pickup_pos
        to_player = ball_pos - pickup_pos
        dist = to_player.length()
        radius = max(0.1, float(getattr(self, "pickup_attract_radius", 3.1)))
        if dist <= 1e-5 or dist > radius:
            return pickup_pos
        proximity = 1.0 - (dist / radius)
        speed = max(0.0, float(getattr(self, "pickup_attract_speed", 8.2)))
        step = min(dist, speed * dt * (0.45 + 0.55 * proximity))
        if step <= 0.0:
            return pickup_pos
        return pickup_pos + to_player * (step / dist)

    def _spawn_sword_powerup(self, world_pos: Vec3) -> None:
        self.sword_pickup_serial += 1
        root = self.world.attachNewNode(f"sword-upgrade-{self.sword_pickup_serial}")
        root.setPos(world_pos)

        pickup_catalog = [
            ("attack", 1.0, (1.0, 0.45, 0.32, 0.98), (1.0, 0.36, 0.26, 0.34)),
            ("defense", 0.95, (0.42, 0.9, 1.0, 0.98), (0.3, 0.86, 1.0, 0.34)),
            ("dex", 0.9, (1.0, 0.93, 0.36, 0.98), (1.0, 0.84, 0.22, 0.34)),
            ("sta", 0.95, (0.44, 1.0, 0.45, 0.98), (0.32, 0.98, 0.36, 0.34)),
            ("int", 0.88, (0.78, 0.56, 1.0, 0.98), (0.74, 0.42, 1.0, 0.34)),
            ("skill_haste", 0.56, (1.0, 0.84, 0.26, 0.98), (1.0, 0.78, 0.24, 0.34)),
            ("skill_longblade", 0.52, (0.32, 0.96, 1.0, 0.98), (0.24, 0.82, 1.0, 0.34)),
            ("skill_fury", 0.48, (1.0, 0.42, 0.66, 0.98), (0.94, 0.3, 0.58, 0.34)),
            ("crit_core", 0.45, (1.0, 0.54, 0.18, 0.98), (1.0, 0.38, 0.12, 0.36)),
            ("skill_critical", 0.42, (1.0, 0.3, 0.24, 0.98), (1.0, 0.22, 0.18, 0.36)),
        ]
        weights = [entry[1] for entry in pickup_catalog]
        pickup_type, _, blade_col, glow_col = random.choices(pickup_catalog, weights=weights, k=1)[0]

        blade = self.box_model.copyTo(root)
        blade.setPos(self.box_norm_offset)
        blade.setScale(self.box_norm_scale)
        blade.setScale(0.08, 0.55, 0.08)
        blade.clearTexture()
        blade.setTexture(self.level_checker_tex, 1)
        blade.setLightOff(1)
        blade.setColor(*blade_col)

        guard = self.box_model.copyTo(root)
        guard.setPos(self.box_norm_offset)
        guard.setScale(self.box_norm_scale)
        guard.setScale(0.34, 0.08, 0.08)
        guard.clearTexture()
        guard.setTexture(self.level_checker_tex, 1)
        guard.setLightOff(1)
        guard.setColor(
            min(1.0, blade_col[0] * 0.78 + 0.22),
            min(1.0, blade_col[1] * 0.78 + 0.22),
            min(1.0, blade_col[2] * 0.78 + 0.22),
            0.95,
        )

        glow = self.sphere_model.copyTo(root)
        glow.setScale(0.34)
        glow.clearTexture()
        glow.setColor(*glow_col)
        glow.setLightOff(1)
        glow.setTransparency(TransparencyAttrib.MAlpha)
        glow.setDepthWrite(False)
        glow.setDepthTest(False)
        glow.setBin("transparent", 36)

        self._register_color_cycle(blade, (0.32, 0.95, 1.0, 0.98), min_speed=0.7, max_speed=2.1)
        self._register_color_cycle(guard, (0.75, 0.92, 1.0, 0.95), min_speed=0.7, max_speed=2.1)

        self.sword_powerups.append(
            {
                "root": root,
                "base": Vec3(world_pos),
                "phase": random.uniform(0.0, math.tau),
                "speed": random.uniform(0.8, 1.5),
                "spin": random.uniform(120.0, 220.0),
                "type": pickup_type,
            }
        )

    def _update_sword_powerups(self, dt: float) -> None:
        if not self.sword_powerups or not hasattr(self, "ball_np"):
            return

        ball_pos = self.ball_np.getPos()
        keep: list[dict] = []
        for item in self.sword_powerups:
            root = item.get("root")
            if root is None or root.isEmpty():
                continue

            base = item.get("base", root.getPos())
            bob = 0.28 * math.sin(self.roll_time * item["speed"] * 3.0 + item["phase"])
            pos = base + Vec3(0, 0, bob)
            pos = self._apply_pickup_attraction(pos, ball_pos, dt, active=True)
            root.setPos(pos)
            root.setHpr(self.roll_time * item["spin"], 0, 0)

            if (ball_pos - root.getPos()).length() <= 0.9:
                self._apply_sword_pickup_effect(item, ball_pos)
                root.removeNode()
                continue

            keep.append(item)

        self.sword_powerups = keep

    def _damage_monster(self, monster: dict, damage: float) -> None:
        if monster.get("dead", False):
            return

        state = str(monster.get("state", "wandering"))
        guard_chance = 0.0
        if state == "guarding":
            guard_chance = 0.58
        elif state == "attacking":
            guard_chance = 0.2
        variant = str(monster.get("variant", "normal"))
        if variant in ("juggernaut", "vanguard"):
            guard_chance += 0.18
        guard_chance = max(0.0, min(0.9, guard_chance))
        if guard_chance > 0.0 and random.random() < guard_chance:
            root = monster.get("root")
            if root is not None and not root.isEmpty():
                self._spawn_floating_text(root.getPos() + Vec3(0, 0, 1.1), "GUARD", (0.36, 1.0, 0.85, 0.95), scale=0.22, life=0.45)
            self._set_monster_state(monster, "guarding", announce=False)
            self._play_sound(self.sfx_monster_guard, volume=0.6, play_rate=1.0 + random.uniform(-0.06, 0.04))
            return

        defense = max(0.35, float(monster.get("defense", 1.0)))
        applied = max(1.0, float(damage) / defense)
        monster["hp"] = max(0.0, monster["hp"] - applied)
        hp_ratio = monster["hp"] / max(1e-6, monster["hp_max"])
        fill = monster.get("hp_fill")
        if fill is not None and not fill.isEmpty():
            fill.setScale(hp_ratio, 1.0, 1.0)

        root = monster.get("root")
        if root is not None and not root.isEmpty():
            self._spawn_floating_text(root.getPos() + Vec3(0, 0, 1.1), f"HP -{int(round(applied))}", (0.3, 0.7, 1.0, 1.0), scale=0.24, life=0.7)

        if monster["hp"] <= 0.0:
            monster["dead"] = True
            self.monsters_slain += 1
            self._update_monster_hud_ui()
            if self.monsters_total > 0 and self.monsters_slain >= self.monsters_total:
                self._trigger_win()
            self._set_monster_state(monster, "dying", announce=True)
            if not self._play_kill_sfx():
                self._play_sound(self.sfx_monster_die, volume=0.88, play_rate=1.0)
            self._grant_kill_protection()
            hum = monster.get("hum_sfx")
            if hum:
                hum.stop()
            monster["hum_active"] = False
            root = monster.get("root")
            if root is not None and not root.isEmpty():
                if random.random() < 0.85:
                    self._spawn_sword_powerup(root.getPos() + Vec3(0, 0, 0.35))
                root.removeNode()
            return

        self._set_monster_state(monster, "hit", announce=True)
        if bool(monster.get("docile_until_attacked", False)):
            monster["awakened"] = True
        if root is not None and not root.isEmpty() and hasattr(self, "ball_np"):
            away = root.getPos() - self.ball_np.getPos()
            away.z = 0.0
            if away.lengthSquared() < 1e-8:
                away = Vec3(getattr(self, "last_move_dir", Vec3(0, 1, 0)))
                away.z = 0.0
            if away.lengthSquared() > 1e-8:
                away.normalize()
                knock_speed = min(9.5, 4.2 + applied * 0.08)
                monster["knockback_vel"] = away * knock_speed + Vec3(0, 0, 0.55)
                root.setPos(root.getPos() + away * 0.12)
                monster["prev_pos"] = Vec3(root.getPos())
        self._play_sound(self.sfx_monster_hit, volume=0.5, play_rate=1.0 + random.uniform(-0.08, 0.08))

    def _apply_attack_hits(self, is_spin: bool) -> None:
        apply_attack_hits(self, is_spin)

    def _update_weapon(self, dt: float) -> None:
        update_weapon_system(self, dt)

    def _resolve_monster_collisions(self, dt: float) -> None:
        if not self.monsters:
            return

        ball_pos = self.ball_np.getPos()
        player_w = float(getattr(self, "player_w", 0.0))
        ball_vel = self.ball_body.getLinearVelocity()
        had_contact = False
        strongest_contact = 0.0
        strongest_crit_chance: float | None = None
        knock_normal_sum = Vec3(0, 0, 0)

        for monster in self.monsters:
            if monster.get("dead", False):
                continue

            root = monster["root"]
            if root is None or root.isEmpty():
                continue

            m_pos = root.getPos()
            monster_w = float(monster.get("w", 0.0))
            delta = ball_pos - m_pos
            delta.z = 0.0
            min_dist = self.ball_radius + monster["radius"] * 0.82

            w_scale = max(0.1, float(getattr(self, "w_dimension_distance_scale", 4.0)))
            dw_scaled = (player_w - monster_w) * w_scale
            planar_sq = delta.lengthSquared()
            dist4d = math.sqrt(max(0.0, planar_sq + dw_scaled * dw_scaled))
            if dist4d >= min_dist:
                continue

            dist = math.sqrt(max(0.0, planar_sq))
            if dist < 1e-6:
                continue

            had_contact = True
            normal = delta / dist
            penetration = min_dist - dist4d

            ball_pos += normal * (penetration * 0.62)
            m_pos -= normal * (penetration * 0.38)
            root.setPos(m_pos)
            monster["velocity"] = Vec3(monster["velocity"]) - normal * min(3.2, penetration * 8.5)
            dormant_docile = bool(monster.get("docile_until_attacked", False)) and (not bool(monster.get("awakened", False)))
            if not dormant_docile:
                self._set_monster_state(monster, "attacking")
            ball_vel += normal * min(5.5, penetration * 11.0)
            knock_normal_sum += normal * max(0.05, penetration)
            if not dormant_docile:
                dmg = float(monster.get("contact_damage", 10.0))
                if dmg > strongest_contact:
                    strongest_contact = dmg
                    strongest_crit_chance = float(monster.get("critical_chance", 0.08))

        if had_contact:
            self.ball_np.setPos(ball_pos)
            self.ball_body.setLinearVelocity(ball_vel)
            if self.monster_contact_sfx_cooldown <= 0.0:
                self._play_sound(self.sfx_monster_hit, volume=0.42, play_rate=0.96 + random.uniform(-0.06, 0.06))
                self.monster_contact_sfx_cooldown = 0.1

            if self.player_damage_cooldown <= 0.0 and strongest_contact > 0.0:
                was_critical = self._apply_player_damage(strongest_contact, crit_chance_override=strongest_crit_chance)
                if knock_normal_sum.lengthSquared() > 1e-8:
                    knock_dir = Vec3(knock_normal_sum)
                    knock_dir.normalize()
                    knock_impulse = 2.8 + strongest_contact * 0.22
                    if was_critical:
                        knock_impulse *= max(1.0, float(getattr(self, "critical_knockback_multiplier", 2.75)))
                        knock_dir += Vec3(0.0, 0.0, 0.25)
                        if knock_dir.lengthSquared() > 1e-8:
                            knock_dir.normalize()
                    self.ball_body.applyCentralImpulse(knock_dir * knock_impulse)
                self.player_damage_cooldown = 0.45

    def _update_monster_health_bars(self) -> None:
        for monster in self.monsters:
            if monster.get("dead", False):
                continue
            fill = monster.get("hp_fill")
            if fill is None or fill.isEmpty():
                continue
            hp_ratio = monster["hp"] / max(1e-6, monster["hp_max"])
            fill.setScale(hp_ratio, 1.0, 1.0)
            fill.setColor(1.0 - hp_ratio * 0.2, 0.28 + hp_ratio * 0.72, 0.22, 1.0)

    def _setup_ball_outline(self) -> None:
        if not hasattr(self, "ball_np"):
            return
        outline = self.sphere_model.copyTo(self.ball_np)
        outline.setScale(self.ball_radius * 1.13)
        outline.setColor(0.01, 0.01, 0.015, 1.0)
        outline.clearTexture()
        outline.setLightOff(1)
        outline.setTwoSided(False)
        outline.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
        outline.setBin("fixed", 0)
        outline.setDepthWrite(True)
        self.ball_outline_np = outline

    def _setup_ball_shadow(self) -> None:
        shadow = self._instance_quad(self.render, "ball-shadow-card", (-1.0, 1.0, -1.0, 1.0))
        shadow.setP(-90)
        shadow.setColor(1, 1, 1, 1)
        shadow.setTexture(self._create_shadow_texture(), 1)
        shadow.setTransparency(TransparencyAttrib.MAlpha)
        shadow.setDepthWrite(False)
        shadow.setDepthTest(True)
        shadow.setBin("transparent", 10)
        shadow.hide()
        self.ball_shadow_np = shadow

    def _update_ball_shadow(self) -> None:
        if self.ball_shadow_np is None:
            return
        ball_pos = self.ball_np.getPos(self.render)
        ray_from = ball_pos + Vec3(0, 0, 0.08)
        ray_to = ball_pos - Vec3(0, 0, 8.0)
        hit = self.physics_world.rayTestClosest(ray_from, ray_to)

        if hit.hasHit() and hit.getNode() == self.ball_body:
            ray_from = ball_pos - Vec3(0, 0, self.ball_radius + 0.03)
            ray_to = ray_from - Vec3(0, 0, 7.7)
            hit = self.physics_world.rayTestClosest(ray_from, ray_to)

        if not hit.hasHit():
            self.ball_shadow_np.hide()
            return

        ground = hit.getHitPos()
        height = max(0.0, ball_pos.z - ground.z - self.ball_radius)
        if height > 6.5:
            self.ball_shadow_np.hide()
            return

        h_norm = min(1.0, height / 3.2)
        base_alpha = 0.78 * (1.0 - h_norm) ** 1.35
        cell_alpha = math.floor(base_alpha * 4.0) / 4.0
        shadow_alpha = max(0.0, min(0.9, cell_alpha))
        shadow_size = (self.ball_radius * 2.35) * (1.0 - 0.52 * h_norm)
        shadow_size = max(self.ball_radius * 0.52, shadow_size)

        self.ball_shadow_np.show()
        self.ball_shadow_np.setPos(ground.x, ground.y, ground.z + 0.012)
        self.ball_shadow_np.setScale(shadow_size, shadow_size, shadow_size)
        self.ball_shadow_np.setAlphaScale(shadow_alpha)

    def _setup_orbit_lights(self) -> None:
        self.orbit_lights: list[dict] = []
        palette = [
            ((0.2, 0.95, 1.0, 1), 1.35, 0.24, 0.0, 4.8),
            ((0.95, 0.45, 1.0, 1), 1.05, 0.3, 2.1, 4.1),
            ((1.0, 0.78, 0.25, 1), 1.2, 0.18, 4.2, 5.2),
        ]
        if self.performance_mode:
            palette = palette[:1]

        for idx, (color, radius, z_offset, phase, speed) in enumerate(palette):
            point = PointLight(f"orbit-light-{idx}")
            point.setColor(color)
            if idx == 0 and self.enable_dynamic_shadows:
                point.setShadowCaster(True, 384, 384)
            point.setAttenuation((1.0, 0.045, 0.01))
            light_np = self.render.attachNewNode(point)
            self.render.setLight(light_np)

            if not self.performance_mode:
                marker = self.sphere_model.copyTo(light_np)
                marker.setScale(0.09)
                marker.setColor(color)
                marker.setTextureOff(1)
                self._register_color_cycle(marker, color, min_speed=0.08, max_speed=0.16)

            self.orbit_lights.append(
                {
                    "np": light_np,
                    "radius": radius,
                    "z": z_offset,
                    "phase": phase,
                    "speed": speed,
                }
            )

    def _update_orbit_lights(self, t: float) -> None:
        if not hasattr(self, "ball_np"):
            return
        center = self.ball_np.getPos()
        for entry in self.orbit_lights:
            angle = t * entry["speed"] + entry["phase"]
            x = center.x + math.cos(angle) * entry["radius"]
            y = center.y + math.sin(angle) * entry["radius"]
            z = center.z + entry["z"] + 0.25 * math.sin(angle * 1.6)
            entry["np"].setPos(x, y, z)

    def _add_static_box_collider(self, pos: Vec3, scale: Vec3, hpr: Vec3 | None = None, visual_holder: NodePath | None = None) -> NodePath:
        shape = BulletBoxShape(scale)
        body = BulletRigidBodyNode("static-box")
        body.setMass(0.0)
        body.setFriction(1.35)
        body.setRestitution(0.02)
        if hasattr(body, "setIntoCollideMask"):
            body.setIntoCollideMask(self.group_level)
        if hasattr(body, "setFromCollideMask"):
            body.setFromCollideMask(self.mask_level_hits)
        body.addShape(shape)
        body_np = self.render.attachNewNode(body)
        body_np.setCollideMask(BitMask32.allOn())
        body_np.setPos(pos)
        if hpr is not None:
            body_np.setHpr(hpr)
        self.physics_world.attachRigidBody(body)
        self.physics_nodes.append(body)
        if visual_holder is not None:
            self.collider_visual_map[id(body)] = visual_holder
        return body_np

    def _register_vertical_mover(self, visual: NodePath, body_np: NodePath | None, group: str) -> None:
        if visual is None or visual.isEmpty():
            return
        g = (group or "").strip().lower()
        if g not in ("room", "platform"):
            return

        if g == "platform":
            amp = random.uniform(self.platform_bob_amp * 0.72, self.platform_bob_amp * 1.24)
            speed = random.uniform(self.platform_bob_speed * 0.75, self.platform_bob_speed * 1.25)
        else:
            amp = random.uniform(self.room_bob_amp * 0.75, self.room_bob_amp * 1.2)
            speed = random.uniform(self.room_bob_speed * 0.78, self.room_bob_speed * 1.22)

        self.vertical_movers.append(
            {
                "group": g,
                "visual": visual,
                "body_np": body_np,
                "base": Vec3(visual.getPos(self.render)),
                "phase": random.uniform(0.0, math.tau),
                "amp": amp,
                "speed": speed,
            }
        )

    def _update_vertical_movers(self, dt: float) -> None:
        if not self.vertical_movers:
            return

        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        ball_pos = self.ball_np.getPos() if hasattr(self, "ball_np") else None
        max_dist_sq = float(self.vertical_mover_max_distance) * float(self.vertical_mover_max_distance)

        keep: list[dict] = []
        for mover in self.vertical_movers:
            visual = mover.get("visual")
            if visual is None or visual.isEmpty():
                continue

            group = str(mover.get("group", ""))

            base = mover["base"]
            if ball_pos is not None and group == "room" and (base - ball_pos).lengthSquared() > max_dist_sq:
                keep.append(mover)
                continue

            phase = mover["phase"]
            amp = mover["amp"]
            speed = mover["speed"]
            offset = math.sin(self.roll_time * speed + phase) * amp
            target = base + up * offset
            visual.setPos(self.render, target)

            body_np = mover.get("body_np")
            should_move_collider = group != "room" or self.move_room_colliders
            if should_move_collider and body_np is not None and not body_np.isEmpty():
                body_np.setPos(target)

            keep.append(mover)

        self.vertical_movers = keep

    def _set_key(self, key: str, state: bool) -> None:
        self.keys[key] = state

    def _set_camera_key(self, key: str, state: bool) -> None:
        self.camera_keys[key] = state

    def _set_hyper_key(self, key: str, state: bool) -> None:
        self.hyper_keys[key] = state

    def _set_hyperspace_gravity_hold(self, state: bool) -> None:
        self.hyperspace_gravity_hold = state

    def _teardown_video_distortion(self) -> None:
        overlay = getattr(self, "video_distort_overlay_np", None)
        if overlay is not None and not overlay.isEmpty():
            overlay.removeNode()
        manager = getattr(self, "video_distort_post_manager", None)
        try:
            cleanup = getattr(manager, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception:
            pass
        self.video_distort_buffer = None
        self.video_distort_post_manager = None
        self.video_distort_overlay_np = None
        self.video_distort_ready = False
        self.video_distort_shader = None

    def _set_dynamic_shadows_enabled(self, enabled: bool) -> None:
        try:
            self.sun_light.setShadowCaster(bool(enabled), 2048, 2048)
        except Exception:
            pass
        try:
            self.up_lamp.setShadowCaster(bool(enabled), 1024, 1024)
        except Exception:
            pass

    def _toggle_performance_mode(self) -> None:
        self.performance_mode = not self.performance_mode
        self.enable_dynamic_shadows = not self.performance_mode
        self.enable_ball_shadow = not self.performance_mode
        self.enable_occlusion_outlines = not self.performance_mode
        self.physics_substeps = 2 if self.performance_mode else 4
        self.physics_fixed_timestep = (1 / 120.0) if self.performance_mode else (1 / 180.0)

        self.scene_cull_interval = 0.12 if self.performance_mode else 0.16
        self.light_update_interval = 1.0 / 20.0 if self.performance_mode else 1.0 / 48.0
        self.monster_collision_update_interval = 1.0 / 24.0 if self.performance_mode else 1.0 / 48.0
        self.hyper_uv_update_interval = 1.0 / 24.0 if self.performance_mode else 1.0 / 48.0

        self.light_update_timer = 0.0
        self.monster_collision_update_timer = 0.0

        self._set_dynamic_shadows_enabled(self.enable_dynamic_shadows)
        self._ensure_performance_player_light()

        if getattr(self, "enable_video_distortion", True):
            if self.video_distort_overlay_np is None or self.video_distort_overlay_np.isEmpty():
                self._setup_video_distortion()
        else:
            self._teardown_video_distortion()

        if self.enable_ball_shadow:
            if self.ball_shadow_np is None or self.ball_shadow_np.isEmpty():
                self._setup_ball_shadow()
        elif self.ball_shadow_np is not None and not self.ball_shadow_np.isEmpty():
            self.ball_shadow_np.hide()

        if not self.enable_occlusion_outlines:
            for outline in self.visual_outline_nodes.values():
                if outline is not None and not outline.isEmpty():
                    outline.hide()

        mode = "PERFORMANCE" if self.performance_mode else "QUALITY"
        print(f"[perf] Mode switched to {mode} (key 1)")

    def _setup_mouse_look(self) -> None:
        if not self.enable_mouse_look or self.win is None:
            return
        props = WindowProperties()
        props.setCursorHidden(True)
        self.win.requestProperties(props)
        px = self.win.getXSize() // 2
        py = self.win.getYSize() // 2
        self.win.movePointer(0, px, py)
        self._mouse_centered = True

    def _consume_mouse_look(self) -> tuple[float, float]:
        if not self.enable_mouse_look or self.win is None:
            return 0.0, 0.0
        if not self.mouseWatcherNode.hasMouse():
            self._mouse_centered = False
            return 0.0, 0.0

        px = self.win.getXSize() // 2
        py = self.win.getYSize() // 2
        pointer = self.win.getPointer(0)
        dx = float(pointer.getX() - px)
        dy = float(pointer.getY() - py)

        self.win.movePointer(0, px, py)
        self._mouse_centered = True

        turn_raw = -dx * self.mouse_look_sensitivity_x
        pitch_sign = -1.0 if self.mouse_look_invert_y else 1.0
        pitch_raw = -dy * self.mouse_look_sensitivity_y * pitch_sign

        smooth = max(0.0, min(0.95, self.mouse_look_smooth))
        keep = 1.0 - smooth
        self._mouse_turn_input = self._mouse_turn_input * smooth + turn_raw * keep
        self._mouse_pitch_input = self._mouse_pitch_input * smooth + pitch_raw * keep
        return self._mouse_turn_input, self._mouse_pitch_input

    def _compute_level_w(self, pos: Vec3) -> float:
        return compute_level_w_scalar(pos.x, pos.y, pos.z, self.corridor_w, self.level_z_step)

    def _add_box(
        self,
        pos: Vec3,
        scale: Vec3,
        color=(0.8, 0.8, 0.8, 1),
        hpr: Vec3 | None = None,
        parent: NodePath | None = None,
        collidable: bool = True,
        w_coord: float | None = None,
        surface_mode: str | None = None,
        motion_group: str | None = None,
    ) -> NodePath:
        target = parent if parent is not None else self.world
        holder = target.attachNewNode("box-holder")
        if surface_mode:
            sm = str(surface_mode).strip().lower()
            holder.setTag("surface_mode", sm)
        holder.setPos(pos)
        holder.setScale(scale)
        if hpr is not None:
            holder.setHpr(hpr)

        node = self.box_model.instanceTo(holder)
        node.setPos(self.box_norm_offset)
        node.setScale(self.box_norm_scale)
        node.setColor(color)
        node.clearTexture()
        node.setTexture(self._get_random_room_texture(), 1)
        self._apply_smart_room_uv(node, pos, scale, surface_mode=surface_mode)

        if self.enable_occlusion_outlines:
            outline = self.box_model.instanceTo(holder)
            outline.setName("occlusion-outline")
            outline.setPos(self.box_norm_offset)
            outline.setScale(self.box_norm_scale * 1.03)
            outline.setColor(0.2, 0.95, 1.0, 0.9)
            outline.clearTexture()
            outline.setLightOff(1)
            outline.setShaderOff(1)
            outline.setDepthWrite(False)
            outline.setDepthTest(False)
            outline.setBin("fixed", 40)
            outline.setTwoSided(True)
            outline.setRenderModeWireframe()
            outline.setRenderModeThickness(2.2)
            outline.setTransparency(TransparencyAttrib.MAlpha)
            outline.hide()

        scale_hint = max(1.0, abs(scale.x) + abs(scale.y) + abs(scale.z))
        target_w = self._compute_level_w(pos) if w_coord is None else w_coord
        self._apply_hypercube_projection(node, target_w, scale_hint=scale_hint)
        if not surface_mode or str(surface_mode).strip().lower() != "water":
            self._register_color_cycle(node, color, min_speed=0.045, max_speed=0.11)
        self._register_scene_visual(holder, target_w)

        body_np = None
        if collidable:
            body_np = self._add_static_box_collider(pos, scale, hpr, holder)

        if collidable and parent is None and motion_group is not None:
            self._register_vertical_mover(holder, body_np, motion_group)
        return holder

    def _register_scene_visual(self, visual: NodePath, w_coord: float = 0.0) -> None:
        if visual is None or visual.isEmpty():
            return
        vid = id(visual)
        self.scene_visuals[vid] = visual
        if not visual.isEmpty():
            outline = visual.find("**/occlusion-outline")
            if not outline.isEmpty():
                self.visual_outline_nodes[vid] = outline
        self.visual_w_map[vid] = w_coord
        self.visual_alpha_targets[vid] = 1.0
        self.visual_alpha_state[vid] = 1.0

    def _clear_inverted_level_echo(self) -> None:
        root = getattr(self, "inverted_level_echo_root", None)
        if root is not None and not root.isEmpty():
            root.removeNode()
        self.inverted_level_echo_root = None

    def _setup_inverted_level_echo(self) -> None:
        self._clear_inverted_level_echo()
        if not bool(getattr(self, "enable_inverted_level_echo", True)):
            return
        if not self.scene_visuals:
            return

        plane_z = float(getattr(self, "inverted_level_echo_plane_z", self.floor_y + 12.0))
        extra_offset = float(getattr(self, "inverted_level_echo_extra_offset", 0.0))
        opacity = self._clamp(float(getattr(self, "inverted_level_echo_opacity", 0.46)), 0.08, 1.0)

        root = self.world.attachNewNode("inverted-level-echo")
        root.setPos(0.0, 0.0, 2.0 * plane_z + extra_offset)
        root.setScale(1.0, 1.0, -1.0)
        root.setTransparency(TransparencyAttrib.MAlpha)
        root.setAlphaScale(opacity)
        root.setShaderOff(1)

        for visual in list(self.scene_visuals.values()):
            if visual is None or visual.isEmpty():
                continue
            try:
                visual.instanceTo(root)
            except Exception:
                continue

        self.inverted_level_echo_root = root

    def _set_visual_occluded(self, visual_id: int, occluded: bool) -> None:
        if occluded:
            self.occluded_visuals.add(visual_id)
        else:
            self.occluded_visuals.discard(visual_id)

    def _update_camera_occlusion(self, camera_pos: Vec3, target_pos: Vec3) -> None:
        for vid in list(self.scene_visuals.keys()):
            self.visual_alpha_targets[vid] = 1.0
        self.occluded_visuals.clear()

        ray_len = max(1e-4, (target_pos - camera_pos).length())
        try:
            hits = self.physics_world.rayTestAll(camera_pos, target_pos)
            if hits.hasHits():
                for i in range(hits.getNumHits()):
                    node = hits.getNode(i)
                    if node is None or node == self.ball_body:
                        continue
                    nid = id(node)
                    if nid in self.collider_visual_map:
                        visual = self.collider_visual_map.get(nid)
                        if visual is not None and not visual.isEmpty() and not visual.isStashed():
                            vid = id(visual)
                            hit_pos = hits.getHitPos(i)
                            hit_dist = (hit_pos - camera_pos).length()
                            d_norm = max(0.0, min(1.0, hit_dist / ray_len))
                            target_alpha = float(self.camera_occlusion_alpha_min) + float(self.camera_occlusion_alpha_range) * d_norm
                            prev = self.visual_alpha_targets.get(vid, 1.0)
                            self.visual_alpha_targets[vid] = min(prev, target_alpha)
                            self.occluded_visuals.add(vid)
        except Exception:
            hit = self.physics_world.rayTestClosest(camera_pos, target_pos)
            if hit.hasHit():
                node = hit.getNode()
                if node is not None and node != self.ball_body:
                    nid = id(node)
                    if nid in self.collider_visual_map:
                        visual = self.collider_visual_map.get(nid)
                        if visual is not None and not visual.isEmpty() and not visual.isStashed():
                            vid = id(visual)
                            hit_dist = (hit.getHitPos() - camera_pos).length()
                            d_norm = max(0.0, min(1.0, hit_dist / ray_len))
                            target_alpha = float(self.camera_occlusion_alpha_min) + float(self.camera_occlusion_alpha_range) * d_norm
                            self.visual_alpha_targets[vid] = min(self.visual_alpha_targets.get(vid, 1.0), target_alpha)
                            self.occluded_visuals.add(vid)

    def _update_ripple_transparency(self, dt: float, grounded_contact: Vec3 | None) -> None:
        if not self.enable_ripple_effect:
            self.ripple_events.clear()
            self.ripple_emit_timer = 0.0
            return
        self.ripple_emit_timer += dt

        if grounded_contact is not None and self.ripple_emit_timer >= self.ripple_emit_interval:
            self.ripple_events.append((Vec3(grounded_contact), self.roll_time))
            self.ripple_emit_timer = 0.0

        self.ripple_events = [
            (origin, t0)
            for origin, t0 in self.ripple_events
            if (self.roll_time - t0) <= self.ripple_max_age
        ]

    def _ripple_alpha_for_point(self, p: Vec3) -> float:
        if not self.enable_ripple_effect:
            return 1.0
        alpha = 1.0
        for origin, t0 in self.ripple_events:
            age = self.roll_time - t0
            if age < 0.0 or age > self.ripple_max_age:
                continue
            radius = self.ripple_speed * age
            dist = math.dist((p.x, p.y), (origin.x, origin.y))
            delta = abs(dist - radius)
            if delta > self.ripple_width:
                continue
            band = 1.0 - (delta / self.ripple_width)
            fade = 1.0 - (age / self.ripple_max_age)
            local_alpha = 1.0 - self.ripple_alpha_strength * band * fade
            alpha = min(alpha, local_alpha)
        return max(0.12, min(1.0, alpha))

    def _refresh_visual_transparency(self) -> None:
        for vid, visual in list(self.scene_visuals.items()):
            if visual is None or visual.isEmpty():
                self.scene_visuals.pop(vid, None)
                self.visual_w_map.pop(vid, None)
                self.visual_alpha_targets.pop(vid, None)
                self.visual_alpha_state.pop(vid, None)
                outline = self.visual_outline_nodes.pop(vid, None)
                if outline is not None and not outline.isEmpty():
                    outline.removeNode()
                continue

            if visual.isStashed():
                outline = self.visual_outline_nodes.get(vid)
                if outline is not None and not outline.isEmpty():
                    outline.hide()
                continue

            target_alpha = self.visual_alpha_targets.get(vid, 1.0)
            prev_alpha = self.visual_alpha_state.get(vid, 1.0)
            alpha = prev_alpha + (target_alpha - prev_alpha) * float(self.camera_occlusion_alpha_smooth)
            self.visual_alpha_state[vid] = alpha

            if alpha < 0.995:
                visual.setTransparency(TransparencyAttrib.MAlpha)
                visual.setAlphaScale(alpha)
            else:
                visual.setAlphaScale(1.0)
                visual.clearTransparency()

            outline = self.visual_outline_nodes.get(vid)
            if outline is not None and not outline.isEmpty():
                if not self.enable_occlusion_outlines:
                    outline.hide()
                elif vid in self.occluded_visuals and alpha < 0.99:
                    outline_alpha = max(0.22, min(0.9, 1.0 - alpha + 0.28))
                    outline.setAlphaScale(outline_alpha)
                    outline.show()
                else:
                    outline.hide()

    def _update_scene_culling(self, hyperspace_active: bool) -> None:
        if self.camera is None:
            return

        cam_pos = self.camera.getPos(self.render)
        cam_forward = self.camera.getQuat(self.render).getForward()
        cam_forward = Vec3(cam_forward.x, cam_forward.y, cam_forward.z)
        if cam_forward.lengthSquared() > 1e-8:
            cam_forward.normalize()

        ball_z = self.ball_np.getZ() if hasattr(self, "ball_np") else 0.0
        speed = 0.0
        if hasattr(self, "ball_body"):
            v = self.ball_body.getLinearVelocity()
            speed = Vec3(v.x, v.y, 0).length()

        max_dist = (132.0 if hyperspace_active else 112.0) + min(34.0, speed * 7.0)
        max_dist_sq = max_dist * max_dist
        base_w_allow = self.hyper_slice + self.hyper_falloff * (1.0 if hyperspace_active else 0.58)
        w_motion_bonus = speed * (0.42 if hyperspace_active else 0.2)
        player_level = self._estimate_room_level_for_z(self.ball_np.getZ()) if hasattr(self, "ball_np") else 0
        level_allow = 2 if hyperspace_active else 1
        hide_miss_threshold = 3 if hyperspace_active else 2
        lens = self.camLens if hasattr(self, "camLens") else None
        use_wall_occlusion = bool(getattr(self, "enable_wall_occlusion_culling", True))
        occlusion_min_dist_sq = float(getattr(self, "scene_cull_occlusion_min_dist", 8.0)) ** 2
        ray_budget = int(max(0, getattr(self, "scene_cull_ray_budget", 220)))
        rays_used = 0
        behind_only = bool(getattr(self, "cull_behind_camera_only", True))
        front_sign = 1.0
        if hasattr(self, "ball_np"):
            ball_rel = self.camera.getRelativePoint(self.render, self.ball_np.getPos(self.render))
            if ball_rel.y < 0.0:
                front_sign = -1.0

        for vid, visual in list(self.scene_visuals.items()):
            if visual is None or visual.isEmpty():
                self.scene_visuals.pop(vid, None)
                self.visual_w_map.pop(vid, None)
                self.scene_cull_hidden.discard(vid)
                self.scene_cull_miss_counts.pop(vid, None)
                continue

            surface_mode = visual.getTag("surface_mode") if not visual.isEmpty() else ""
            if surface_mode in ("wall", "water"):
                self.scene_cull_miss_counts[vid] = 0
                if vid in self.scene_cull_hidden:
                    visual.unstash()
                    self.scene_cull_hidden.discard(vid)
                continue

            hide_candidate = False
            vis_pos = visual.getPos(self.render)
            to_vis = vis_pos - cam_pos
            dist_sq = to_vis.lengthSquared()
            rel = self.camera.getRelativePoint(self.render, vis_pos)
            if (not behind_only) and dist_sq > max_dist_sq:
                hide_candidate = True
            elif dist_sq > 4.0:
                dist = math.sqrt(dist_sq)
                rel_y = float(rel.y) * front_sign
                behind_camera = rel_y < -0.15
                if behind_camera and dist > 10.0:
                    hide_candidate = True

            in_frustum = False
            if lens is not None:
                rel_y = float(rel.y) * front_sign
                if rel_y > 0.0:
                    ndc = Point2()
                    if lens.project(rel, ndc):
                        if abs(ndc.x) <= 1.22 and abs(ndc.y) <= 1.22:
                            in_frustum = True

            if not behind_only:
                w_coord = self.visual_w_map.get(vid, 0.0)
                w_delta = abs(w_coord - self.player_w)
                height_bias = max(0.0, vis_pos.z - (ball_z + self.camera_height_offset))
                w_allow = base_w_allow + w_motion_bonus + (height_bias / max(0.35, self.hyper_height_taper_range)) * self.hyper_height_taper_strength
                if w_delta > w_allow:
                    hide_candidate = True

                vis_level = self._estimate_room_level_for_z(vis_pos.z)
                if abs(vis_level - player_level) > level_allow:
                    hide_candidate = True

            if in_frustum:
                hide_candidate = False

            occluded_by_wall = False
            if (
                (not behind_only)
                and
                use_wall_occlusion
                and in_frustum
                and not hide_candidate
                and dist_sq > occlusion_min_dist_sq
                and rays_used < ray_budget
                and hasattr(self, "physics_world")
            ):
                rays_used += 1
                try:
                    hit = self.physics_world.rayTestClosest(cam_pos, vis_pos)
                    if hit.hasHit():
                        node = hit.getNode()
                        if node is not None and node != self.ball_body:
                            nid = id(node)
                            hit_visual = self.collider_visual_map.get(nid)
                            if hit_visual is None:
                                occluded_by_wall = True
                            elif id(hit_visual) != vid:
                                occluded_by_wall = True
                except Exception:
                    occluded_by_wall = False

            if occluded_by_wall:
                hide_candidate = True

            if hide_candidate:
                miss = self.scene_cull_miss_counts.get(vid, 0) + 1
                self.scene_cull_miss_counts[vid] = miss
                hide = miss >= hide_miss_threshold
            else:
                self.scene_cull_miss_counts[vid] = 0
                hide = False

            if hide:
                if vid not in self.scene_cull_hidden:
                    visual.stash()
                    self.scene_cull_hidden.add(vid)
                    self.occluded_visuals.discard(vid)
            else:
                if vid in self.scene_cull_hidden:
                    visual.unstash()
                    self.scene_cull_hidden.discard(vid)

    def _update_hyperspace_bounce(self) -> None:
        pos = self.ball_np.getPos()
        vel = self.ball_body.getLinearVelocity()
        r = self.ball_radius

        min_x = 0.55 + r
        max_x = self.map_w - 0.55 - r
        min_y = 0.55 + r
        max_y = self.map_d - 0.55 - r
        min_z = self.hyper_bounds_bottom_z + r + 0.12
        max_z = self.hyper_bounds_top_z - r - 0.12

        bounced = False
        if pos.x < min_x and vel.x < 0:
            pos.x = min_x
            vel.x = abs(vel.x) * self.hyperspace_bounce_gain
            bounced = True
        elif pos.x > max_x and vel.x > 0:
            pos.x = max_x
            vel.x = -abs(vel.x) * self.hyperspace_bounce_gain
            bounced = True

        if pos.y < min_y and vel.y < 0:
            pos.y = min_y
            vel.y = abs(vel.y) * self.hyperspace_bounce_gain
            bounced = True
        elif pos.y > max_y and vel.y > 0:
            pos.y = max_y
            vel.y = -abs(vel.y) * self.hyperspace_bounce_gain
            bounced = True

        if pos.z < min_z and vel.z < 0:
            pos.z = min_z
            vel.z = abs(vel.z) * self.hyperspace_bounce_gain
            bounced = True
        elif pos.z > max_z and vel.z > 0:
            pos.z = max_z
            vel.z = -abs(vel.z) * self.hyperspace_bounce_gain
            bounced = True

        if bounced:
            self.ball_np.setPos(pos)
            self.ball_body.setLinearVelocity(vel)

    def _extract_contact_point_normal(self, contact) -> tuple[Vec3 | None, Vec3 | None]:
        point_a = None
        point_b = None
        normal = None
        try:
            manifold = contact.getManifoldPoint()
            point_a = manifold.getPositionWorldOnA()
            point_b = manifold.getPositionWorldOnB()
            normal = manifold.getNormalWorldOnB()
        except Exception:
            return None, None

        point = None
        if point_b is not None:
            point = Vec3(point_b)
        elif point_a is not None:
            point = Vec3(point_a)

        if normal is None:
            return point, None

        normal_vec = Vec3(normal)
        if normal_vec.lengthSquared() < 1e-10:
            return point, None
        normal_vec.normalize()

        if hasattr(self, "ball_np") and point is not None:
            to_ball = self.ball_np.getPos() - point
            if to_ball.lengthSquared() > 1e-10 and normal_vec.dot(to_ball) < 0.0:
                normal_vec = -normal_vec

        return point, normal_vec

    def _analyze_ball_contacts(self, contact_result=None) -> dict:
        if contact_result is None:
            contact_result = self.physics_world.contactTest(self.ball_body)

        analysis = {
            "grounded": False,
            "ceiling": False,
            "wall": False,
            "ground_point": None,
            "best_ground_dot": -1.0,
        }
        if contact_result.getNumContacts() <= 0:
            return analysis

        gravity_up = self._get_gravity_up()
        if gravity_up.lengthSquared() < 1e-8:
            gravity_up = Vec3(0, 0, 1)
        else:
            gravity_up.normalize()

        for contact in contact_result.getContacts():
            point, normal = self._extract_contact_point_normal(contact)
            if normal is None:
                continue

            up_dot = float(normal.dot(gravity_up))
            if up_dot >= 0.38:
                analysis["grounded"] = True
                if up_dot > analysis["best_ground_dot"]:
                    analysis["best_ground_dot"] = up_dot
                    analysis["ground_point"] = point
            elif up_dot <= -0.38:
                analysis["ceiling"] = True
            else:
                analysis["wall"] = True

        return analysis

    def _resolve_folded_world_collision(self, compression_factor: float, hyperspace_active: bool) -> None:
        if not hasattr(self, "ball_np"):
            return

        pos = self.ball_np.getPos()
        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        fold_strength = max(0.0, 1.0 - float(compression_factor))
        if hyperspace_active:
            fold_strength = max(fold_strength, 0.18)

        skin = 0.04 + fold_strength * 0.2
        probe = 1.0 + fold_strength * 1.8
        min_clear = self.ball_radius + skin

        corrected = Vec3(pos)

        ray_down_from = corrected + up * (self.ball_radius + 0.18)
        ray_down_to = corrected - up * (self.ball_radius + probe)
        down_hit = self.physics_world.rayTestClosest(ray_down_from, ray_down_to)
        if down_hit.hasHit() and down_hit.getNode() is not None and down_hit.getNode() != self.ball_body:
            floor_point = Vec3(down_hit.getHitPos())
            floor_gap = (corrected - floor_point).dot(up)
            if floor_gap < min_clear:
                corrected += up * (min_clear - floor_gap)

        ray_up_from = corrected - up * (self.ball_radius + 0.12)
        ray_up_to = corrected + up * (self.ball_radius + probe)
        up_hit = self.physics_world.rayTestClosest(ray_up_from, ray_up_to)
        if up_hit.hasHit() and up_hit.getNode() is not None and up_hit.getNode() != self.ball_body:
            ceil_point = Vec3(up_hit.getHitPos())
            ceil_gap = (ceil_point - corrected).dot(up)
            if ceil_gap < min_clear:
                corrected -= up * (min_clear - ceil_gap)

        lateral_ref = Vec3(self.ball_body.getLinearVelocity())
        lateral_ref -= up * lateral_ref.dot(up)
        if lateral_ref.lengthSquared() < 1e-8:
            lateral_ref = Vec3(1, 0, 0)
            if abs(lateral_ref.dot(up)) > 0.85:
                lateral_ref = Vec3(0, 1, 0)
        lateral_ref = lateral_ref - up * lateral_ref.dot(up)
        if lateral_ref.lengthSquared() < 1e-8:
            lateral_ref = Vec3(1, 0, 0)
        lateral_ref.normalize()
        side = up.cross(lateral_ref)
        if side.lengthSquared() < 1e-8:
            side = Vec3(0, 1, 0)
        else:
            side.normalize()

        for axis in (lateral_ref, -lateral_ref, side, -side):
            wall_from = corrected
            wall_to = corrected + axis * (self.ball_radius + probe)
            wall_hit = self.physics_world.rayTestClosest(wall_from, wall_to)
            if not wall_hit.hasHit() or wall_hit.getNode() is None or wall_hit.getNode() == self.ball_body:
                continue
            hit_pos = Vec3(wall_hit.getHitPos())
            hit_dist = (hit_pos - corrected).length()
            if hit_dist < min_clear:
                corrected -= axis * (min_clear - hit_dist)

        delta = corrected - pos
        if delta.lengthSquared() > 1e-9:
            self.ball_np.setPos(corrected)
            vel = self.ball_body.getLinearVelocity()
            v_up = up * vel.dot(up)
            v_lat = vel - v_up
            self.ball_body.setLinearVelocity(v_lat * 0.92 + v_up * 0.98)

    def _prevent_ball_tunneling(self, prev_pos: Vec3) -> None:
        if not hasattr(self, "ball_np") or not hasattr(self, "ball_body"):
            return

        curr_pos = self.ball_np.getPos()
        travel = curr_pos - prev_pos
        travel_dist = travel.length()
        if travel_dist <= max(0.02, self.ball_radius * 0.06):
            return

        hit = self.physics_world.rayTestClosest(prev_pos, curr_pos)
        if not hit.hasHit():
            return

        node = hit.getNode()
        if node is None or node == self.ball_body:
            return

        normal = Vec3(hit.getHitNormal())
        if normal.lengthSquared() < 1e-8:
            normal = -travel.normalized()
        else:
            normal.normalize()

        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        if abs(normal.dot(up)) > 0.72:
            return

        corrected = Vec3(hit.getHitPos()) + normal * (self.ball_radius + 0.02)
        self.ball_np.setPos(corrected)

        vel = Vec3(self.ball_body.getLinearVelocity())
        vn_mag = vel.dot(normal)
        if vn_mag < 0.0:
            vn = normal * vn_mag
            vt = vel - vn
            self.ball_body.setLinearVelocity(vt * 0.985)

    def _enforce_ball_floor_clearance(self, compression_factor: float) -> None:
        if not hasattr(self, "ball_np"):
            return

        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        pos = self.ball_np.getPos()
        fold_strength = max(0.0, 1.0 - float(compression_factor))
        skin = 0.05 + fold_strength * 0.16
        target_clear = self.ball_radius + skin

        ray_from = pos + up * (self.ball_radius * 0.35)
        ray_to = pos - up * (self.ball_radius + 1.9)
        hit = self.physics_world.rayTestClosest(ray_from, ray_to)
        if not hit.hasHit():
            return

        node = hit.getNode()
        if node is None or node == self.ball_body:
            return

        floor_point = Vec3(hit.getHitPos())
        gap = (pos - floor_point).dot(up)
        if gap >= target_clear:
            return

        correction = up * (target_clear - gap)
        self.ball_np.setPos(pos + correction)

        vel = self.ball_body.getLinearVelocity()
        v_up_mag = vel.dot(up)
        v_up = up * max(0.0, v_up_mag)
        v_lat = vel - up * v_up_mag
        self.ball_body.setLinearVelocity(v_lat * 0.98 + v_up)

    def _suppress_wall_climb_velocity(self) -> None:
        contacts = self.physics_world.contactTest(self.ball_body)
        analysis = self._analyze_ball_contacts(contacts)
        if not analysis.get("wall", False):
            return

        vel = self.ball_body.getLinearVelocity()
        max_up_speed = 0.42 if self.zero_g_mode else 0.9
        if vel.z > max_up_speed:
            self.ball_body.setLinearVelocity(Vec3(vel.x, vel.y, max_up_speed))

        ang = self.ball_body.getAngularVelocity()
        self.ball_body.setAngularVelocity(Vec3(ang.x * 0.86, ang.y * 0.86, ang.z * 0.55))

    def _wrap_xy_position(self, pos: Vec3, margin: float = 0.0, wrap_z: bool = False) -> tuple[Vec3, Vec3]:
        wrapped = Vec3(pos)
        delta = Vec3(0, 0, 0)
        span_x = float(self.map_w)
        span_y = float(self.map_d)
        span_z = float(max(0.001, self.hyper_bounds_top_z - self.hyper_bounds_bottom_z))

        low_x = -margin
        high_x = span_x + margin
        low_y = -margin
        high_y = span_y + margin
        low_z = self.hyper_bounds_bottom_z - margin
        high_z = self.hyper_bounds_top_z + margin

        if wrapped.x < low_x:
            wrapped.x += span_x
            delta.x += span_x
        elif wrapped.x > high_x:
            wrapped.x -= span_x
            delta.x -= span_x

        if wrapped.y < low_y:
            wrapped.y += span_y
            delta.y += span_y
        elif wrapped.y > high_y:
            wrapped.y -= span_y
            delta.y -= span_y

        if wrap_z:
            if wrapped.z < low_z:
                wrapped.z += span_z
                delta.z += span_z
            elif wrapped.z > high_z:
                wrapped.z -= span_z
                delta.z -= span_z

        return wrapped, delta

    def _apply_world_wrap(self) -> None:
        wrap_margin = max(0.0, float(getattr(self, "world_wrap_margin", 0.35)))
        wrapped_pos, delta = self._wrap_xy_position(self.ball_np.getPos(), margin=wrap_margin, wrap_z=True)
        if delta.lengthSquared() < 1e-12:
            return

        self.ball_np.setPos(wrapped_pos)
        if getattr(self, "camera_parented_to_ball", False):
            self.camera_smoothed_pos = Vec3(self.camera.getPos(self.render))
            return
        if self.camera_smoothed_pos is not None:
            self.camera_smoothed_pos = self.camera_smoothed_pos + delta
        self.camera.setPos(self.camera.getPos() + delta)

    def _setup_outside_islands_and_warps(self) -> None:
        self.outside_islands.clear()
        self.warp_links.clear()
        show_markers = bool(getattr(self, "show_portal_markers", False))
        if getattr(self, "subtractive_maze_mode", False):
            self._setup_subtractive_maze_portals()
            return
        if not self.rooms:
            return

        if not self.allow_outside_island_warps:
            island_count = 0
        else:
            island_count = 4

        cx = self.map_w * 0.5
        cy = self.map_d * 0.5
        ring_radius = max(self.map_w, self.map_d) * 0.72 + 70.0

        selected_rooms: list[int] = []
        if island_count > 0:
            room_pick_step = max(1, len(self.rooms) // island_count)
            selected_rooms = [min(i * room_pick_step, len(self.rooms) - 1) for i in range(island_count)]

        for idx in range(island_count):
            ang = (math.tau * idx) / island_count
            base_pos = Vec3(cx + math.cos(ang) * ring_radius, cy + math.sin(ang) * ring_radius, self.floor_y + 4.0)
            root = self.world.attachNewNode(f"outside-island-{idx}")
            root.setPos(base_pos)

            colliders: list[dict] = []
            room_local_centers: list[Vec3] = []
            room_nodes = 4
            local_ring = 16.0
            for k in range(room_nodes):
                a = (math.tau * k) / room_nodes
                local = Vec3(math.cos(a) * local_ring, math.sin(a) * local_ring, math.sin(a * 2.0) * 0.8)
                room_local_centers.append(local)
                half = Vec3(6.0, 6.0, self.floor_t * 0.5)

                holder = self._add_box(local, half, color=(0.2, 0.26, 0.33, 1.0), parent=root, collidable=False)
                body_np = self._add_static_box_collider(base_pos + local, half, visual_holder=holder)
                colliders.append({"body_np": body_np, "local": Vec3(local)})

            for k in range(room_nodes):
                a = room_local_centers[k]
                b = room_local_centers[(k + 1) % room_nodes]
                mid = (a + b) * 0.5
                seg = b - a
                seg_len = max(0.1, seg.length())
                h = math.degrees(math.atan2(seg.y, seg.x))
                bridge_half = Vec3(seg_len * 0.5, 1.55, self.floor_t * 0.45)
                holder = self._add_box(mid, bridge_half, color=(0.18, 0.22, 0.28, 1.0), hpr=Vec3(h, 0, 0), parent=root, collidable=False)
                body_np = self._add_static_box_collider(base_pos + mid, bridge_half, hpr=Vec3(h, 0, 0), visual_holder=holder)
                colliders.append({"body_np": body_np, "local": Vec3(mid), "hpr": Vec3(h, 0, 0)})

            island = {
                "root": root,
                "base": Vec3(base_pos),
                "phase": random.uniform(0.0, math.tau),
                "speed": random.uniform(0.15, 0.32),
                "amp": random.uniform(5.0, 9.0),
                "z_amp": random.uniform(0.45, 1.1),
                "colliders": colliders,
                "warp_local": Vec3(room_local_centers[0] + Vec3(0, 0, 0.42)),
            }
            self.outside_islands.append(island)

            room_idx = selected_rooms[idx]
            room = self.rooms[room_idx]
            base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            room_warp_pos = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + 0.42)

            if show_markers:
                pad = self.sphere_model.copyTo(self.world)
                pad.setScale(0.34)
                pad.setPos(room_warp_pos)
                pad.setColor(0.95, 0.42, 1.0, 0.9)
                pad.setLightOff(1)
                pad.setTransparency(TransparencyAttrib.MAlpha)
                pad.setCollideMask(BitMask32.allOff())
                self._register_color_cycle(pad, (0.95, 0.42, 1.0, 0.9), min_speed=0.35, max_speed=0.9)

                island_pad = self.sphere_model.copyTo(root)
                island_pad.setScale(0.38)
                island_pad.setPos(island["warp_local"])
                island_pad.setColor(0.35, 0.95, 1.0, 0.92)
                island_pad.setLightOff(1)
                island_pad.setTransparency(TransparencyAttrib.MAlpha)
                island_pad.setCollideMask(BitMask32.allOff())
                self._register_color_cycle(island_pad, (0.35, 0.95, 1.0, 0.92), min_speed=0.35, max_speed=0.9)

            self.warp_links.append(
                {
                    "mode": "island",
                    "room_pos": room_warp_pos,
                    "island_idx": idx,
                    "radius": 1.05,
                    "radius_sq": 1.05 * 1.05,
                    "room_idx": room_idx,
                }
            )

        room_count = len(self.rooms)
        if room_count < 4:
            return

        pair_count = max(1, min(4, room_count // 5))
        used_rooms: set[int] = set()
        sorted_rooms = sorted(
            range(room_count),
            key=lambda i: (self.rooms[i].w * self.rooms[i].h),
            reverse=True,
        )

        for i in range(pair_count):
            a_idx = sorted_rooms[i % len(sorted_rooms)]
            b_idx = sorted_rooms[-(i + 1)]
            if a_idx == b_idx or a_idx in used_rooms or b_idx in used_rooms:
                continue

            room_a = self.rooms[a_idx]
            room_b = self.rooms[b_idx]
            a_level = self._level_base_z(self.room_levels.get(a_idx, 0))
            b_level = self._level_base_z(self.room_levels.get(b_idx, 0))
            a_pos = Vec3(room_a.x + room_a.w * 0.5, room_a.y + room_a.h * 0.5, a_level + 0.46)
            b_pos = Vec3(room_b.x + room_b.w * 0.5, room_b.y + room_b.h * 0.5, b_level + 0.46)

            if (a_pos - b_pos).length() < max(self.corridor_w * 1.5, 24.0):
                continue

            if show_markers:
                a_pad = self.sphere_model.copyTo(self.world)
                a_pad.setScale(0.42)
                a_pad.setPos(a_pos)
                a_pad.setColor(0.98, 0.38, 0.92, 0.9)
                a_pad.setLightOff(1)
                a_pad.setTransparency(TransparencyAttrib.MAlpha)
                a_pad.setCollideMask(BitMask32.allOff())
                self._register_color_cycle(a_pad, (0.98, 0.38, 0.92, 0.9), min_speed=0.36, max_speed=0.88)

                b_pad = self.sphere_model.copyTo(self.world)
                b_pad.setScale(0.42)
                b_pad.setPos(b_pos)
                b_pad.setColor(0.34, 0.95, 1.0, 0.9)
                b_pad.setLightOff(1)
                b_pad.setTransparency(TransparencyAttrib.MAlpha)
                b_pad.setCollideMask(BitMask32.allOff())
                self._register_color_cycle(b_pad, (0.34, 0.95, 1.0, 0.9), min_speed=0.36, max_speed=0.88)

            self.warp_links.append(
                {
                    "mode": "room_fold",
                    "a_pos": a_pos,
                    "b_pos": b_pos,
                    "radius": 1.2,
                    "radius_sq": 1.2 * 1.2,
                    "a_room_idx": a_idx,
                    "b_room_idx": b_idx,
                }
            )
            used_rooms.update({a_idx, b_idx})

        if self.mobius_twist_enabled:
            center_x = self.map_w * 0.5
            center_y = self.map_d * 0.5
            middle_band = self.map_d * max(0.08, min(0.45, self.mobius_middle_band_ratio))
            half_band = middle_band * 0.5

            left_candidates: list[int] = []
            right_candidates: list[int] = []
            for idx, room in enumerate(self.rooms):
                if idx in used_rooms:
                    continue
                room_cx = room.x + room.w * 0.5
                room_cy = room.y + room.h * 0.5
                if abs(room_cy - center_y) > half_band:
                    continue
                if room_cx < center_x:
                    left_candidates.append(idx)
                else:
                    right_candidates.append(idx)

            left_candidates.sort(key=lambda ridx: (abs((self.rooms[ridx].y + self.rooms[ridx].h * 0.5) - center_y), -self.rooms[ridx].w * self.rooms[ridx].h))
            right_candidates.sort(key=lambda ridx: (abs((self.rooms[ridx].y + self.rooms[ridx].h * 0.5) - center_y), -self.rooms[ridx].w * self.rooms[ridx].h))

            mobius_pairs = min(
                max(1, int(self.mobius_loop_count)),
                len(left_candidates),
                len(right_candidates),
            )
            for i in range(mobius_pairs):
                a_idx = left_candidates[i]
                b_idx = right_candidates[i]
                if a_idx == b_idx:
                    continue

                room_a = self.rooms[a_idx]
                room_b = self.rooms[b_idx]
                a_level = self._level_base_z(self.room_levels.get(a_idx, 0))
                b_level = self._level_base_z(self.room_levels.get(b_idx, 0))
                a_pos = Vec3(room_a.x + room_a.w * 0.5, room_a.y + room_a.h * 0.5, a_level + 0.46)
                b_pos = Vec3(room_b.x + room_b.w * 0.5, room_b.y + room_b.h * 0.5, b_level + 0.46)

                if (a_pos - b_pos).length() < max(self.corridor_w * 1.9, 34.0):
                    continue

                if show_markers:
                    a_pad = self.sphere_model.copyTo(self.world)
                    a_pad.setScale(0.5)
                    a_pad.setPos(a_pos)
                    a_pad.setColor(1.0, 0.76, 0.2, 0.94)
                    a_pad.setLightOff(1)
                    a_pad.setTransparency(TransparencyAttrib.MAlpha)
                    a_pad.setCollideMask(BitMask32.allOff())
                    self._register_color_cycle(a_pad, (1.0, 0.76, 0.2, 0.94), min_speed=0.42, max_speed=1.0)

                    b_pad = self.sphere_model.copyTo(self.world)
                    b_pad.setScale(0.5)
                    b_pad.setPos(b_pos)
                    b_pad.setColor(0.25, 1.0, 0.72, 0.94)
                    b_pad.setLightOff(1)
                    b_pad.setTransparency(TransparencyAttrib.MAlpha)
                    b_pad.setCollideMask(BitMask32.allOff())
                    self._register_color_cycle(b_pad, (0.25, 1.0, 0.72, 0.94), min_speed=0.42, max_speed=1.0)

                self.warp_links.append(
                    {
                        "mode": "room_fold",
                        "a_pos": a_pos,
                        "b_pos": b_pos,
                        "radius": 1.28,
                        "radius_sq": 1.28 * 1.28,
                        "a_room_idx": a_idx,
                        "b_room_idx": b_idx,
                        "mobius": True,
                        "mobius_phase": (math.tau * i) / max(1, mobius_pairs),
                    }
                )
                used_rooms.update({a_idx, b_idx})

            self._refresh_room_fold_thread_cache()

    def _setup_subtractive_maze_portals(self) -> None:
        points = list(getattr(self, "maze_portal_points", []))
        if len(points) < 2:
            return
        show_markers = bool(getattr(self, "show_portal_markers", False))

        random.shuffle(points)
        pairs = min(4, len(points) // 2)
        for i in range(pairs):
            a_pos = Vec3(points[i * 2]) + Vec3(0, 0, 0.22)
            b_pos = Vec3(points[i * 2 + 1]) + Vec3(0, 0, 0.22)
            if (a_pos - b_pos).lengthSquared() < 20.0:
                continue

            if show_markers:
                a_pad = self.sphere_model.copyTo(self.world)
                a_pad.setScale(0.34)
                a_pad.setPos(a_pos)
                a_pad.setColor(1.0, 0.42, 0.88, 0.86)
                a_pad.setLightOff(1)
                a_pad.setTransparency(TransparencyAttrib.MAlpha)
                a_pad.setCollideMask(BitMask32.allOff())
                self._register_color_cycle(a_pad, (1.0, 0.42, 0.88, 0.86), min_speed=0.28, max_speed=0.64)

                b_pad = self.sphere_model.copyTo(self.world)
                b_pad.setScale(0.34)
                b_pad.setPos(b_pos)
                b_pad.setColor(0.35, 0.95, 1.0, 0.86)
                b_pad.setLightOff(1)
                b_pad.setTransparency(TransparencyAttrib.MAlpha)
                b_pad.setCollideMask(BitMask32.allOff())
                self._register_color_cycle(b_pad, (0.35, 0.95, 1.0, 0.86), min_speed=0.28, max_speed=0.64)

            self.warp_links.append(
                {
                    "mode": "maze_portal",
                    "a_pos": a_pos,
                    "b_pos": b_pos,
                    "radius": 1.05,
                    "radius_sq": 1.05 * 1.05,
                }
            )

        self._refresh_room_fold_thread_cache()

    def _apply_room_fold_twist(self, link: dict, fold_push: Vec3, up: Vec3, vel: Vec3) -> Vec3:
        out_vel = vel * 0.85 + fold_push * 2.4
        if not bool(link.get("mobius", False)):
            return out_vel
        phase = float(link.get("mobius_phase", 0.0))
        twist = max(0.0, min(1.0, float(getattr(self, "mobius_twist_strength", 0.9))))
        out_vx, out_vy, out_vz, out_w = compute_mobius_fold_twist(
            (vel.x, vel.y, vel.z),
            (fold_push.x, fold_push.y, fold_push.z),
            (up.x, up.y, up.z),
            self.player_w,
            self.roll_time,
            phase,
            twist,
            self.hyper_w_limit,
        )
        self.player_w = self._clamp(out_w, -self.hyper_w_limit, self.hyper_w_limit)
        return Vec3(out_vx, out_vy, out_vz)

    def _start_room_fold_worker_if_needed(self) -> None:
        if not self.enable_room_fold_thread:
            return
        if self._room_fold_worker is not None and self._room_fold_worker.is_alive():
            return
        self._room_fold_stop_event.clear()
        self._room_fold_worker = threading.Thread(
            target=self._room_fold_worker_loop,
            name="room-fold-worker",
            daemon=True,
        )
        self._room_fold_worker.start()
        self.room_fold_thread_ready = True

    def _replace_queue_item(self, q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass

    def _refresh_room_fold_thread_cache(self) -> None:
        cache: list[tuple[int, float, float, float, float, float, float, float]] = []
        for idx, link in enumerate(self.warp_links):
            if link.get("mode") != "room_fold":
                continue
            a_pos = link.get("a_pos")
            b_pos = link.get("b_pos")
            if a_pos is None or b_pos is None:
                continue
            r2 = float(link.get("radius_sq", float(link.get("radius", 1.0)) ** 2))
            cache.append((idx, float(a_pos.x), float(a_pos.y), float(a_pos.z), float(b_pos.x), float(b_pos.y), float(b_pos.z), r2))

        with self._room_fold_links_lock:
            self._room_fold_links_thread_cache = cache
            self._room_fold_links_version += 1

    def _room_fold_worker_loop(self) -> None:
        while not self._room_fold_stop_event.is_set():
            try:
                req = self._room_fold_probe_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            while True:
                try:
                    req = self._room_fold_probe_queue.get_nowait()
                except queue.Empty:
                    break

            req_version, bx, by, bz = req
            with self._room_fold_links_lock:
                version = self._room_fold_links_version
                local_cache = list(self._room_fold_links_thread_cache)

            hit_idx = -1
            hit_dir = -1
            for link_idx, ax, ay, az, cx, cy, cz, radius_sq in local_cache:
                da = (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az)
                if da <= radius_sq:
                    hit_idx = link_idx
                    hit_dir = 0
                    break
                db = (bx - cx) * (bx - cx) + (by - cy) * (by - cy) + (bz - cz) * (bz - cz)
                if db <= radius_sq:
                    hit_idx = link_idx
                    hit_dir = 1
                    break

            self._replace_queue_item(self._room_fold_result_queue, (req_version, version, hit_idx, hit_dir))

    def _apply_room_fold_warp(self, link: dict, from_a: bool, up: Vec3) -> bool:
        a_pos = link.get("a_pos")
        b_pos = link.get("b_pos")
        if a_pos is None or b_pos is None:
            return False

        a_room_idx = int(link.get("a_room_idx", -1))
        b_room_idx = int(link.get("b_room_idx", -1))
        if from_a:
            out_pos = self._safe_room_spawn_pos(b_room_idx if b_room_idx >= 0 else 0, z_lift=0.34)
            fold_push = Vec3(b_pos) - Vec3(a_pos)
        else:
            out_pos = self._safe_room_spawn_pos(a_room_idx if a_room_idx >= 0 else 0, z_lift=0.34)
            fold_push = Vec3(a_pos) - Vec3(b_pos)

        self.ball_np.setPos(out_pos)
        vel = Vec3(self.ball_body.getLinearVelocity())
        fold_push = fold_push - up * fold_push.dot(up)
        if fold_push.lengthSquared() > 1e-8:
            fold_push.normalize()
        self.ball_body.setLinearVelocity(self._apply_room_fold_twist(link, fold_push, up, vel))
        self.warp_cooldown = 0.46
        return True

    def _apply_maze_portal_warp(self, link: dict, from_a: bool, up: Vec3) -> bool:
        a_pos = link.get("a_pos")
        b_pos = link.get("b_pos")
        if a_pos is None or b_pos is None:
            return False
        out_pos = Vec3(b_pos) if from_a else Vec3(a_pos)
        out_pos += up * (self.ball_radius + 0.2)
        self.ball_np.setPos(out_pos)
        vel = Vec3(self.ball_body.getLinearVelocity())
        self.ball_body.setLinearVelocity(vel * 0.75)
        self.ball_body.setAngularVelocity(self.ball_body.getAngularVelocity() * 0.8)
        self.warp_cooldown = 0.4
        return True

    def _pick_landmark_model(self) -> str | None:
        if self.landmark_model_candidates is None:
            model_extensions = (".bam", ".egg", ".gltf", ".glb", ".obj")
            model_paths: list[str] = []
            roots = ["graphics/landmarks", "graphics"]
            for root_dir in roots:
                if not os.path.isdir(root_dir):
                    continue
                for root, _, files in os.walk(root_dir):
                    for name in files:
                        if name.lower().endswith(model_extensions):
                            path = os.path.join(root, name).replace("\\", "/")
                            if path not in model_paths:
                                model_paths.append(path)
            self.landmark_model_candidates = model_paths

        if self.landmark_model_candidates:
            return random.choice(self.landmark_model_candidates)

        return None

    def _setup_room_landmarks(self) -> None:
        self.room_landmarks.clear()
        if not self.rooms:
            return

        room_indices = sorted(
            range(len(self.rooms)),
            key=lambda idx: self.rooms[idx].w * self.rooms[idx].h,
            reverse=True,
        )
        for room_idx in room_indices:
            room = self.rooms[room_idx]
            room_area = float(room.w * room.h)
            spawn_count = 1
            if room_area >= 180.0:
                spawn_count += 1
            if not self.performance_mode and room_area >= 320.0:
                spawn_count += 1

            for landmark_idx in range(spawn_count):
                base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
                if spawn_count == 1:
                    center = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + 0.06)
                else:
                    margin = 0.16 + 0.05 * min(2, landmark_idx)
                    center = Vec3(
                        random.uniform(room.x + room.w * margin, room.x + room.w * (1.0 - margin)),
                        random.uniform(room.y + room.h * margin, room.y + room.h * (1.0 - margin)),
                        base_z + 0.06,
                    )

                model_path = self._pick_landmark_model()
                landmark_root = self.world.attachNewNode(f"landmark-{room_idx}-{landmark_idx}")
                landmark_root.setPos(center)
                landmark_root.setH(random.uniform(0.0, 360.0))
                landmark_root.setCollideMask(BitMask32.allOn())

                model_np = None
                if model_path is not None:
                    try:
                        model_np = self.loader.loadModel(model_path)
                    except Exception:
                        model_np = None

                if model_np is not None and not model_np.isEmpty():
                    model_np.reparentTo(landmark_root)
                    if bool(getattr(self, "landmark_linux_axis_fix", False)):
                        path_l = str(model_path).lower() if model_path is not None else ""
                        if path_l.endswith((".gltf", ".glb", ".obj")):
                            model_np.setP(float(getattr(self, "landmark_linux_axis_fix_degrees", -90.0)))
                    room_span = max(1.0, min(room.w, room.h))
                    model_np.setScale(max(0.55, min(1.45, room_span * 0.04)))
                    model_np.setColor(0.86, 0.92, 1.0, 1.0)
                    model_np.setTexture(self._get_random_room_texture(), 1)
                    uv_scale = Vec3(max(0.25, room_span * 0.2), max(0.25, room_span * 0.2), max(0.25, room_span * 0.2))
                    self._apply_smart_room_uv(model_np, center, uv_scale, surface_mode="floor")
                else:
                    pylon = self._add_box(
                        Vec3(0, 0, 0.62),
                        Vec3(0.32, 0.32, 0.62),
                        color=(0.7, 0.78, 0.88, 1.0),
                        parent=landmark_root,
                        collidable=False,
                        surface_mode="floor",
                    )
                    orb = self.sphere_model.copyTo(landmark_root)
                    orb.setScale(0.34)
                    orb.setPos(0, 0, 1.4)
                    orb.setColor(0.26, 0.95, 1.0, 0.95)
                    orb.setTransparency(TransparencyAttrib.MAlpha)
                    orb.setLightOff(1)
                    self._register_color_cycle(orb, (0.26, 0.95, 1.0, 0.95), min_speed=0.08, max_speed=0.2)
                    if pylon is not None and not pylon.isEmpty():
                        pylon.setCollideMask(BitMask32.allOff())

                collider_half = Vec3(0.34, 0.34, 0.62)
                collider_center_local = Vec3(0, 0, 0.62)
                bounds = landmark_root.getTightBounds()
                if bounds is not None:
                    bmin, bmax = bounds
                    if bmin is not None and bmax is not None:
                        center_local = (bmin + bmax) * 0.5
                        size = bmax - bmin
                        collider_center_local = Vec3(center_local)
                        collider_half = Vec3(
                            max(0.18, abs(size.x) * 0.5),
                            max(0.18, abs(size.y) * 0.5),
                            max(0.28, abs(size.z) * 0.5),
                        )

                landmark_quat = landmark_root.getQuat(self.render)
                collider_center_world = landmark_root.getPos(self.render) + landmark_quat.xform(collider_center_local)
                collider_np = self._add_static_box_collider(
                    collider_center_world,
                    collider_half,
                    hpr=landmark_root.getHpr(self.render),
                    visual_holder=landmark_root,
                )
                collider_np.setName(f"landmark-collider-{room_idx}-{landmark_idx}")

                w_coord = self._compute_level_w(center)
                self._apply_hypercube_projection(landmark_root, w_coord, scale_hint=max(room.w, room.h) * 0.08)
                self._register_scene_visual(landmark_root, w_coord)
                self.room_landmarks.append({"room_idx": room_idx, "root": landmark_root, "collider": collider_np})

    def _spawn_roaming_black_holes(self, count: int = 6) -> None:
        for entry in getattr(self, "black_holes", []):
            loop_sfx = entry.get("loop_sfx")
            if loop_sfx:
                try:
                    loop_sfx.stop()
                except Exception:
                    pass
            root = entry.get("root")
            if root is not None and not root.isEmpty():
                root.removeNode()
        self.black_holes = []

        if not self.rooms:
            return

        if self.black_hole_suck_outline_tex is None:
            self.black_hole_suck_outline_tex = self._create_radial_mirrored_rainbow_texture(256)
        if self.black_hole_blow_dial_tex is None:
            self.black_hole_blow_dial_tex = self._create_angular_dial_texture(256)

        spawn_count = max(1, int(count))
        blower_ratio = self._clamp(float(getattr(self, "black_hole_blower_ratio", 0.46)), 0.0, 1.0)
        for idx in range(spawn_count):
            room_idx = random.randrange(len(self.rooms))
            room = self.rooms[room_idx]
            base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            margin = 1.1
            x = random.uniform(room.x + margin, room.x + room.w - margin)
            y = random.uniform(room.y + margin, room.y + room.h - margin)

            kind = "blow" if random.random() < blower_ratio else "suck"
            root = self.world.attachNewNode(f"anomaly-{kind}-{idx}")
            root.setPos(x, y, base_z + 1.35)
            root.setTransparency(TransparencyAttrib.MAlpha)
            root.setBin("transparent", 44)
            root.setDepthWrite(False)

            core = self.sphere_model.copyTo(root)
            core.setScale(max(0.22, self.black_hole_visual_radius * 0.52))
            core.clearTexture()
            core.setLightOff(1)
            core.setTransparency(TransparencyAttrib.MAlpha)
            core.setBin("transparent", 47)
            core.setDepthWrite(False)

            lens = self.sphere_model.copyTo(root)
            lens.setScale(max(0.55, self.black_hole_visual_radius * 1.18))
            lens.clearTexture()
            lens.setTexture(self._get_random_room_texture(), 1)
            lens.setTexScale(TextureStage.getDefault(), 3.5, 3.5)
            lens.setLightOff(1)
            lens.setTransparency(TransparencyAttrib.MAlpha)
            lens.setBin("transparent", 45)
            lens.setDepthWrite(False)
            lens.setAttrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.MAdd,
                    ColorBlendAttrib.OIncomingAlpha,
                    ColorBlendAttrib.OOne,
                ),
                1,
            )

            corona = self.sphere_model.copyTo(root)
            corona.setScale(max(0.72, self.black_hole_visual_radius * 1.62))
            corona.clearTexture()
            corona.setTexture(self._get_random_room_texture(), 1)
            corona.setTexScale(TextureStage.getDefault(), 5.2, 5.2)
            corona.setLightOff(1)
            corona.setTransparency(TransparencyAttrib.MAlpha)
            corona.setBin("transparent", 46)
            corona.setDepthWrite(False)

            outline = self.sphere_model.copyTo(root)
            outline.setScale(max(0.85, self.black_hole_visual_radius * 1.95))
            outline.clearTexture()
            outline.setLightOff(1)
            outline.setTransparency(TransparencyAttrib.MAlpha)
            outline.setBin("transparent", 48)
            outline.setDepthWrite(False)
            outline.setAttrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.MAdd,
                    ColorBlendAttrib.OIncomingAlpha,
                    ColorBlendAttrib.OOne,
                ),
                1,
            )
            outline_stage = TextureStage(f"anomaly-outline-{idx}")
            outline_stage.setMode(TextureStage.MModulate)

            if kind == "blow":
                core.setColor(0.0, 0.0, 0.0, 0.97)
                lens.setColor(0.06, 0.08, 0.12, 0.26)
                corona.setColor(0.18, 0.22, 0.36, 0.16)
                outline.setTexture(outline_stage, self.black_hole_blow_dial_tex, 1)
                dial_scale = max(0.9, self.black_hole_visual_radius * 2.25)
                outline.setScale(dial_scale, dial_scale, max(0.06, dial_scale * 0.11))
            else:
                core.setColor(0.0, 0.0, 0.0, 0.98)
                lens.setColor(0.04, 0.06, 0.09, 0.28)
                corona.setColor(0.12, 0.16, 0.28, 0.15)
                outline.setTexture(outline_stage, self.black_hole_suck_outline_tex, 1)
            outline.setColor(1.0, 1.0, 1.0, 0.36)

            self.black_holes.append(
                {
                    "root": root,
                    "core": core,
                    "lens": lens,
                    "corona": corona,
                    "outline": outline,
                    "outline_stage": outline_stage,
                    "kind": kind,
                    "room_idx": room_idx,
                    "base_z": base_z + 1.35,
                    "phase": random.uniform(0.0, math.tau),
                    "spin": random.uniform(30.0, 70.0) * (-1.0 if kind == "blow" else 1.0),
                    "vel": Vec3(
                        random.uniform(-self.black_hole_roam_speed, self.black_hole_roam_speed),
                        random.uniform(-self.black_hole_roam_speed, self.black_hole_roam_speed),
                        0.0,
                    ),
                    "pull": random.uniform(self.black_hole_pull_strength * 0.8, self.black_hole_pull_strength * 1.35),
                    "radius": random.uniform(self.black_hole_influence_radius * 0.85, self.black_hole_influence_radius * 1.25),
                    "distort_gain": random.uniform(0.55, 1.2),
                    "loop_sfx": None,
                }
            )

        self._attach_black_hole_loop_sounds()

    def _update_roaming_black_holes(self, dt: float) -> None:
        if not self.black_holes:
            self.black_hole_distort_intensity = max(0.0, self.black_hole_distort_intensity - dt * 2.0)
            return

        ball_pos = self.ball_np.getPos() if hasattr(self, "ball_np") and self.ball_np is not None else None
        listener_pos = Vec3(ball_pos) if ball_pos is not None else (self.camera.getPos(self.render) if hasattr(self, "camera") else Vec3(0, 0, 0))
        listener_vel = self.ball_body.getLinearVelocity() if hasattr(self, "ball_body") and self.ball_body is not None else Vec3(0, 0, 0)
        live_monsters: list[dict] = []
        for monster in getattr(self, "monsters", []):
            if monster.get("dead", False):
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            monster["cosmic_warp_cooldown"] = max(0.0, float(monster.get("cosmic_warp_cooldown", 0.0)) - dt)
            live_monsters.append(monster)

        max_distort = 0.0
        monster_force_scale = max(0.001, float(getattr(self, "black_hole_monster_force_scale", 0.032)))
        spit_speed = max(0.5, float(getattr(self, "black_hole_monster_spit_speed", 8.2)))
        warp_cd = max(0.2, float(getattr(self, "black_hole_monster_warp_cooldown", 2.4)))

        for entry in self.black_holes:
            root = entry.get("root")
            if root is None or root.isEmpty():
                loop_sfx = entry.get("loop_sfx")
                if loop_sfx:
                    try:
                        loop_sfx.stop()
                    except Exception:
                        pass
                    entry["loop_sfx"] = None
                continue

            kind = str(entry.get("kind", "suck")).lower()
            pos = root.getPos()
            pos += Vec3(entry.get("vel", Vec3(0, 0, 0))) * dt

            room_idx = int(entry.get("room_idx", 0))
            room_idx = max(0, min(len(self.rooms) - 1, room_idx))
            room = self.rooms[room_idx]
            x_min = room.x + 1.0
            x_max = room.x + room.w - 1.0
            y_min = room.y + 1.0
            y_max = room.y + room.h - 1.0

            vel = Vec3(entry.get("vel", Vec3(0, 0, 0)))
            if pos.x < x_min or pos.x > x_max:
                vel.x = -vel.x
            if pos.y < y_min or pos.y > y_max:
                vel.y = -vel.y
            pos.x = max(x_min, min(x_max, pos.x))
            pos.y = max(y_min, min(y_max, pos.y))
            entry["vel"] = vel

            phase = float(entry.get("phase", 0.0))
            pos.z = float(entry.get("base_z", pos.z)) + 0.42 * math.sin(self.roll_time * 1.8 + phase)
            root.setPos(pos)
            root.setHpr(self.roll_time * float(entry.get("spin", 42.0)), self.roll_time * 9.0, 0.0)

            loop_sfx = entry.get("loop_sfx")
            if loop_sfx:
                to_listener = listener_pos - pos
                dist = max(1e-5, to_listener.length())
                dir_to_listener = to_listener / dist
                anomaly_vel = Vec3(entry.get("vel", Vec3(0, 0, 0)))
                rel_radial = (anomaly_vel - listener_vel).dot(dir_to_listener)
                exaggeration = float(getattr(self, "black_hole_doppler_exaggeration", 13.5))
                rate = 1.0 + (rel_radial * 0.016 * exaggeration)
                if kind == "suck":
                    rate *= 0.93 + 0.14 * (0.5 + 0.5 * math.sin(self.roll_time * 2.4 + phase))
                else:
                    rate *= 0.9 + 0.18 * (0.5 + 0.5 * math.sin(self.roll_time * 3.1 + phase))
                loop_sfx.setPlayRate(self._clamp(rate, 0.35, 3.6))
                max_dist = max(1.0, float(getattr(self, "black_hole_sfx_max_distance", 520.0)))
                prox = self._clamp(1.0 - (dist / max_dist), 0.0, 1.0)
                base_vol = float(getattr(self, "black_hole_sfx_volume", 0.9))
                shaped = 0.2 + 0.8 * (prox ** 1.6)
                loop_sfx.setVolume(self._clamp(base_vol * shaped, 0.0, 1.0))

            lens = entry.get("lens")
            corona = entry.get("corona")
            outline = entry.get("outline")
            pulse = 0.5 + 0.5 * math.sin(self.roll_time * 4.6 + phase)
            if lens is not None and not lens.isEmpty():
                lens.setTexOffset(TextureStage.getDefault(), (self.roll_time * 0.31 + phase * 0.11) % 1.0, (self.roll_time * -0.27 + phase * 0.17) % 1.0)
                lens.setScale(max(0.2, self.black_hole_visual_radius * (0.94 + pulse * 0.28)))
                if kind == "blow":
                    lens.setColor(0.08 + pulse * 0.1, 0.12 + pulse * 0.1, 0.18 + pulse * 0.12, 0.1 + pulse * 0.14)
                else:
                    lens.setColor(0.06 + pulse * 0.08, 0.08 + pulse * 0.09, 0.12 + pulse * 0.1, 0.11 + pulse * 0.14)
            if corona is not None and not corona.isEmpty():
                corona.setTexOffset(TextureStage.getDefault(), (self.roll_time * -0.2 + phase * 0.27) % 1.0, (self.roll_time * 0.22 + phase * 0.09) % 1.0)
                corona.setScale(max(0.3, self.black_hole_visual_radius * (1.15 + pulse * 0.42)))
                if kind == "blow":
                    corona.setColor(0.18 + pulse * 0.12, 0.26 + pulse * 0.12, 0.44 + pulse * 0.14, 0.08 + pulse * 0.1)
                else:
                    corona.setColor(0.14 + pulse * 0.1, 0.2 + pulse * 0.11, 0.34 + pulse * 0.12, 0.08 + pulse * 0.1)
            if outline is not None and not outline.isEmpty():
                glow = 0.68 + 0.32 * pulse
                outline_stage = entry.get("outline_stage")
                if kind == "blow":
                    dial_scale = max(0.9, self.black_hole_visual_radius * (2.2 + pulse * 0.18))
                    outline.setScale(dial_scale, dial_scale, max(0.06, dial_scale * 0.11))
                    outline.setColor(0.95 * glow, 0.98 * glow, 1.0 * glow, 0.2 + pulse * 0.24)
                    if outline_stage is not None:
                        outline.setTexRotate(outline_stage, (self.roll_time * -72.0 + phase * 57.0) % 360.0)
                        outline.setTexOffset(outline_stage, (phase * 0.13 + self.roll_time * 0.06) % 1.0, 0.0)
                else:
                    outline.setScale(max(0.4, self.black_hole_visual_radius * (1.86 + pulse * 0.3)))
                    outline.setColor(1.0 * glow, 1.0 * glow, 1.0 * glow, 0.24 + pulse * 0.26)
                    if outline_stage is not None:
                        outline.setTexRotate(outline_stage, (self.roll_time * 108.0 + phase * 91.0) % 360.0)
                        outline.setTexOffset(outline_stage, (phase * 0.07 + self.roll_time * 0.03) % 1.0, 0.0)

            radius = max(0.8, float(entry.get("radius", self.black_hole_influence_radius)))
            pull_strength = max(0.0, float(entry.get("pull", self.black_hole_pull_strength)))
            distort_gain = max(0.0, float(entry.get("distort_gain", 1.0)))

            if ball_pos is not None and hasattr(self, "ball_body") and self.ball_body is not None:
                to_anomaly = pos - ball_pos
                dist = max(1e-5, to_anomaly.length())
                if dist <= radius:
                    direction = to_anomaly / dist
                    proximity = 1.0 - (dist / radius)
                    force_dir = -direction if kind == "blow" else direction
                    force_mag = pull_strength * (0.16 + proximity * proximity * (1.95 if kind == "blow" else 1.7))
                    if kind == "suck":
                        soften = self._clamp(float(getattr(self, "black_hole_suck_escape_soften", 0.38)), 0.1, 0.9)
                        soften_dist = max(0.3, radius * soften)
                        center_scale = self._clamp(dist / soften_dist, 0.0, 1.0)
                        force_mag *= (0.22 + 0.78 * center_scale)
                        force_mag = min(force_mag, float(getattr(self, "black_hole_suck_max_force", 165.0)))
                    self.ball_body.applyCentralForce(force_dir * force_mag)
                    max_distort = max(max_distort, proximity * distort_gain)

            for monster in live_monsters:
                m_root = monster.get("root")
                if m_root is None or m_root.isEmpty():
                    continue
                m_pos = m_root.getPos()
                to_anomaly_m = pos - m_pos
                m_dist = max(1e-5, to_anomaly_m.length())
                if m_dist > radius:
                    continue

                m_dir = to_anomaly_m / m_dist
                proximity_m = 1.0 - (m_dist / radius)
                move_dir = -m_dir if kind == "blow" else m_dir
                shove = pull_strength * (0.14 + proximity_m * proximity_m * 1.65) * monster_force_scale
                existing_knock = Vec3(monster.get("knockback_vel", Vec3(0, 0, 0)))
                monster["knockback_vel"] = existing_knock + move_dir * shove

                if kind == "suck" and proximity_m >= 0.94 and float(monster.get("cosmic_warp_cooldown", 0.0)) <= 0.0:
                    target_room_idx = random.randrange(len(self.rooms))
                    out_pos = self._safe_room_spawn_pos(target_room_idx, z_lift=random.uniform(0.32, 0.82))
                    m_root.setPos(out_pos)
                    monster["prev_pos"] = Vec3(out_pos)
                    monster["base_z"] = float(out_pos.z)
                    spit_dir = Vec3(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(0.12, 0.46))
                    if spit_dir.lengthSquared() < 1e-8:
                        spit_dir = Vec3(1.0, 0.0, 0.22)
                    spit_dir.normalize()
                    monster["knockback_vel"] = spit_dir * random.uniform(spit_speed * 0.7, spit_speed * 1.35)
                    monster["w"] = random.uniform(-self.hyper_w_limit * 0.95, self.hyper_w_limit * 0.95)
                    monster["w_vel"] = float(monster.get("w_vel", 0.0)) * 0.2 + random.uniform(-2.4, 2.4)
                    monster["fold_jump_cooldown"] = max(float(monster.get("fold_jump_cooldown", 0.0)), 0.8)
                    monster["cosmic_warp_cooldown"] = warp_cd
                    self._set_monster_state(monster, "guarding")

        response = min(1.0, dt * 7.5)
        self.black_hole_distort_intensity += (max_distort - self.black_hole_distort_intensity) * response

    def _update_outside_islands(self, dt: float) -> None:
        if not self.outside_islands:
            return
        for island in self.outside_islands:
            root = island.get("root")
            if root is None or root.isEmpty():
                continue
            phase = island["phase"]
            speed = island["speed"]
            amp = island["amp"]
            z_amp = island["z_amp"]
            base = island["base"]

            sway = Vec3(
                math.sin(self.roll_time * speed + phase) * amp,
                math.cos(self.roll_time * speed * 0.92 + phase * 1.3) * amp,
                math.sin(self.roll_time * speed * 1.4 + phase * 0.7) * z_amp,
            )
            target = base + sway
            if not self._is_outside_main_rooms(target, margin=10.0):
                target = Vec3(base)

            root.setPos(target)
            for entry in island["colliders"]:
                body_np = entry.get("body_np")
                if body_np is None or body_np.isEmpty():
                    continue
                local = entry.get("local", Vec3(0, 0, 0))
                body_np.setPos(target + local)
                if "hpr" in entry:
                    body_np.setHpr(entry["hpr"])

    def _update_warp_links(self, dt: float) -> None:
        self.warp_cooldown = max(0.0, self.warp_cooldown - dt)
        if self.warp_cooldown > 0.0 or not self.warp_links or not hasattr(self, "ball_np"):
            return

        ball_pos = self.ball_np.getPos()
        up = self._get_gravity_up()
        if up.lengthSquared() < 1e-8:
            up = Vec3(0, 0, 1)
        else:
            up.normalize()

        if self.enable_room_fold_thread and self.room_fold_thread_ready:
            with self._room_fold_links_lock:
                req_version = self._room_fold_links_version
            self._replace_queue_item(self._room_fold_probe_queue, (req_version, float(ball_pos.x), float(ball_pos.y), float(ball_pos.z)))

            result = None
            while True:
                try:
                    result = self._room_fold_result_queue.get_nowait()
                except queue.Empty:
                    break
            if result is not None:
                _req_version, result_version, hit_idx, hit_dir = result
                with self._room_fold_links_lock:
                    current_version = self._room_fold_links_version
                if result_version == current_version and 0 <= int(hit_idx) < len(self.warp_links) and int(hit_dir) in (0, 1):
                    link = self.warp_links[int(hit_idx)]
                    if link.get("mode") == "room_fold":
                        if self._apply_room_fold_warp(link, from_a=(int(hit_dir) == 0), up=up):
                            return
        else:
            for link in self.warp_links:
                mode = link.get("mode")
                if mode == "maze_portal":
                    a_pos = link.get("a_pos")
                    b_pos = link.get("b_pos")
                    if a_pos is None or b_pos is None:
                        continue
                    radius_sq = float(link.get("radius_sq", float(link.get("radius", 1.0)) ** 2))
                    if (ball_pos - a_pos).lengthSquared() <= radius_sq:
                        if self._apply_maze_portal_warp(link, from_a=True, up=up):
                            return
                    if (ball_pos - b_pos).lengthSquared() <= radius_sq:
                        if self._apply_maze_portal_warp(link, from_a=False, up=up):
                            return
                    continue

                if mode != "room_fold":
                    continue
                a_pos = link.get("a_pos")
                b_pos = link.get("b_pos")
                if a_pos is None or b_pos is None:
                    continue
                radius_sq = float(link.get("radius_sq", float(link.get("radius", 1.0)) ** 2))
                if (ball_pos - a_pos).lengthSquared() <= radius_sq:
                    if self._apply_room_fold_warp(link, from_a=True, up=up):
                        return
                if (ball_pos - b_pos).lengthSquared() <= radius_sq:
                    if self._apply_room_fold_warp(link, from_a=False, up=up):
                        return

        for link in self.warp_links:
            mode = link.get("mode", "island")
            radius = link.get("radius", 1.0)
            radius_sq = float(link.get("radius_sq", float(radius) * float(radius)))

            if mode == "room_fold":
                continue
            if mode == "maze_portal":
                a_pos = link.get("a_pos")
                b_pos = link.get("b_pos")
                if a_pos is None or b_pos is None:
                    continue
                if (ball_pos - a_pos).lengthSquared() <= radius_sq:
                    if self._apply_maze_portal_warp(link, from_a=True, up=up):
                        return
                if (ball_pos - b_pos).lengthSquared() <= radius_sq:
                    if self._apply_maze_portal_warp(link, from_a=False, up=up):
                        return
                continue

            room_pos = link.get("room_pos")
            island_idx = int(link.get("island_idx", -1))
            if room_pos is None or island_idx < 0:
                continue

            island = self.outside_islands[island_idx] if island_idx < len(self.outside_islands) else None
            if island is None:
                continue
            island_root = island.get("root")
            if island_root is None or island_root.isEmpty():
                continue

            island_world = island_root.getPos(self.render) + island.get("warp_local", Vec3(0, 0, 0))
            room_idx = int(link.get("room_idx", 0))
            if room_idx < 0 or room_idx >= len(self.rooms):
                continue
            room = self.rooms[room_idx]
            room_target = self._safe_room_spawn_pos(room_idx, z_lift=0.36)

            if (ball_pos - room_pos).lengthSquared() <= radius_sq:
                self.ball_np.setPos(island_world + up * (self.ball_radius + 0.28))
                self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
                self.warp_cooldown = 0.55
                return

            if (ball_pos - island_world).lengthSquared() <= radius_sq:
                self.ball_np.setPos(room_target + up * (self.ball_radius + 0.24))
                self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
                self.warp_cooldown = 0.55
                return

    def _initialize_infinite_world_goal(self) -> None:
        if not self.infinite_level_mode:
            return
        self._stream_infinite_exterior(force=True)
        self._place_goal_in_world(force_new=True)
        self._rebuild_goal_path_string()

    def _chunk_key_from_pos(self, pos: Vec3) -> tuple[int, int, int]:
        return (
            int(math.floor(pos.x / self.exterior_chunk_size)),
            int(math.floor(pos.y / self.exterior_chunk_size)),
            int(math.floor((pos.z - self.floor_y) / self.exterior_chunk_z_step)),
        )

    def _chunk_center(self, key: tuple[int, int, int]) -> Vec3:
        kx, ky, kz = key
        return Vec3(
            (kx + 0.5) * self.exterior_chunk_size,
            (ky + 0.5) * self.exterior_chunk_size,
            self.floor_y + (kz + 0.5) * self.exterior_chunk_z_step,
        )

    def _is_outside_main_rooms(self, p: Vec3, margin: float = 6.0) -> bool:
        return (p.x < -margin or p.x > self.map_w + margin or p.y < -margin or p.y > self.map_d + margin)

    def _generate_exterior_chunk(self, key: tuple[int, int, int]) -> None:
        if key in self.exterior_chunks:
            return

        center = self._chunk_center(key)
        outside = self._is_outside_main_rooms(center)
        if not outside and random.random() < 0.45:
            self.exterior_chunks[key] = {"center": center, "platforms": []}
            return

        chunk_platforms: list[Vec3] = []
        mirror_signs = [
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, -1.0, -1.0),
        ]

        base_nodes = random.randint(2, 4)
        half_xy = self.exterior_chunk_size * 0.28
        half_z = self.exterior_chunk_z_step * 0.28
        used: set[tuple[int, int, int]] = set()

        for _ in range(base_nodes):
            local = Vec3(
                random.uniform(8.0, half_xy),
                random.uniform(8.0, half_xy),
                random.uniform(2.5, half_z),
            )
            plat_half = Vec3(
                random.uniform(2.8, 6.6),
                random.uniform(2.8, 6.6),
                random.uniform(0.22, 0.8),
            )

            for sx, sy, sz in mirror_signs:
                offset = Vec3(local.x * sx, local.y * sy, local.z * sz)
                world_pos = center + offset
                key_pos = (int(round(world_pos.x)), int(round(world_pos.y)), int(round(world_pos.z)))
                if key_pos in used:
                    continue
                used.add(key_pos)

                col = (
                    random.uniform(0.52, 0.78),
                    random.uniform(0.6, 0.9),
                    random.uniform(0.72, 1.0),
                    1.0,
                )
                self._add_box(world_pos, plat_half, color=col, motion_group="platform")
                chunk_platforms.append(Vec3(world_pos))

                if random.random() < 0.38:
                    island_top = world_pos + Vec3(0, 0, plat_half.z + random.uniform(0.9, 2.1))
                    self._add_box(
                        island_top,
                        Vec3(plat_half.x * 0.62, plat_half.y * 0.62, plat_half.z * 0.62),
                        color=(0.3, 0.88, 1.0, 1.0),
                        motion_group="platform",
                    )
                    chunk_platforms.append(Vec3(island_top))

        self.exterior_chunks[key] = {"center": center, "platforms": chunk_platforms}

    def _stream_infinite_exterior(self, force: bool = False) -> None:
        if not self.infinite_level_mode or not hasattr(self, "ball_np"):
            return
        origin = self._chunk_key_from_pos(self.ball_np.getPos())
        ox, oy, oz = origin
        desired: list[tuple[int, int, int, int]] = []
        for dz in range(-self.exterior_stream_radius_z, self.exterior_stream_radius_z + 1):
            for dy in range(-self.exterior_stream_radius_xy, self.exterior_stream_radius_xy + 1):
                for dx in range(-self.exterior_stream_radius_xy, self.exterior_stream_radius_xy + 1):
                    key = (ox + dx, oy + dy, oz + dz)
                    if key in self.exterior_chunks or key in self.exterior_stream_pending:
                        continue
                    dist = abs(dx) + abs(dy) + abs(dz)
                    desired.append((dist, key[0], key[1], key[2]))

        desired.sort(key=lambda item: item[0])
        for _, kx, ky, kz in desired:
            key = (kx, ky, kz)
            self.exterior_stream_backlog.append(key)
            self.exterior_stream_pending.add(key)

        budget = self.exterior_gen_budget_force if force else self.exterior_gen_budget_tick
        while budget > 0 and self.exterior_stream_backlog:
            key = self.exterior_stream_backlog.popleft()
            self.exterior_stream_pending.discard(key)
            self._generate_exterior_chunk(key)
            budget -= 1

    def _place_goal_in_world(self, force_new: bool = False) -> None:
        if not self.infinite_level_mode:
            return

        if not self.exterior_chunks:
            self._stream_infinite_exterior(force=True)

        ball_key = self._chunk_key_from_pos(self.ball_np.getPos())
        candidates: list[tuple[int, tuple[int, int, int], Vec3]] = []
        for key, data in self.exterior_chunks.items():
            platforms = data.get("platforms", [])
            if not platforms:
                continue
            dist = abs(key[0] - ball_key[0]) + abs(key[1] - ball_key[1]) + abs(key[2] - ball_key[2])
            if dist < 2:
                continue
            candidates.append((dist, key, random.choice(platforms)))

        if not candidates:
            return

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, goal_key, goal_pos = candidates[0]
        if (not force_new) and self.goal_chunk_key == goal_key:
            return

        self.goal_chunk_key = goal_key
        self.goal_pos = Vec3(goal_pos + Vec3(0, 0, 1.15))

        if self.goal_np is None or self.goal_np.isEmpty():
            self.goal_np = self.sphere_model.copyTo(self.world)
            self.goal_np.setScale(0.35)
            self.goal_np.setLightOff(1)
            self.goal_np.setCollideMask(BitMask32.allOff())
            self.goal_np.setTransparency(TransparencyAttrib.MAlpha)
            self.goal_np.setColor(1.0, 0.3, 0.9, 0.95)
            self._register_color_cycle(self.goal_np, (1.0, 0.3, 0.9, 0.95), min_speed=0.32, max_speed=0.72)
        self.goal_np.setPos(self.goal_pos)
        self._force_non_solid_visual(self.goal_np)

    def _heuristic_key_dist(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _find_chunk_path(self, start_key: tuple[int, int, int], goal_key: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        if start_key == goal_key:
            return [start_key]

        valid = {k for k, v in self.exterior_chunks.items() if v.get("platforms")}
        if start_key not in valid or goal_key not in valid:
            return []

        open_heap: list[tuple[float, tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_key))
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score = {start_key: 0.0}
        visited: set[tuple[int, int, int]] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_key:
                break

            cx, cy, cz = current
            neighbors = [
                (cx + 1, cy, cz),
                (cx - 1, cy, cz),
                (cx, cy + 1, cz),
                (cx, cy - 1, cz),
                (cx, cy, cz + 1),
                (cx, cy, cz - 1),
            ]
            for nxt in neighbors:
                if nxt not in valid:
                    continue
                tentative = g_score[current] + 1.0
                if tentative >= g_score.get(nxt, 1e18):
                    continue
                came_from[nxt] = current
                g_score[nxt] = tentative
                f_score = tentative + self._heuristic_key_dist(nxt, goal_key)
                heapq.heappush(open_heap, (f_score, nxt))

        if goal_key not in came_from and goal_key != start_key:
            return []

        path = [goal_key]
        while path[-1] != start_key:
            prev = came_from.get(path[-1])
            if prev is None:
                return []
            path.append(prev)
        path.reverse()
        return path

    def _rebuild_goal_path_string(self) -> None:
        if self.goal_path_np is not None and not self.goal_path_np.isEmpty():
            self.goal_path_np.removeNode()
            self.goal_path_np = None

        start_pos = Vec3(self.ball_np.getPos()) if hasattr(self, "ball_np") and self.ball_np is not None else None
        if start_pos is None:
            return

        nearest_monster_pos: Vec3 | None = None
        nearest_monster_dist = float("inf")
        player_w = float(getattr(self, "player_w", 0.0))
        for monster in getattr(self, "monsters", []):
            if monster.get("dead", False):
                continue
            root = monster.get("root")
            if root is None or root.isEmpty():
                continue
            m_pos = root.getPos()
            m_w = float(monster.get("w", 0.0))
            d = self._distance4d(start_pos, player_w, m_pos, m_w)
            if d < nearest_monster_dist:
                nearest_monster_dist = d
                nearest_monster_pos = Vec3(m_pos)

        points: list[Vec3] = []
        if nearest_monster_pos is not None:
            up = self._get_gravity_up()
            if up.lengthSquared() < 1e-8:
                up = Vec3(0, 0, 1)
            else:
                up.normalize()

            start = Vec3(start_pos)
            end = Vec3(nearest_monster_pos)
            chord = end - start
            chord_len = max(0.001, chord.length())
            arch_height = self._clamp(chord_len * 0.16, 0.7, 5.0)
            control = (start + end) * 0.5 + up * arch_height

            points = [start]
            samples = 16
            for i in range(1, samples):
                t = i / float(samples)
                omt = 1.0 - t
                p = (start * (omt * omt)) + (control * (2.0 * omt * t)) + (end * (t * t))
                points.append(Vec3(p))
            points.append(end)
        else:
            if self.goal_np is None or self.goal_np.isEmpty() or self.goal_chunk_key is None:
                points = []

        if not points and self.goal_np is not None and not self.goal_np.isEmpty() and self.goal_chunk_key is not None:
            start_key = self._chunk_key_from_pos(self.ball_np.getPos())
            if start_key not in self.exterior_chunks:
                self._generate_exterior_chunk(start_key)

            key_path = self._find_chunk_path(start_key, self.goal_chunk_key)
            points = [Vec3(self.ball_np.getPos())]
            if key_path:
                for key in key_path[1:-1]:
                    data = self.exterior_chunks.get(key)
                    if data is None:
                        continue
                    points.append(Vec3(data.get("center", self._chunk_center(key))))
            points.append(Vec3(self.goal_pos))

        landmark_points: list[Vec3] = []
        nearest_landmark_pos: Vec3 | None = None
        nearest_landmark_dist_sq = float("inf")
        for entry in getattr(self, "room_landmarks", []):
            root = entry.get("root") if isinstance(entry, dict) else None
            if root is None or root.isEmpty():
                continue
            lp = root.getPos(self.render)
            d_sq = (lp - start_pos).lengthSquared()
            if d_sq < nearest_landmark_dist_sq:
                nearest_landmark_dist_sq = d_sq
                nearest_landmark_pos = Vec3(lp)

        if nearest_landmark_pos is not None:
            up = self._get_gravity_up()
            if up.lengthSquared() < 1e-8:
                up = Vec3(0, 0, 1)
            else:
                up.normalize()
            l_start = Vec3(start_pos)
            l_end = Vec3(nearest_landmark_pos)
            l_chord = l_end - l_start
            l_len = max(0.001, l_chord.length())
            l_arch_height = self._clamp(l_len * 0.2, 0.8, 6.0)
            l_control = (l_start + l_end) * 0.5 + up * l_arch_height
            landmark_points = [l_start]
            l_samples = 22
            for i in range(1, l_samples):
                t = i / float(l_samples)
                omt = 1.0 - t
                p = (l_start * (omt * omt)) + (l_control * (2.0 * omt * t)) + (l_end * (t * t))
                landmark_points.append(Vec3(p))
            landmark_points.append(l_end)

        if len(points) < 2 and len(landmark_points) < 2:
            return

        segs = LineSegs("goal-path-string")
        segs.setThickness(1.2)

        if len(points) >= 2:
            hue = (self.roll_time * 0.27) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            segs.setColor(r, g, b, 0.95)
            segs.moveTo(points[0])
            for p in points[1:]:
                segs.drawTo(p)

        if len(landmark_points) >= 2:
            dotted_count = len(landmark_points) - 1
            for i in range(dotted_count):
                if i % 2 == 1:
                    continue
                a = landmark_points[i]
                b_pt = landmark_points[i + 1]
                hue = (self.roll_time * 0.45 + (i / max(1.0, float(dotted_count)))) % 1.0
                rr, rg, rb = colorsys.hsv_to_rgb(hue, 0.92, 1.0)
                segs.setColor(rr, rg, rb, 0.95)
                segs.moveTo(a)
                segs.drawTo(b_pt)

        self.goal_path_np = self.render.attachNewNode(segs.create(True))
        self.goal_path_np.setLightOff(1)
        self.goal_path_np.setShaderOff(1)
        self.goal_path_np.setCollideMask(BitMask32.allOff())
        self.goal_path_np.setTransparency(TransparencyAttrib.MAlpha)
        self.goal_path_np.setDepthWrite(False)
        self.goal_path_np.setBin("fixed", 55)
        self._force_non_solid_visual(self.goal_path_np)

    def _update_infinite_world_goal(self, dt: float) -> None:
        if not self.infinite_level_mode:
            return

        self.exterior_stream_timer -= dt
        if self.exterior_stream_timer <= 0.0:
            self._stream_infinite_exterior(force=False)
            self.exterior_stream_timer = self.exterior_stream_interval

        if self.goal_np is None or self.goal_np.isEmpty() or self.goal_chunk_key is None:
            self._place_goal_in_world(force_new=True)

        if self.goal_chunk_key is not None:
            dist_to_goal = (self.ball_np.getPos() - self.goal_pos).length()
            if dist_to_goal <= max(1.3, self.ball_radius * 3.2):
                self._place_goal_in_world(force_new=True)

        self.goal_path_update_timer -= dt
        if self.goal_path_update_timer <= 0.0:
            self._rebuild_goal_path_string()
            self.goal_path_update_timer = self.goal_path_update_interval

    def _add_decor_box(
        self,
        pos: Vec3,
        scale: Vec3,
        color=(0.8, 0.8, 0.8, 1),
        hpr: Vec3 | None = None,
        w_coord: float | None = None,
    ) -> NodePath:
        return self._add_box(pos, scale, color=color, hpr=hpr, collidable=False, w_coord=w_coord)

    def _force_non_solid_visual(self, node: NodePath | None) -> None:
        if node is None or node.isEmpty():
            return
        try:
            node.setCollideMask(BitMask32.allOff())
        except Exception:
            pass
        for child in node.findAllMatches("**"):
            if child is None or child.isEmpty():
                continue
            try:
                child.setCollideMask(BitMask32.allOff())
            except Exception:
                continue

    def _compute_model_normalization(self, model: NodePath) -> tuple[Vec3, Vec3]:
        bounds = model.getTightBounds()
        if bounds is None:
            return Vec3(1, 1, 1), Vec3(0, 0, 0)

        bmin, bmax = bounds
        size = bmax - bmin
        center = (bmin + bmax) * 0.5

        sx = 2.0 / max(size.x, 1e-6)
        sy = 2.0 / max(size.y, 1e-6)
        sz = 2.0 / max(size.z, 1e-6)
        norm_scale = Vec3(sx, sy, sz)
        norm_offset = Vec3(-center.x * sx, -center.y * sy, -center.z * sz)
        return norm_scale, norm_offset

    def _get_cycle_texture_pool(self, base_tex: Texture | None) -> list[Texture]:
        pool: list[Texture] = []

        def _push(tex: Texture | None) -> None:
            if tex is None:
                return
            for existing in pool:
                if existing is tex:
                    return
            pool.append(tex)

        _push(base_tex)
        for _ in range(8):
            _push(self._get_random_room_texture())
            if len(pool) >= 3:
                break
        _push(self.floor_fractal_tex_a)
        _push(self.floor_fractal_tex_b)
        _push(self.level_checker_tex)
        if not pool:
            pool.append(self.level_checker_tex)
        return pool

    def _build_texture_cycle_layers(self, node: NodePath) -> list[dict]:
        if getattr(self, "single_texture_per_cube", False):
            return []
        if not self.texture_layer_cycle_enabled or node is None or node.isEmpty():
            return []

        default_stage = TextureStage.getDefault()
        base_tex = node.getTexture(default_stage)
        if base_tex is None:
            base_tex = node.getTexture()
        if base_tex is None:
            base_tex = self.level_checker_tex
            node.setTexture(default_stage, base_tex, 1)

        tex_pool = self._get_cycle_texture_pool(base_tex)
        node.setTransparency(TransparencyAttrib.MAlpha)

        layers: list[dict] = []
        phase_step = math.tau / 3.0
        dir_pairs = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0))
        for idx in range(3):
            stage = TextureStage(f"cycle-layer-{len(self.color_cycle_nodes)}-{idx}")
            stage.setMode(TextureStage.MAdd)
            stage.setSort(20 + idx)
            tex = tex_pool[idx % len(tex_pool)]
            node.setTexture(stage, tex, 20 + idx)

            u0 = random.random()
            v0 = random.random()
            node.setTexOffset(stage, u0, v0)

            du, dv = dir_pairs[idx]
            gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
            stage.setColor((gain, gain, gain, 0.05))
            layers.append(
                {
                    "stage": stage,
                    "u0": u0,
                    "v0": v0,
                    "du": du,
                    "dv": dv,
                    "phase": idx * phase_step,
                }
            )
        return layers

    def _register_color_cycle(self, node: NodePath, base_color: tuple[float, float, float, float], min_speed: float = 0.05, max_speed: float = 0.12) -> None:
        r, g, b, a = base_color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        tex_layers = self._build_texture_cycle_layers(node)
        self.color_cycle_nodes.append(
            {
                "node": node,
                "h": h,
                "s": s,
                "v": v,
                "a": a,
                "phase": random.random(),
                "speed": random.uniform(min_speed, max_speed),
                "tex_layers": tex_layers,
            }
        )

    def _update_color_cycle(self, dt: float) -> None:
        self.color_cycle_time += dt
        keep: list[dict] = []
        for entry in self.color_cycle_nodes:
            node = entry["node"]
            if node is None or node.isEmpty():
                continue
            if node.isStashed() or node.isHidden():
                keep.append(entry)
                continue
            h = (entry["h"] + self.color_cycle_time * entry["speed"] + entry["phase"]) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, entry["s"], entry["v"])
            node.setColor(r, g, b, entry["a"])

            tex_layers = entry.get("tex_layers") or []
            if tex_layers:
                min_a = max(0.0, min(1.0, float(self.texture_layer_alpha_min)))
                max_a = max(min_a, min(1.0, float(self.texture_layer_alpha_max)))
                scroll_u = float(self.texture_layer_scroll_u)
                scroll_v = float(self.texture_layer_scroll_v)
                fade_speed = float(self.texture_layer_fade_speed)
                for layer in tex_layers:
                    stage = layer.get("stage")
                    if stage is None:
                        continue
                    phase = float(layer.get("phase", 0.0)) + float(entry["phase"]) * math.tau
                    fade_wave = 0.5 + 0.5 * math.sin(self.color_cycle_time * fade_speed + phase)
                    alpha = min_a + (max_a - min_a) * fade_wave
                    gain = float(getattr(self, "texture_layer_additive_gain", 0.42))
                    stage.setColor((gain, gain, gain, alpha))

                    u = (float(layer.get("u0", 0.0)) + self.color_cycle_time * scroll_u * float(layer.get("du", 1.0))) % 1.0
                    v = (float(layer.get("v0", 0.0)) + self.color_cycle_time * scroll_v * float(layer.get("dv", 1.0))) % 1.0
                    node.setTexOffset(stage, u, v)
            keep.append(entry)
        self.color_cycle_nodes = keep

    def _spawn_hypercube_monsters(self, count: int = 8) -> None:
        if not self.rooms:
            return
        self.monsters_slain = 0

        hp_bg_template = self._get_quad_template("monster-hp-bg", (-0.5, 0.5, -0.045, 0.045))
        hp_fill_template = self._get_quad_template("monster-hp-fill", (0.0, 0.94, -0.03, 0.03))

        room_total = len(self.rooms)
        min_per_room = 1
        max_per_room = 2 if self.performance_mode else 3
        approx_per_room = max(min_per_room, min(max_per_room, int(round(max(1, count) / max(1, room_total)))))

        spawn_plan: list[tuple[int, Vec3]] = []
        start_room_idx = int(getattr(self, "start_room_idx", 0)) if self.rooms else 0
        start_room_idx = max(0, min(len(self.rooms) - 1, start_room_idx))

        if bool(getattr(self, "four_d_obstacle_arena_mode", False)) and self.arena_platform_points:
            points = [Vec3(p) for p in self.arena_platform_points]
            points.sort(key=lambda p: p.z)
            target_count = min(max(1, count), len(points))
            step = max(1, len(points) // target_count)
            for i in range(target_count):
                p = points[min(len(points) - 1, i * step)]
                room_idx = self._get_current_room_idx_for_pos(p)
                if room_idx is None:
                    room_idx = start_room_idx
                stagger = (i % 5) * 0.55
                spawn_plan.append(
                    (
                        room_idx,
                        Vec3(
                            p.x + random.uniform(-1.4, 1.4),
                            p.y + random.uniform(-1.4, 1.4),
                            p.z + 0.35 + stagger,
                        ),
                    )
                )

        for room_idx, room in enumerate(self.rooms):
            room_base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            room_center = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, room_base_z + 0.92)
            spread = max(0.45, min(room.w, room.h) * 0.14)
            per_room_target = approx_per_room
            if room_idx == start_room_idx:
                per_room_target = 0
            for _ in range(per_room_target):
                for _try in range(16):
                    ox = random.uniform(-spread, spread)
                    oy = random.uniform(-spread, spread)
                    sx = room_center.x + ox
                    sy = room_center.y + oy
                    if self._is_in_room_entry_lane(room_idx, room, sx, sy, extra=0.5):
                        continue
                    spawn_plan.append((room_idx, Vec3(sx, sy, room_center.z)))
                    break
                else:
                    spawn_plan.append((room_idx, Vec3(room_center)))

        if len(spawn_plan) > max(1, count):
            random.shuffle(spawn_plan)
            spawn_plan = spawn_plan[: max(1, count)]

        if not spawn_plan:
            return

        for idx, (room_idx, spawn_pos) in enumerate(spawn_plan):
            spawn_pos = self._clamp_point_inside_room(room_idx, Vec3(spawn_pos), inset=0.78)
            root = self.world.attachNewNode(f"cube-monster-{idx}")
            root.setPos(spawn_pos)

            variant = "normal"
            hp_mult = 1.0
            defense_mult = 1.0
            speed_mult = 1.0
            guard_mult = 1.0
            attack_mult = 1.0
            cycle_min, cycle_max = 0.25, 0.7
            variant_roll = random.random()
            if variant_roll < 0.12:
                variant = "juggernaut"
                hp_mult = 3.2
                defense_mult = 2.8
                speed_mult = 0.78
                guard_mult = 1.55
                attack_mult = 1.85
                cycle_min, cycle_max = 1.1, 2.2
            elif variant_roll < 0.24:
                variant = "vanguard"
                hp_mult = 2.35
                defense_mult = 2.2
                speed_mult = 0.95
                guard_mult = 2.4
                attack_mult = 1.45
                cycle_min, cycle_max = 1.0, 1.9
            elif variant_roll < 0.34:
                variant = "raider"
                hp_mult = 1.5
                defense_mult = 1.0
                speed_mult = 2.2
                guard_mult = 1.3
                attack_mult = 1.2
                cycle_min, cycle_max = 1.5, 2.8

            if random.random() < float(getattr(self, "monster_giant_spawn_ratio", 0.08)):
                variant = "giant"
                hp_mult = max(hp_mult, float(getattr(self, "monster_giant_hp_mult", 6.5)))
                defense_mult = max(defense_mult, 2.4)
                speed_mult = min(speed_mult, 0.62)
                guard_mult = max(guard_mult, 0.9)
                attack_mult = max(attack_mult, 1.5)
                cycle_min, cycle_max = 0.45, 1.15

            part_count = random.randint(2, 4)
            parts: list[dict] = []
            for _ in range(part_count):
                part_holder = root.attachNewNode("monster-part-holder")
                part = self.box_model.copyTo(part_holder)
                part.setPos(self.box_norm_offset)
                part.setScale(self.box_norm_scale)
                hue = random.random()
                sat = random.uniform(0.55, 0.9)
                val = random.uniform(0.72, 1.0)
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                part.setColor(r, g, b, 1.0)
                part.clearTexture()
                part.setTexture(self.level_checker_tex, 1)
                self._apply_hypercube_projection(
                    part,
                    random.uniform(-self.hyper_w_limit, self.hyper_w_limit),
                    scale_hint=random.uniform(1.3, 3.2),
                )
                self._register_color_cycle(part, (r, g, b, 1.0), min_speed=cycle_min, max_speed=cycle_max)

                part_outline = self.box_model.copyTo(part)
                part_outline.setPos(self.box_norm_offset)
                part_outline.setScale(self.box_norm_scale)
                part_outline.setColor(0.2, 0.9, 1.0, 1.0)
                part_outline.clearTexture()
                part_outline.setLightOff(1)
                part_outline.setTwoSided(False)
                part_outline.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
                part_outline.setBin("fixed", 0)
                part_outline.hide()

                axis = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.8, 0.8))
                if axis.lengthSquared() < 1e-6:
                    axis = Vec3(1, 0, 0)
                axis.normalize()

                parts.append(
                    {
                        "holder": part_holder,
                        "node": part,
                        "outline": part_outline,
                        "base_offset": axis * random.uniform(0.08, 0.5),
                        "min_scale": random.uniform(0.08, 0.16),
                        "max_scale": random.uniform(0.22, 0.5),
                        "phase": random.uniform(0.0, math.tau),
                        "speed": random.uniform(2.0, 4.7),
                    }
                )

            velocity = Vec3(random.uniform(-2.2, 2.2), random.uniform(-2.2, 2.2), 0)
            if velocity.lengthSquared() < 0.2:
                velocity = Vec3(1.8, -1.4, 0)

            size_scale = random.uniform(0.72, 1.35)
            speed_scale = random.uniform(0.66, 1.42)
            hp_scale = random.uniform(0.76, 1.78)
            defense = random.uniform(0.75, 1.45)
            detect_range_mult = random.uniform(0.72, 1.6)
            range_profile_roll = random.random()
            if range_profile_roll < 0.24:
                detect_range_mult *= random.uniform(0.62, 0.88)
            elif range_profile_roll > 0.76:
                detect_range_mult *= random.uniform(1.2, 1.65)
            monster_crit_chance = random.uniform(0.04, 0.2)
            if variant == "raider":
                monster_crit_chance = random.uniform(0.18, 0.34)
            elif variant in ("juggernaut", "vanguard"):
                monster_crit_chance = random.uniform(0.06, 0.18)
            elif variant == "giant":
                monster_crit_chance = random.uniform(0.02, 0.08)
            fast_speed_boost = 1.0
            if random.random() < float(getattr(self, "monster_fast_ratio", 0.22)):
                fast_speed_boost = random.uniform(
                    float(getattr(self, "monster_fast_speed_min", 2.0)),
                    float(getattr(self, "monster_fast_speed_max", 3.4)),
                )
            speed_scale *= speed_mult
            speed_scale *= fast_speed_boost
            hp_scale *= hp_mult
            defense *= defense_mult
            velocity *= speed_scale
            root.setScale(size_scale)

            hp_anchor = root.attachNewNode("hp-anchor")
            hp_anchor.setPos(0, 0, 1.16)
            hp_anchor.setBillboardPointEye()

            hp_bg = hp_bg_template.instanceTo(hp_anchor)
            hp_bg.setColor(0.05, 0.06, 0.08, 0.88)
            hp_bg.setTransparency(TransparencyAttrib.MAlpha)
            hp_bg.setBin("transparent", 20)
            hp_bg.setDepthWrite(False)

            hp_fill_origin = hp_anchor.attachNewNode("hp-fill-origin")
            hp_fill_origin.setPos(-0.47, 0.001, 0.0)
            hp_fill = hp_fill_template.instanceTo(hp_fill_origin)
            hp_fill.setColor(0.25, 1.0, 0.25, 0.92)
            hp_fill.setTransparency(TransparencyAttrib.MAlpha)
            hp_fill.setBin("transparent", 21)
            hp_fill.setDepthWrite(False)

            state_text = TextNode(f"monster-state-{idx}")
            if variant == "giant":
                state_text.setText("GIANT")
            else:
                state_text.setText("ELITE" if variant != "normal" else "WANDER")
            state_text.setTextColor(0.65, 0.8, 1.0, 0.85)
            state_text.setAlign(TextNode.ACenter)
            state_np = hp_anchor.attachNewNode(state_text)
            state_np.setPos(0.0, 0.001, 0.09)
            state_np.setScale(0.09)
            state_np.setBillboardPointEye()

            monster = {
                "root": root,
                "parts": parts,
                "velocity": velocity,
                "knockback_vel": Vec3(0, 0, 0),
                "jump_vel": 0.0,
                "jump_cooldown": random.uniform(0.0, 0.45),
                "base_z": float(spawn_pos.z),
                "prev_pos": Vec3(root.getPos()),
                "radius": random.uniform(0.85, 1.35) * size_scale,
                "w": random.uniform(-self.hyper_w_limit * 0.85, self.hyper_w_limit * 0.85),
                "w_vel": random.uniform(-1.4, 1.4),
                "phase": random.uniform(0.0, math.tau),
                "spin": random.uniform(28.0, 74.0),
                "hp_max": self.monster_max_hp * hp_scale,
                "hp": self.monster_max_hp * hp_scale,
                "dead": False,
                "hp_fill": hp_fill_origin,
                "hum_sfx": None,
                "hum_active": False,
                "variant": variant,
                "defense": defense,
                "critical_chance": monster_crit_chance,
                "contact_damage": random.choice([8.0, 10.0, 12.0, 14.0]) * attack_mult,
                "outline_radius": random.uniform(4.0, 7.4),
                "outline_phase": random.uniform(0.0, math.tau),
                "state": "guarding" if variant in ("juggernaut", "vanguard") else "wandering",
                "state_timer": 0.0,
                "state_announce_cooldown": random.uniform(0.0, 0.45),
                "state_label": state_np,
                "ai_state": "guarding" if variant in ("juggernaut", "vanguard") else "wandering",
                "ai_char": None,
                "ai_behaviors": None,
                "ai_hunt_range": random.uniform(9.0, 14.5) * guard_mult * detect_range_mult,
                "ai_attack_range": random.uniform(1.5, 2.4) * max(1.0, speed_mult * 0.9),
                "ai_guard_range": random.uniform(15.0, 20.0) * guard_mult * detect_range_mult,
                "detection_range_mult": detect_range_mult,
                "docile_until_attacked": variant == "giant",
                "awakened": variant != "giant",
                "fold_target_idx": None,
                "fold_jump_cooldown": random.uniform(0.0, 0.55),
                "liminal_phase": random.uniform(0.0, math.tau),
                "speed_boost": fast_speed_boost,
            }
            self.monsters.append(monster)
            self._register_scene_visual(root, monster["w"])

        self.monster_spawn_count = len(self.monsters)
        self.monsters_total = self.monster_spawn_count
        self._update_monster_hud_ui()
        print(f"Spawned monsters: {self.monster_spawn_count} ({approx_per_room} per room across {room_total} rooms)")

    def _monster_move_blocked(self, start: Vec3, end: Vec3, radius: float) -> bool:
        if (end - start).lengthSquared() < 1e-10:
            return False

        ray_start = Vec3(start.x, start.y, start.z + 0.38)
        ray_end = Vec3(end.x, end.y, end.z + 0.38)
        hit = self.physics_world.rayTestClosest(ray_start, ray_end)
        if not hit.hasHit():
            return False

        node = hit.getNode()
        if node is None or node == self.ball_body:
            return False

        normal = hit.getHitNormal()
        if abs(normal.z) > 0.8:
            return False

        travel = (ray_end - ray_start).length()
        hit_dist = (hit.getHitPos() - ray_start).length()
        return hit_dist <= (travel + radius * 0.15)

    def _update_hypercube_monsters(self, dt: float) -> None:
        if not self.monsters:
            return

        ball_pos = self.ball_np.getPos() if hasattr(self, "ball_np") else Vec3(0, 0, 0)
        if self.ai_world is not None:
            try:
                self.ai_world.update()
            except Exception:
                self.ai_world = None
        alive_monsters: list[dict] = []
        for monster in self.monsters:
            if monster.get("dead", False):
                continue
            root = monster["root"]
            if root is None or root.isEmpty():
                hum = monster.get("hum_sfx")
                if hum:
                    hum.stop()
                continue
            monster["state_announce_cooldown"] = max(0.0, float(monster.get("state_announce_cooldown", 0.0)) - dt)
            pos = root.getPos()
            prev_pos = Vec3(monster.get("prev_pos", pos))
            monster["jump_cooldown"] = max(0.0, float(monster.get("jump_cooldown", 0.0)) - dt)

            jump_vel = float(monster.get("jump_vel", 0.0))
            base_z = float(monster.get("base_z", pos.z))
            if jump_vel > 0.0 or pos.z > base_z + 1e-3:
                jump_vel -= float(getattr(self, "monster_ai_jump_gravity", 12.0)) * dt
                pos.z += jump_vel * dt
                if pos.z <= base_z and jump_vel <= 0.0:
                    pos.z = base_z
                    jump_vel = 0.0
                root.setPos(pos)
                monster["jump_vel"] = jump_vel

            knockback_vel = Vec3(monster.get("knockback_vel", Vec3(0, 0, 0)))
            if knockback_vel.lengthSquared() > 1e-8:
                pos += knockback_vel * dt
                root.setPos(pos)
                monster["knockback_vel"] = knockback_vel * max(0.0, 1.0 - dt * 7.5)

            dist_sq = self._distance4d_sq(pos, float(monster.get("w", 0.0)), ball_pos, float(getattr(self, "player_w", 0.0)))
            hp_ratio = monster.get("hp", 1.0) / max(1e-6, monster.get("hp_max", 1.0))

            attack_range = float(monster.get("ai_attack_range", 2.0))
            hunt_range = float(monster.get("ai_hunt_range", 11.5))
            guard_range = float(monster.get("ai_guard_range", 17.0))

            if monster.get("state") == "hit":
                monster["state_timer"] = float(monster.get("state_timer", 0.0)) + dt
                if monster["state_timer"] >= 0.22:
                    self._set_monster_state(monster, "guarding")

            desired_state = monster.get("state", "wandering")
            if desired_state not in ("hit", "dying"):
                dormant_docile = bool(monster.get("docile_until_attacked", False)) and (not bool(monster.get("awakened", False)))
                if dormant_docile:
                    desired_state = "wandering"
                else:
                    if hp_ratio < 0.24 and dist_sq < (hunt_range * 1.6) ** 2:
                        desired_state = "running"
                    elif dist_sq <= attack_range * attack_range:
                        desired_state = "attacking"
                    elif dist_sq <= hunt_range * hunt_range:
                        desired_state = "hunting"
                    elif dist_sq <= guard_range * guard_range:
                        desired_state = "guarding"
                    else:
                        desired_state = "wandering"
                self._set_monster_state(monster, desired_state)

            if Vec3(monster.get("knockback_vel", Vec3(0, 0, 0))).lengthSquared() < 0.03:
                self._apply_monster_ai_state(monster, monster.get("state", "wandering"))

            monster["fold_jump_cooldown"] = max(0.0, float(monster.get("fold_jump_cooldown", 0.0)) - dt)
            state = monster.get("state", "wandering")
            if state in ("hunting", "attacking", "running", "guarding") and self.liminal_fold_nodes:
                m_idx = self._nearest_liminal_fold_idx(pos, float(monster.get("w", 0.0)))
                b_idx = self._nearest_liminal_fold_idx(ball_pos, float(getattr(self, "player_w", 0.0)))
                if m_idx is not None and b_idx is not None:
                    next_hop = self._next_liminal_fold_hop(m_idx, b_idx)
                    if next_hop is not None:
                        monster["fold_target_idx"] = next_hop

            fold_idx = monster.get("fold_target_idx")
            if isinstance(fold_idx, int) and 0 <= fold_idx < len(self.liminal_fold_nodes):
                node = self.liminal_fold_nodes[fold_idx]
                node_pos = Vec3(node["pos"])
                to_node = node_pos - pos
                to_node.z *= 0.6
                dist_to_node = to_node.length()
                if dist_to_node > 1e-4:
                    step = min(dist_to_node, dt * (3.8 if state in ("hunting", "attacking") else 2.4))
                    pos += to_node.normalized() * step
                    root.setPos(pos)

                if dist_to_node <= 1.55 and float(monster.get("fold_jump_cooldown", 0.0)) <= 0.0:
                    ghost = 0.32 + 0.22 * math.sin(self.roll_time * 8.0 + monster.get("liminal_phase", 0.0))
                    jump_offset = Vec3(random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35), ghost)
                    root.setPos(node_pos + jump_offset)
                    pos = root.getPos()
                    monster["base_z"] = float(pos.z)
                    target_w = float(node.get("w", 0.0))
                    monster["w"] += (target_w - float(monster.get("w", 0.0))) * 0.78
                    monster["w_vel"] = float(monster.get("w_vel", 0.0)) * 0.4 + random.uniform(-1.6, 1.6)
                    monster["fold_jump_cooldown"] = random.uniform(0.45, 0.95)

                    m_idx2 = self._nearest_liminal_fold_idx(pos, float(monster.get("w", 0.0)))
                    b_idx2 = self._nearest_liminal_fold_idx(ball_pos, float(getattr(self, "player_w", 0.0)))
                    if m_idx2 is not None and b_idx2 is not None:
                        monster["fold_target_idx"] = self._next_liminal_fold_hop(m_idx2, b_idx2)

            pos = root.getPos()
            is_in_jump_arc = float(monster.get("jump_vel", 0.0)) > 0.05
            if (not is_in_jump_arc) and self._monster_move_blocked(prev_pos, pos, monster["radius"]):
                fold_idx = monster.get("fold_target_idx")
                if isinstance(fold_idx, int) and 0 <= fold_idx < len(self.liminal_fold_nodes):
                    node_pos = Vec3(self.liminal_fold_nodes[fold_idx]["pos"])
                    root.setPos(node_pos + Vec3(0, 0, 0.28))
                    pos = root.getPos()
                    monster["base_z"] = float(pos.z)
                    target_w = float(self.liminal_fold_nodes[fold_idx].get("w", 0.0))
                    monster["w"] += (target_w - float(monster.get("w", 0.0))) * 0.82
                    monster["fold_jump_cooldown"] = random.uniform(0.35, 0.8)
                else:
                    state_now = monster.get("state", "wandering")
                    can_jump = (
                        bool(getattr(self, "monster_ai_jump_enabled", True))
                        and state_now in ("hunting", "attacking", "running")
                        and float(monster.get("jump_cooldown", 0.0)) <= 0.0
                    )
                    if can_jump:
                        up_impulse = max(0.6, float(getattr(self, "monster_ai_jump_impulse", 4.8)))
                        monster["jump_vel"] = max(float(monster.get("jump_vel", 0.0)), up_impulse)
                        monster["jump_cooldown"] = float(getattr(self, "monster_ai_jump_cooldown_duration", 0.9))
                        root.setPos(prev_pos + Vec3(0, 0, 0.04))
                        pos = root.getPos()
                    else:
                        root.setPos(prev_pos)
                        pos = Vec3(prev_pos)
                        self._set_monster_state(monster, "guarding")

            pos, _ = self._wrap_xy_position(pos, margin=1.6)

            speed_boost = max(1.0, float(monster.get("speed_boost", 1.0)))
            if speed_boost > 1.01:
                planar_delta = Vec3(pos.x - prev_pos.x, pos.y - prev_pos.y, 0.0)
                if planar_delta.lengthSquared() > 1e-8:
                    extra = planar_delta * (speed_boost - 1.0)
                    max_extra = max(0.0, dt * (2.6 + 3.8 * speed_boost))
                    extra_len = extra.length()
                    if extra_len > max_extra and max_extra > 0.0:
                        extra *= max_extra / extra_len
                    pos += extra

            root.setPos(pos)

            vel = (pos - prev_pos) / max(1e-4, dt)
            monster["prev_pos"] = Vec3(pos)

            monster["w"] += monster["w_vel"] * dt
            if monster["w"] > self.hyper_w_limit:
                monster["w"] = self.hyper_w_limit
                monster["w_vel"] = -abs(monster["w_vel"])
            elif monster["w"] < -self.hyper_w_limit:
                monster["w"] = -self.hyper_w_limit
                monster["w_vel"] = abs(monster["w_vel"])

            t = self.roll_time + monster["phase"]
            facing = Vec3(ball_pos.x - pos.x, ball_pos.y - pos.y, 0)
            if facing.lengthSquared() > 1e-6:
                facing.normalize()
                look_h = math.degrees(math.atan2(-facing.x, facing.y))
            else:
                look_h = t * monster["spin"]

            m_state = monster.get("state", "wandering")
            if m_state == "attacking":
                root.setHpr(look_h + math.sin(t * 18.0) * 20.0, 0, 0)
            elif m_state == "running":
                root.setHpr(look_h + 180.0, 0, 0)
            elif m_state in ("guarding", "hunting"):
                root.setHpr(look_h, 0, 0)
            else:
                root.setHpr(t * monster["spin"], 0, 0)

            monster["velocity"] = vel
            self.visual_w_map[id(root)] = monster["w"]
            self._update_monster_hum(monster, ball_pos)
            dist_sq = (pos - ball_pos).lengthSquared()

            near = dist_sq <= (monster.get("outline_radius", 5.5) ** 2)
            hue = (self.roll_time * 3.8 + monster.get("outline_phase", 0.0)) % 1.0
            rr, rg, rb = colorsys.hsv_to_rgb(hue, 0.88, 1.0)
            for part in monster["parts"]:
                outline = part.get("outline")
                if outline is None or outline.isEmpty():
                    continue
                if near and not root.isStashed():
                    outline.show()
                    outline.setColor(rr, rg, rb, 1.0)
                else:
                    outline.hide()
            heavy_anim = (not root.isStashed()) and (dist_sq < 54.0 * 54.0 or (self.monster_anim_tick % 2 == 0))
            if heavy_anim:
                for part in monster["parts"]:
                    part_holder = part.get("holder")
                    if part_holder is None or part_holder.isEmpty():
                        continue
                    pulse = 0.5 + 0.5 * math.sin(t * part["speed"] + part["phase"])
                    hyper_mix = 0.5 + 0.5 * math.sin((t + monster["w"] * 1.7) * (part["speed"] * 0.7) - part["phase"])
                    scale = part["min_scale"] + (part["max_scale"] - part["min_scale"]) * pulse
                    offset_factor = 0.12 + hyper_mix * 1.65
                    part_holder.setPos(part["base_offset"] * offset_factor + Vec3(0, 0, math.sin(t + part["phase"]) * 0.06))
                    part_holder.setScale(scale)
                    outline = part.get("outline")
                    if outline is not None and not outline.isEmpty():
                        outline.setScale(1.18)
            else:
                for part in monster["parts"]:
                    part_holder = part.get("holder")
                    if part_holder is None or part_holder.isEmpty():
                        continue
                    part_holder.setPos(part["base_offset"] * 0.42)
                    part_holder.setScale((part["min_scale"] + part["max_scale"]) * 0.5)
                    outline = part.get("outline")
                    if outline is not None and not outline.isEmpty():
                        outline.setScale(1.18)

            alive_monsters.append(monster)

        self.monsters = alive_monsters

    def _build_dungeon(self, force: bool = False) -> None:
        if self.dungeon_built_once and not force:
            return
        self.dungeon_build_count += 1
        if self.dungeon_build_count > 1:
            print(f"[perf] Ignored repeated dungeon build request (count={self.dungeon_build_count})")
        self.pending_floor_rects.clear()
        self.pending_ceiling_rects.clear()
        self.pending_floor_holes.clear()
        self.pending_ceiling_holes.clear()
        self.water_surfaces.clear()
        self._clear_inverted_level_echo()

        if getattr(self, "four_d_obstacle_arena_mode", False):
            self._build_four_d_obstacle_arena()
            self._rebuild_entrance_debug_overlay()
            self._setup_inverted_level_echo()
            self.dungeon_built_once = True
            return

        if self.platform_only_mode:
            self._build_platform_only_dungeon()
            self._rebuild_entrance_debug_overlay()
            self._setup_inverted_level_echo()
            self.dungeon_built_once = True
            return

        if getattr(self, "subtractive_maze_mode", False):
            self._build_subtractive_cube_maze()
            self._rebuild_entrance_debug_overlay()
            self._setup_inverted_level_echo()
            self.dungeon_built_once = True
            return

        self._build_floor_and_bounds()

        self._plan_hallways_and_doors()
        self._plan_floor_ceiling_openings()

        for idx, room in enumerate(self.rooms):
            level = self.room_levels[idx]
            self._build_room(idx, room, level)

        for p1, p2 in self.corridor_segments:
            self._build_corridor_segment(p1, p2)
        for jx, jy, jz in self.corridor_joints:
            self._build_corridor_joint(jx, jy, jz)

        self._commit_floor_union()
        self._commit_ceiling_union()
        self._rebuild_entrance_debug_overlay()
        self._setup_inverted_level_echo()
        self.dungeon_built_once = True

    def _build_subtractive_cube_maze(self) -> None:
        self.maze_portal_points.clear()
        gx = 5 if self.performance_mode else 7
        gy = 5 if self.performance_mode else 7
        gz = 4 if self.performance_mode else 5

        start = (gx // 2, gy // 2, gz - 1)
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        visited: set[tuple[int, int, int]] = {start}
        stack: list[tuple[int, int, int]] = [start]
        passages: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

        def _edge(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
            return (a, b) if a <= b else (b, a)

        while stack:
            cx, cy, cz = stack[-1]
            neighbors: list[tuple[int, int, int]] = []
            random.shuffle(dirs)
            for dx, dy, dz in dirs:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz and (nx, ny, nz) not in visited:
                    neighbors.append((nx, ny, nz))
            if neighbors:
                nxt = random.choice(neighbors)
                passages.add(_edge((cx, cy, cz), nxt))
                visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()

        cells = list(visited)
        extra_links = max(2, len(cells) // 6)
        for _ in range(extra_links):
            cx, cy, cz = random.choice(cells)
            random.shuffle(dirs)
            for dx, dy, dz in dirs:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz and (nx, ny, nz) in visited:
                    passages.add(_edge((cx, cy, cz), (nx, ny, nz)))
                    break

        lx = gx * 2 + 1
        ly = gy * 2 + 1
        lz = gz * 2 + 1
        carved: set[tuple[int, int, int]] = set()

        for cx, cy, cz in visited:
            carved.add((cx * 2 + 1, cy * 2 + 1, cz * 2 + 1))
        for a, b in passages:
            ax, ay, az = a
            bx, by, bz = b
            carved.add((ax + bx + 1, ay + by + 1, az + bz + 1))

        max_drill_radius = max(1, int(getattr(self, "subtractive_drill_max_radius", 4)))
        carved_expanded: set[tuple[int, int, int]] = set()
        for ix, iy, iz in carved:
            hash_seed = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
            drill_radius = 1 + (abs(hash_seed) % max_drill_radius)
            for dx in range(-drill_radius, drill_radius + 1):
                for dy in range(-drill_radius, drill_radius + 1):
                    nx = ix + dx
                    ny = iy + dy
                    nz = iz
                    if 0 <= nx < lx and 0 <= ny < ly and 0 <= nz < lz:
                        carved_expanded.add((nx, ny, nz))
        shell = 2
        carved = {
            (ix, iy, iz)
            for (ix, iy, iz) in carved_expanded
            if (shell <= ix < (lx - shell)) and (shell <= iy < (ly - shell)) and (shell <= iz < (lz - shell))
        }

        cube_step = 6.45
        cube_half = Vec3(cube_step * 0.48, cube_step * 0.48, cube_step * 0.48)
        center = Vec3(self.map_w * 0.5, self.map_d * 0.5, self.floor_y + 7.5)

        def _to_world(ix: int, iy: int, iz: int) -> Vec3:
            return Vec3(
                center.x + (ix - (lx - 1) * 0.5) * cube_step,
                center.y + (iy - (ly - 1) * 0.5) * cube_step,
                center.z + (iz - (lz - 1) * 0.5) * cube_step,
            )

        carved_cells_world: list[Vec3] = []
        for cx, cy, cz in visited:
            carved_cells_world.append(_to_world(cx * 2 + 1, cy * 2 + 1, cz * 2 + 1))

        solid_voxels: set[tuple[int, int, int]] = set()
        for ix in range(lx):
            for iy in range(ly):
                for iz in range(lz):
                    if (ix, iy, iz) not in carved:
                        solid_voxels.add((ix, iy, iz))

        consumed: set[tuple[int, int, int]] = set()
        merged_blocks: list[tuple[int, int, int, int, int, int]] = []

        for iz in range(lz):
            for iy in range(ly):
                for ix in range(lx):
                    seed_cell = (ix, iy, iz)
                    if seed_cell not in solid_voxels or seed_cell in consumed:
                        continue

                    x1 = ix
                    while x1 + 1 < lx:
                        nxt = (x1 + 1, iy, iz)
                        if nxt not in solid_voxels or nxt in consumed:
                            break
                        x1 += 1

                    y1 = iy
                    while y1 + 1 < ly:
                        ok = True
                        yy = y1 + 1
                        for xx in range(ix, x1 + 1):
                            c = (xx, yy, iz)
                            if c not in solid_voxels or c in consumed:
                                ok = False
                                break
                        if not ok:
                            break
                        y1 += 1

                    z1 = iz
                    while z1 + 1 < lz:
                        ok = True
                        zz = z1 + 1
                        for yy in range(iy, y1 + 1):
                            for xx in range(ix, x1 + 1):
                                c = (xx, yy, zz)
                                if c not in solid_voxels or c in consumed:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                        z1 += 1

                    for zz in range(iz, z1 + 1):
                        for yy in range(iy, y1 + 1):
                            for xx in range(ix, x1 + 1):
                                consumed.add((xx, yy, zz))

                    merged_blocks.append((ix, x1, iy, y1, iz, z1))

        solid_count = 0
        for ix0, ix1, iy0, iy1, iz0, iz1 in merged_blocks:
            cx = center.x + (((ix0 + ix1) * 0.5) - (lx - 1) * 0.5) * cube_step
            cy = center.y + (((iy0 + iy1) * 0.5) - (ly - 1) * 0.5) * cube_step
            cz = center.z + (((iz0 + iz1) * 0.5) - (lz - 1) * 0.5) * cube_step
            hx = max(0.06, (ix1 - ix0 + 1) * cube_step * 0.5 * 0.98)
            hy = max(0.06, (iy1 - iy0 + 1) * cube_step * 0.5 * 0.98)
            hz = max(0.06, (iz1 - iz0 + 1) * cube_step * 0.5 * 0.98)

            z_idx = ((iz0 + iz1) * 0.5)
            z_norm = 0.0 if lz <= 1 else max(0.0, min(1.0, z_idx / float(lz - 1)))
            gradient_scale = 0.78 + 0.44 * z_norm
            hx *= gradient_scale
            hy *= gradient_scale
            hz *= gradient_scale

            col = (
                0.26 + 0.08 * ((ix0 + iy0) % 2),
                0.31 + 0.07 * ((iy0 + iz0) % 2),
                0.38 + 0.08 * ((ix0 + iz0) % 2),
                1.0,
            )
            mover = None
            if (ix1 - ix0) <= 0 and (iy1 - iy0) <= 0 and (iz1 - iz0) <= 0:
                mover = "platform" if random.random() < 0.06 and 0 < iz0 < (lz - 1) else None
            self._add_box(Vec3(cx, cy, cz), Vec3(hx, hy, hz), color=col, motion_group=mover, surface_mode="wall")
            solid_count += 1

        spawn_world = _to_world(start[0] * 2 + 1, start[1] * 2 + 1, start[2] * 2 + 1)
        spawn_r = float(getattr(self, "ball_radius", 0.68))
        self.platform_course_spawn_pos = spawn_world + Vec3(0, 0, spawn_r + 0.18)

        low_z = center.z - (lz * 0.5) * cube_step - 1.8
        high_z = center.z + (lz * 0.5) * cube_step + 1.8
        self.hyper_bounds_bottom_z = low_z
        self.hyper_bounds_top_z = high_z

        maze_half_x = lx * cube_step * 0.5
        maze_half_y = ly * cube_step * 0.5
        maze_half_z = max(2.0, (high_z - low_z) * 0.5 + cube_step)
        wall_t = max(0.6, cube_step * 0.45)
        wall_center_z = (high_z + low_z) * 0.5
        self._add_static_box_collider(
            Vec3(center.x - maze_half_x - wall_t, center.y, wall_center_z),
            Vec3(wall_t, maze_half_y + wall_t, maze_half_z),
        )
        self._add_static_box_collider(
            Vec3(center.x + maze_half_x + wall_t, center.y, wall_center_z),
            Vec3(wall_t, maze_half_y + wall_t, maze_half_z),
        )
        self._add_static_box_collider(
            Vec3(center.x, center.y - maze_half_y - wall_t, wall_center_z),
            Vec3(maze_half_x + wall_t, wall_t, maze_half_z),
        )
        self._add_static_box_collider(
            Vec3(center.x, center.y + maze_half_y + wall_t, wall_center_z),
            Vec3(maze_half_x + wall_t, wall_t, maze_half_z),
        )
        self._add_static_box_collider(
            Vec3(center.x, center.y, low_z - wall_t),
            Vec3(maze_half_x + wall_t, maze_half_y + wall_t, wall_t),
        )

        if carved_cells_world:
            shuffled = list(carved_cells_world)
            random.shuffle(shuffled)
            portal_count = min(8, max(4, len(shuffled) // 6))
            self.maze_portal_points = shuffled[:portal_count]

        print(f"[maze] subtractive cube maze built: {solid_count} merged solids (drill 3..{max_drill_radius * 2 + 1}), {len(visited)} rooms, {len(passages)} hall links")

    def _build_four_d_obstacle_arena(self) -> None:
        self.infinite_level_mode = True
        self.allow_outside_island_warps = False
        self.arena_platform_points = []
        margin = 8.0
        inner_w = self.map_w - margin * 2.0
        inner_h = self.map_d - margin * 2.0
        half_w = inner_w * 0.5
        half_h = inner_h * 0.5
        self.rooms = [
            Room(margin, margin, half_w, half_h),
            Room(margin + half_w, margin, half_w, half_h),
            Room(margin, margin + half_h, half_w, half_h),
            Room(margin + half_w, margin + half_h, half_w, half_h),
        ]
        self.edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        self.room_levels = {0: 0, 1: 0, 2: 0, 3: 0}
        self.start_room_idx = 0
        self.room_doors = {idx: {"left": [], "right": [], "bottom": [], "top": []} for idx in range(len(self.rooms))}
        self._setup_room_gravity_zones()
        self._setup_room_dimension_fields()
        self._setup_room_compression_pockets()
        self.room_bounds_cache = None
        self.zone_bounds_cache = None
        self._rebuild_spatial_acceleration_caches()

        loop_half = max(14.0, float(self.platform_loop_range) * 0.85)
        self.hyper_bounds_bottom_z = self.floor_y - loop_half
        self.hyper_bounds_top_z = self.floor_y + loop_half
        center = Vec3(self.map_w * 0.5, self.map_d * 0.5, self.floor_y + 2.0)
        self.platform_course_spawn_pos = Vec3(center.x, center.y, self.floor_y + 6.1)

        self._build_floor_and_bounds()

        hub_half = Vec3(4.6, 4.6, 0.55)
        self._add_box(
            center + Vec3(0, 0, 2.6),
            hub_half,
            color=(0.28, 0.46, 0.68, 1.0),
            motion_group="platform",
            w_coord=0.0,
            surface_mode="floor",
        )
        self.arena_platform_points.append(Vec3(center.x, center.y, center.z + 3.5))

        ring_count = 4 if self.performance_mode else 5
        seg_base = 8 if self.performance_mode else 10
        arena_radius_max = max(8.0, min(inner_w, inner_h) * 0.44)
        arena_radius_min = max(6.0, arena_radius_max * 0.32)
        for ring in range(ring_count):
            ring_t = ring / max(1, ring_count - 1)
            radius = arena_radius_min + (arena_radius_max - arena_radius_min) * ring_t
            segments = seg_base + ring * 2
            z_base = self.floor_y + 2.4 + ring * 2.1
            for seg in range(segments):
                ang = (math.tau * seg) / max(1, segments)
                arc_jitter = random.uniform(-0.22, 0.22)
                x = center.x + math.cos(ang + arc_jitter) * radius
                y = center.y + math.sin(ang + arc_jitter) * radius
                z = z_base + math.sin(ang * 2.0 + ring_t * math.pi) * 1.1
                hx = random.uniform(1.2, 2.0)
                hy = random.uniform(1.2, 2.0)
                hz = random.uniform(0.24, 0.46)
                w_coord = self._clamp(
                    math.sin(ang * 2.0 + ring * 0.73) * self.hyper_w_limit * (0.35 + 0.52 * ring_t),
                    -self.hyper_w_limit,
                    self.hyper_w_limit,
                )
                moving = (seg % 3 == 0) and (random.random() < 0.5)
                color = (
                    0.3 + 0.24 * ring_t,
                    0.42 + 0.32 * (0.5 + 0.5 * math.sin(ang + ring_t)),
                    0.7 + 0.24 * (0.5 + 0.5 * math.cos(ang * 1.4)),
                    1.0,
                )
                self._add_box(
                    Vec3(x, y, z),
                    Vec3(hx, hy, hz),
                    color=color,
                    motion_group=("platform" if moving else None),
                    w_coord=w_coord,
                    surface_mode="floor",
                )
                self.arena_platform_points.append(Vec3(x, y, z + hz + 0.35))

                if seg % 2 == 0:
                    pillar_h = random.uniform(0.9, 2.2)
                    self._add_box(
                        Vec3(x, y, z + hz + pillar_h * 0.5),
                        Vec3(0.22, 0.22, pillar_h * 0.5),
                        color=(0.22, 0.66, 0.92, 1.0),
                        w_coord=w_coord,
                        surface_mode="floor",
                    )

        self.liminal_fold_nodes.clear()
        self.liminal_fold_links.clear()
        fold_nodes = 8
        fold_radius = 19.0
        for idx in range(fold_nodes):
            ang = (math.tau * idx) / fold_nodes
            npos = Vec3(center.x + math.cos(ang) * fold_radius, center.y + math.sin(ang) * fold_radius, self.floor_y + 4.0 + math.sin(ang * 2.0) * 1.2)
            nw = self._clamp(math.sin(ang * 1.5) * self.hyper_w_limit * 0.86, -self.hyper_w_limit, self.hyper_w_limit)
            self.liminal_fold_nodes.append({"pos": npos, "w": nw, "room_idx": idx % len(self.rooms)})
            self.liminal_fold_links[idx] = []

        for idx in range(fold_nodes):
            nxt = (idx + 1) % fold_nodes
            cross = (idx + (fold_nodes // 2)) % fold_nodes
            self.liminal_fold_links[idx].append(nxt)
            self.liminal_fold_links[nxt].append(idx)
            self.liminal_fold_links[idx].append(cross)
            self.liminal_fold_links[cross].append(idx)

    def _build_platform_only_dungeon(self) -> None:
        self.infinite_level_mode = True
        loop_half = max(10.0, float(self.platform_loop_range) * 0.5)
        self.hyper_bounds_bottom_z = self.floor_y - loop_half
        self.hyper_bounds_top_z = self.floor_y + loop_half
        self.platform_course_spawn_pos = Vec3(self.map_w * 0.5, self.map_d * 0.5, self.hyper_bounds_top_z - 1.8)

        t = 0.4
        edge_h = max(self.wall_h + 10.0, self.platform_loop_range)
        edge_center_z = (self.hyper_bounds_top_z + self.hyper_bounds_bottom_z) * 0.5
        edge_half_h = edge_h * 0.5
        self._add_static_box_collider(
            Vec3(self.map_w * 0.5, -t, edge_center_z),
            Vec3(self.map_w * 0.5 + 1.0, t + 0.25, edge_half_h),
        )
        self._add_static_box_collider(
            Vec3(self.map_w * 0.5, self.map_d + t, edge_center_z),
            Vec3(self.map_w * 0.5 + 1.0, t + 0.25, edge_half_h),
        )
        self._add_static_box_collider(
            Vec3(-t, self.map_d * 0.5, edge_center_z),
            Vec3(t + 0.25, self.map_d * 0.5 + 1.0, edge_half_h),
        )
        self._add_static_box_collider(
            Vec3(self.map_w + t, self.map_d * 0.5, edge_center_z),
            Vec3(t + 0.25, self.map_d * 0.5 + 1.0, edge_half_h),
        )

        placed: list[tuple[float, float, float, float, float, float]] = []

        start_half = Vec3(4.2, 4.2, 0.52)
        start_pos = Vec3(self.platform_course_spawn_pos.x, self.platform_course_spawn_pos.y, self.platform_course_spawn_pos.z - 0.9)
        self._add_box(start_pos, start_half, color=(0.28, 0.46, 0.68, 1.0), motion_group="platform")
        placed.append((start_pos.x - start_half.x, start_pos.x + start_half.x, start_pos.y - start_half.y, start_pos.y + start_half.y, start_pos.z - start_half.z, start_pos.z + start_half.z))

        count = max(24, int(self.platform_course_count))
        pad = max(0.2, float(self.platform_overlap_padding))
        min_z = self.hyper_bounds_bottom_z + 1.2
        max_z = self.hyper_bounds_top_z - 1.4
        for _ in range(count):
            success = False
            for _try in range(24):
                hx = random.uniform(self.platform_min_span * 0.5, self.platform_max_span * 0.5)
                hy = random.uniform(self.platform_min_span * 0.5, self.platform_max_span * 0.5)
                hz = random.uniform(0.2, 0.56)
                cx = random.uniform(1.8 + hx, self.map_w - 1.8 - hx)
                cy = random.uniform(1.8 + hy, self.map_d - 1.8 - hy)
                zq = random.randint(0, max(1, int(self.platform_loop_range / max(0.5, self.platform_vertical_step))))
                cz = min(max_z, max(min_z, self.hyper_bounds_bottom_z + 1.0 + zq * self.platform_vertical_step + random.uniform(-0.35, 0.35)))

                x0, x1 = cx - hx - pad, cx + hx + pad
                y0, y1 = cy - hy - pad, cy + hy + pad
                z0, z1 = cz - hz - 0.5, cz + hz + 0.5

                overlap = False
                for ax0, ax1, ay0, ay1, az0, az1 in placed:
                    if x0 <= ax1 and x1 >= ax0 and y0 <= ay1 and y1 >= ay0 and z0 <= az1 and z1 >= az0:
                        overlap = True
                        break
                if overlap:
                    continue

                moving = random.random() < self.platform_mover_ratio
                color = (
                    random.uniform(0.34, 0.72),
                    random.uniform(0.44, 0.84),
                    random.uniform(0.6, 0.98),
                    1.0,
                )
                self._add_box(Vec3(cx, cy, cz), Vec3(hx, hy, hz), color=color, motion_group=("platform" if moving else None))

                rail_h = max(0.16, float(self.platform_guardrail_height))
                rail_t = max(0.06, float(self.platform_guardrail_thickness))
                post_z = cz + hz + rail_h * 0.5
                post_scale = Vec3(rail_t, rail_t, rail_h * 0.5)
                self._add_box(Vec3(cx - hx + rail_t, cy - hy + rail_t, post_z), post_scale, color=(0.25, 0.7, 0.92, 0.72), collidable=False)
                self._add_box(Vec3(cx + hx - rail_t, cy - hy + rail_t, post_z), post_scale, color=(0.25, 0.7, 0.92, 0.72), collidable=False)
                self._add_box(Vec3(cx - hx + rail_t, cy + hy - rail_t, post_z), post_scale, color=(0.25, 0.7, 0.92, 0.72), collidable=False)
                self._add_box(Vec3(cx + hx - rail_t, cy + hy - rail_t, post_z), post_scale, color=(0.25, 0.7, 0.92, 0.72), collidable=False)

                placed.append((cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz))
                success = True
                break

            if not success:
                continue

    def _build_floor_and_bounds(self) -> None:
        if getattr(self, "four_d_obstacle_arena_mode", False):
            overscan = max(0.0, float(getattr(self, "water_loop_overscan", 6.0)))
            water_half_x = self.map_w * 0.5 + overscan
            water_half_y = self.map_d * 0.5 + overscan
            water_half_z = max(0.08, self.floor_t * 0.32)
            water_center_z = self.floor_y + self.water_surface_raise
            water_holder = self._add_box(
                Vec3(self.map_w * 0.5, self.map_d * 0.5, water_center_z),
                Vec3(water_half_x, water_half_y, water_half_z),
                color=(0.26, 0.52, 0.78, 0.24),
                collidable=False,
                surface_mode="water",
            )
            if water_holder is not None and not water_holder.isEmpty():
                water_holder.setTransparency(TransparencyAttrib.MAlpha)
                water_holder.setDepthWrite(False)
                water_holder.setBin("transparent", 33)
                self._register_water_surface(
                    water_holder,
                    -overscan,
                    float(self.map_w) + overscan,
                    -overscan,
                    float(self.map_d) + overscan,
                    float(water_center_z),
                )
        else:
            floor_scale = Vec3(self.map_w / 2, self.map_d / 2, self.floor_t)
            floor_pos = Vec3(self.map_w / 2, self.map_d / 2, self.floor_y - self.floor_t - self.water_level_offset)
            floor_holder = self._add_box(floor_pos, floor_scale, color=(0.14, 0.16, 0.2, 1), surface_mode="wall")
            self.hyper_enclosure_ids.add(id(floor_holder))

        t = 0.4
        h = self.wall_h + 4
        if not getattr(self, "four_d_obstacle_arena_mode", False):
            bound_0 = self._add_box(Vec3(self.map_w / 2, -t, h / 2), Vec3(self.map_w / 2, t, h / 2), color=(0.2, 0.24, 0.3, 1), collidable=False)
            bound_1 = self._add_box(Vec3(self.map_w / 2, self.map_d + t, h / 2), Vec3(self.map_w / 2, t, h / 2), color=(0.2, 0.24, 0.3, 1), collidable=False)
            bound_2 = self._add_box(Vec3(-t, self.map_d / 2, h / 2), Vec3(t, self.map_d / 2, h / 2), color=(0.2, 0.24, 0.3, 1), collidable=False)
            bound_3 = self._add_box(Vec3(self.map_w + t, self.map_d / 2, h / 2), Vec3(t, self.map_d / 2, h / 2), color=(0.2, 0.24, 0.3, 1), collidable=False)
            self.hyper_enclosure_ids.update({id(bound_0), id(bound_1), id(bound_2), id(bound_3)})

            boundary_half_h = max(self.wall_h + 14.0, self.level_z_step * 3.0)
            boundary_center_z = self.floor_y + boundary_half_h - 2.0
            edge_thickness = t * 0.5 + 0.35
            self._add_static_box_collider(
                Vec3(self.map_w * 0.5, -t, boundary_center_z),
                Vec3(self.map_w * 0.5 + 1.0, edge_thickness, boundary_half_h),
            )
            self._add_static_box_collider(
                Vec3(self.map_w * 0.5, self.map_d + t, boundary_center_z),
                Vec3(self.map_w * 0.5 + 1.0, edge_thickness, boundary_half_h),
            )
            self._add_static_box_collider(
                Vec3(-t, self.map_d * 0.5, boundary_center_z),
                Vec3(edge_thickness, self.map_d * 0.5 + 1.0, boundary_half_h),
            )
            self._add_static_box_collider(
                Vec3(self.map_w + t, self.map_d * 0.5, boundary_center_z),
                Vec3(edge_thickness, self.map_d * 0.5 + 1.0, boundary_half_h),
            )

        self._add_static_box_collider(
            Vec3(self.map_w * 0.5, self.map_d * 0.5, self.hyper_bounds_top_z + 0.11),
            Vec3(self.map_w * 0.5, self.map_d * 0.5, 0.11),
        )
        self._add_static_box_collider(
            Vec3(self.map_w * 0.5, self.map_d * 0.5, self.hyper_bounds_bottom_z - 0.11),
            Vec3(self.map_w * 0.5, self.map_d * 0.5, 0.11),
        )

    def _build_room(self, room_idx: int, room: Room, level: int) -> None:
        base_z = self._level_base_z(level)

        wt = self.wall_t
        wx = room.x
        wy = room.y
        ww = room.w
        wh = room.h
        wall_color = (0.78, 0.82, 0.88, 1)

        room_plane_scale = Vec3(ww / 2 + wt / 2, wh / 2 + wt / 2, self.floor_t / 2)

        floor_pos = Vec3(wx + ww / 2, wy + wh / 2, self._floor_input_z(base_z))
        floor_scale = Vec3(room_plane_scale)
        self._queue_floor_rect(floor_pos, floor_scale, color=(0.19, 0.22, 0.28, 1))

        ceiling_pos = Vec3(wx + ww / 2, wy + wh / 2, self._ceiling_input_z(base_z))
        ceiling_scale = Vec3(room_plane_scale)
        self._queue_ceiling_rect(ceiling_pos, ceiling_scale, color=(0.9, 0.93, 0.98, 1))

        self._build_room_walls_with_doors(room_idx, room, base_z, wall_color)
        if room.w >= 14 and room.h >= 14 and random.random() < self.gen.angled_room_ratio:
            self._add_angled_room_walls(room_idx, room, base_z)

        self._add_pillars(room_idx, room, base_z)
        self._add_room_obstacles(room_idx, room, base_z)
        self._add_wall_decor(room_idx, room, base_z)
        if random.random() > 0.45:
            self._add_dome(room, base_z)
        self._add_room_hyper_layers(room, base_z)

    def _add_room_hyper_layers(self, room: Room, base_z: float) -> None:
        center = Vec3(room.x + room.w * 0.5, room.y + room.h * 0.5, base_z + self.wall_h * 0.5)
        base_w = self._compute_level_w(center)
        count = max(1, self.room_hyper_layer_count)
        mid = (count - 1) * 0.5
        inner_x_min = room.x + self.wall_t * 0.5
        inner_x_max = room.x + room.w - self.wall_t * 0.5
        inner_y_min = room.y + self.wall_t * 0.5
        inner_y_max = room.y + room.h - self.wall_t * 0.5

        for layer_idx in range(count):
            layer_delta = layer_idx - mid
            layer_w = base_w + layer_delta * self.room_hyper_w_spacing
            layer_abs = abs(layer_delta)

            inset = 0.14 + layer_abs * 0.22
            half_x = max(0.12, (inner_x_max - inner_x_min) * 0.5 - inset)
            half_y = max(0.12, (inner_y_max - inner_y_min) * 0.5 - inset)
            z = base_z + self.wall_h * (0.24 + 0.21 * (layer_idx / max(1.0, count - 1)))
            thickness = max(0.02, self.wall_t * 0.2)

            y_low = inner_y_min + inset + thickness * 0.5
            y_high = inner_y_max - inset - thickness * 0.5
            x_low = inner_x_min + inset + thickness * 0.5
            x_high = inner_x_max - inset - thickness * 0.5

            if y_high <= y_low or x_high <= x_low:
                continue

            shell_color = (
                0.55 + 0.18 * math.sin(layer_w * 0.7),
                0.63 + 0.12 * math.cos(layer_w * 0.5),
                0.92,
                1.0,
            )
            beam_color = (0.22, 0.95, 1.0, 1.0)
            room_shell_color = (
                0.33 + 0.14 * math.sin(layer_w * 0.43 + 0.7),
                0.44 + 0.13 * math.cos(layer_w * 0.37),
                0.86,
                1.0,
            )

            room_half_x = max(0.22, half_x * 0.98)
            room_half_y = max(0.22, half_y * 0.98)
            floor_thickness = max(0.03, thickness * 0.9)
            wall_thickness = max(0.04, thickness * 1.15)
            wall_half_z = max(0.22, self.wall_h * 0.34)
            room_floor_z = base_z + floor_thickness
            room_ceil_z = base_z + self.wall_h - floor_thickness

            canonical_floor_z = self._floor_input_z(base_z)
            canonical_ceil_z = self._ceiling_input_z(base_z)
            floor_coplanar = abs(room_floor_z - canonical_floor_z) <= max(0.03, self.floor_t * 0.42)
            ceil_coplanar = abs(room_ceil_z - canonical_ceil_z) <= max(0.03, self.floor_t * 0.42)

            if floor_coplanar:
                self._queue_floor_rect(
                    Vec3(center.x, center.y, canonical_floor_z),
                    Vec3(room_half_x, room_half_y, self.floor_t * 0.5),
                    color=room_shell_color,
                )
            else:
                self._add_decor_box(
                    Vec3(center.x, center.y, room_floor_z),
                    Vec3(room_half_x, room_half_y, floor_thickness),
                    color=room_shell_color,
                    w_coord=layer_w,
                )

            if ceil_coplanar:
                self._queue_ceiling_rect(
                    Vec3(center.x, center.y, canonical_ceil_z),
                    Vec3(room_half_x, room_half_y, self.floor_t * 0.5),
                    color=room_shell_color,
                )
            else:
                self._add_decor_box(
                    Vec3(center.x, center.y, room_ceil_z),
                    Vec3(room_half_x, room_half_y, floor_thickness),
                    color=room_shell_color,
                    w_coord=layer_w,
                )
            self._add_decor_box(
                Vec3(center.x - room_half_x, center.y, center.z),
                Vec3(wall_thickness, room_half_y, wall_half_z),
                color=room_shell_color,
                w_coord=layer_w,
            )
            self._add_decor_box(
                Vec3(center.x + room_half_x, center.y, center.z),
                Vec3(wall_thickness, room_half_y, wall_half_z),
                color=room_shell_color,
                w_coord=layer_w,
            )
            self._add_decor_box(
                Vec3(center.x, center.y - room_half_y, center.z),
                Vec3(room_half_x, wall_thickness, wall_half_z),
                color=room_shell_color,
                w_coord=layer_w,
            )
            self._add_decor_box(
                Vec3(center.x, center.y + room_half_y, center.z),
                Vec3(room_half_x, wall_thickness, wall_half_z),
                color=room_shell_color,
                w_coord=layer_w,
            )

            self._add_decor_box(
                Vec3(center.x, y_low, z),
                Vec3(half_x, thickness, thickness),
                color=shell_color,
                w_coord=layer_w,
            )
            self._add_decor_box(
                Vec3(center.x, y_high, z),
                Vec3(half_x, thickness, thickness),
                color=shell_color,
                w_coord=layer_w,
            )
            self._add_decor_box(
                Vec3(x_low, center.y, z),
                Vec3(thickness, half_y, thickness),
                color=shell_color,
                w_coord=layer_w,
            )
            self._add_decor_box(
                Vec3(x_high, center.y, z),
                Vec3(thickness, half_y, thickness),
                color=shell_color,
                w_coord=layer_w,
            )

            if layer_idx % 2 == 0:
                self._add_decor_box(
                    Vec3(center.x, center.y, z + 0.08),
                    Vec3(half_x * 0.8, thickness * 0.8, thickness * 0.8),
                    color=beam_color,
                    hpr=Vec3(45, 0, 0),
                    w_coord=layer_w,
                )
            else:
                self._add_decor_box(
                    Vec3(center.x, center.y, z + 0.08),
                    Vec3(half_y * 0.8, thickness * 0.8, thickness * 0.8),
                    color=beam_color,
                    hpr=Vec3(-45, 0, 0),
                    w_coord=layer_w,
                )

    def _add_angled_room_walls(self, room_idx: int, room: Room, base_z: float) -> None:
        cut = min(room.w, room.h) * random.uniform(0.18, 0.25)
        thickness = self.wall_t * 0.76
        half_len = cut * 0.3
        z = base_z + self.wall_h * 0.5
        col = (0.7, 0.75, 0.83, 1)
        clearance = max(2.4, self.corridor_w * 0.95)

        corner_specs = [
            {
                "name": "bottom_left",
                "center": (room.x + cut * 0.58, room.y + cut * 0.58),
                "angle": 45,
                "sides": [("bottom", room.x), ("left", room.y)],
            },
            {
                "name": "bottom_right",
                "center": (room.x + room.w - cut * 0.58, room.y + cut * 0.58),
                "angle": -45,
                "sides": [("bottom", room.x + room.w), ("right", room.y)],
            },
            {
                "name": "top_left",
                "center": (room.x + cut * 0.58, room.y + room.h - cut * 0.58),
                "angle": -45,
                "sides": [("top", room.x), ("left", room.y + room.h)],
            },
            {
                "name": "top_right",
                "center": (room.x + room.w - cut * 0.58, room.y + room.h - cut * 0.58),
                "angle": 45,
                "sides": [("top", room.x + room.w), ("right", room.y + room.h)],
            },
        ]

        valid_specs = []
        for spec in corner_specs:
            ok = True
            for side, corner_coord in spec["sides"]:
                for door_pos in self.room_doors[room_idx][side]:
                    if abs(door_pos - corner_coord) < clearance:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid_specs.append(spec)

        if not valid_specs:
            return

        random.shuffle(valid_specs)
        use_count = min(len(valid_specs), random.choice([1, 2, 2, 3]))
        for spec in valid_specs[:use_count]:
            cx, cy = spec["center"]
            ang = spec["angle"]
            self._add_box(
                Vec3(cx, cy, z),
                Vec3(half_len, thickness * 0.5, self.wall_h * 0.5),
                color=col,
                hpr=Vec3(ang, 0, 0),
            )
            self._add_angled_wall_decor(cx, cy, z, half_len, thickness, ang)

    def _add_angled_wall_decor(self, cx: float, cy: float, z: float, half_len: float, thickness: float, angle: float) -> None:
        decor_color = (0.24, 0.96, 1.0, 1)
        strip_half = half_len * 0.5
        strip_thickness = max(0.012, min(0.022, thickness * 0.16))
        strip_height = max(0.022, self.wall_h * 0.018)
        decor_w = self._compute_level_w(Vec3(cx, cy, z))

        for h_off in (-self.wall_h * 0.2, self.wall_h * 0.18):
            self._add_decor_box(
                Vec3(cx, cy, z + h_off),
                Vec3(strip_half, strip_thickness, strip_height),
                color=decor_color,
                hpr=Vec3(angle, 0, 0),
                w_coord=decor_w,
            )

    def _door_opening_width(self, room: Room) -> float:
        room_min = max(2.2, min(room.w, room.h))
        max_open = max(1.6, room_min - 2.0)
        target = self.corridor_w + self.wall_t * 1.1
        return max(1.6, min(max_open, target))

    def _effective_entry_opening_width(self, room: Room) -> float:
        base = self._door_opening_width(room)
        ball_r = float(getattr(self, "ball_radius", 0.32))
        clearance = max(0.28, self.wall_t * 1.4, ball_r * 1.6)
        room_min = max(2.2, min(room.w, room.h))
        max_open = max(1.8, room_min - 1.45)
        return min(max_open, base + clearance)

    def _is_in_room_entry_lane(self, room_idx: int, room: Room, x: float, y: float, extra: float = 0.0) -> bool:
        doors = self.room_doors.get(room_idx)
        if not doors:
            return False

        door_w = self._effective_entry_opening_width(room)
        lane_half = max(door_w * 0.56, self.corridor_w * 0.58) + extra
        inset = max(1.05, self.corridor_w * 0.42) + extra

        if x <= room.x + inset:
            for door_pos in doors["left"]:
                if abs(y - door_pos) <= lane_half:
                    return True
        if x >= room.x + room.w - inset:
            for door_pos in doors["right"]:
                if abs(y - door_pos) <= lane_half:
                    return True
        if y <= room.y + inset:
            for door_pos in doors["bottom"]:
                if abs(x - door_pos) <= lane_half:
                    return True
        if y >= room.y + room.h - inset:
            for door_pos in doors["top"]:
                if abs(x - door_pos) <= lane_half:
                    return True

        return False

    def _add_pillars(self, room_idx: int, room: Room, base_z: float) -> None:
        px1 = room.x + room.w * 0.25
        px2 = room.x + room.w * 0.75
        py1 = room.y + room.h * 0.25
        py2 = room.y + room.h * 0.75

        for px in [px1, px2]:
            for py in [py1, py2]:
                if self._is_in_room_entry_lane(room_idx, room, px, py, extra=0.42):
                    continue
                self._add_box(
                    Vec3(px, py, base_z + self.wall_h * 0.5),
                    Vec3(0.3, 0.3, self.wall_h * 0.5),
                    color=(0.66, 0.71, 0.79, 1),
                )
                self._add_decor_box(
                    Vec3(px, py, base_z + self.wall_h * 0.58),
                    Vec3(0.34, 0.34, 0.04),
                    color=(0.2, 0.95, 1.0, 1),
                )

    def _add_room_obstacles(self, room_idx: int, room: Room, base_z: float) -> None:
        if room.w < 10.0 or room.h < 10.0:
            return

        room_area = room.w * room.h
        base_density = 0.08 + self.gen.decor_density * 0.18
        target_count = max(5, int(room_area * base_density / 50.0))
        max_count = max(8, int(room_area / 22.0))
        target_count = min(target_count, max_count)

        center_x = room.x + room.w * 0.5
        center_y = room.y + room.h * 0.5
        center_clear = max(2.6, min(room.w, room.h) * 0.11)
        margin = max(1.2, self.corridor_w * 0.14)

        placed = 0
        attempts = target_count * 10
        while placed < target_count and attempts > 0:
            attempts -= 1
            ox = random.uniform(room.x + margin, room.x + room.w - margin)
            oy = random.uniform(room.y + margin, room.y + room.h - margin)

            if (ox - center_x) ** 2 + (oy - center_y) ** 2 < center_clear * center_clear:
                continue

            obstacle_color = (
                random.uniform(0.56, 0.78),
                random.uniform(0.62, 0.84),
                random.uniform(0.72, 0.92),
                1.0,
            )

            if random.random() < 0.42:
                sx = random.uniform(0.22, 0.46)
                sy = random.uniform(0.22, 0.46)
                sz = self.wall_h * random.uniform(0.28, 0.62)
            else:
                sx = random.uniform(0.36, 1.1)
                sy = random.uniform(0.36, 1.1)
                sz = self.wall_h * random.uniform(0.08, 0.24)

            footprint = max(sx, sy)
            if self._is_in_room_entry_lane(room_idx, room, ox, oy, extra=footprint + 0.22):
                continue

            self._add_box(
                Vec3(ox, oy, base_z + sz),
                Vec3(sx, sy, sz),
                color=obstacle_color,
            )

            if random.random() < 0.55:
                cap_h = 0.04
                self._add_decor_box(
                    Vec3(ox, oy, base_z + sz * 2.0 + cap_h + 0.002),
                    Vec3(max(0.12, sx * 0.72), max(0.12, sy * 0.72), cap_h),
                    color=(0.2, 0.96, 1.0, 1.0),
                )

            placed += 1

    def _add_dome(self, room: Room, base_z: float) -> None:
        dome = self.sphere_model.copyTo(self.world)
        dome.setScale(room.w * 0.18, room.h * 0.18, 0.9)
        dome_pos = Vec3(room.x + room.w / 2, room.y + room.h / 2, base_z + self.wall_h - 0.05)
        dome.setPos(dome_pos)
        dome.setColor(0.82, 0.88, 0.97, 1)
        dome.clearTexture()
        dome.setTexture(self.level_checker_tex, 1)
        self._apply_hypercube_projection(dome, self._compute_level_w(dome_pos), scale_hint=2.6)
        self._register_color_cycle(dome, (0.82, 0.88, 0.97, 1.0), min_speed=0.04, max_speed=0.1)
        self._register_scene_visual(dome, self._compute_level_w(dome_pos))

    def _add_wall_decor(self, room_idx: int, room: Room, base_z: float) -> None:
        trigger = min(0.97, self.gen.decor_density * 1.22)
        if random.random() > trigger:
            return
        self._add_wall_panels(room_idx, room, base_z)
        self._add_light_strips(room_idx, room, base_z)
        if random.random() < min(0.9, 0.36 + self.gen.decor_density * 0.9):
            self._add_qr_aztec_wall_decor(room_idx, room, base_z)
        if random.random() < min(0.85, self.gen.decor_density * 0.72):
            self._add_light_strips(room_idx, room, base_z + self.wall_h * 0.03)
        if random.random() < min(0.75, self.gen.decor_density * 0.56):
            self._add_wall_panels(room_idx, room, base_z + self.wall_h * 0.02)

    def _add_qr_aztec_wall_decor(self, room_idx: int, room: Room, base_z: float) -> None:
        door_w = self._effective_entry_opening_width(room)
        wall_pad = 0.24
        depth = min(0.018, self.wall_t * 0.22)

        bottom_ranges = self._get_wall_solid_ranges(room.x + 0.34, room.x + room.w - 0.34, self.room_doors[room_idx]["bottom"], door_w, wall_pad)
        top_ranges = self._get_wall_solid_ranges(room.x + 0.34, room.x + room.w - 0.34, self.room_doors[room_idx]["top"], door_w, wall_pad)
        left_ranges = self._get_wall_solid_ranges(room.y + 0.34, room.y + room.h - 0.34, self.room_doors[room_idx]["left"], door_w, wall_pad)
        right_ranges = self._get_wall_solid_ranges(room.y + 0.34, room.y + room.h - 0.34, self.room_doors[room_idx]["right"], door_w, wall_pad)

        motif_seed = room_idx * 92821 + int(base_z * 127.0)
        for a, b in bottom_ranges:
            seg = b - a
            if seg < 1.28:
                continue
            self._stamp_qr_aztec_segment("bottom", a, b, room.y + depth, base_z, depth, motif_seed + 11)

        for a, b in top_ranges:
            seg = b - a
            if seg < 1.28:
                continue
            self._stamp_qr_aztec_segment("top", a, b, room.y + room.h - depth, base_z, depth, motif_seed + 23)

        for a, b in left_ranges:
            seg = b - a
            if seg < 1.28:
                continue
            self._stamp_qr_aztec_segment("left", a, b, room.x + depth, base_z, depth, motif_seed + 37)

        for a, b in right_ranges:
            seg = b - a
            if seg < 1.28:
                continue
            self._stamp_qr_aztec_segment("right", a, b, room.x + room.w - depth, base_z, depth, motif_seed + 53)

    def _stamp_qr_aztec_segment(self, side: str, a: float, b: float, fixed_axis: float, base_z: float, depth: float, motif_seed: int) -> None:
        seg = b - a
        cols = max(9, min(15, int(seg / 0.32)))
        if cols % 2 == 0:
            cols += 1
        rows = 9
        cell_w = seg / cols
        z_low = base_z + self.wall_h * 0.34
        z_high = base_z + self.wall_h * 0.7
        cell_h = (z_high - z_low) / rows

        def _finder(ix: int, iy: int, ox: int, oy: int, size: int) -> bool:
            dx = ix - ox
            dy = iy - oy
            if dx < 0 or dy < 0 or dx >= size or dy >= size:
                return False
            edge = dx == 0 or dy == 0 or dx == size - 1 or dy == size - 1
            core = dx == size // 2 and dy == size // 2
            return edge or core

        cx = (cols - 1) * 0.5
        cy = (rows - 1) * 0.5

        for ix in range(cols):
            for iy in range(rows):
                finder = (
                    _finder(ix, iy, 0, 0, 5)
                    or _finder(ix, iy, cols - 5, 0, 5)
                    or _finder(ix, iy, 0, rows - 5, 5)
                )
                timing = (iy == rows // 2 and ix % 2 == 0) or (ix == cols // 2 and iy % 2 == 0)
                ring = int(max(abs(ix - cx), abs(iy - cy)))
                aztec_ring = ring in (1, 2, 3) and (((ix + iy + motif_seed) % 2) == 0)
                aztec_steps = abs(ix - cx) + abs(iy - cy) in (3, 5, 7)
                accent_hash = ((ix * 13 + iy * 17 + motif_seed) % 11) in (0, 1)
                active = finder or timing or aztec_ring or aztec_steps or accent_hash
                if not active:
                    continue

                x_or_y = a + (ix + 0.5) * cell_w
                z = z_low + (iy + 0.5) * cell_h
                squash_z = max(0.018, cell_h * (0.2 if finder else 0.16))
                squash_xy = max(0.03, cell_w * (0.46 if finder else 0.38))
                glow = finder or aztec_ring
                color = (0.22, 0.95, 1.0, 1.0) if glow else (0.64, 0.72, 0.84, 1.0)

                if side == "bottom" or side == "top":
                    self._add_decor_box(Vec3(x_or_y, fixed_axis, z), Vec3(squash_xy, depth, squash_z), color=color)
                else:
                    self._add_decor_box(Vec3(fixed_axis, x_or_y, z), Vec3(depth, squash_xy, squash_z), color=color)

    def _get_wall_solid_ranges(self, low: float, high: float, doors: list[float], door_w: float, pad: float) -> list[tuple[float, float]]:
        opens = self._open_intervals(doors, low, high, door_w + pad)
        solid: list[tuple[float, float]] = []
        cursor = low
        for a, b in opens:
            if a > cursor:
                solid.append((cursor, a))
            cursor = max(cursor, b)
        if high > cursor:
            solid.append((cursor, high))
        return solid

    def _add_wall_panels(self, room_idx: int, room: Room, base_z: float) -> None:
        panel_color = (0.75, 0.8, 0.88, 1)
        groove_color = (0.58, 0.64, 0.74, 1)
        panel_depth = min(0.016, self.wall_t * 0.2)
        door_w = self._effective_entry_opening_width(room)
        wall_pad = 0.22

        z_low = base_z + self.wall_h * 0.3
        z_high = base_z + self.wall_h * 0.72
        panel_h = max(0.55, self.wall_h * 0.16)
        groove_h = 0.035

        bottom_ranges = self._get_wall_solid_ranges(room.x + 0.3, room.x + room.w - 0.3, self.room_doors[room_idx]["bottom"], door_w, wall_pad)
        top_ranges = self._get_wall_solid_ranges(room.x + 0.3, room.x + room.w - 0.3, self.room_doors[room_idx]["top"], door_w, wall_pad)
        left_ranges = self._get_wall_solid_ranges(room.y + 0.3, room.y + room.h - 0.3, self.room_doors[room_idx]["left"], door_w, wall_pad)
        right_ranges = self._get_wall_solid_ranges(room.y + 0.3, room.y + room.h - 0.3, self.room_doors[room_idx]["right"], door_w, wall_pad)

        for a, b in bottom_ranges:
            seg = b - a
            if seg < 0.82:
                continue
            cx = (a + b) * 0.5
            self._add_decor_box(Vec3(cx, room.y + panel_depth, z_low), Vec3(seg * 0.42, panel_depth, panel_h), color=panel_color)
            self._add_decor_box(Vec3(cx, room.y + panel_depth, z_high), Vec3(seg * 0.46, panel_depth, groove_h), color=groove_color)

        for a, b in top_ranges:
            seg = b - a
            if seg < 0.82:
                continue
            cx = (a + b) * 0.5
            self._add_decor_box(Vec3(cx, room.y + room.h - panel_depth, z_low), Vec3(seg * 0.42, panel_depth, panel_h), color=panel_color)
            self._add_decor_box(Vec3(cx, room.y + room.h - panel_depth, z_high), Vec3(seg * 0.46, panel_depth, groove_h), color=groove_color)

        for a, b in left_ranges:
            seg = b - a
            if seg < 0.82:
                continue
            cy = (a + b) * 0.5
            self._add_decor_box(Vec3(room.x + panel_depth, cy, z_low), Vec3(panel_depth, seg * 0.42, panel_h), color=panel_color)
            self._add_decor_box(Vec3(room.x + panel_depth, cy, z_high), Vec3(panel_depth, seg * 0.46, groove_h), color=groove_color)

        for a, b in right_ranges:
            seg = b - a
            if seg < 0.82:
                continue
            cy = (a + b) * 0.5
            self._add_decor_box(Vec3(room.x + room.w - panel_depth, cy, z_low), Vec3(panel_depth, seg * 0.42, panel_h), color=panel_color)
            self._add_decor_box(Vec3(room.x + room.w - panel_depth, cy, z_high), Vec3(panel_depth, seg * 0.46, groove_h), color=groove_color)

    def _add_light_strips(self, room_idx: int, room: Room, base_z: float) -> None:
        strip_color = (0.2, 0.94, 1.0, 1)
        strip_depth = min(0.014, self.wall_t * 0.18)
        strip_h = 0.04
        z = base_z + self.wall_h * 0.82
        door_w = self._effective_entry_opening_width(room)
        wall_pad = 0.18

        bottom_ranges = self._get_wall_solid_ranges(room.x + 0.25, room.x + room.w - 0.25, self.room_doors[room_idx]["bottom"], door_w, wall_pad)
        top_ranges = self._get_wall_solid_ranges(room.x + 0.25, room.x + room.w - 0.25, self.room_doors[room_idx]["top"], door_w, wall_pad)
        left_ranges = self._get_wall_solid_ranges(room.y + 0.25, room.y + room.h - 0.25, self.room_doors[room_idx]["left"], door_w, wall_pad)
        right_ranges = self._get_wall_solid_ranges(room.y + 0.25, room.y + room.h - 0.25, self.room_doors[room_idx]["right"], door_w, wall_pad)

        for a, b in bottom_ranges:
            seg = b - a
            if seg < 0.72:
                continue
            cx = (a + b) * 0.5
            self._add_decor_box(Vec3(cx, room.y + strip_depth, z), Vec3(seg * 0.48, strip_depth, strip_h), color=strip_color)

        for a, b in top_ranges:
            seg = b - a
            if seg < 0.72:
                continue
            cx = (a + b) * 0.5
            self._add_decor_box(Vec3(cx, room.y + room.h - strip_depth, z), Vec3(seg * 0.48, strip_depth, strip_h), color=strip_color)

        for a, b in left_ranges:
            seg = b - a
            if seg < 0.72:
                continue
            cy = (a + b) * 0.5
            self._add_decor_box(Vec3(room.x + strip_depth, cy, z), Vec3(strip_depth, seg * 0.48, strip_h), color=strip_color)

        for a, b in right_ranges:
            seg = b - a
            if seg < 0.72:
                continue
            cy = (a + b) * 0.5
            self._add_decor_box(Vec3(room.x + room.w - strip_depth, cy, z), Vec3(strip_depth, seg * 0.48, strip_h), color=strip_color)

    def _plan_hallways_and_doors(self) -> None:
        self.corridor_segments.clear()
        self.corridor_joints.clear()
        for idx in range(len(self.rooms)):
            self.room_doors[idx] = {"left": [], "right": [], "bottom": [], "top": []}

        for a, b in self.edges:
            room_a = self.rooms[a]
            room_b = self.rooms[b]
            ax, ay = room_a.center
            bx, by = room_b.center

            a_anchor = self._get_room_anchor(room_a, bx, by)
            b_anchor = self._get_room_anchor(room_b, ax, ay)

            self.room_doors[a][a_anchor[2]].append(a_anchor[3])
            self.room_doors[b][b_anchor[2]].append(b_anchor[3])

            za = self._level_base_z(self.room_levels.get(a, 0))
            zb = self._level_base_z(self.room_levels.get(b, 0))
            z = 0.5 * (za + zb)
            p1 = (a_anchor[0], a_anchor[1], z)
            p2 = (b_anchor[0], b_anchor[1], z)
            mid = (p2[0], p1[1], z)

            self.corridor_segments.append((p1, mid))
            self.corridor_segments.append((mid, p2))
            self.corridor_joints.append(mid)

    def _get_room_anchor(self, room: Room, tx: float, ty: float) -> tuple[float, float, str, float]:
        dx = tx - (room.x + room.w * 0.5)
        dy = ty - (room.y + room.h * 0.5)

        if abs(dx) >= abs(dy):
            y = self._clamp(ty, room.y + 1.0, room.y + room.h - 1.0)
            if dx >= 0:
                return room.x + room.w, y, "right", y
            return room.x, y, "left", y

        x = self._clamp(tx, room.x + 1.0, room.x + room.w - 1.0)
        if dy >= 0:
            return x, room.y + room.h, "top", x
        return x, room.y, "bottom", x

    def _build_room_walls_with_doors(self, room_idx: int, room: Room, base_z: float, wall_color: tuple[float, float, float, float]) -> None:
        wt = self.wall_t
        x0 = room.x
        x1 = room.x + room.w
        y0 = room.y
        y1 = room.y + room.h
        door_w = self._effective_entry_opening_width(room)

        self._build_vertical_wall_segments(x0 - wt / 2, y0 - wt / 2, y1 + wt / 2, base_z, wt, wall_color, self.room_doors[room_idx]["left"], door_w)
        self._build_vertical_wall_segments(x1 + wt / 2, y0 - wt / 2, y1 + wt / 2, base_z, wt, wall_color, self.room_doors[room_idx]["right"], door_w)
        self._build_horizontal_wall_segments(y0 - wt / 2, x0 - wt / 2, x1 + wt / 2, base_z, wt, wall_color, self.room_doors[room_idx]["bottom"], door_w)
        self._build_horizontal_wall_segments(y1 + wt / 2, x0 - wt / 2, x1 + wt / 2, base_z, wt, wall_color, self.room_doors[room_idx]["top"], door_w)

    def _build_vertical_wall_segments(
        self,
        x: float,
        y0: float,
        y1: float,
        base_z: float,
        wt: float,
        color: tuple[float, float, float, float],
        doors: list[float],
        door_w: float,
    ) -> None:
        intervals = self._open_intervals(doors, y0, y1, door_w)
        cursor = y0
        for a, b in intervals:
            if a > cursor:
                seg = a - cursor
                self._add_box(Vec3(x, cursor + seg / 2, base_z + self.wall_h / 2), Vec3(wt / 2, seg / 2, self.wall_h / 2), color=color)
            cursor = max(cursor, b)
        if y1 > cursor:
            seg = y1 - cursor
            self._add_box(Vec3(x, cursor + seg / 2, base_z + self.wall_h / 2), Vec3(wt / 2, seg / 2, self.wall_h / 2), color=color)

    def _build_horizontal_wall_segments(
        self,
        y: float,
        x0: float,
        x1: float,
        base_z: float,
        wt: float,
        color: tuple[float, float, float, float],
        doors: list[float],
        door_w: float,
    ) -> None:
        intervals = self._open_intervals(doors, x0, x1, door_w)
        cursor = x0
        for a, b in intervals:
            if a > cursor:
                seg = a - cursor
                self._add_box(Vec3(cursor + seg / 2, y, base_z + self.wall_h / 2), Vec3(seg / 2, wt / 2, self.wall_h / 2), color=color)
            cursor = max(cursor, b)
        if x1 > cursor:
            seg = x1 - cursor
            self._add_box(Vec3(cursor + seg / 2, y, base_z + self.wall_h / 2), Vec3(seg / 2, wt / 2, self.wall_h / 2), color=color)

    def _open_intervals(self, centers: list[float], low: float, high: float, width: float) -> list[tuple[float, float]]:
        if not centers:
            return []
        intervals = []
        half = width * 0.5
        for c in centers:
            a = self._clamp(c - half, low + 0.3, high - 0.3)
            b = self._clamp(c + half, low + 0.3, high - 0.3)
            if b > a:
                intervals.append((a, b))
        intervals.sort(key=lambda t: t[0])

        merged: list[tuple[float, float]] = []
        for a, b in intervals:
            if not merged or a > merged[-1][1]:
                merged.append((a, b))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        return merged

    def _clear_entrance_debug_overlay(self) -> None:
        overlay = getattr(self, "entrance_debug_np", None)
        if overlay is not None and not overlay.isEmpty():
            overlay.removeNode()
        self.entrance_debug_np = None

    def _rebuild_entrance_debug_overlay(self) -> None:
        self._clear_entrance_debug_overlay()
        if not self.enable_entrance_debug_overlay:
            return
        if not self.rooms or not self.room_doors:
            return

        segs = LineSegs("entrance-debug-overlay")
        segs.setThickness(3.0)

        wt = self.wall_t
        z_pad = 0.08
        for room_idx, room in enumerate(self.rooms):
            doors = self.room_doors.get(room_idx)
            if not doors:
                continue
            base_z = self._level_base_z(self.room_levels.get(room_idx, 0))
            z0 = base_z + z_pad
            z1 = base_z + self.wall_h - z_pad
            door_w = self._effective_entry_opening_width(room)
            half = door_w * 0.5

            def draw_rect(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, color: tuple[float, float, float, float]) -> None:
                segs.setColor(*color)
                segs.moveTo(p0)
                segs.drawTo(p1)
                segs.drawTo(p2)
                segs.drawTo(p3)
                segs.drawTo(p0)

            for x in doors["bottom"]:
                a = self._clamp(x - half, room.x + 0.3, room.x + room.w - 0.3)
                b = self._clamp(x + half, room.x + 0.3, room.x + room.w - 0.3)
                if b <= a:
                    continue
                y = room.y - wt * 0.5
                draw_rect(Vec3(a, y, z0), Vec3(b, y, z0), Vec3(b, y, z1), Vec3(a, y, z1), (0.2, 1.0, 0.35, 0.95))

            for x in doors["top"]:
                a = self._clamp(x - half, room.x + 0.3, room.x + room.w - 0.3)
                b = self._clamp(x + half, room.x + 0.3, room.x + room.w - 0.3)
                if b <= a:
                    continue
                y = room.y + room.h + wt * 0.5
                draw_rect(Vec3(a, y, z0), Vec3(b, y, z0), Vec3(b, y, z1), Vec3(a, y, z1), (0.2, 1.0, 0.35, 0.95))

            for y in doors["left"]:
                a = self._clamp(y - half, room.y + 0.3, room.y + room.h - 0.3)
                b = self._clamp(y + half, room.y + 0.3, room.y + room.h - 0.3)
                if b <= a:
                    continue
                x = room.x - wt * 0.5
                draw_rect(Vec3(x, a, z0), Vec3(x, b, z0), Vec3(x, b, z1), Vec3(x, a, z1), (0.98, 0.74, 0.22, 0.95))

            for y in doors["right"]:
                a = self._clamp(y - half, room.y + 0.3, room.y + room.h - 0.3)
                b = self._clamp(y + half, room.y + 0.3, room.y + room.h - 0.3)
                if b <= a:
                    continue
                x = room.x + room.w + wt * 0.5
                draw_rect(Vec3(x, a, z0), Vec3(x, b, z0), Vec3(x, b, z1), Vec3(x, a, z1), (0.98, 0.74, 0.22, 0.95))

        node = self.render.attachNewNode(segs.create(True))
        node.setLightOff(1)
        node.setShaderOff(1)
        node.setCollideMask(BitMask32.allOff())
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthWrite(False)
        node.setBin("fixed", 60)
        self._force_non_solid_visual(node)
        self.entrance_debug_np = node

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _space_rooms_apart(self, min_gap: float = 1.8, iterations: int = 12) -> None:
        if not self.rooms:
            return

        gap = max(0.4, float(min_gap))
        for _ in range(max(1, int(iterations))):
            moved = False
            for i in range(len(self.rooms)):
                a = self.rooms[i]
                acx, acy = a.center
                for j in range(i + 1, len(self.rooms)):
                    b = self.rooms[j]
                    bcx, bcy = b.center

                    dx = bcx - acx
                    dy = bcy - acy
                    req_x = (a.w * 0.5 + b.w * 0.5 + gap) - abs(dx)
                    req_y = (a.h * 0.5 + b.h * 0.5 + gap) - abs(dy)
                    if req_x <= 0.0 or req_y <= 0.0:
                        continue

                    moved = True
                    if req_x < req_y:
                        push = req_x * 0.52
                        sign = 1.0 if dx >= 0.0 else -1.0
                        a.x -= sign * push * 0.5
                        b.x += sign * push * 0.5
                    else:
                        push = req_y * 0.52
                        sign = 1.0 if dy >= 0.0 else -1.0
                        a.y -= sign * push * 0.5
                        b.y += sign * push * 0.5

                    a.x = self._clamp(a.x, 1.0, max(1.0, self.map_w - a.w - 1.0))
                    a.y = self._clamp(a.y, 1.0, max(1.0, self.map_d - a.h - 1.0))
                    b.x = self._clamp(b.x, 1.0, max(1.0, self.map_w - b.w - 1.0))
                    b.y = self._clamp(b.y, 1.0, max(1.0, self.map_d - b.h - 1.0))

            if not moved:
                break

    def _is_outside_main_rooms(self, pos: Vec3, margin: float = 8.0) -> bool:
        m = max(0.0, float(margin))
        for room in self.rooms:
            if (room.x - m) <= pos.x <= (room.x + room.w + m) and (room.y - m) <= pos.y <= (room.y + room.h + m):
                return False
        return True

    def _reshape_edges_dungeon_style(self) -> None:
        if len(self.rooms) < 3:
            return

        cx = self.map_w * 0.5
        cy = self.map_d * 0.5
        room_indices = list(range(len(self.rooms)))
        room_indices.sort(
            key=lambda i: (
                math.dist(self.rooms[i].center, (cx, cy)),
                -(self.rooms[i].w * self.rooms[i].h),
            )
        )

        hub = room_indices[0]
        attached = {hub}
        new_edges: set[tuple[int, int]] = set()

        while len(attached) < len(self.rooms):
            best_pair = None
            best_cost = float("inf")
            for b in attached:
                bx, by = self.rooms[b].center
                for a in room_indices:
                    if a in attached:
                        continue
                    ax, ay = self.rooms[a].center
                    d = math.dist((ax, ay), (bx, by))
                    hub_bias = math.dist((ax, ay), self.rooms[hub].center) * 0.28
                    cost = d + hub_bias
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            new_edges.add(tuple(sorted((a, b))))
            attached.add(a)

        extra_loops = max(1, min(6, len(self.rooms) // 4))
        candidates: list[tuple[float, int, int]] = []
        for i in range(len(self.rooms)):
            ix, iy = self.rooms[i].center
            for j in range(i + 1, len(self.rooms)):
                if (i, j) in new_edges:
                    continue
                jx, jy = self.rooms[j].center
                d = math.dist((ix, iy), (jx, jy))
                if d < max(16.0, self.corridor_w * 2.1):
                    candidates.append((d, i, j))
        candidates.sort(key=lambda t: t[0])
        for _, a, b in candidates[:extra_loops]:
            new_edges.add((a, b))

        self.edges = [tuple(edge) for edge in sorted(new_edges)]

    def _setup_phase_room_overlays(self) -> None:
        self.phase_room_overlays.clear()
        if len(self.rooms) < 2:
            return

        pairs = max(2, min(8, len(self.rooms) // 2))
        idxs = list(range(len(self.rooms)))
        idxs.sort(key=lambda i: self.rooms[i].w * self.rooms[i].h, reverse=True)

        used: set[int] = set()
        for i in range(pairs):
            a_idx = idxs[i % len(idxs)]
            b_idx = idxs[-(i + 1)]
            if a_idx == b_idx or a_idx in used or b_idx in used:
                continue

            room_a = self.rooms[a_idx]
            room_b = self.rooms[b_idx]
            level_a = self._level_base_z(self.room_levels.get(a_idx, 0))
            level_b = self._level_base_z(self.room_levels.get(b_idx, 0))
            center_a = Vec3(room_a.x + room_a.w * 0.5, room_a.y + room_a.h * 0.5, level_a + self.wall_h * 0.48)
            center_b = Vec3(room_b.x + room_b.w * 0.5, room_b.y + room_b.h * 0.5, level_b + self.wall_h * 0.48)

            overlay_a = self.world.attachNewNode(f"phase-overlay-a-{a_idx}-{b_idx}")
            overlay_a.setPos(center_a)
            overlay_a.setCollideMask(BitMask32.allOff())
            shell_a = self._add_box(
                Vec3(0, 0, 0),
                Vec3(max(0.9, room_b.w * 0.22), max(0.9, room_b.h * 0.22), max(0.38, self.wall_h * 0.24)),
                color=(0.36, 0.92, 1.0, 0.34),
                parent=overlay_a,
                collidable=False,
            )
            if shell_a is not None and not shell_a.isEmpty():
                shell_a.setTransparency(TransparencyAttrib.MAlpha)
                shell_a.setLightOff(1)

            overlay_b = self.world.attachNewNode(f"phase-overlay-b-{b_idx}-{a_idx}")
            overlay_b.setPos(center_b)
            overlay_b.setCollideMask(BitMask32.allOff())
            shell_b = self._add_box(
                Vec3(0, 0, 0),
                Vec3(max(0.9, room_a.w * 0.22), max(0.9, room_a.h * 0.22), max(0.38, self.wall_h * 0.24)),
                color=(0.95, 0.42, 1.0, 0.32),
                parent=overlay_b,
                collidable=False,
            )
            if shell_b is not None and not shell_b.isEmpty():
                shell_b.setTransparency(TransparencyAttrib.MAlpha)
                shell_b.setLightOff(1)

            self.phase_room_overlays.append({"root": overlay_a, "phase": random.uniform(0.0, math.tau)})
            self.phase_room_overlays.append({"root": overlay_b, "phase": random.uniform(0.0, math.tau)})

            self.warp_links.append(
                {
                    "mode": "room_fold",
                    "a_pos": Vec3(center_a.x, center_a.y, level_a + 0.46),
                    "b_pos": Vec3(center_b.x, center_b.y, level_b + 0.46),
                    "radius": 1.3,
                    "radius_sq": 1.3 * 1.3,
                    "a_room_idx": a_idx,
                    "b_room_idx": b_idx,
                }
            )
            used.update({a_idx, b_idx})

        self._refresh_room_fold_thread_cache()

    def _update_phase_room_overlays(self, dt: float) -> None:
        if not self.phase_room_overlays:
            return
        amount = min(1.0, max(0.0, (abs(self.player_w) - self.hyperspace_threshold) / max(0.001, self.hyper_w_limit - self.hyperspace_threshold)))
        for entry in self.phase_room_overlays:
            root = entry.get("root")
            if root is None or root.isEmpty():
                continue
            phase = float(entry.get("phase", 0.0))
            pulse = 0.72 + 0.28 * math.sin(self.roll_time * 2.6 + phase)
            alpha = 0.06 + amount * 0.62 * pulse
            root.setAlphaScale(max(0.0, min(0.88, alpha)))

    def _choose_start_room_index(self) -> int:
        if not self.rooms:
            return 0
        cx = self.map_w * 0.5
        cy = self.map_d * 0.5
        min_level = min((int(self.room_levels.get(i, 0)) for i in range(len(self.rooms))), default=0)
        degree: dict[int, int] = {i: 0 for i in range(len(self.rooms))}
        for a, b in self.edges:
            if a in degree:
                degree[a] += 1
            if b in degree:
                degree[b] += 1

        best_idx = 0
        best_score = float("-inf")
        for idx, room in enumerate(self.rooms):
            level = int(self.room_levels.get(idx, 0))
            if level != min_level:
                continue

            area = float(room.w * room.h)
            room_cx, room_cy = room.center
            center_dist = math.dist((room_cx, room_cy), (cx, cy))
            edge_margin = min(room_cx, room_cy, self.map_w - room_cx, self.map_d - room_cy)
            connectivity = float(degree.get(idx, 0))
            score = area * 1.7 + edge_margin * 2.0 + connectivity * 5.0 - center_dist * 0.6

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _level_base_z(self, level: int) -> float:
        return float(level) * float(self.level_z_step)

    def _floor_input_z(self, base_z: float) -> float:
        floor_z = float(base_z) - self.floor_t * 0.5 + self.flush_eps
        ceiling_z = float(base_z) + self.wall_h + self.floor_t * 0.5 - self.flush_eps
        return ceiling_z if getattr(self, "swap_floor_and_ceiling", False) else floor_z

    def _ceiling_input_z(self, base_z: float) -> float:
        floor_z = float(base_z) - self.floor_t * 0.5 + self.flush_eps
        ceiling_z = float(base_z) + self.wall_h + self.floor_t * 0.5 - self.flush_eps
        return floor_z if getattr(self, "swap_floor_and_ceiling", False) else ceiling_z

    def _build_corridor_segment(self, p1: tuple[float, float, float], p2: tuple[float, float, float]) -> None:
        x1, y1, z = p1
        x2, y2, _ = p2
        width = self.corridor_w
        wt = self.wall_t

        floor_color = (0.3, 0.31, 0.35, 1)
        wall_color = (0.74, 0.79, 0.87, 1)
        ceil_color = (0.9, 0.93, 0.98, 1)
        floor_color = (0.18, 0.21, 0.27, 1)

        if abs(x2 - x1) >= abs(y2 - y1):
            length = abs(x2 - x1)
            if length < 0.6:
                return
            x = min(x1, x2) + length / 2
            y = y1
            self._queue_floor_rect(Vec3(x, y, self._floor_input_z(z)), Vec3(length / 2 + wt / 2, width / 2 + wt / 2, self.floor_t / 2), color=floor_color)
            self._queue_ceiling_rect(Vec3(x, y, self._ceiling_input_z(z)), Vec3(length / 2 + wt / 2, width / 2 + wt / 2, self.floor_t / 2), color=ceil_color)
            self._add_box(Vec3(x, y - width / 2 - wt / 2, z + self.wall_h / 2), Vec3(length / 2 + wt / 2, wt / 2, self.wall_h / 2), color=wall_color)
            self._add_box(Vec3(x, y + width / 2 + wt / 2, z + self.wall_h / 2), Vec3(length / 2 + wt / 2, wt / 2, self.wall_h / 2), color=wall_color)
            hall_strip_depth = min(0.014, wt * 0.18)
            self._add_decor_box(Vec3(x, y - width / 2 + hall_strip_depth, z + self.wall_h * 0.76), Vec3(length * 0.45, hall_strip_depth, 0.03), color=(0.2, 0.94, 1.0, 1))
            self._add_decor_box(Vec3(x, y + width / 2 - hall_strip_depth, z + self.wall_h * 0.76), Vec3(length * 0.45, hall_strip_depth, 0.03), color=(0.2, 0.94, 1.0, 1))
        else:
            length = abs(y2 - y1)
            if length < 0.6:
                return
            x = x1
            y = min(y1, y2) + length / 2
            self._queue_floor_rect(Vec3(x, y, self._floor_input_z(z)), Vec3(width / 2 + wt / 2, length / 2 + wt / 2, self.floor_t / 2), color=floor_color)
            self._queue_ceiling_rect(Vec3(x, y, self._ceiling_input_z(z)), Vec3(width / 2 + wt / 2, length / 2 + wt / 2, self.floor_t / 2), color=ceil_color)
            self._add_box(Vec3(x - width / 2 - wt / 2, y, z + self.wall_h / 2), Vec3(wt / 2, length / 2 + wt / 2, self.wall_h / 2), color=wall_color)
            self._add_box(Vec3(x + width / 2 + wt / 2, y, z + self.wall_h / 2), Vec3(wt / 2, length / 2 + wt / 2, self.wall_h / 2), color=wall_color)
            hall_strip_depth = min(0.014, wt * 0.18)
            self._add_decor_box(Vec3(x - width / 2 + hall_strip_depth, y, z + self.wall_h * 0.76), Vec3(hall_strip_depth, length * 0.45, 0.03), color=(0.2, 0.94, 1.0, 1))
            self._add_decor_box(Vec3(x + width / 2 - hall_strip_depth, y, z + self.wall_h * 0.76), Vec3(hall_strip_depth, length * 0.45, 0.03), color=(0.2, 0.94, 1.0, 1))

    def _build_corridor_joint(self, x: float, y: float, z: float) -> None:
        w = self.corridor_w
        wt = self.wall_t

        self._queue_floor_rect(
            Vec3(x, y, self._floor_input_z(z)),
            Vec3(w / 2 + wt / 2, w / 2 + wt / 2, self.floor_t / 2),
            color=(0.18, 0.21, 0.27, 1),
        )
        self._queue_ceiling_rect(
            Vec3(x, y, self._ceiling_input_z(z)),
            Vec3(w / 2 + wt / 2, w / 2 + wt / 2, self.floor_t / 2),
            color=(0.9, 0.93, 0.98, 1),
        )

        ox = w / 2 + wt / 2
        oy = w / 2 + wt / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                self._add_box(
                    Vec3(x + sx * ox, y + sy * oy, z + self.wall_h / 2),
                    Vec3(wt / 2, wt / 2, self.wall_h / 2),
                    color=(0.74, 0.79, 0.87, 1),
                )

    def _canonical_union_plane_z(self, z: float) -> float:
        return round(float(z) * 200.0) / 200.0

    def _queue_floor_rect(self, pos: Vec3, scale: Vec3, color: tuple[float, float, float, float]) -> None:
        floor_z = pos.z - max(0.0, float(getattr(self, "flush_eps", 0.0))) - float(getattr(self, "water_level_offset", 0.0))
        floor_z = self._canonical_union_plane_z(floor_z)
        self.pending_floor_rects.append(
            (
                pos.x - scale.x,
                pos.x + scale.x,
                pos.y - scale.y,
                pos.y + scale.y,
                floor_z,
                color,
            )
        )

    def _queue_ceiling_rect(self, pos: Vec3, scale: Vec3, color: tuple[float, float, float, float]) -> None:
        ceil_z = self._canonical_union_plane_z(pos.z)
        self.pending_ceiling_rects.append(
            (
                pos.x - scale.x,
                pos.x + scale.x,
                pos.y - scale.y,
                pos.y + scale.y,
                ceil_z,
                color,
            )
        )

    def _queue_floor_hole(self, pos: Vec3, scale: Vec3) -> None:
        floor_z = pos.z - max(0.0, float(getattr(self, "flush_eps", 0.0))) - float(getattr(self, "water_level_offset", 0.0))
        floor_z = self._canonical_union_plane_z(floor_z)
        self.pending_floor_holes.append(
            (
                pos.x - scale.x,
                pos.x + scale.x,
                pos.y - scale.y,
                pos.y + scale.y,
                floor_z,
            )
        )

    def _queue_ceiling_hole(self, pos: Vec3, scale: Vec3) -> None:
        ceil_z = self._canonical_union_plane_z(pos.z)
        self.pending_ceiling_holes.append(
            (
                pos.x - scale.x,
                pos.x + scale.x,
                pos.y - scale.y,
                pos.y + scale.y,
                ceil_z,
            )
        )

    def _plan_floor_ceiling_openings(self) -> None:
        if not self.corridor_joints:
            return
        half = max(0.5, self.corridor_w * 0.22)
        for jx, jy, jz in self.corridor_joints:
            floor_pos = Vec3(jx, jy, self._floor_input_z(jz))
            ceil_pos = Vec3(jx, jy, self._ceiling_input_z(jz))
            hole_scale = Vec3(half, half, self.floor_t / 2)
            if not self.disable_floor_holes:
                self._queue_floor_hole(floor_pos, hole_scale)
            if not self.disable_ceiling_holes:
                self._queue_ceiling_hole(ceil_pos, hole_scale)

    def _build_union_occupancy(
        self,
        rects: list[tuple[float, float, float, float, float, tuple[float, float, float, float]]],
        holes: list[tuple[float, float, float, float, float]],
        cell: float,
    ) -> list[tuple[float, float, float, float, float, tuple[float, float, float, float]]]:
        grouped: dict[int, list[tuple[float, float, float, float, float, tuple[float, float, float, float]]]] = {}
        for rect in rects:
            z_key = int(round(rect[4] * 1000))
            grouped.setdefault(z_key, []).append(rect)

        holes_grouped: dict[int, list[tuple[float, float, float, float, float]]] = {}
        for hole in holes:
            z_key = int(round(hole[4] * 1000))
            holes_grouped.setdefault(z_key, []).append(hole)

        merged_rects: list[tuple[float, float, float, float, float, tuple[float, float, float, float]]] = []
        for z_key, z_rects in grouped.items():
            occupied: dict[int, set[int]] = {}
            z = z_rects[0][4]
            color = z_rects[0][5]

            for x0, x1, y0, y1, _, _ in z_rects:
                ix0 = int(math.floor(x0 / cell))
                ix1 = int(math.floor((x1 - 1e-6) / cell))
                iy0 = int(math.floor(y0 / cell))
                iy1 = int(math.floor((y1 - 1e-6) / cell))
                for iy in range(iy0, iy1 + 1):
                    row = occupied.setdefault(iy, set())
                    for ix in range(ix0, ix1 + 1):
                        row.add(ix)

            for hx0, hx1, hy0, hy1, _ in holes_grouped.get(z_key, []):
                ix0 = int(math.floor(hx0 / cell))
                ix1 = int(math.floor((hx1 - 1e-6) / cell))
                iy0 = int(math.floor(hy0 / cell))
                iy1 = int(math.floor((hy1 - 1e-6) / cell))
                for iy in range(iy0, iy1 + 1):
                    row = occupied.get(iy)
                    if row is None:
                        continue
                    for ix in range(ix0, ix1 + 1):
                        row.discard(ix)

            active: dict[tuple[int, int], tuple[int, int]] = {}
            finished: list[tuple[int, int, int, int]] = []
            all_rows = sorted(occupied.keys())

            def row_runs(ix_set: set[int]) -> list[tuple[int, int]]:
                vals = sorted(ix_set)
                if not vals:
                    return []
                runs: list[tuple[int, int]] = []
                start = vals[0]
                prev = vals[0]
                for ix in vals[1:]:
                    if ix == prev + 1:
                        prev = ix
                        continue
                    runs.append((start, prev))
                    start = ix
                    prev = ix
                runs.append((start, prev))
                return runs

            for iy in all_rows:
                current = row_runs(occupied[iy])
                current_keys = set(current)

                to_close = [key for key in active.keys() if key not in current_keys]
                for key in to_close:
                    y_start, y_end = active.pop(key)
                    finished.append((key[0], key[1], y_start, y_end))

                for run in current:
                    if run in active:
                        y_start, _ = active[run]
                        active[run] = (y_start, iy)
                    else:
                        active[run] = (iy, iy)

            for key, (y_start, y_end) in active.items():
                finished.append((key[0], key[1], y_start, y_end))

            for ix0, ix1, iy0, iy1 in finished:
                wx0 = ix0 * cell
                wx1 = (ix1 + 1) * cell
                wy0 = iy0 * cell
                wy1 = (iy1 + 1) * cell
                merged_rects.append((wx0, wx1, wy0, wy1, z, color))

        return merged_rects

    def _commit_floor_union(self) -> None:
        if not self.pending_floor_rects:
            return
        cell = 0.35
        merged_rects = self._build_union_occupancy(self.pending_floor_rects, self.pending_floor_holes, cell)
        for x0, x1, y0, y1, z, color in merged_rects:
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            hx = max(0.02, (x1 - x0) * 0.5)
            hy = max(0.02, (y1 - y0) * 0.5)
            holder = self._add_box(Vec3(cx, cy, z), Vec3(hx, hy, self.floor_t * 0.5), color=color, surface_mode="floor")

            support_half_z = max(self.floor_t * 0.5, 0.22)
            support_center_z = z - (support_half_z - self.floor_t * 0.5)
            self._add_static_box_collider(
                Vec3(cx, cy, support_center_z),
                Vec3(hx, hy, support_half_z),
                visual_holder=holder,
            )

            if (x1 - x0) * (y1 - y0) >= float(getattr(self, "water_overlay_min_area", 2.5)):
                water_holder = self._add_box(
                    Vec3(cx, cy, z + self.water_surface_raise),
                    Vec3(hx, hy, max(0.01, self.floor_t * 0.18)),
                    color=(0.26, 0.5, 0.72, 0.22),
                    collidable=False,
                    surface_mode="water",
                )
                if water_holder is not None and not water_holder.isEmpty():
                    water_holder.setTransparency(TransparencyAttrib.MAlpha)
                    water_holder.setDepthWrite(False)
                    water_holder.setBin("transparent", 33)

        self.pending_floor_rects.clear()
        self.pending_floor_holes.clear()

    def _commit_ceiling_union(self) -> None:
        if not self.pending_ceiling_rects:
            return
        cell = 0.35
        merged_rects = self._build_union_occupancy(self.pending_ceiling_rects, self.pending_ceiling_holes, cell)
        for x0, x1, y0, y1, z, color in merged_rects:
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            hx = max(0.02, (x1 - x0) * 0.5)
            hy = max(0.02, (y1 - y0) * 0.5)
            self._add_box(Vec3(cx, cy, z), Vec3(hx, hy, self.floor_t * 0.5), color=color, surface_mode="ceiling")

        self.pending_ceiling_rects.clear()
        self.pending_ceiling_holes.clear()

    def _add_stairs(self, x: float, y: float, z_start: float, dz: float) -> None:
        steps = max(4, int(abs(dz) * 4))
        direction = 1 if dz > 0 else -1
        for i in range(steps):
            t = i / steps
            z = z_start + t * dz
            self._add_box(
                Vec3(x + (i - steps / 2) * 0.35, y, z + 0.05),
                Vec3(0.17, self.corridor_w * 0.45, 0.05),
                color=(0.62, 0.68, 0.76, 1),
                hpr=Vec3(0 if direction > 0 else 180, 0, 0),
            )

    def _add_ramp(self, x: float, y: float, z_start: float, dz: float) -> None:
        length = max(3.0, abs(dz) * 2.8)
        pitch = math.degrees(math.atan2(dz, length))
        self._add_box(
            Vec3(x, y, z_start + dz / 2),
            Vec3(length / 2, self.corridor_w * 0.45, 0.08),
            color=(0.62, 0.69, 0.77, 1),
            hpr=Vec3(0, -pitch, 0),
        )

    def update(self, task):
        dt = globalClock.getDt()
        dt = min(dt, 1 / 30)
        self._update_infinite_world_goal(dt)
        self.monster_anim_tick += 1
        self.hit_cooldown = max(0.0, self.hit_cooldown - dt)
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.monster_contact_sfx_cooldown = max(0.0, self.monster_contact_sfx_cooldown - dt)
        self.player_damage_cooldown = max(0.0, self.player_damage_cooldown - dt)
        self.roll_time += dt
        self._update_water_surface_waves(self.roll_time)
        self._update_water_crystals(dt)
        self._update_ball_texture_scroll(dt)
        self.vertical_mover_update_timer -= dt
        if self.vertical_mover_update_timer <= 0.0:
            self._update_vertical_movers(self.vertical_mover_update_interval)
            self.vertical_mover_update_timer = self.vertical_mover_update_interval
        self._update_outside_islands(dt)
        self._update_roaming_black_holes(dt)
        self._update_warp_links(dt)
        self._update_phase_room_overlays(dt)
        self.color_cycle_update_timer -= dt
        if self.color_cycle_update_timer <= 0.0:
            self._update_color_cycle(dt)
            self.color_cycle_update_timer = 1.0 / 30.0
        self.hyper_uv_update_timer -= dt
        if self.hyper_uv_update_timer <= 0.0:
            self._update_hyper_uv_projection(self.roll_time)
            self._update_dynamic_room_uv(self.roll_time)
            next_uv_interval = self.hyper_uv_update_interval
            dyn_count = len(self.dynamic_room_uv_nodes)
            if dyn_count > 1200:
                next_uv_interval = max(next_uv_interval, 1.0 / 10.0)
            elif dyn_count > 700:
                next_uv_interval = max(next_uv_interval, 1.0 / 14.0)
            elif dyn_count > 300:
                next_uv_interval = max(next_uv_interval, 1.0 / 18.0)
            self.hyper_uv_update_timer = next_uv_interval
        self.light_update_timer -= dt
        if self.light_update_timer <= 0.0:
            self._update_orbit_lights(self.roll_time)
            self._update_atmosphere_lights(self.roll_time)
            self.light_update_timer = self.light_update_interval
        self._update_hypercube_monsters(dt)
        self._update_combat_stat_buffs(dt)
        self._update_weapon(dt)
        self.health_ui_update_timer -= dt
        if self.health_ui_update_timer <= 0.0:
            self._update_monster_health_bars()
            self._update_player_health_ui()
            self._update_monster_hud_ui()
            self.health_ui_update_timer = 1.0 / 20.0
        self._update_input_hud(dt)
        self._update_holographic_map_ui(dt)

        if self.game_over_active or self.win_active:
            if self.game_over_active:
                self._update_game_over_countdown(dt)
            if hasattr(self, "ball_body") and self.ball_body is not None:
                self.ball_body.setLinearVelocity(Vec3(0, 0, 0))
                self.ball_body.setAngularVelocity(Vec3(0, 0, 0))
            self.audio3d.update()
            return task.cont

        self._update_health_powerups(dt)
        self._update_kill_protection_visuals(dt)
        self._update_sword_powerups(dt)
        self._update_floating_texts(dt)
        self._update_hyperbomb(dt)
        self._update_magic_missiles(dt)
        self._update_magic_missile_trails(dt)
        if self.enable_particles and self.enable_motion_trails:
            self._update_motion_trails(dt)
        elif self.motion_trails:
            for entry in self.motion_trails:
                node = entry.get("node")
                if node is not None and not node.isEmpty():
                    node.removeNode()
            self.motion_trails.clear()
        self.audio3d.update()

        mouse_turn_deg, mouse_pitch_deg = self._consume_mouse_look()

        hyper_input = 0.0
        if self.hyper_keys["q"]:
            hyper_input -= 2.0
        if self.hyper_keys["e"]:
            hyper_input += 2.0
        if abs(hyper_input) <= 1e-6 and bool(getattr(self, "player_ai_enabled", False)):
            target_w = float(getattr(self, "player_ai_target_w", self.player_w))
            w_delta = target_w - self.player_w
            if abs(w_delta) > 0.02:
                hyper_input += self._clamp(w_delta * 0.85, -1.25, 1.25)

        self.player_w += (mouse_pitch_deg / 24.0) * self.hyper_mouse_w_speed * dt
        self.player_w += (mouse_turn_deg / 32.0) * self.hyper_turn_w_speed * dt
        self.player_w = max(-self.hyper_w_limit, min(self.hyper_w_limit, self.player_w))

        decay_alpha = min(1.0, dt * self.scroll_lift_decay)
        self.scroll_lift_target += (0.0 - self.scroll_lift_target) * decay_alpha
        lift_alpha = min(1.0, dt * self.scroll_lift_smooth)
        self.scroll_lift_value += (self.scroll_lift_target - self.scroll_lift_value) * lift_alpha

        hyperspace_active = abs(self.player_w) > self.hyperspace_threshold
        hyperspace_amount = min(
            1.0,
            max(
                0.0,
                (abs(self.player_w) - self.hyperspace_threshold)
                / max(0.001, self.hyper_w_limit - self.hyperspace_threshold),
            ),
        )
        self.audio_hyper_mix = hyperspace_amount
        self._update_hyperspace_background(self.roll_time, hyperspace_amount)
        self._update_hyperspace_illumination(hyperspace_active, hyperspace_amount)

        if self.enable_particles and hasattr(self, "star_root"):
            if hyperspace_amount > 0.03:
                if self.star_root.isStashed():
                    self.star_root.unstash()
                self.star_update_timer -= dt
                if self.star_update_timer <= 0.0:
                    self._update_star_particles(dt, self.roll_time)
                    self.star_update_timer = 0.05
            else:
                if not self.star_root.isStashed():
                    self.star_root.stash()
        elif hasattr(self, "star_root") and not self.star_root.isEmpty() and not self.star_root.isStashed():
            self.star_root.stash()

        zone_gravity = Vec3(0, 0, 0) if self.zero_g_mode else self._get_room_zone_gravity(self.ball_np.getPos())
        if self.enable_gravity_particles:
            self.gravity_particles_update_timer -= dt
            if self.gravity_particles_update_timer <= 0.0:
                self._update_room_gravity_particles(dt, self.ball_np.getPos())
                self.gravity_particles_update_timer = self.gravity_particles_update_interval
        hyperspace_gravity_override = hyperspace_active and self.hyperspace_gravity_hold
        self.ball_body.setFriction(self.ball_friction_shift if self.hyperspace_gravity_hold else self.ball_friction_default)
        if self.zero_g_mode:
            target_gravity = Vec3(0, 0, 0)
        else:
            zone_mag = max(1e-4, zone_gravity.length())
            zone_dir = Vec3(zone_gravity) / zone_mag
            if hyperspace_active and not hyperspace_gravity_override:
                mag_scale = 0.72 + 0.28 * math.sin(self.roll_time * 1.35)
                target_gravity = zone_dir * (zone_mag * mag_scale)
            else:
                target_gravity = Vec3(zone_gravity)

        grav_blend = min(1.0, dt * self.gravity_blend_speed)
        self.current_gravity = self.current_gravity + (target_gravity - self.current_gravity) * grav_blend
        self.physics_world.setGravity(self.current_gravity)

        if abs(hyper_input) > 1e-6:
            gravity_up = self._get_gravity_up()
            cam_forward = self.camera.getQuat(self.render).getForward()
            planar_forward = Vec3(cam_forward.x, cam_forward.y, cam_forward.z)
            planar_forward = planar_forward - gravity_up * planar_forward.dot(gravity_up)
            if planar_forward.lengthSquared() < 1e-8:
                planar_forward = Vec3(0, 1, 0)
            else:
                planar_forward.normalize()

            side = gravity_up.cross(planar_forward)
            if side.lengthSquared() < 1e-8:
                side = Vec3(1, 0, 0)
            else:
                side.normalize()

            w_phase = 0.0 if self.hyper_w_limit <= 1e-6 else max(-1.0, min(1.0, self.player_w / self.hyper_w_limit))
            axis_angle = w_phase * (math.pi * 0.5)
            hyper_axis = planar_forward * math.cos(axis_angle) + side * math.sin(axis_angle)
            if hyper_axis.lengthSquared() < 1e-8:
                hyper_axis = planar_forward
            else:
                hyper_axis.normalize()

            input_sign = -1.0 if hyper_input < 0.0 else 1.0
            force = hyper_axis * (abs(hyper_input) * self.hyper_force_strength)
            force += gravity_up * (input_sign * abs(hyper_input) * self.hyper_force_strength * self.hyper_force_lift)
            self.ball_body.applyCentralForce(force)

        if abs(self.scroll_lift_value) > 1e-4:
            gravity_up = self._get_gravity_up()
            force_mag = self.scroll_lift_up_force if self.scroll_lift_value > 0.0 else self.scroll_lift_down_force
            self.ball_body.applyCentralForce(gravity_up * (self.scroll_lift_value * force_mag))

        self._apply_water_buoyancy(dt)

        if self.zero_g_mode:
            self.ball_body.setRestitution(self.normal_restitution)
        else:
            self.ball_body.setRestitution(self.hyperspace_restitution if (hyperspace_active and not hyperspace_gravity_override) else self.normal_restitution)

        pre_physics_pos = Vec3(self.ball_np.getPos())
        pre_physics_speed = self.ball_body.getLinearVelocity().length()
        adaptive_substeps = self.physics_substeps + int(min(4.0, pre_physics_speed / 12.0))
        if hyperspace_active:
            adaptive_substeps += 1
        adaptive_substeps = max(self.physics_substeps, min(10, adaptive_substeps))

        self.physics_world.doPhysics(dt, adaptive_substeps, self.physics_fixed_timestep)
        self._prevent_ball_tunneling(pre_physics_pos)
        self._suppress_wall_climb_velocity()
        if hyperspace_active and not hyperspace_gravity_override:
            self._update_hyperspace_bounce()
        if self.infinite_level_mode:
            self._apply_world_wrap()
        self.monster_collision_update_timer -= dt
        if self.monster_collision_update_timer <= 0.0:
            self._resolve_monster_collisions(dt)
            self.monster_collision_update_timer = self.monster_collision_update_interval

        manual_turn = 0.0
        if self.camera_keys["arrow_left"]:
            manual_turn += 1.0
        if self.camera_keys["arrow_right"]:
            manual_turn -= 1.0
        manual_pitch = 0.0
        if self.camera_keys["arrow_up"]:
            manual_pitch += 1.0
        if self.camera_keys["arrow_down"]:
            manual_pitch -= 1.0

        heading_delta = manual_turn * self.camera_orbit_speed * dt + mouse_turn_deg
        pitch_delta = manual_pitch * self.camera_pitch_speed * dt + mouse_pitch_deg

        if abs(heading_delta) > 0.0 or abs(pitch_delta) > 0.0:
            self.heading += heading_delta
            self.pitch += pitch_delta
            self.pitch = max(6.0, min(80.0, self.pitch))
            self.camera_manual_turn_hold = 0.22
        else:
            self.camera_manual_turn_hold = max(0.0, self.camera_manual_turn_hold - dt)

        move = Vec3(0, 0, 0)
        if self.keys["w"]:
            move.y += 1
        if self.keys["s"]:
            move.y -= 1
        if self.keys["a"]:
            move.x += 1
        if self.keys["d"]:
            move.x -= 1

        manual_move_active = move.lengthSquared() > 0
        if manual_move_active:
            self.player_ai_enabled = False
            self.player_ai_idle_timer = 0.0
            self.player_ai_target_id = None
            self.player_ai_lock_target_id = None
            self.player_ai_retarget_timer = 0.0
            self.player_ai_combo_step = 0
            self.player_ai_combo_timer = 0.0
            self.player_ai_camera_target_pos = None
            self.player_ai_target_w = float(getattr(self, "player_w", 0.0))
            self.player_ai_room_path = []
            self.player_ai_room_path_goal = None
            self.player_ai_room_path_recalc_timer = 0.0
        else:
            self.player_ai_idle_timer = min(self.player_ai_idle_delay, self.player_ai_idle_timer + dt)
            if self.player_ai_idle_timer >= self.player_ai_idle_delay:
                self.player_ai_enabled = True

        input_up = self._get_gravity_up()
        desired_move_dir: Vec3 | None = None
        if manual_move_active:
            move.normalize()
            cam_forward = self.camera.getQuat(self.render).getForward()
            forward = Vec3(cam_forward.x, cam_forward.y, cam_forward.z)
            forward -= input_up * forward.dot(input_up)
            if forward.lengthSquared() < 1e-6:
                ref_axis = Vec3(0, 1, 0)
                if abs(ref_axis.dot(input_up)) > 0.95:
                    ref_axis = Vec3(1, 0, 0)
                ref_axis -= input_up * ref_axis.dot(input_up)
                if ref_axis.lengthSquared() < 1e-8:
                    ref_axis = Vec3(1, 0, 0)
                ref_axis.normalize()
                forward = self._rotate_around_axis(ref_axis, input_up, math.radians(self.heading))
            else:
                forward.normalize()
            if self._is_ceiling_mode():
                forward = -forward
            right = input_up.cross(forward)
            if right.lengthSquared() < 1e-8:
                right = Vec3(1, 0, 0)
            else:
                right.normalize()
            move_dir = (forward * move.y + right * move.x)
            if move_dir.lengthSquared() > 0:
                move_dir.normalize()
                desired_move_dir = Vec3(move_dir)
                self.last_move_dir = Vec3(move_dir)
                torque_axis = input_up.cross(move_dir)
                torque = torque_axis * self.roll_torque
                self.ball_body.applyTorque(torque)
                self.ball_body.applyCentralForce(move_dir * (self.roll_force * 0.56))

        if desired_move_dir is None:
            ai_dir = self._update_player_ai(dt)
            if ai_dir is not None and ai_dir.lengthSquared() > 1e-8:
                ai_dir.normalize()
                desired_move_dir = Vec3(ai_dir)
                self.last_move_dir = Vec3(ai_dir)
                torque_axis = input_up.cross(ai_dir)
                self.ball_body.applyTorque(torque_axis * self.roll_torque)
                self.ball_body.applyCentralForce(ai_dir * (self.roll_force * 0.56))

        velocity = self.ball_body.getLinearVelocity()

        if not self.grounded:
            vertical_speed = velocity.dot(input_up)
            horizontal = Vec3(velocity) - input_up * vertical_speed
            if self.jump_float_timer > 0.0 and vertical_speed > 0.0:
                vertical_speed *= max(0.0, 1.0 - dt * self.jump_float_drag)
            elif vertical_speed < 0.0:
                vertical_speed *= max(0.0, 1.0 - dt * self.float_fall_drag)
            velocity = horizontal + input_up * vertical_speed
            self.ball_body.setLinearVelocity(velocity)

        if self.enable_dimensional_compression:
            raw_compression_factor = self._compression_factor_at(self.ball_np.getPos(), self.roll_time)
        else:
            raw_compression_factor = 1.0
        smooth_alpha = min(1.0, dt * self.compression_smooth_speed)
        self.compression_factor_smoothed += (raw_compression_factor - self.compression_factor_smoothed) * smooth_alpha
        compression_factor = self.compression_factor_smoothed
        self._enforce_ball_floor_clearance(compression_factor)
        self._update_timespace_tone(compression_factor, dt)
        self._update_camera_dimension_from_compression(dt, compression_factor, velocity.length(), hyperspace_active)
        if compression_factor < 0.999:
            vertical_speed = velocity.dot(input_up)
            horizontal = Vec3(velocity) - input_up * vertical_speed
            h_drag = max(0.76, 1.0 - (1.0 - compression_factor) * dt * 1.45)
            v_drag = max(0.84, 1.0 - (1.0 - compression_factor) * dt * 0.82)
            if desired_move_dir is not None:
                h_drag += (1.0 - h_drag) * 0.52
                v_drag += (1.0 - v_drag) * 0.36
            horizontal *= h_drag
            vertical_speed *= v_drag
            velocity = horizontal + input_up * vertical_speed
            self.ball_body.setLinearVelocity(velocity)
        elif compression_factor > 1.001:
            vertical_speed = velocity.dot(input_up)
            horizontal = Vec3(velocity) - input_up * vertical_speed
            gain = compression_factor - 1.0
            h_boost = min(1.9, 1.0 + gain * dt * 2.4)
            v_boost = min(1.6, 1.0 + gain * dt * 1.4)
            horizontal *= h_boost
            vertical_speed *= v_boost
            velocity = horizontal + input_up * vertical_speed
            self.ball_body.setLinearVelocity(velocity)

        self._resolve_folded_world_collision(compression_factor, hyperspace_active)

        if self.hyperspace_gravity_hold:
            brake_factor = max(0.0, 1.0 - dt * self.shift_brake_drag)
            self.ball_body.setLinearVelocity(Vec3(velocity) * brake_factor)
            self.ball_body.setAngularVelocity(self.ball_body.getAngularVelocity() * max(0.0, 1.0 - dt * (self.shift_brake_drag * 0.5)))
            velocity = self.ball_body.getLinearVelocity()

        horizontal = Vec3(velocity) - input_up * velocity.dot(input_up)
        if desired_move_dir is not None:
            desired_velocity = desired_move_dir * self.max_ball_speed
            steer = desired_velocity - horizontal
            self.ball_body.applyCentralForce(steer * self.link_control_gain)
        else:
            brake = max(0.0, 1.0 - dt * self.link_brake_drag)
            velocity = input_up * velocity.dot(input_up) + horizontal * brake
            self.ball_body.setLinearVelocity(velocity)
            self.ball_body.setAngularVelocity(self.ball_body.getAngularVelocity() * max(0.0, 1.0 - dt * (self.link_brake_drag * 0.82)))
            horizontal = Vec3(velocity) - input_up * velocity.dot(input_up)
        speed = horizontal.length()
        if self.enable_particles and self.enable_motion_trails:
            self.motion_trail_emit_timer -= dt
        if self.enable_particles and self.enable_motion_trails and speed > 0.55 and self.motion_trail_emit_timer <= 0.0:
            back_dir = Vec3(horizontal)
            if back_dir.lengthSquared() > 1e-8:
                back_dir.normalize()
            else:
                back_dir = Vec3(-self.last_move_dir)
            ball_pos = self.ball_np.getPos()
            self._spawn_motion_trail(
                ball_pos - back_dir * 0.14 + Vec3(0, 0, -0.02),
                scale=self.ball_radius * 0.98,
                color=(0.35, 0.82, 1.0, 0.42),
                life=0.22,
                vel=-back_dir * 0.55,
                use_box=False,
            )
            if self.attack_mode != "idle" and hasattr(self, "sword_tip"):
                self._spawn_motion_trail(
                    self.sword_tip.getPos(self.render),
                    scale=0.08,
                    color=(0.28, 0.95, 1.0, 0.36),
                    life=0.16,
                    vel=-back_dir * 0.35 + Vec3(0, 0, 0.12),
                    use_box=True,
                )
            self.motion_trail_emit_timer = 0.016 if self.attack_mode == "spin" else 0.028
        if speed > self.max_ball_speed:
            horizontal *= self.max_ball_speed / speed
            self.ball_body.setLinearVelocity(Vec3(horizontal.x, horizontal.y, velocity.z))

        contact_result = self.physics_world.contactTest(self.ball_body)
        contact_analysis = self._analyze_ball_contacts(contact_result)
        self.grounded = bool(contact_analysis.get("grounded", False))
        grounded_contact: Vec3 | None = None

        if self.grounded:
            self.jumps_used = 0
            grounded_contact = self._get_ball_floor_contact_point()
            if grounded_contact is None:
                analysis_point = contact_analysis.get("ground_point")
                if analysis_point is not None:
                    grounded_contact = Vec3(analysis_point)
            if grounded_contact is None:
                closest_z = float("inf")
                for contact in contact_result.getContacts():
                    point = None
                    try:
                        point = contact.getManifoldPoint().getPositionWorldOnB()
                    except Exception:
                        point = None
                    if point is None:
                        try:
                            point = contact.getManifoldPoint().getPositionWorldOnA()
                        except Exception:
                            point = None
                    if point is None:
                        continue
                    if point.z < closest_z:
                        closest_z = point.z
                        grounded_contact = Vec3(point)

                if grounded_contact is None:
                    gravity_up = self._get_gravity_up()
                    grounded_contact = self.ball_np.getPos() - gravity_up * (self.ball_radius + 0.02)

        if self.jump_queued:
            if self.infinite_jumps or self.jumps_used < self.max_jumps:
                jump_up = self._get_gravity_up()
                if jump_up.lengthSquared() < 1e-8:
                    jump_up = Vec3(0, 0, 1)
                else:
                    jump_up.normalize()

                boost_dir = Vec3(velocity) - jump_up * Vec3(velocity).dot(jump_up)
                if boost_dir.lengthSquared() < 1e-6:
                    boost_dir = Vec3(self.last_move_dir)
                if boost_dir.lengthSquared() > 1e-6:
                    boost_dir.normalize()

                impulse = jump_up * (self.jump_impulse * self.jump_rise_boost)
                if boost_dir.lengthSquared() > 1e-6:
                    impulse += boost_dir * (self.space_boost_impulse * 0.2)

                self.ball_body.applyCentralImpulse(impulse)
                if not self.infinite_jumps:
                    self.jumps_used += 1
                self.grounded = False
                self.jump_float_timer = self.jump_float_duration
                self._play_sound(self.sfx_jump, volume=0.72, play_rate=1.2)
                grounded_contact = None
        self.jump_queued = False
        self.jump_float_timer = max(0.0, self.jump_float_timer - dt)

        self._update_floor_contact_pulses(dt, grounded_contact if self.grounded else None)

        self._update_floor_wet_shader_inputs(dt, grounded_contact if self.grounded else None, speed)
        self._update_water_surface_shader_inputs()

        self._update_ripple_transparency(dt, grounded_contact if self.grounded else None)

        if self.sfx_roll:
            angular_speed = self.ball_body.getAngularVelocity().length() * self.ball_radius
            roll_metric = max(horizontal.length(), angular_speed)
            grounded_speed = roll_metric if self.grounded else 0.0
            if grounded_speed > 0.08:
                if self.sfx_roll.status() != self.sfx_roll.PLAYING:
                    self.sfx_roll.play()
                norm = min(1.0, grounded_speed / max(0.01, self.max_ball_speed))
                log_norm = math.log1p(9.0 * norm) / math.log1p(9.0)
                wobble_freq = 5.0 + 11.0 * norm
                wobble = math.sin(self.roll_time * wobble_freq)
                vol_wobble = (0.015 + 0.13 * norm) * wobble
                pitch_wobble = (0.02 + 0.12 * norm) * wobble
                vol_jitter = random.uniform(-(0.012 + 0.06 * norm), 0.012 + 0.06 * norm)
                pitch_jitter = random.uniform(-(0.01 + 0.09 * norm), 0.01 + 0.09 * norm)

                base_volume = 0.1 + log_norm * 0.86
                base_rate = 0.72 + norm * 1.05
                mix = max(0.0, min(1.0, float(self.audio_hyper_mix)))
                roll_vol = max(0.0, min(1.0, base_volume + vol_wobble + vol_jitter))
                roll_rate = max(0.5, min(2.4, base_rate + pitch_wobble + pitch_jitter))
                roll_vol = roll_vol * (1.0 - 0.34 * mix) + 0.26 * mix
                roll_rate = roll_rate * (1.0 - 0.24 * mix) + 1.0 * (0.24 * mix)
                self.sfx_roll.setVolume(max(0.0, min(1.0, roll_vol)))
                self.sfx_roll.setPlayRate(max(0.5, min(2.4, roll_rate)))
            else:
                self.sfx_roll.setVolume(max(0.0, self.sfx_roll.getVolume() - dt * 1.6))

        delta_v = (velocity - self.prev_ball_velocity).length()
        impact_speed = delta_v / max(dt, 1e-4)
        has_contact = contact_result.getNumContacts() > 0
        just_landed = self.grounded and (not self.prev_grounded)
        vertical_impact_speed = abs(self.prev_ball_velocity.z)
        is_real_impact = just_landed or vertical_impact_speed > 2.2
        if has_contact and is_real_impact and impact_speed > 5.5 and self.hit_cooldown <= 0.0:
            intensity = min(1.0, impact_speed / 24.0)
            hit_rate = 0.72 + intensity * 0.9 + random.uniform(-0.06, 0.08)
            self._play_sound(self.sfx_hit, volume=0.18 + intensity * 0.72, play_rate=hit_rate)
            self.hit_cooldown = 0.08

        self.prev_grounded = self.grounded
        self.prev_ball_velocity = velocity

        if getattr(self, "camera_parented_to_ball", False):
            ball_pos = self.ball_np.getPos(self.render)
            gravity_up = self._get_gravity_up()
            ref_forward = Vec3(0, 1, 0)
            if abs(ref_forward.dot(gravity_up)) > 0.95:
                ref_forward = Vec3(1, 0, 0)
            ref_forward = ref_forward - gravity_up * ref_forward.dot(gravity_up)
            if ref_forward.lengthSquared() < 1e-8:
                ref_forward = Vec3(1, 0, 0)
            ref_forward.normalize()

            heading_offset = 180.0 if self._is_ceiling_mode() else 0.0
            yaw = math.radians(self.heading + heading_offset)
            orbit_planar = self._rotate_around_axis(-ref_forward, gravity_up, yaw)
            if orbit_planar.lengthSquared() < 1e-8:
                orbit_planar = Vec3(0, -1, 0)
            else:
                orbit_planar.normalize()
            target_rel = orbit_planar * self.camera_follow_distance + gravity_up * self.camera_height_offset
            current_rel = self.camera.getPos(self.camera_anchor) if hasattr(self, "camera_anchor") else self.camera.getPos()
            blend = min(1.0, dt * 8.0)
            new_rel = current_rel + (target_rel - current_rel) * blend
            corrected_world = self._enforce_camera_above_ball(
                ball_pos,
                self.camera_anchor.getPos(self.render) + new_rel,
                gravity_up,
                min_up_offset=max(0.5, self.camera_height_offset * 0.45),
            )
            corrected_world = self._clamp_camera_to_current_room_bounds(corrected_world, ball_pos)
            self.camera.setPos(self.camera_anchor, corrected_world)
            self.camera.lookAt(ball_pos + gravity_up * self.camera_height_offset, gravity_up)

            cam_pos = self.camera.getPos(self.render)
            self.camera_smoothed_pos = Vec3(cam_pos)
            horizontal_speed = (Vec3(velocity) - self._get_gravity_up() * velocity.dot(self._get_gravity_up())).length()
            if self.enable_scene_culling:
                self.scene_cull_timer -= dt
                if self.scene_cull_timer <= 0.0:
                    self._update_scene_culling(hyperspace_active)
                    self.scene_cull_timer = self.scene_cull_interval
            elif self.scene_cull_hidden:
                for vid in list(self.scene_cull_hidden):
                    visual = self.scene_visuals.get(vid)
                    if visual is not None and not visual.isEmpty():
                        visual.unstash()
                self.scene_cull_hidden.clear()
                self.scene_cull_miss_counts.clear()
            self.occlusion_update_timer -= dt
            self.transparency_update_timer -= dt
            if self.occlusion_update_timer <= 0.0:
                self._update_camera_occlusion(cam_pos, ball_pos)
                self.occlusion_update_timer = self.occlusion_update_interval
            if self.transparency_update_timer <= 0.0:
                self._refresh_visual_transparency()
                self.transparency_update_timer = self.transparency_update_interval
            if self.enable_ball_shadow:
                self.shadow_update_timer -= dt
                if self.shadow_update_timer <= 0.0:
                    self._update_ball_shadow()
                    self.shadow_update_timer = 1.0 / 16.0 if self.performance_mode else 1.0 / 24.0
            self._update_video_distortion(dt, horizontal_speed)
            return task.cont

        ball_pos = self.ball_np.getPos()
        gravity_up = self._get_gravity_up()
        ref_forward = Vec3(0, 1, 0)
        if abs(ref_forward.dot(gravity_up)) > 0.95:
            ref_forward = Vec3(1, 0, 0)
        ref_forward = ref_forward - gravity_up * ref_forward.dot(gravity_up)
        if ref_forward.lengthSquared() < 1e-8:
            ref_forward = Vec3(1, 0, 0)
        ref_forward.normalize()

        planar_vel = velocity - gravity_up * velocity.dot(gravity_up)
        planar_speed = planar_vel.length()
        if self.camera_manual_turn_hold <= 0.0:
            desired_planar_dir: Vec3 | None = None
            if (not manual_move_active) and bool(getattr(self, "player_ai_enabled", False)):
                ai_target = getattr(self, "player_ai_camera_target_pos", None)
                if isinstance(ai_target, Vec3):
                    to_target = Vec3(ai_target) - ball_pos
                    to_target -= gravity_up * to_target.dot(gravity_up)
                    if to_target.lengthSquared() > 1e-6:
                        to_target.normalize()
                        desired_planar_dir = to_target
            if desired_planar_dir is None and planar_speed > self.camera_auto_align_min_speed:
                desired_planar_dir = Vec3(planar_vel)
                desired_planar_dir.normalize()
            if desired_planar_dir is not None:
                sin_term = gravity_up.dot(ref_forward.cross(desired_planar_dir))
                cos_term = max(-1.0, min(1.0, ref_forward.dot(desired_planar_dir)))
                desired_heading = math.degrees(math.atan2(sin_term, cos_term))
                delta_heading = (desired_heading - self.heading + 180.0) % 360.0 - 180.0
                self.heading += delta_heading * min(1.0, dt * self.camera_auto_align_speed)

        heading_offset = 180.0 if self._is_ceiling_mode() else 0.0
        yaw = math.radians(self.heading + heading_offset)
        orbit_dir = self._rotate_around_axis(-ref_forward, gravity_up, yaw)
        if orbit_dir.lengthSquared() < 1e-8:
            orbit_dir = Vec3(0, -1, 0)
        else:
            orbit_dir.normalize()

        target = ball_pos + gravity_up * self.camera_height_offset
        desired_cam_pos = target + orbit_dir * self.camera_follow_distance

        if self.camera_smoothed_pos is None:
            self.camera_smoothed_pos = Vec3(desired_cam_pos)
        else:
            alpha = 1.0 - math.exp(-dt * 8.5)
            self.camera_smoothed_pos = self.camera_smoothed_pos + (desired_cam_pos - self.camera_smoothed_pos) * alpha

        resolved_cam = self._resolve_camera_tight(target, self.camera_smoothed_pos)
        resolved_cam = self._enforce_camera_above_ball(
            ball_pos,
            resolved_cam,
            gravity_up,
            min_up_offset=max(0.5, self.camera_height_offset * 0.45),
        )
        resolved_cam = self._clamp_camera_to_current_room_bounds(resolved_cam, ball_pos)

        self.camera_smoothed_pos = Vec3(resolved_cam)

        to_cam_planar = self.camera_smoothed_pos - ball_pos
        to_cam_planar -= gravity_up * to_cam_planar.dot(gravity_up)
        cam_dist_planar = to_cam_planar.length()
        min_cam_dist = self.ball_radius + self.camera_ball_clearance
        if cam_dist_planar < min_cam_dist:
            if cam_dist_planar < 1e-6:
                to_cam_planar = Vec3(orbit_dir)
                cam_dist_planar = to_cam_planar.length()
            to_cam_planar *= min_cam_dist / max(cam_dist_planar, 1e-6)
            desired_up = gravity_up * (self.camera_smoothed_pos - ball_pos).dot(gravity_up)
            self.camera_smoothed_pos = ball_pos + to_cam_planar + desired_up

        self.camera.setPos(self.camera_smoothed_pos)
        if self.enable_scene_culling:
            self.scene_cull_timer -= dt
            if self.scene_cull_timer <= 0.0:
                self._update_scene_culling(hyperspace_active)
                self.scene_cull_timer = self.scene_cull_interval
        elif self.scene_cull_hidden:
            for vid in list(self.scene_cull_hidden):
                visual = self.scene_visuals.get(vid)
                if visual is not None and not visual.isEmpty():
                    visual.unstash()
            self.scene_cull_hidden.clear()
            self.scene_cull_miss_counts.clear()
        self.occlusion_update_timer -= dt
        self.transparency_update_timer -= dt
        if self.occlusion_update_timer <= 0.0:
            self._update_camera_occlusion(self.camera_smoothed_pos, ball_pos)
            self.occlusion_update_timer = self.occlusion_update_interval
        if self.transparency_update_timer <= 0.0:
            self._refresh_visual_transparency()
            self.transparency_update_timer = self.transparency_update_interval
        if self.enable_ball_shadow:
            self.shadow_update_timer -= dt
            if self.shadow_update_timer <= 0.0:
                self._update_ball_shadow()
                self.shadow_update_timer = 1.0 / 16.0 if self.performance_mode else 1.0 / 24.0
        self._update_video_distortion(dt, planar_speed)
        self.camera.lookAt(target, gravity_up)

        return task.cont


def run() -> None:
    _configure_display_prc()
    _configure_gpu_prc()
    app = SoulSymphony()
    app.run()


if __name__ == "__main__":
    run()
