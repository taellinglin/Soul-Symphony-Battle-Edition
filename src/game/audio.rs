use macroquad::rand::gen_range;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
pub struct AudioEvents {
    pub jump: bool,
    pub rocket: bool,
    pub bomb: bool,
    pub spin: bool,
    pub throw_attack: bool,
    pub boss_enter: bool,
}

#[derive(Clone)]
struct SoundData {
    bytes: Arc<Vec<u8>>,
}

impl SoundData {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::new(bytes),
        }
    }
}

#[derive(Default)]
struct SoundBank {
    clips: Vec<SoundData>,
    last_idx: Option<usize>,
}

impl SoundBank {
    fn pick(&mut self) -> Option<SoundData> {
        if self.clips.is_empty() {
            return None;
        }
        if self.clips.len() == 1 {
            self.last_idx = Some(0);
            return Some(self.clips[0].clone());
        }
        let mut idx = gen_range(0, self.clips.len());
        if let Some(last) = self.last_idx {
            if idx == last {
                idx = (idx + 1) % self.clips.len();
            }
        }
        self.last_idx = Some(idx);
        Some(self.clips[idx].clone())
    }
}

#[allow(dead_code)]
pub struct AudioSystem {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    bgm_tracks: Vec<(String, SoundData)>,
    bgm_track_path: Option<String>,
    bgm_track: Option<SoundData>,
    bgm_sink: Option<Sink>,
    bgm_volume: f32,
    bgm_mode: String,
    soundtrack_override: Option<SoundData>,
    voiceover_timer: f32,
    voiceover_min_gap: f32,
    voiceover_active: Option<SoundData>,
    voiceover_sink: Option<Sink>,
    voiceover_volume_scale: f32,
    boss_bank: SoundBank,
    win_bank: SoundBank,
    gameover_bank: SoundBank,
    levelup_bank: SoundBank,
    kill_bank: SoundBank,
    critical_bank: SoundBank,
    sfx_jump: Option<SoundData>,
    sfx_attack: Option<SoundData>,
    sfx_attack_spin: Option<SoundData>,
    sfx_attack_bomb: Option<SoundData>,
    sfx_attack_homing: Option<SoundData>,
    sfx_warp: Option<SoundData>,
    sfx_pickup: Option<SoundData>,
    sfx_monster_hit: Option<SoundData>,
    sfx_monster_guard: Option<SoundData>,
    sfx_monster_die: Option<SoundData>,
    active_sfx: Vec<Sink>,
}

#[allow(dead_code)]
impl AudioSystem {
    pub async fn new() -> Self {
        let (_stream, stream_handle) = OutputStream::try_default()
            .expect("Failed to initialize audio device");
        let bgm_tracks = load_bgm_tracks(&[
            "assets/audio/bgm",
            "assets/audio/potential_tracks",
        ])
        .await;
        let soundtrack_override = load_first_sound(&[
            "assets/audio/bgm/Soundtrack.ogg",
            "assets/audio/potential_tracks/Soundtrack.ogg",
        ])
        .await;
        let boss_bank = load_bank("assets/audio/soundfx/boss_").await;
        let win_bank = load_bank("assets/audio/soundfx/win_").await;
        let gameover_bank = load_bank("assets/audio/soundfx/gameover_").await;
        let levelup_bank = load_bank("assets/audio/soundfx/levelup_").await;
        let kill_bank = load_bank("assets/audio/soundfx/kill_").await;
        let critical_bank = load_bank("assets/audio/soundfx/critical_damage_").await;

        let sfx_jump = load_first_sound(&[
            "assets/audio/soundfx/qigongjump.wav",
            "assets/audio/soundfx/qigongjump.ogg",
        ])
        .await;
        let sfx_attack = load_first_sound(&["assets/audio/soundfx/attack.wav"]).await;
        let sfx_attack_spin = load_first_sound(&["assets/audio/soundfx/attackspin.wav"]).await;
        let sfx_attack_bomb = load_first_sound(&["assets/audio/soundfx/attackbomb.wav"]).await;
        let sfx_attack_homing = load_first_sound(&["assets/audio/soundfx/attackhomingmissle.wav"])
            .await;
        let sfx_warp = load_first_sound(&["assets/audio/soundfx/warp.wav"]).await;
        let sfx_pickup = load_first_sound(&[
            "assets/audio/soundfx/pickup.wav",
            "assets/audio/soundfx/pickuphealth.wav",
        ])
        .await;
        let sfx_monster_hit = load_first_sound(&["assets/audio/soundfx/monsterhit.wav"]).await;
        let sfx_monster_guard =
            load_first_sound(&["assets/audio/soundfx/monsterguard.wav"]).await;
        let sfx_monster_die = load_first_sound(&["assets/audio/soundfx/monsterdie.wav"]).await;

        let mut audio = Self {
            _stream,
            stream_handle,
            bgm_tracks,
            bgm_track_path: None,
            bgm_track: None,
            bgm_sink: None,
            bgm_volume: 0.48,
            bgm_mode: "".to_string(),
            soundtrack_override,
            voiceover_timer: 0.0,
            voiceover_min_gap: 0.9,
            voiceover_active: None,
            voiceover_sink: None,
            voiceover_volume_scale: 1.0,
            boss_bank,
            win_bank,
            gameover_bank,
            levelup_bank,
            kill_bank,
            critical_bank,
            sfx_jump,
            sfx_attack,
            sfx_attack_spin,
            sfx_attack_bomb,
            sfx_attack_homing,
            sfx_warp,
            sfx_pickup,
            sfx_monster_hit,
            sfx_monster_guard,
            sfx_monster_die,
            active_sfx: Vec::new(),
        };
        audio.set_bgm_mode("normal");
        audio
    }

    pub fn update(&mut self, dt: f32) {
        if self.voiceover_timer > 0.0 {
            self.voiceover_timer = (self.voiceover_timer - dt).max(0.0);
        }
        self.active_sfx.retain(|sink| !sink.empty());
        if let Some(voiceover) = &self.voiceover_sink {
            if voiceover.empty() {
                self.voiceover_sink = None;
            }
        }
    }

    pub fn apply_events(&mut self, events: AudioEvents) {
        let sfx_jump = self.sfx_jump.clone();
        let sfx_attack = self.sfx_attack.clone();
        let sfx_attack_spin = self.sfx_attack_spin.clone();
        let sfx_attack_bomb = self.sfx_attack_bomb.clone();
        let sfx_attack_homing = self.sfx_attack_homing.clone();

        if events.jump {
            self.play_sfx(sfx_jump, 0.9);
        }
        if events.rocket {
            self.play_sfx(sfx_attack_homing.or(sfx_attack.clone()), 0.78);
        }
        if events.bomb {
            self.play_sfx(sfx_attack_bomb.or(sfx_attack.clone()), 0.8);
        }
        if events.spin {
            self.play_sfx(sfx_attack_spin.or(sfx_attack.clone()), 0.8);
        }
        if events.throw_attack {
            self.play_sfx(sfx_attack, 0.72);
        }
        if events.boss_enter {
            self.play_boss_voiceover();
        }
    }

    pub fn set_bgm_mode(&mut self, mode: &str) {
        let mode_key = mode.trim().to_lowercase();
        if mode_key == self.bgm_mode {
            return;
        }
        self.bgm_mode = mode_key.clone();

        if mode_key == "boss" {
            if let Some(sound) = self.find_bgm_by_stem("boss") {
                self.play_bgm(sound, "boss".to_string());
                return;
            }
        }

        if mode_key != "boss" {
            if let Some(sound) = self.soundtrack_override.clone() {
                self.play_bgm(sound, "soundtrack".to_string());
                return;
            }
        }

        let target_stem = if mode_key == "boss" { "boss" } else { "soundtrack" };
        if let Some(sound) = self.find_bgm_by_stem(target_stem) {
            self.play_bgm(sound, target_stem.to_string());
            return;
        }

        self.play_random_bgm();
    }

    fn play_bgm(&mut self, sound: SoundData, path: String) {
        if let Some(current) = &self.bgm_sink {
            current.stop();
        }
        self.bgm_track = Some(sound.clone());
        self.bgm_track_path = Some(path);
        self.bgm_sink = self
            .build_sink(&sound, self.bgm_volume.clamp(0.0, 1.0), true);
    }

    fn play_random_bgm(&mut self) {
        if self.bgm_tracks.is_empty() {
            self.bgm_track = None;
            self.bgm_track_path = None;
            return;
        }
        let idx = gen_range(0, self.bgm_tracks.len());
        let (path, sound) = self.bgm_tracks[idx].clone();
        self.play_bgm(sound, path);
    }

    fn find_bgm_by_stem(&self, stem: &str) -> Option<SoundData> {
        let stem_key = stem.to_lowercase();
        for (path, sound) in &self.bgm_tracks {
            let name = Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if name == stem_key {
                return Some(sound.clone());
            }
        }
        None
    }

    fn play_boss_voiceover(&mut self) {
        if let Some(clip) = self.boss_bank.pick() {
            self.play_voiceover(clip);
        }
    }

    pub fn play_win_voiceover(&mut self) {
        if let Some(clip) = self.win_bank.pick() {
            self.play_voiceover(clip);
        }
    }

    pub fn play_gameover_voiceover(&mut self) {
        if let Some(clip) = self.gameover_bank.pick() {
            self.play_voiceover(clip);
        }
    }

    pub fn play_levelup_voiceover(&mut self) {
        if let Some(clip) = self.levelup_bank.pick() {
            self.play_voiceover(clip);
        }
    }

    pub fn play_kill_voiceover(&mut self) {
        if let Some(clip) = self.kill_bank.pick() {
            self.play_voiceover(clip);
        }
    }

    pub fn play_critical_voiceover(&mut self) {
        if let Some(clip) = self.critical_bank.pick() {
            self.play_voiceover(clip);
        }
    }

    fn play_voiceover(&mut self, clip: SoundData) {
        if self.voiceover_timer > 0.0 {
            return;
        }
        if let Some(active) = &self.voiceover_sink {
            active.stop();
        }
        self.voiceover_active = Some(clip.clone());
        self.voiceover_timer = self.voiceover_min_gap;
        self.voiceover_sink = self.build_sink(
            &clip,
            (0.95 * self.voiceover_volume_scale).clamp(0.0, 1.0),
            false,
        );
    }

    fn play_sfx(&mut self, sound: Option<SoundData>, volume: f32) {
        if let Some(sfx) = sound {
            if let Some(sink) = self.build_sink(&sfx, volume.clamp(0.0, 1.0), false) {
                self.active_sfx.push(sink);
            }
        }
    }

    fn build_sink(&self, sound: &SoundData, volume: f32, looped: bool) -> Option<Sink> {
        let cursor = Cursor::new((*sound.bytes).clone());
        let decoder = Decoder::new(cursor).ok()?;
        let sink = Sink::try_new(&self.stream_handle).ok()?;
        sink.set_volume(volume);
        if looped {
            sink.append(decoder.repeat_infinite());
        } else {
            sink.append(decoder);
        }
        Some(sink)
    }
}

async fn load_first_sound(paths: &[&str]) -> Option<SoundData> {
    for path in paths {
        if Path::new(path).exists() {
            if !is_audio_ext(Path::new(path)) {
                continue;
            }
            if let Ok(bytes) = std::fs::read(path) {
                return Some(SoundData::new(bytes));
            }
        }
    }
    None
}

async fn load_bank(dir: &str) -> SoundBank {
    let mut bank = SoundBank::default();
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if is_audio_ext(&path) {
                files.push(path);
            }
        }
    }
    files.sort();
    for path in files {
        if let Some(path_str) = path.to_str() {
            if let Ok(bytes) = std::fs::read(path_str) {
                bank.clips.push(SoundData::new(bytes));
            }
        }
    }
    bank
}

async fn load_bgm_tracks(dirs: &[&str]) -> Vec<(String, SoundData)> {
    let mut tracks: Vec<(String, SoundData)> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if is_audio_ext(&path) {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    for path in files {
        if let Some(path_str) = path.to_str() {
            if let Ok(bytes) = std::fs::read(path_str) {
                tracks.push((path_str.to_string(), SoundData::new(bytes)));
            }
        }
    }
    tracks
}

fn is_audio_ext(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => matches!(ext.to_lowercase().as_str(), "wav" | "ogg"),
        None => false,
    }
}
