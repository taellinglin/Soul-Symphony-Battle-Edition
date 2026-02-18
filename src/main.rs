use macroquad::prelude::*;

mod game;
mod world;

fn window_conf() -> Conf {
    Conf {
        window_title: "Soul Symphony Battle Edition".to_string(),
        fullscreen: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut game = game::Game::new().await;

    loop {
        let dt = get_frame_time();
        game.update(dt);
        if game.should_quit() {
            break;
        }

        clear_background(BLACK);
        game.draw();

        next_frame().await
    }
}
