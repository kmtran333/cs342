from typing import List, Tuple

import pandas as pd
import numpy as np
import pystk
from PIL import Image, ImageDraw
from os import makedirs

NUM_PLAYERS = 2
TRACK_NAME = 'icy_soccer_field'

SCREEN_WIDTH = 128
SCREEN_HEIGHT = 96


def init_pystk():
    game_config = pystk.GraphicsConfig.ld()
    game_config.screen_width = SCREEN_WIDTH
    game_config.screen_height = SCREEN_HEIGHT
    pystk.init(game_config)


def create_soccer_match() -> pystk.Race:
    race_config = pystk.RaceConfig(track=TRACK_NAME,
                                   mode=pystk.RaceConfig.RaceMode.SOCCER,
                                   num_kart=2 * NUM_PLAYERS)
    race_config.players.pop()
    for i in range(2 * NUM_PLAYERS):
        team = 0 if i < NUM_PLAYERS else 1
        player_config = pystk.PlayerConfig(
            controller=pystk.PlayerConfig.Controller.AI_CONTROL, team=team)
        race_config.players.append(player_config)
    race = pystk.Race(race_config)
    return race


def calculate_middle_of_goal_line(goal_left: List[float],
                                  goal_right: List[float]) -> List[float]:
    x = (goal_left[0] + goal_right[0] / 2)
    y = (goal_left[1] + goal_right[1] / 2)
    z = (goal_left[2] + goal_right[2] / 2)
    return [x, y, z]


def world_loc_on_screen(camera: pystk.Camera, loc: List[float]) -> List[float]:
    proj = np.array(camera.projection).T
    view = np.array(camera.view).T

    projected_view_in_2d = proj @ view @ np.array(list(loc) + [1])
    return np.clip(
        np.array([
            projected_view_in_2d[0] / projected_view_in_2d[-1],
            -projected_view_in_2d[1] / projected_view_in_2d[-1]
        ]), -1, 1)


def composite_and_show_camera_view_and_loc(camera: pystk.Camera, im: Image,
                                           *locs):
    middle_of_screen = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
    radius = 1

    locs = [world_loc_on_screen(camera, loc) for loc in locs]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    for idx, loc in enumerate(locs):
        cords = middle_of_screen * (1 + loc)
        cords = [
            cords[0] - radius, 
            cords[1] - radius, 
            cords[0] + radius,
            cords[1] + radius
        ]

        draw = ImageDraw.Draw(im)
        draw.ellipse((cords[0], cords[1], cords[2], cords[3]),
                     fill=colors[idx])
    return im

def save_training_data(images: List, aim_points: List[Tuple[float]]):
    try:
        makedirs('data')
    except OSError:
        pass

    for idx, (image, aim_point) in enumerate(zip(images, aim_points)):
        image.save('data/{}_{}.png'.format(TRACK_NAME, idx))

        df = pd.DataFrame()
        df['puck_x'] = [aim_point[1][0]]
        df['puck_y'] = [aim_point[1][1]]
        df.to_csv('data/{}_{}.csv'.format(TRACK_NAME, idx), header=False, index=False)
        
if __name__ == '__main__':
    init_pystk()
    race = create_soccer_match()

    n_steps = 6000
    try:
        state = pystk.WorldState()

        race.start()
        race.step()

        images = []
        aim_points = [] # Kart, Goal, Puck
        for step in range(n_steps):
            state.update()
            soccer = state.soccer

            player = state.players[0]
            camera = player.camera
            kart = player.kart

            kart_loc = kart.location
            goal_loc = calculate_middle_of_goal_line(soccer.goal_line[1][0],
                                                     soccer.goal_line[1][1])
            puck_loc = soccer.ball.location

            # ball_goal_distance = np.linalg.norm(
            #     np.array(puck_loc) - np.array(goal_loc))

            race.step()

            image_arr = np.array(race.render_data[0].image)
            image = Image.fromarray(image_arr)

            # image = composite_and_show_camera_view_and_loc(
            #     camera, image, kart_loc, puck_loc, goal_loc)
            images.append(image)
            aim_points.append((world_loc_on_screen(camera, kart_loc), 
                               world_loc_on_screen(camera, puck_loc), 
                               world_loc_on_screen(camera, goal_loc)))

        race.stop()
        save_training_data(images, aim_points)
        # images[0].save('./result.gif',
        #                save_all=True,
        #                optimize=False,
        #                append_images=images[1:],
        #                loop=0)

    except Exception as e:
        print(e)
    finally:
        race.stop()
        del race

    pystk.clean()
