import numpy as np
import torch
from data_extraction.data import world_loc_on_screen

from image_agent.image_net import load_model


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.image_net = load_model()
        self.team = 1
        self.num_players = 2

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image, soccer_state=None):
        steer = 0
        drift = False
        acceleration = 0.5
        brake = False

        player_image_tensor = torch.Tensor(player_image[0]).permute(2, 0, 1)
        prediction = self.image_net(player_image_tensor)
        aim_point = prediction[0].detach().numpy()
        
        actual_aim_point = world_loc_on_screen(player_state[0]['camera'], soccer_state['ball']['location'])
        print(prediction, actual_aim_point)
        
        steer = np.tanh(aim_point[0] / aim_point[1])
        # if aim_point[0] < 0 and steer > 0:
        #     steer *= -1
        # elif aim_point[0] > 0 and steer < 0:
        #     steer *= -1
            
        drifting = np.abs(steer) >= 0.820

        if drifting:
            drift = True
            if np.abs(steer) >= 0.870:
                brake = True
                acceleration = 0.011
            elif np.abs(steer) >= 0.95:
                brake = True
                acceleration = 0.001
            else:
                # acceleration = 0.
                pass

        return [
            dict(acceleration=acceleration,
                 steer=steer,
                 brake=brake,
                 drift=drift),
            # Remove this one
             dict(acceleration=acceleration,
                 steer=steer,
                 brake=brake,
                 drift=drift)
        ]
