import numpy as np
import math


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None

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

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def act(self, player_state, player_image, soccer_state=None):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight

        p1_state = player_state[0]
        p2_state = player_state[1]

        # Actions for Player 1 only
        proj = np.array(p1_state['camera']['projection']).T
        view = np.array(p1_state['camera']['view']).T

        cur_vel_1 = np.linalg.norm(p1_state['kart']['velocity'])
        if cur_vel_1 < 10:
            acc_factor = 1.0 - (cur_vel_1 / 40)
        else:
            acc_factor = 0

        img_loc_puck = self._to_image(soccer_state['ball']['location'], proj, view)
        puck_x = img_loc_puck[0]
        puck_y = img_loc_puck[1]

        go_right = True if puck_x > 0 else False
        drift = False

        steer_angle = -1.0 * math.degrees(math.atan(puck_y / abs(puck_x)))

        if steer_angle < 30:
            drift = True

        if steer_angle > 80:
            acc_factor = 1  # acc_factor * 3
            steer_angle = 90

        steer_factor = 1 - (abs(steer_angle) / 90.0)

        if not go_right:
            steer_factor = steer_factor * -1

        if drift:
            acc_factor = acc_factor / 5
            # steer_factor = -1.0 if steer_factor * 2 < -1.0 else 1.0 if steer_factor * 2 > 1.0 else steer_factor * 2
            steer_factor = steer_factor * 2

        p1_output = [dict(acceleration=0.2, steer=steer_factor, drift=drift)]
        return p1_output + [dict(acceleration=0, steer=0)]
