import pystk
import math


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    if current_vel < 40:
        acc_factor = 1.0 - (current_vel / 40)
    else:
        acc_factor = 0

    go_right = True if aim_point[0] > 0 else False

    # Gives angle from -90 deg to 90 deg
    steer_angle = -1.0 * math.degrees(math.atan(aim_point[1]/abs(aim_point[0])))

    if steer_angle < 30:
        action.drift = True

    if steer_angle > 80:
        acc_factor = 1 #acc_factor * 3
        steer_angle = 90
        action.nitro = True

    steer_factor = 1 - (abs(steer_angle) / 90.0)

    if not go_right:
        steer_factor = steer_factor * -1

    if action.drift:
        acc_factor = acc_factor / 5
        # steer_factor = -1.0 if steer_factor * 2 < -1.0 else 1.0 if steer_factor * 2 > 1.0 else steer_factor * 2
        steer_factor = steer_factor * 2
    action.acceleration = acc_factor
    action.steer = steer_factor

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
