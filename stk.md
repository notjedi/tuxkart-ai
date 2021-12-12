General ideas:

    - resize image to 480p res?
    - send 5 previous frames to the network
    - evaluate postion? (position difference and log loss? idk come up with better one)
    - 10 frames per second?
    - 1 lap and end it after x seconds if there is no progress
    - train mini and normal version of the game (mini with less number of params and generalizing to the actual game)
    - include kart properties as features so that it learns for heavier or lighter karts
    - remove track from list if it has been cleared
    - how does the agent learn how to correctly use different powerups?
    - use log penalty for difference between finish time (small difference low penalty, huge difference high penalty)

Reward Function:

    - direction
        * +1 for moving forward
        * -1 for moving backward (brake)
    - speed (use speed instead of acceleration - works better in grass and other conditions)
        (prolly using velocity or distance_down_track)
        * +1 for acceleration of 1
        * -1 for acceleration of 0
    - drift
        * +3 for drifting
    - nitro
        * +2 for using nitro
    - collecting items & zipper
        * +3 for collecting items
    - position (using finish time of different karts)
        * +10 for finishing the race * race position
        * +3 if position increases
        * -3 if position decreases
    - overtaking karts (using kart positions)
        * +3 if position increases
        * -3 if position decreases
    - using powerups
        * +5 for correctly using powerups (only if powerups are present)
    - jumping (consider velocity and distance)
        * -5


TODO:

    - re-evaluate the function so it doesn't do some unexpected things
    - check if kart is on track? (using SEMANTIC view?)
    - how to efficiently use powerups?
    - add more data about the track, race etc for a bigger model?

    https://github.com/philkr/pystk/blob/master/pystk_cpp/state.cpp
    https://github.com/philkr/pystk/blob/master/pystk_cpp/binding.cpp
    - encode Attachments:
        - NOTHING
		- PARACHUTE
		- ANVIL
		- BOMB
		- SWATTER
		- BUBBLEGUM_SHIELD

    - encode Powerup:
        - NOTHING
		- BUBBLEGUM
		- CAKE
		- BOWLING
		- ZIPPER
		- PLUNGER
		- SWITCH
		- SWATTER
		- RUBBERBALL
		- PARACHUTE
		- ANVIL

    - encode Kart details:
			race_result
			powerup
			finish_time
			jumping - what is this?

            # prolly in a bigger model than the normal one?
			wheel_base?
			velocity?
			shield_time?
			location?
			rotation?
			attachment?
			max_steer_angle?

Training Approach (REINFORCE):

    - collect data from N environments to the replay buffer using multiprocessing
    - play K episodes and save (state, action, reward, new_state)
    - only add data to replay buffer if it's useful
    - init weights
    - train on collected data using PPO
    - calc loss and backprop

Loss function:

    - Enropy bonus?
    TODO
