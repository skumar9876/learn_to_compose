from objects_env import World
import math
import numpy as np
import random


def init_test():
    """
    Tests that the environment is initialized properly. There should be 4
    5x5 blobs of non-zero entities.
    """
    all_grid_pos = [[x,y] for x in range(int(math.sqrt(World.GRID_SIZE))) \
        for y in range(int(math.sqrt(World.GRID_SIZE)))]
    # random object positions
    object_pos = [all_grid_pos[ind] \
        for ind in np.random.choice(World.GRID_SIZE, 3, replace=False)]
    env = World(object_pos=object_pos)
    img = env.reset(object_pos=object_pos)

    count = 0
    for y in range(len(img)):
        for x in range(len(img[0])):
            if img[y][x] != 0:
                count += 1
                n = img[y][x]

                image_window = np.squeeze(
                    img[y:y+env.DILATION_FACTOR,
                    x:x+env.DILATION_FACTOR]
                )
                test_window = n \
                    * np.ones((env.DILATION_FACTOR, env.DILATION_FACTOR))
                assert np.array_equal(image_window, test_window)

                img[y:y+env.DILATION_FACTOR, x:x+env.DILATION_FACTOR] = \
                    np.expand_dims(
                        np.zeros((env.DILATION_FACTOR, env.DILATION_FACTOR)),
                        axis=-1
                    )
    assert count == env.NUM_OBJECTS + 1


def find_agent(env, img):
    """Finds and returns the upper left coordinates of the agent's position."""
    for y in range(len(img)):
        for x in range(len(img[0])):
            if img[y][x] == env.AGENT_COLOR:
                return y, x


def find_specified_object(env, img, goal):
    """Finds and returns the upper left coordinates of the object's position
    specified by its goal type (-1: avoid, 0: neutral, 1:seek"""
    specified_obj_ind = env.goal_arr.index(goal)
    specified_obj_color = env.object_dict_inverted[specified_obj_ind]
    for y in range(len(img)):
        for x in range(len(img[0])):
            if img[y][x] == specified_obj_color:
                return y, x


def object_movement_test():
    """ test that avoid object motion occurs successfully
    target object remains stationary
    for now even avoid object remains stationary"""
    # Create test environment.
    env = World()
    img = env.reset()
    target_obj_pos = find_specified_object(env, img, 1)
    avoid_obj_pos = find_specified_object(env, img, -1)
    # move objects
    env.move_avoid_object()
    img = env.construct_image()
    new_target_obj_pos = find_specified_object(env, img, 1)
    new_avoid_obj_pos = find_specified_object(env, img, -1)
    if not new_avoid_obj_pos:
        # avoid object is under agent
        assert [env.y, env.x] in env.object_positions_2d
        new_avoid_obj_pos = \
            [env.y*env.DILATION_FACTOR, env.x*env.DILATION_FACTOR]
    if not new_target_obj_pos:
        # avoid object is on top of target object
        assert env.object_positions_2d.count(
            [pos/env.DILATION_FACTOR for pos in new_avoid_obj_pos]) == 2
        new_target_obj_pos = new_avoid_obj_pos

    # target object should remain stationary
    assert new_target_obj_pos == target_obj_pos
    # avoid object position should be a single step away
    assert new_avoid_obj_pos[0] == avoid_obj_pos[0] \
        and (new_avoid_obj_pos[1] == avoid_obj_pos[1] + env.DILATION_FACTOR \
            or new_avoid_obj_pos[1] == avoid_obj_pos[1] - env.DILATION_FACTOR) \
        or new_avoid_obj_pos[1] == avoid_obj_pos[1] \
        and (new_avoid_obj_pos[0] == avoid_obj_pos[0] + env.DILATION_FACTOR \
            or new_avoid_obj_pos[0] == avoid_obj_pos[0] - env.DILATION_FACTOR)


def movement_test():
    """Tests that the agent moves in each direction correctly."""

    # Create test environment.
    env = World()
    # The min value that x or y position can have.
    min_pos = 0
    # The max value that x or y position can have.
    max_pos = env.DILATION_FACTOR * math.sqrt(env.GRID_SIZE) \
        - env.DILATION_FACTOR

    # Test that the agent moves up correctly.
    img = env.reset()
    agent_y, agent_x = find_agent(env, img)
    img, _, _= env.step(env.UP)
    assert find_agent(env, img) == \
        (max(min_pos, agent_y - env.DILATION_FACTOR), agent_x)

    # Test that the agent moves down correctly.
    env = World()
    img = env.reset()
    agent_y, agent_x = find_agent(env, img)
    img, _, _ = env.step(env.DOWN)
    assert find_agent(env, img) == \
        (min(max_pos, agent_y + env.DILATION_FACTOR), agent_x)

    # Test that the agent moves left correctly.
    env = World()
    img = env.reset()
    agent_y, agent_x = find_agent(env, img)
    img, _, _ = env.step(env.LEFT)
    assert find_agent(env, img) == \
        (agent_y, max(min_pos, agent_x - env.DILATION_FACTOR))

    # Test that the agent moves right correctly.
    env = World()
    img = env.reset()
    agent_y, agent_x = find_agent(env, img)
    img, _, _ = env.step(env.RIGHT)
    assert find_agent(env, img) == \
        (agent_y, min(max_pos, agent_x + env.DILATION_FACTOR))


def collision_test():
    """
    Tests that when the agent collides with an object, it takes the place of
    that object. When the agent then moves again, the object should re-appear.
    """
    # Test that the agent can move on top of stationary objects
    env = World()
    # The min value that x or y position can have.
    min_pos = 0
    # The max value that x or y position can have.
    max_pos = env.DILATION_FACTOR * math.sqrt(env.GRID_SIZE) \
        - env.DILATION_FACTOR
    img = env.reset([[1,1],[2,2],[3,3]])
    agent_y, agent_x = find_agent(env, img)
    # object positions are hardcoded at (1,1), (2,2), (3,3)
    if agent_y > 2*env.DILATION_FACTOR:
        while agent_y > 2*env.DILATION_FACTOR:
            img, _, _ = env.step(env.UP)
            assert find_agent(env, img) == \
                (max(min_pos, agent_y - env.DILATION_FACTOR), agent_x)
            agent_y, agent_x = find_agent(env, img)
    if agent_y < 2*env.DILATION_FACTOR:
        while agent_y < 2*env.DILATION_FACTOR:
            img, _, _ = env.step(env.DOWN)
            assert find_agent(env, img) == \
                (min(max_pos, agent_y + env.DILATION_FACTOR), agent_x)
            agent_y, agent_x = find_agent(env, img)
    if agent_x > 2*env.DILATION_FACTOR:
        while agent_x > 2*env.DILATION_FACTOR:
            img, _, _ = env.step(env.LEFT)
            assert find_agent(env, img) == \
                (agent_y, max(min_pos, agent_x - env.DILATION_FACTOR))
            agent_y, agent_x = find_agent(env, img)
    if agent_x < 2*env.DILATION_FACTOR:
        while agent_x < 2*env.DILATION_FACTOR:
            img, _, _ = env.step(env.RIGHT)
            assert find_agent(env, img) == \
                (agent_y, min(max_pos, agent_x + env.DILATION_FACTOR))
            agent_y, agent_x = find_agent(env, img)
    # now the agent is at (1,1)
    assert find_agent(env, img) == \
        (2*env.DILATION_FACTOR, 2*env.DILATION_FACTOR)
    # now move the agent to the left and confirm that the object reappears
    img, _, _ = env.step(env.LEFT)
    assert find_agent(env, img) == (2*env.DILATION_FACTOR, env.DILATION_FACTOR)
    assert img[2*env.DILATION_FACTOR, 2*env.DILATION_FACTOR] == \
        env.object_dict_inverted[1]

    # target object moves over stationary object then moves back it reappers
    env = World()
    img = env.reset([[1,1],[2,2],[3,3]])
    target_obj_pos = find_specified_object(env, img, 1)
    avoid_obj_pos = find_specified_object(env, img, -1)
    # fool avoid object into thinking agent is on target object
    env.y, env.x = [pos/env.DILATION_FACTOR for pos in target_obj_pos]
    # move avoid object until it's on top of target object
    while avoid_obj_pos != target_obj_pos:
        env.move_avoid_object()
        img = env.construct_image()
        avoid_obj_pos = find_specified_object(env, img, -1)

    # now move one more time and verify that target object reappears
    env.y, env.x = [(pos-1)/env.DILATION_FACTOR for pos in target_obj_pos]
    env.move_avoid_object()
    img = env.construct_image()
    avoid_obj_pos = find_specified_object(env, img, -1)
    new_target_obj_pos = find_specified_object(env, img, 1)
    assert new_target_obj_pos == target_obj_pos


def reward_terminality_test():
    """
    Tests the reward and terminal conditions of each task
    """
    # avoid task
    env = World(task=['target'])
    # get a random assignment of object positions
    goal_arr = [-1,0,1]
    random.shuffle(goal_arr)
    img = env.reset(goal_arr=goal_arr)
    target_obj_pos = env.object_positions_2d[env.target_object_index]
    # run the episode until terminal condition
    done = False
    r = env.STEP_COST
    while not done:
        assert r == env.STEP_COST
        # move agent towards 'target' object
        if env.x < target_obj_pos[1]:
            a = env.RIGHT
        elif env.x > target_obj_pos[1]:
            a = env.LEFT
        elif env.y < target_obj_pos[0]:
            a = env.DOWN
        elif env.y > target_obj_pos[0]:
            a = env.UP
        img, r, done = env.step(a)
    # target object must be hidden under agent
    target_obj_pos = find_specified_object(env, img, 1)
    assert target_obj_pos == None
    agent_pos = find_agent(env, img)
    assert agent_pos != None
    assert r == 1

    # target task
    env = World(task=['avoid'])
    # get a random assignment of object positions
    goal_arr = [-1,0,1]
    random.shuffle(goal_arr)
    img = env.reset(goal_arr=goal_arr)
    # run the episode until terminal condition
    env.MAX_STEPS = 10
    done = False
    r = 0
    while not done:
        assert r == 0
        img, r, done = env.step(random.choice(range(env.NUM_ACTIONS)))
    if env.num_steps == env.MAX_STEPS:
        if r > 0:
            # agent escaped!
            avoid_obj_pos = find_specified_object(env, img, -1)
            assert avoid_obj_pos != None
            agent_pos = find_agent(env, img)
            assert agent_pos != None
            assert avoid_obj_pos != agent_pos
            assert r == 1
        else:
            # avoid object must be hidden under agent
            avoid_obj_pos = find_specified_object(env, img, -1)
            assert avoid_obj_pos == None
            agent_pos = find_agent(env, img)
            assert agent_pos != None
            assert r == -1
    else:
        # avoid object must be hidden under agent
        avoid_obj_pos = find_specified_object(env, img, -1)
        assert avoid_obj_pos == None
        agent_pos = find_agent(env, img)
        assert agent_pos != None
        assert r == -1


# Test initialization.
for i in xrange(100):
    init_test()
print "---- INITIALIZATION TESTS SUCCESSFUL..."

# Test agent movement.
for i in xrange(100):
    movement_test()
print "---- AGENT MOVEMENT TESTS SUCCESSFUL..."

for i in xrange(100):
    object_movement_test()
print "---- OBJECT MOVEMENT TESTS SUCCESSFUL..."

for i in xrange(100):
    collision_test()
print "---- COLLISION TESTS SUCCESSFUL..."

for i in xrange(100):
    reward_terminality_test()
print "---- REWARD/TERMINAL TEST SUCCESSFUL!..."


print "---- ALL TESTS SUCCEED! :D ----"
