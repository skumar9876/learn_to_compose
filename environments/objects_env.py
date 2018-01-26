import math
import numpy as np
import random
from copy import deepcopy


class World():

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    AGENT_COLOR = 10
    EMPTY_COLOR = 0
    STEP_COST = -0.01
    GRID_SIZE = 25
    MAX_STEPS = 2 * GRID_SIZE
    DILATION_FACTOR = 3
    NUM_ACTIONS = 4
    NUM_OBJECTS = 3

    def __init__(
            self, task=['avoid'], object_pos=None,
            goal_arr=[-1, 0, 1], teleport=False):
        """Initializes the environment.

        Args:
            task: Array of tasks in the environment - may contain 'target'
                  and/or 'avoid.'
            goal_arr: Array indicating index of object to target and/or index
                      of object to avoid.
        """
        self.task = task
        self.goal_arr = deepcopy(goal_arr)
        self.reset(object_pos)
        self.teleport = teleport

        assert math.sqrt(self.GRID_SIZE) - int(math.sqrt(self.GRID_SIZE)) == 0

    def reset(self, object_pos=None, max_steps=None):
        if not max_steps:
            self.max_steps = self.MAX_STEPS
        else:
            self.max_steps = max_steps

        self.side_len = int(math.sqrt(self.GRID_SIZE))
        if not object_pos:
            # if object positions not manually specified, drawn randomly
            object_pos_flat = np.random.choice(
                self.GRID_SIZE, self.NUM_OBJECTS, replace=False)
            object_pos_2d = np.unravel_index(
                object_pos_flat, (self.side_len, self.side_len))
            self.object_positions_2d = [[object_pos_2d[0][i], object_pos_2d[1][i]] \
                for i in xrange(self.NUM_OBJECTS)]
        else:
            self.object_positions_2d = deepcopy(object_pos)

        self.object_dict = {(i + 1) * (255 / float(self.NUM_OBJECTS)) : i \
            for i in xrange(self.NUM_OBJECTS)}
        self.object_dict_inverted = {y:x \
            for x,y in self.object_dict.iteritems()}

        self.world = self.generate_map()
        self.image = self.construct_image()

        self.num_steps = 0

        self.target_object_index, self.avoid_object_index = \
            self.get_specified_object_indices()

        return np.expand_dims(self.image.copy(), axis=-1)

    def reset_test(self, object_pos=None, goal_arr=[-1, 0, 1]):
        """Version of reset that returns both the dilated image and the original world."""
        self.reset()
        return np.expand_dims(self.world.copy(), axis=-1), np.expand_dims(
            self.image.copy(), axis=-1)

    def get_state_size(self):
        return [np.shape(self.image)[0], np.shape(self.image)[1]]

    def get_num_actions(self):
        return self.NUM_ACTIONS

    def generate_map(self):
        world = []

        # Generate the 1-dimensional world and put the objects.
        world = self.EMPTY_COLOR * np.ones(self.GRID_SIZE)

        # convert 2d object positions in 1-d world
        object_positions = np.ravel_multi_index(
            ([obj[0] for obj in self.object_positions_2d],\
                [obj[1] for obj in self.object_positions_2d]),
            (self.side_len, self.side_len)
        )

        for i in xrange(len(object_positions)):
            pos = object_positions[i]
            world[pos] = self.object_dict_inverted[i]

        # Agent starting position.
        agent_pos = 0  # random.randint(0, self.side_len - 1)

        # Agent cannot start at an object position.
        while world[agent_pos] != self.EMPTY_COLOR:
            agent_pos = random.randint(0, self.side_len - 1)

        # Make a 2-dimensional copy of the world.
        world_2d = world.reshape((self.side_len, self.side_len))

        self.x = agent_pos % self.side_len
        self.y = agent_pos / self.side_len

        world_2d[self.y][self.x] = self.AGENT_COLOR

        return world_2d

    def construct_image(self):
        row_dilated = np.repeat(self.world, self.DILATION_FACTOR, axis=0)
        col_dilated = np.repeat(row_dilated, self.DILATION_FACTOR, axis=1)
        return col_dilated

    def get_specified_object_indices(self):
        """
        Get the indices of the specified target object
        and the object to avoid.
        Value of 1 means agent should try to go towards that object.
        Value of -1 means agent should try to avoid that object.
        """
        return self.goal_arr.index(1), self.goal_arr.index(-1)

    def move_avoid_object(self):
        """Moves the avoid object towards the agent."""
        obj = self.object_positions_2d[self.avoid_object_index]
        old_x = obj[1]
        old_y = obj[0]

        delta_x = 0
        delta_y = 0

        # Determine which delta x direction to move the avoid object.
        if old_x < self.x:
            delta_x = 1
        elif old_x > self.x:
            delta_x = -1

        if old_y < self.y:
            delta_y = 1
        elif old_y > self.y:
            delta_y = -1

        if delta_x and delta_y:
            # Whether to move the object in the x or y direction.
            move_val = random.randint(0, 1)

            if not move_val:
                delta_y = 0
            else:
                delta_x = 0

        new_x = old_x + delta_x
        new_y = old_y + delta_y

        if self.teleport:
            # Avoid object teleports across the borders.
            new_x = new_x % self.side_len
            new_y = new_y % self.side_len

        # Checks that the new_x and new_y values are within
        # the bounds of the map.
        if (new_x >= 0 and new_x < self.side_len
                and new_y >= 0 and new_y < self.side_len):

            if self.world[old_y, old_x] != self.AGENT_COLOR:
                # Avoid object could have been on top of stationary object,
                # in which case the stationary object should re-appear when
                # the avoid object moves off of that location.
                self.world[old_y][old_x] = self.EMPTY_COLOR
                self.object_positions_2d[self.avoid_object_index] = [new_y, new_x]
                try:
                    obj_ind = self.object_positions_2d.index([old_y, old_x])
                except ValueError:
                    obj_ind = -1
                if obj_ind >= 0 and not(obj_ind == self.avoid_object_index):
                    self.world[old_y][old_x] = self.object_dict_inverted[obj_ind]


            if self.world[new_y, new_x] != self.AGENT_COLOR:
                self.world[new_y, new_x] = self.object_dict_inverted[self.avoid_object_index]

        else:
            new_x = old_x
            new_y = old_y


    def move_agent(self, action):
        new_x = self.x
        new_y = self.y

        if action == self.UP:
            new_y = self.y - 1
        if action == self.DOWN:
            new_y = self.y + 1
        if action == self.LEFT:
            new_x = self.x - 1
        if action == self.RIGHT:
            new_x = self.x + 1

        if self.teleport:
            # Agent teleports across the borders.
            new_x = new_x % self.side_len
            new_y = new_y % self.side_len

        if (new_x >= 0 and new_x < self.side_len
                and new_y >= 0 and new_y < self.side_len):
            # Agent could have been on top of stationary object,
            # in which case the stationary object should re-appear when
            # the agent moves off of that location.
            try:
                obj_ind = self.object_positions_2d.index([self.y,self.x])
            except ValueError:
                obj_ind = -1
            if obj_ind >= 0:
                self.world[self.y][self.x] = self.object_dict_inverted[obj_ind]
            else:
                self.world[self.y][self.x] = self.EMPTY_COLOR

            self.x = new_x
            self.y = new_y

            self.world[self.y][self.x] = self.AGENT_COLOR

    def step(self, action):
        self.move_agent(action)
        if 'avoid' in self.task:
            self.move_avoid_object()

        self.num_steps += 1

        reward = self.reward()
        done = self.isTerminal()
        action_list = ['up', 'down', 'left', 'right']
        # print "Action taken " + action_list[action]
        # print "position of objects is {}".format(self.object_positions_2d)
        # print self.world.copy()
        # print reward
        # print done
        # print ''

        self.image = self.construct_image()

        return np.expand_dims(self.image.copy(), axis=-1), reward, done

    def step_test(self, action):
        """Version of step that returns both the dilated image and the original world."""
        image, reward, done = self.step(action)
        return np.expand_dims(self.world.copy(), axis=-1), np.expand_dims(
            self.image.copy(), axis=-1), reward, done

    def reward(self):
        target_object_index, avoid_object_index = \
            self.get_specified_object_indices()
        if ('avoid' in self.task
                and self.object_positions_2d[avoid_object_index] == \
                    [self.y, self.x]):
            return -1

        if ('target' in self.task
                and self.object_positions_2d[target_object_index] == \
                    [self.y, self.x]):
            return 1


        if self.num_steps >= self.max_steps:
            if 'target' in self.task:
                return -1
            if 'avoid' in self.task:
                return 1

        # If the task is to ONLY avoid a particular object,
        # then the agent gets 0 reward each time step.
        if 'target' not in self.task:
            return 0

        # Otherwise, the agent should try to reach the target
        # object as fast as possible, so it receives a negative
        # step cost at each time step.
        return self.STEP_COST

    def isTerminal(self):
        target_object_index, avoid_object_index = \
            self.get_specified_object_indices()
        if ('avoid' in self.task
                and self.object_positions_2d[avoid_object_index] == \
                    [self.y, self.x]):
            return True

        if ('target' in self.task
                and self.object_positions_2d[target_object_index] == \
                    [self.y, self.x]):
            return True

        if self.num_steps >= self.max_steps:
            return True

        return False
