import numpy as np

class TransitionTable:

    # basic funcs

    def __init__(self, state_siz, act_num, hist_len, max_transitions):
        self.state_siz = state_siz
        self.act_num = act_num
        self.hist_len  = hist_len
        self.max_transitions = max_transitions

        # memory for state transitions
        self.states  = np.zeros((max_transitions, state_siz*hist_len))
        self.actions = np.zeros((max_transitions, act_num))
        self.rewards = np.zeros((max_transitions, 1))
        self.top = 0
        self.bottom = 0
        self.size = 0

    # helper funcs
    def add(self, state, action):
        self.states[self.top] = state
        self.actions[self.top] = action
        if self.size == self.max_transitions:
            self.bottom = (self.bottom + 1) % self.max_transitions
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_transitions

    def one_hot_action(self, actions):
        actions = np.atleast_2d(actions)
        one_hot_actions = np.zeros((actions.shape[0], self.act_num))
        for i in range(len(actions)):
            one_hot_actions[i, int(actions[i])] = 1
        return one_hot_actions

    def sample_minibatch(self, batch_size):
        state      = np.zeros((batch_size, self.state_siz*self.hist_len), dtype=np.float32)
        action     = np.zeros((batch_size, self.act_num), dtype=np.float32)
        for i in range(batch_size):
            state[i]         = self.states.take(i, axis=0, mode='wrap')
            action[i]        = self.actions.take(i, axis=0, mode='wrap')
        return state, action
