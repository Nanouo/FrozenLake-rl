import numpy as np
import gymnasium as gym

class RLModel:
    def __init__(self):
        self.learning_rate = 0.8 
        #This is the rate at which the AI will trust new information over old information 1 means overwrite old information, 0 means ignore new information

        self.gamma = 0.95 
        #The discount factor is a measure of how much the AI values future rewards over immediate rewards. A value of 0 means the AI only cares about immediate rewards, while a value of 1 means the AI values future rewards just as much as immediate rewards.

        self.epsilon = 1.0
        #Epsilon is a measure of how much the AI will explore new actions versus exploiting known actions. A value of 1 means the AI will always explore, while a value of 0 means the AI will always exploit known actions.

        self.epsilon_decay = 0.995
        #The epsilon decay is a measure of how quickly the AI will reduce its exploration over time. A value of 1 means the AI will not reduce its exploration, while a value of 0 means the AI will immediately stop exploring.

        self.epsilon_min = 0.01
        #The minimum value of epsilon. Once epsilon reaches this value, it will not decay further.

        self.episodes = 2000 
        #The number of episodes the AI will train for. An episode is a single run of the environment from start to finish.

        self.env, self.n_states, self.n_actions = self.setup_env()

        self.q_table = np.zeros((self.n_states, self.n_actions))
        #A 16x4 matrix of zeros that will be used to store the Q-values for each state-action pair. The Q-values represent the expected future rewards for taking a given action in a given state. 16x4 rows x columns 

        


    '''
    At the start (np.zeros((16, 4)) — the agent knows nothing):

            LEFT  DOWN  RIGHT  UP
    s0  [  0.0,  0.0,  0.0,  0.0 ]
    s1  [  0.0,  0.0,  0.0,  0.0 ]
    s2  [  0.0,  0.0,  0.0,  0.0 ]
    s3  [  0.0,  0.0,  0.0,  0.0 ]
    s4  [  0.0,  0.0,  0.0,  0.0 ]
    s5  [  0.0,  0.0,  0.0,  0.0 ]
    s6  [  0.0,  0.0,  0.0,  0.0 ]
    s7  [  0.0,  0.0,  0.0,  0.0 ]
    s8  [  0.0,  0.0,  0.0,  0.0 ]
    s9  [  0.0,  0.0,  0.0,  0.0 ]
    s10 [  0.0,  0.0,  0.0,  0.0 ]
    s11 [  0.0,  0.0,  0.0,  0.0 ]
    s12 [  0.0,  0.0,  0.0,  0.0 ]
    s13 [  0.0,  0.0,  0.0,  0.0 ]
    s14 [  0.0,  0.0,  0.0,  0.0 ]
    s15 [  0.0,  0.0,  0.0,  0.0 ]

    16 rows, 4 numbers each = 64 values total.

    After training it fills in with real numbers, e.g. (illustrative):

            LEFT   DOWN   RIGHT   UP
    s0  [ 0.52,  0.48,  0.53,  0.51 ]   <- argmax = col 2 (RIGHT), value 0.53
    s1  [ 0.31,  0.29,  0.34,  0.50 ]   <- argmax = col 3 (UP)
    ...
    s14 [ 0.61,  0.72,  0.95,  0.60 ]   <- argmax = col 2 (RIGHT) -> toward goal!
    s15 [ 0.0,   0.0,   0.0,   0.0  ]   <- goal: terminal, never acted from, stays 0

    How you actually touch it in code

    - One row = q_table[state] → the 4 numbers for that tile, e.g. q_table[0] → [0.52, 0.48, 0.53, 0.51]. That's what choose_action reads.
    - One cell = q_table[state][action] (or q_table[state, action]) → a single number, e.g. q_table[0][2] → 0.53. That's what update_q_table nudges.
    - np.argmax(q_table[state]) → the index (0–3) of the biggest number in that row = the best action.
    '''

    def setup_env(self, is_slippery = False):
        self.env = gym.make(
            'FrozenLake-v1',
            desc = None,
            map_name = "4x4",
            is_slippery = is_slippery,
            )
        return self.env, self.env.observation_space.n, self.env.action_space.n
    
    #This is creating the frozen lake enviorment using gymnasium and it returns the enviorment, the number of states, and the number of actions.
    #is_slippery = is_slippery — passes your parameter through to the env. This is the single most important flag in the whole project, so it's worth dwelling on:
    #- False → deterministic. Action "Right" always moves right. The world is predictable.
    #- True → stochastic. The ice is slippery: the intended action only happens 1/3 of the time; the other 2/3 you slide to one of the perpendicular directions. "Right" might send you up or down.
    #That slipperiness is what makes FrozenLake a real RL problem rather than a maze-solve — and it's exactly the feature the GeeksforGeeks gridworld you learned from didn't have. Starting with False to get things working is a smart choice; flipping it to True later is where the interesting learning happens.
    
    def choose_action(self, state):
        random_n = np.random.random() #draws a random float between 0 and 1
        if random_n < self.epsilon: #if the random number is less than epsilon, the AI will explore a new action
            return self.env.action_space.sample() #returns a random action
        return np.argmax(self.q_table[state]) 
    #If not then trust the q-table 
    #self.q_table[state] grabs the row for the current square: 4 numbers one per action
    #np.argmax() returns the index of the largest number in that row, which is the action with the highest expected future reward.
    
    def update_q_table(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.max(self.q_table[next_state]) #target = reward 
        if done:
            target = reward 
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (target - self.q_table[state][action])
    
    def train(self):
        count = 0
        for episode in range(self.episodes):
            state, info = self.env.reset() #Reset puts the agent back on tile 0 and hands you sthe starting state 
            done = False #done = False arms the intter while loop 
            while not done:
                action = self.choose_action(state) #runs choose_action to get the next action based on the current state
                next_state, reward, terminated, truncated, info = self.env.step(action) #This is setting these variables from the step function creates terminated and truncated
                done = terminated or truncated #End this episode by changing done 
                self.update_q_table(state, action, reward, next_state, done) #runs update_q_table to update the q_table based on the current state, action, reward, next_state, and done
                state = next_state #Update state to next state when AI ends episode
                if reward == 1.0: #Reached end goal
                    count += 1
                    
            self.epsilon = self.epsilon*self.epsilon_decay #epislon is now decayed so less randomness in the next episode
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
            if episode % 200 == 0:
                print(f"Episode: {episode} && Win rate: {(count/(episode - 1)) *100}%")
                
            

if __name__ == "__main__":
    model = RLModel()
    model.train()