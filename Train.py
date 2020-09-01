
import random
from Network import DQN_Operator
from Environment import ReplayBuffer, breakout

# settings
train_episode_num      = 10000
learning_rate          = 0.00001
gamma                  = 0.99
buffer_capacity        = 500000
batch_size             = 32
replay_start_size      = 50000
final_exploration_step = 1000000
update_interval        = 10000 # target net
update_frequency       = 4  # the number of actions selected by the agent between successive SGD updates
preprocess_height      = 84
preprocess_width       = 84
model_path             = './DRQN.model'

def main():
    env = breakout(preprocess_height, preprocess_width)
    Operator = DQN_Operator(preprocess_height, preprocess_width, 4, learning_rate, model_path)
    buffer = ReplayBuffer(buffer_capacity)

    score_sum = 0.0
    max_score = 0.0
    step = 1

    for n_epi in range(train_episode_num):
        epsilon = max(0.1, 1.0 - (0.9/final_exploration_step) * step)
        s = env.reset()
        h, c = Operator.init_hidden()
        terminate = False
        score = 0

        while not terminate:
            a , (h_prime, c_prime) = Operator.action_epsilon_greedy(epsilon, s, (h, c))
            s_prime, r, terminate, _ = env.step(a)
            t = 0.0 if terminate else 1.0
            buffer.push((s, a, r, t))

            s = s_prime
            h = h_prime
            c = c_prime

            score += r
            step += 1
            env.render()

            if step%update_frequency==0 and buffer.size() > replay_start_size:
                s_batch, a_batch, r_batch, t_batch = buffer.sample(batch_size)
                Operator.train(s_batch, a_batch, r_batch, t_batch, gamma)
            
            if step % update_interval==0 and buffer.size() > replay_start_size:
                Operator.update_targetPolicy()
                Operator.save(model_path)
        
        score_sum += score
        if score > max_score:
            max_score = score
        
        if n_epi % 10==0 and n_epi != 0: # average last 10 game
            print('frame : {}, episode : {}, avg score : {}, max score : {}, buffer size : {}, epsilon : {:.3f}%'.format(step, n_epi, score_sum/10, max_score, buffer.size(), epsilon*100))
            score_sum = 0.0
    env.close()

if __name__ == "__main__":
    main()