import time
from Network import DQN_Operator
from Environment import atari_env

play_epi_num           = 100
preprocess_height      = 84
preprocess_width       = 84
name                   = 'Pong-v0'
model_path             = './DRQN.model'


def main():
    env = atari_env(preprocess_height, preprocess_width, name)
    act_space = env.env.action_space.n
    Operator = DQN_Operator(preprocess_height, preprocess_width,
                            act_space, 0, model_path)

    score_sum = 0.0
    max_score = 0.0
    for epi in range(play_epi_num):
        epsilon = 0.0
        s = env.reset()
        h, c = Operator.init_hidden()
        terminate = False
        score = 0

        while not terminate:
            a , (h_prime, c_prime) = Operator.action_epsilon_greedy(epsilon, s, (h, c))
            s_prime, r, terminate, _ = env.step(a)

            s = s_prime
            h = h_prime
            c = c_prime

            score += r
            env.render()
            time.sleep(0)

        score_sum += score
        if score > max_score:
            max_score = score

        print('episode : {}, avg score : {:.1f}, max score : {}'.format(epi, score_sum/(epi+1), max_score))
    env.close()


if __name__ == "__main__":
    main()
