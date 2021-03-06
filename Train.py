from Network import DQN_Operator
from Environment import ReplayBuffer, atari_env
from Config import *


def main():
    env = atari_env(preprocess_height, preprocess_width, name)
    act_space = env.env.action_space.n
    Operator = DQN_Operator(preprocess_height, preprocess_width, act_space,
                            learning_rate, model_path)
    buffer = ReplayBuffer(buffer_capacity)

    score_sum = 0.0
    max_score = 0.0
    step = start_step  # 1

    for n_epi in range(start_ep, train_episode_num):
        epsilon = max(0.1, 1.0 - (0.9/final_exploration_step) * step)
        s = env.reset()
        h, c = Operator.init_hidden()
        terminate = False
        score = 0

        while not terminate:
            h, c = Operator.init_hidden()
            a, (h_prime, c_prime) = Operator.action_epsilon_greedy(epsilon, s, (h, c))
            s_prime, r, terminate, _ = env.step(a)
            #history = history[1:] + [s_prime]
            t = 0.0 if terminate else 1.0
            # normalize reward
            r_norm = r
            if r < 0:
                r_norm = -1.0
            elif r > 0:
                r_norm = 1.0
            buffer.push((s, a, r_norm, t))

            s = s_prime
            h = h_prime
            c = c_prime

            score += r
            step += 1
            # env.render()

            if step % update_frequency == 0 and buffer.size() > replay_start_size:
                s_batch1, a_batch1, r_batch1, t_batch1 = [], [], [], []
                for k in range(real_batch_size):
                    s_batch, a_batch, r_batch, t_batch = buffer.sample(batch_size)
                    s_batch1.append(s_batch)
                    a_batch1.append(a_batch)
                    r_batch1.append(r_batch)
                    t_batch1.append(t_batch)
                Operator.train(s_batch1, a_batch1, r_batch1, t_batch1, gamma)

            if step % update_interval == 0 and buffer.size() > replay_start_size:
                Operator.update_targetPolicy()
                Operator.save(model_path)

        score_sum += score
        if score > max_score:
            max_score = score

        if n_epi % 10 == 0 and n_epi != 0:  # average last 10 game
            print('frame : {}, episode : {}, avg score : {}, max score : {}, buffer size : {}, epsilon : {:.3f}%'.format(step, n_epi, score_sum/10, max_score, buffer.size(), epsilon*100))
            score_sum = 0.0
    env.close()


if __name__ == "__main__":
    main()
