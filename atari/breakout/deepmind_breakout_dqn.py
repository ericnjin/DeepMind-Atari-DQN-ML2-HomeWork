from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import gym


CONST_EPISODES = 5000
CONST_EPISILON = 1.
CONST_EPISILON_START = 1.0 
CONST_EPISILON_END = 0.1
CONST_EXPLORATION_STEPS = 1000000.

CONST_BATCH_SIZE = 32
CONST_TRAIN_START = 50000
CONST_UPDATE_TARGET_RATE = 10000
CONST_DISCOUNT_FACTOR = 0.99
CONST_NO_OP_STEPS = 30



class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.epsilon = CONST_EPISILON
        self.epsilon_start = CONST_EPISILON_START
        self.epsilon_end = CONST_EPISILON_END
        self.exploration_steps = CONST_EXPLORATION_STEPS
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = CONST_BATCH_SIZE
        self.train_start = CONST_TRAIN_START
        self.update_target_rate = CONST_UPDATE_TARGET_RATE
        self.discount_factor = CONST_DISCOUNT_FACTOR

        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = CONST_NO_OP_STEPS

        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의, stable optimizer
    # [-1 ~ 1]사이에서는 2차함수, 그외의 부분에서는 1차함수, 범위를 넘어가는 큰오류에 대해 안정적으로 반응   
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    
    # 상태가 입력, 큐함수가 출력인 인공신경망 생성, non-linear function approximation
    #  필터를 적용 , 32, 64, 64.. ==>  Agent가 image에서 특정정보를 얻어내기 위함
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    
    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    
    # 보상을 최대로 주는..입실론 탐욕 정책으로 행동 선택
    # 픽셀변경(0~255) ==> (0~1)
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else: 
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    
    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장하고 향후 ,experiance replay에 사용함 
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))


    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    # 충분희 스텝이 지난 후부터 trainnig시작 , 학습시작후 epsilon은 time step마다 reduce..
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 미니배치, 샘플중에서 배치크기만큼의 샘플을 무작위로 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if dead[i]: # agent is dead ?!
                target[i] = reward[i]  
            else: # batch 크기만큼의 정답을 생성
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 학습속도를 높이기 위해 흑백화면으로 전처리
# before [ 201, 160, 3] ==> after [84,84,1]
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    # openAI에서 제공한는 gym 이용
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(CONST_EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # agent초반에 일정동안 아무것도 하지않는 무작위 설정, 1 means do nothing
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)


        # 매 스텝마다 environment 는 agent에게 새로운 observe를 줌 ==> history에 추가
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            # 액션을 수행하면 environment는 next observe, 보상, 에피소드 끊난지 여부 , life (게임몇번더할 수 있나))등의 정보를 return함
            observe, reward, done, info = env.step(real_action)

            #매 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # 일종의 normalization (0~255) ==> (0~1) 
            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            # breaout game인 한에피소드당 경우 5개의 까지 게임가능, 정보는  info에서 가져올 수 잇음 
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # generalization...다른 게임에도 적요가능하도록 , 즉, 게임마다 점수를 주는 기준이 틀리기때문에..1로 통일
            reward = np.clip(reward, -1., 1.)

            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 time step마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

          
            if dead:
                dead = False
            else:
                history = next_history

            if done:
                # 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn.h5")
