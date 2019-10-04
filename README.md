## Deepmind 논문
https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning

## 필요한 라이브러리들 (Dependencies)
1. Python 3.x
2. Tensorflow 
3. Keras 
4. numpy
5. pandas
6. pillow
7. matplot
8. Skimage
9. h5py

## 참조 자료 
http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf

http://llcao.net/cu-deeplearning15/presentation/DeepMindNature-preso-w-David-Silver-RL.pdf

http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf

https://www.youtube.com/watch?v=V7_cNTfm2i8&feature=youtu.be&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS

http://sanghyukchun.github.io/74/ 

http://arxiv.org/pdf/cs/0308031.pdf


## 논문요약 
###Abstract
* First deep learning model using reinforcement learning
- Successfully learn control policies directly
- From high-dimensional sensory input (pixels)  (환경에서 나오는 정보 즉, 게임화면에서 나오는 [가로픽셀수] * [세로픽셀수] * [3 RGB] ) 

* CNN model trained on a variant of Q-learning
- Input: raw pixel, output: a value function estimating future reward
- Output: a value function (미래의 보상을 가늠할 수 있는 가치함수가 output )

* Applied seven Atari 2600 games (no adjustment)
- Outperforms ALL previous approaches on six games
- Surpasses a human expert on three games


###1. Intrduction
*Challenge 
Most DL requires hand labeled training data
- RL must learn from a scalar reward signal
- Reward signal is often sparse, noisy, and delayed
- delay between actions and resulting rewards can be thousand time steps

* Most DL assumes data samples are independent
- RL encounters sequences of highly correlated states (연관성이 매우깊다)


* Our Solution
- CNN with a variant Q-learning
- Experience replay


###2. Background
* 환경(Environment)에 대한 순차적 행동결정이있고 그것은 MDP(Markov Decision Process)로 정의 한다고 가정
* MDP have five features
1. State : Agent 자신의 상황에 대한 관 
   St = x1, a1, x2 .... a(t-1), xt  ==> sequences of actions and observation
2. Action
3. Reward ==> futures rewards are discounted factor per time stemp
4. State Transition Probability ==> 시간t에 agent가 있을 상태가 확정지을 수가 없고 확률적으로만 존재한다.
5. Discounter factor (할인률)


* The goal of agent is finding an action-value function i.e. solving Bellman equation.
단, action-value function을 직접구할 수 는 없고 근사를 해야한다. ==> a function approximation.
  non-linear function ==> neural network ==> Q-network : it can trained by minimising a sequence of loss function


###3. Deep Reinforcemnet Learning
we utilize a techinqure experinece replay

* Experience replay : 각 time-step별로 얻은 sample(experience)들을 아래와 같은 tuple형태로 data set에 저장해두고, 
randomly draw하여 mini-batch를 구성하고 update하는 방법 , data usage efficiency를 도모하여 correation state를 해결.

**프로그램 실행 방법** -

- ~DeepMind-Atari-DQN-ML2-HomeWork/atari/breakout/ python3 deepmind_breakout_dqn.py 
 
- 
