# RL Note Book

## 强化学习基础概念

状态 state: 用于描述agent在环境中的某个状态
动作 action: 用于描述agent在某种状态下的动作
奖励 reward: 用于描述agent在执行某个状态某个动作后获得的奖励
状态转移函数 state transition: 用于描述agent从一个状态到另一个状态的转换
奖励转换 reward transition: 用于描述agent在执行某个状态某个动作后获得的奖励
策略 policy: 用于描述agent在某个状态下应该采取的动作
环境 environment: 用于描述agent与之交互的外部世界
折扣因子 discount factor: 用于描述agent对未来奖励的折扣程度
轨迹 Trajectory: 用于描述agent在环境中从初始状态到终止状态的整个过程

强化学习的目标是让agent在一个环境中学习到最优的策略，使其能够在长期的过程中获得最大的奖励。

MDP

1. 集合
   * 状态集合 state space：状态$S$的集合
   * 动作集合 action space: 与状态$s\in S$ 相关的动作集合$A(s)$
   * 奖励集合 reward space: $R(s, a)$

2. 概率分布
   * 状态转换概率：在状态$s$的情况下，采取动作$a$，转换到状态$s^{\prime} 的概率P(s^{\prime}|s,a)$
   * 奖励概率：在状态$s$的情况下，采取动作$a$，得到奖励$r$的概率$P(r|s,a)$
3. 策略
   * 在状态$s$的情况下，选择动作$a$的概率$\pi(a|s)$
4. 马尔可夫性质：无记忆性质
5. 累积折扣汇报：$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+2}+...$



## 贝尔曼公式

状态价值：代表这个状态的不同轨迹下能拿到最终奖励总和的期望
$$
\begin{aligned}
V_\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]&=\mathbb{E}[R_{t+1}|S_t=s]+\gamma \mathbb{E}[G_{t+1}|S_t=s]\\
&=\mathbb{E}[R_{t+1}|S_t=s]+\gamma \sum_{v^{\prime}}v_{\pi}(s^{\prime})p(s^{\prime}|s)\\
&=\mathbb{E}[R_{t+1}|S_t=s]+\gamma\sum_{s^{\prime}\in S}p(s^{\prime}|s,a)\sum_{a\in A}\pi(a|s)v_\pi(s^{\prime})\\
&=\sum_{a\in A}\pi(a|s)[\sum_{r\in R}p(r|s,a)r+\gamma\sum_{s^{\prime}\in S}p(s^{\prime} |s,a)v_\pi(s^{\prime})] \\

v_{\pi}&=r_{\pi}+\gamma P_{\pi}v_{\pi}
\end{aligned}
$$


动作价值：代表在某一个状态除非并采取某个动作后能得到的平均汇报
$$
q_{\pi}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a] \\
\mathbb{E}[G_t|S_t=s]=\sum_a\mathbb{E}[G_t|S_t=s,A_t=a]\pi(a|s)\\
\underbrace{\mathbb{E}\left[G_{t} \mid S_{t}=s\right]}_{v_{\pi}(s)} = \sum_{a} \underbrace{\mathbb{E}\left[G_{t} \mid S_{t}=s, A_{t}=a\right]}_{q_{\pi}(s, a)} \pi(a \mid s)
$$


动作价值可以理解成状态价值中一个动作的数值
$$
\begin{aligned}
v_{\pi}(s)&=\sum_a\pi(a|s)q_{\pi}(s,a) \\
&=\sum_a\pi(a|s)[\sum_rp(r|s,a)r+\gamma \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})] \\
q_{\pi}(s,a)&=\sum_rp(r|s,a)r+\gamma \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})
\end{aligned}
$$
我们可以通过对比某个状态的不同动作的动作价值，选择最大的动作价值作为最优策略





## 贝尔曼最优策略

最优策略是存在两个策略，它们在每个状态都有各自的State Value。对于所有的状态，$\pi1$的State Value都要比$\pi2$的State Value都要好，那么策略$\pi1$要比$\pi2$要好。
$$
\forall s \in S, V_{\pi*}(s) \geq V_{\pi}(s), 那么\pi*是最优策略。
$$
贝尔曼最优方程
$$
\begin{aligned}
v(s)&=\max_{\pi}\sum_{a}\pi(a|s)(\sum_rp(r|s,a)r+\gamma \sum_{s^{\prime}}p(s^{\prime}|s, a)v(s^{\prime})), s \in S \space\space\space\space\space 公式1\\
&=\max_{\pi}\sum_a\pi(a|s)q(s,a), s \in S \space\space\space\space\space 公式2
\end{aligned}
$$
方程里有已知的数值，$p(r|s,a), p(s^{\prime}|s,a),r,\gamma$都是已知的(model-based)。$v(s), v(s^{\prime}), \pi(s)$都是未知的。贝尔曼公式依赖一个给定的$\pi$，而贝尔曼公式的${\pi}$是没有确定的，需要求解。

如果想要$v(s)$的数值最大，那么等号右侧就是求解两数值乘积最大的情况。当知道一个状态的不同动作的动作价值后，我们可以挑选其中最大的$q(s,a)$代入贝尔曼最优方程，然后$\pi(a|s)$在这个最大的动作上的取值为1，其他动作的取值为0。

按照公式1进行迭代
$$
v_{k+1}(s)=\max_{\pi}\sum_{a}\pi(a|s)(\sum_rp(r|s,a)r+\gamma \sum_{s^{\prime}}p(s^{\prime}|s, a)v_k(s^{\prime}))
$$


1. 对于任意状态$s$，我们对解有一个估计，这个估计就是$v_k(s)$，这个可以是任意的一个值，也可以当作最开始的$v_0(s)$

2. 对任意状态$s$下的每一个动作，用第一步给定的$v_k(s^{\prime})$，求解$q_k(s,a)$。
3. 根据第二步得到的不同动作的$q_k(s,a)$，我们做贪心策略。对于最大的动作价值的动作，$\pi_{k+1}(a|s)$等于1，其余的动作概率等于0。$a_k^*(s)==argmax_aq_k(s,a)$
4. 根据第三步的数值更新$v_{k+1}(s)$

这上面这几步被称为值迭代算法，每轮循环通过迭代状态价值不断逼近最优策略。

上面这个公式也可以写成，通过上一轮的$v_k$更新当前轮的策略，得到当前轮的$v_{k+1}$。
$$
v_{k+1}=f(v_k)=max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{k}), k=1, 2, 3 {\space}...
$$
每轮的算法可以拆分成两部分-策略更新和价值更新。

1. 在给定$v_k$的情况下进行**策略更新（policy update）**，求解$\pi$，可以得到$\pi_{k+1}$。这个步骤是贝尔曼最优方程右边的优化问题。

$$
\pi_{k+1} = argmax_{\pi}(r_{\pi}+\gamma P_{\pi}v_k)
$$

2. 把上一步得到$\pi_{k+1}$带入式子，把式子中的$\pi_k$换成$\pi_{k+1}$，根据给定的$v_k$求解$v_{k+1}$，这一步是**值更新**。
   $$
   v_{k+1}=r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}}v_k
   $$

这里和贝尔曼方程的区别在于，$v_k$并不能当作一个状态值来理解，它是一个数值，在贝尔曼最优方程中是不断迭代的一个数值，并不是一个状态值（State Value）。

* $v_k$是某次迭代没有收敛的一个数值
* $v_k$是估计的状态值，后面可以求解出最优的状态值

## 值迭代算法

随便给出一个$v_k$，通过交替执行策略更新和值更新这两步骤，不断优化$v_k$和$\pi$，最终得到一个收敛状态
$$
\begin{aligned}
&初始化整个算法，假设这个算法是model-based，其中p(r|s,a)和p(s^{\prime}|s,a)对于所有的(s,a)状态动作都是已知的,初始化一个v_0。\\
&目标是通过迭代不断求解贝尔曼最优方程找到一个最优化的状态值和一个最优化策略\\
&while{\space}v_k{\space}has{\space}not{\space}converged{\space}in{\space}the{\space}sense{\space}that{\space}\big\|v_k-v_{k-1}\big\|{\space}is{\space}greater{\space}than{\space}perdefined{\space}small{\space}threshold,{\space}for{\space}the{\space}kth{\space}iteration,{\space}do:\\
&\qquad For{\space}every{\space}state{\space}s \in S,{\space}do\\
&\qquad \qquad For{\space}every{\space}action{\space}a \in A(s),{\space}do \\
&\qquad \qquad \qquad q-value:{\space}q_k(s,a)=\sum_rp(r|s,a)r+\gamma \sum_{s^{\prime}}p(s^{\prime}|s,a)v_k(s^{\prime})\\
&\qquad \qquad Maximum action value: a_k^*(s)=argmax_aq_k(q,s)\\
&\qquad \qquad Policy update:\pi_{k+1}(a|s)=1{\space}if{\space}a=a_k^*,{\space}and{\space}\pi_{k+1}(a|s)=0{\space}otherwise\\
&\qquad \qquad Value update:v_{k+1}(s)=max_aq_k(a,s)
\end{aligned}
$$




## 策略迭代算法

值迭代算法是通过给定一个初始的$v_k$，开始不断迭代的算法。我们也可以直接给定一个初始策略的方式$\pi_0$进入不断迭代的算法。策略迭代相当于比值迭代先进行了半步，先进行了**值更新**。策略迭代和值迭代的区别在于我们把哪一项当作主体。

步骤一策略评估$v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}$

步骤二策略更新$\pi_{k+1}=argmax_{\pi}(r_{\pi}+\gamma P_{\pi}v_{\pi_k})$

看上去就是值迭代算法，其中我们还留意到策略评估里还少了一个需要被知道的数值$v_{\pi_k}$，另外的策略迭代中也包含一层值迭代算法用于求解这个数值。这个数值有两种求解方式。

方式一，其中$P_{\pi_k}$和$r_{\pi_k}$都是已知的。这个方式比较麻烦，需要求解逆矩阵。更加常用的算法是方式二。
$$
v_{\pi_k}=(I-\gamma P_{\pi_k})^{-1}r_{\pi_k}
$$
方式二，迭代算法，随便给一个$v_{\pi_k}^0$的数值，然后进行$j$轮迭代或者$\big\|v_{\pi_k}^{(j+1)}-v_{\pi_k}^{(j)} \big\|$小于一个数值。这一步就相当于给出策略评估中需要用到的$v_{\pi_k}$。
$$
v_{\pi_k}^{(j+1)}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}^{(j)}, j=0,1,2, ...
$$
策略迭代算法整体步骤

1. 随便给定一个初始策略$\pi_0$

2. 进行一个小的迭代算用于求解策略评估中的$v_{\pi _k}$，这个迭代算法就是上面写的方式二。

3. 拿到得到的$v_{\pi_ k}$进行策略更新。
   $$
   \begin{aligned}
   &初始化整个算法，假设这个算法是model-based，其中p(r|s,a)和p(s^{\prime}|s,a)对于所有的(s,a)状态动作都是已知的,初始化一个策略\pi_0。 \\
   &我们的目标是通过迭代找到所有的最优状态价值和一个最优策略。\\
   &while{\space}v_{\pi_k}{\space}has{\space}no{\space}converged,{\space}for{\space}kth{\space}iteration,{\space}do\\
   &\qquad \textsf{Policy}{\space}\textsf{evaluation：}\\
   &\qquad initialization： an{\space}arbitrary{\space}initial{\space}guess{\space}v_{\pi_ k}^{(0)}\\
   &\qquad For{\space}every{\space}state{\space}s \in S,{\space}do\\
   &\qquad \qquad v_{\pi_ k}^{(k+1)}(s)=\sum_a \pi_k(a|s)[\sum_rp(r|s,a)r+\gamma\sum_s^{\prime}p(s^{\prime}|s,a)v_{\pi_ k}^{j}s^{\prime}] \\
   &\qquad \textsf{Policy}{\space}\textsf{improvement:}\\
   &\qquad For{\space}every{\space}state{\space}s \in S,{\space}do\\
   &\qquad \qquad For{\space}every{\space}action{\space}a \in A,{\space}do\\
   &\qquad \qquad \qquad q_{\pi_ k}(s,a)=\sum_rp(r|s,a)r+\gamma \sum_{r^{\prime}}p(s^{\prime}|s,a)v_{\pi_ k}(s^{\prime})\\
   &\qquad \qquad a_k^*(s)=argmax_aq_{\pi_ k}(s,a)\\
   &\qquad \qquad \pi_{k+1}(a|s)=1{\space}if{\space}a=a_k^*,{\space}and{\space}\pi_{k+1}(a|s)=0{\space}otherwise
   \end{aligned}
   $$

策略迭代和值迭代的不同

|               | Policy iteration algorithm                                   | Value iteration algorithm                                    | Comments                                                     |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| step 1 Policy | $\pi _0$(给定一个初始策略$\pi _0$)                           | N/A                                                          |                                                              |
| step 2 Value  | $v_{\pi_0}=r_{\pi_0}+\gamma P_{\pi_0}v_{\pi_0}$(通过贝尔曼公式求出$v_{\pi_0}$) | $v_0=v_{\pi_0}$（直接给定一个初始值，在这里为了方便比较我们使用和策略迭代相同的数值） |                                                              |
| step 3 Policy | $\pi_1=arg \max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{\pi_0})$      | $\pi_1=arg \max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{0})$          | 此时两者的策略更新都是相同的                                 |
| step 4 Value  | $v_{\pi_1}=r_{\pi_1}+\gamma  P_{\pi_1}v_{\pi_1}$(通过贝尔曼公式求出$v_{\pi_1}$) | $v_{1}=r_{\pi_1}+\gamma P_{\pi_1}v_{0}$                      | $v_{\pi1} \ge v_{1}{\space}since{\space}v_{\pi_1}\ge v_{\pi_0}$ |
| step 5 Policy | $\pi_2=arg \max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{\pi_1})$      | $\pi_2^{\prime}=arg \max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{1})$ |                                                              |
| ...           | ...                                                          | ...                                                          |                                                              |



当策略迭代和值迭代都从相同的条件出发时，它们的前三步是相同的，第四步开始不同了。因为在策略迭代中，值更新（Value Upate）需要通过算法迭代才能得到，而值迭代算法只需要进行一次

Value iteration: v-p-v-p-v-p-......

Policy iteration:p-vvvvv-p-vvvvv-p-vvvvv-p-vvvvv-......

其实Policy iteration和Value iteration都是Truncated Policy iteration的特殊情况。Truncated Policy iteration是指Policy iteration中通过贝尔曼公式求解$v_{\pi_0}^{(j+1)}$

这一步不以$\big\|v_{\pi_k}^{(j+1)}-v_{\pi_k}^{(j)} \big\|$小于一个数值，即收敛最为结束条件，而是进行$j$步迭代。那么Policy iteration就是$j \to \infty$的特殊例子，同理Value iteration是$j=1$的特殊例子。



## 蒙特卡洛（Mento Carlo)

蒙特卡洛估计是指依靠重复随机抽样来解决近似问题一大类技术。蒙特卡洛不需要模型的特点。

策略迭代有两个步骤：

1. 策略评估：通过一个初始策略${\pi}_0$ , 找到它的状态值 $v_{\pi_0}=r_{\pi_0}+\gamma P_{\pi_0}v_{\pi_0}$(通过贝尔曼公式求出$v_{\pi_0}$)。
2. 策略改进：通过$v_{\pi_ k}$可以通过贝尔曼公式求解新的策略$v_{k+ 1}$ , 通过选择最大的$q_{\pi_ k}(s,a)$。

策略提升步骤：

$\pi_{k+1}(s) = {argmax}_{\pi}\sum_{a}\pi(a|s)q_{\pi _k}(s,a)$

而

$q_{\pi_ k} (s,a)=\sum_rp(r|s,a) + \gamma\sum_{s^\prime}p(s^\prime|s,a)v_{\pi_ k}(s^\prime)$

$q_{\pi_k}(s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]$

$q_{\pi_k}$有两种方式计算得到，第一个公式是需要$p(r|s,a)$和$p(s^\prime|s,a)$这两个方程，第二个公式不需要，仅仅需要大量的样本进行计算。

Mento Carlo basic algorithm

这个是策略迭代的变种，也有两个步骤，策略评估和策略提升。

在策略提升的阶段，通过收集从$(s,a)$出发的每个状态的每个动作的价值，在这个数据上求解一个均值。选择最大的$q_{\pi_ k}$的策略当作最优策略进行下一步的优化。
$$
\begin{aligned}
&初始化整个算法，假设这个算法是model-free，初始化一个策略\pi_0。 \\
&我们的目标是找到一个一个最优策略。\\
&while{\space}the{\space}value{\space}estimate{\space}has{\space}not{\space}converged,{\space}for{\space}the{\space}kth{\space}iteration,{\space}do\\
&\qquad For{\space}every{\space}state{\space}s \in S,{\space}do\\

&\qquad \qquad For{\space}every{\space}action{\space} a\in A(s),{\space}do \\
&\qquad \qquad \qquad Collect{\space}sufficiently{\space}many{\space}episodes{\space}starting{\space}from{\space}(s,a){\space}following{\space}\pi_k \\
&\qquad \qquad \qquad MC-based{\space}policy{\space}evaluation{\space}step:\\
&\qquad \qquad \qquad q_{\pi_ k}(s,a) = average{\space}return{\space}of{\space}all{\space}the{\space}episodes{\space}starting{\space}from{\space}(s,a)\\
&\qquad \qquad \textsf{Policy}{\space}\textsf{improvement step:}\\
&\qquad \qquad a_k^*(s) = argmax_a q_{\pi _k}(s,a) \\
&\qquad \qquad \pi_{k+1}(a|s) = 1{\space}if{\space}a=a_k^*,{\space}and{\space}{\pi}_{k+1}(a|s)=0{\space}otherwise
\end{aligned}
$$


## 时序差分方法（Temporal-difference learning)

1. 蒙特卡洛（Mento Carlo）方法是 model-free 的方法，Temporal-difference learning（TD learning）是第二种 model-free 的方法。
2. 蒙特卡洛（Mento Carlo）是一个非递增的方法，Temporal-difference learning(TD learning)是一种迭代的方法

$$
V(s_t)\gets V(s_t)+a[G_t-V(s_t)]
$$

$a$代表对价值估计更新的补偿，可以将$a$取为一个整数，此时更新方式不再像蒙特卡洛方法那样严格地取期望。蒙特卡洛方法必须要等整个序列结束之后才能计算这次回报$G_t$，而时序差分方法只需要当前步结束即可进行计算。具体来说，时序差分算法用当时获得地奖励加上下一个状态

时序差分方法也是和蒙特卡洛一样在通过预测动作价值进行最优策略的更新。由此衍生出两类算法$sarsa$和$Q-Learning$
$$
\begin{align*}V_{\pi}(s) &= \mathbb{E}_{\pi} [G_{t} \vert S_{t} = s] \\&= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^{k} R_{t+k} \middle\vert S_{t} = s \right] \\&= \mathbb{E}_{\pi} \left[ R_{t} + \gamma \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \middle\vert S_{t} = s \right] \\&= \mathbb{E}_{\pi} \left[ R_{t} + \gamma V_{\pi}(S_{t+1}) \middle\vert S_{t} = s \right]\end{align*}
$$

#### Sarsa

$Q(s_t,a_t)\gets Q(s_t,a_t)+a[r_t+ \gamma Q(s_{t+1},a_{t+1}-Q(s_t,a_t)]$

* 初始化$Q(s,a)$
* for 序列$e=1\to E{\space}do$
* ​    得到初始状态$s$
* ​    用$\epsilon-greedy$策略根据$Q$选择当前状态$s$下的动作$a$
* ​    for 时间步$t=1\to T {\space}do$:
* ​         得到环境反馈的$r,s^{\prime}$
* ​         用$\epsilon-greedy$策略根据$Q$选择当前状态$s^{\prime}$下的动作$a^{\prime}$
* ​         $Q(s,a)\gets Q(s,a)+\alpha[r+\gamma Q(s^{\prime},a^{\prime})-Q(s,a)]$
* ​         $s\gets s^{\prime}, a\gets a^{\prime}$
* ​     end for
* end for

#### Q-Learning

$Q(s_t,a_t)\gets Q(s_t,a_t)+\alpha[R_t+\gamma \max_aQ(s_{t+1},a) - Q(s_t,a_t)]$

* 初始化$Q(s,a)$
* for 序列$e=1 \to E{\space}do$:
* ​    得到初始状态$s$
* ​    for 时间步$t=1\to T{\space}do$
* ​         用$\epsilon -greedy$策略根据$Q$选择当前状态$s$下的动作$a$
* ​         得到环境反馈的$r,s^{\prime}$
* ​         $Q(s,a)\gets Q(s,a)+\alpha[r+\gamma \max_{a^{\prime}}Q(s^{\prime},a^{\prime})-Q(s,a)]$
* ​         $s\gets s^{\prime}$
* ​     end for
* end for



### DQN

使用函数拟合的方式代替表格形式的$Q(s,a)$，

## 状态值估计的算法（Algorithm for state value estimation)

1. 目标函数（object function)
   * $v_{\pi}(s)$是一个真实的state value，${\hat v}(s,w)$是它的估计值，我们的目标是让估计值接近真实值
   * 当${\hat v} (s,w)$函数的结构确定的时候，（假设它是一个神经网络）能调整其中的$w$，让估计值接近真实值
   * 那么这个问题实质上是一个policy evaluation的问题，就是从一个策略，得到要给近似的${\hat v}$，让它接近更加真实的state value。

2. 目标函数定义
   $$
   J(w)=\mathbb{E}[(v_{\pi}(S)-{\hat v}(S,w))^2]
   $$
   我们的目标通过$w$优化目标函数，得到最小的$J(w)$，使用梯度下降方法
   $$
   w_{k+1} = w_k -a_k \triangledown _wJ(w_k)
   $$
   在目标函数里的期望数值是关于随机比哪里$s \in S$，这里面$S$是一个随机变了，随机变量一定是有probability distribution的，那么$S$的概率分布会影响我们的期望数值

   * 如果假设这个是平均分布，那么有
     $$
     J(w)=\mathbb{E}[(v_{\pi} - {\hat v}(S,w))^2]=\frac{1}{|S|}{\sum}_{s \in S}(v_{\pi}(s)-{\hat v}(s,w))^2
     $$
     平均分布的状态函数意味着各种状态都同等重要，但是各种状态的重要性不尽相同。靠近最终目标和初始状态的状态会更加重要，距离目标状态较远的状态不太重要。

   * 如果假设这个是平稳分布（ **the stationary distribution.**）平稳分布是本课程中经常使用的一个重要概念。简而言之，它描述了马尔可夫过程的长期行为。就是我从某一个状态出发，然后我按照策略采取 action，然后我不断地去和环境进行交互，然后我一直采取这个策略，采取非常多次之后，我就达到了一种平稳的状态，在那个平稳的状态下我能够告诉你，在每一个状态 agent 出现它的概率是多少。之后会通过一个例子更清晰的了解，现在我们就先知道反正它是一个概率分布。
     $$
     J(w)=\mathbb{E}[(v_{\pi})(S)-{\hat v}(S,w))^2]=\sum_{s\in S}d_{\pi}(s)(v_{\pi}(s)-{\hat v}(s,w))^2
     $$
     

3. 优化目标函数的算法
   $$
   w_{k+1}=w_k-a_k\triangledown_wJ(w_k)
   $$
   梯度公式

