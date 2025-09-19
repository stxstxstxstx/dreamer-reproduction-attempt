import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers


def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.') # 日志保存目录，这里的 pathlib.Path(. ) 表示当前工作目录
  config.seed = 0  # 随机种子，保证实验可复现
  config.steps = 5e6  # 训练总步数
  config.eval_every = 1e4  # 每多少步进行一次评估
  config.log_every = 1e3  # 每多少步记录一次日志
  config.log_scalars = True  # 是否记录标量日志
  config.log_images = True  # 是否记录图像日志
  config.gpu_growth = True  # 是否允许GPU内存动态增长
  config.precision = 16  # 训练精度（16位或32位）
  # Environment.
  config.task = 'dmc_walker_walk'  # 任务名称（DeepMind Control Suite中的walker walk任务）
  config.envs = 1  # 并行环境数量
  config.parallel = 'none'  # 并行方式
  config.action_repeat = 2  # 动作重复次数
  config.time_limit = 1000  # 环境时间限制
  config.prefill = 5000  # 预填充经验回放缓冲区的步数
  config.eval_noise = 0.0  # 评估时的探索噪声
  config.clip_rewards = 'none'  # 奖励裁剪方式
  # Model.
  config.deter_size = 200  # 确定性状态维度
  config.stoch_size = 30  # 随机状态维度
  config.num_units = 400  # 全连接层单元数
  config.dense_act = 'elu'  # 全连接层激活函数
  config.cnn_act = 'relu'  # 卷积层激活函数
  config.cnn_depth = 32  # 卷积层深度
  config.pcont = False  # 是否使用持续概率预测
  config.free_nats = 3.0  # KL散度的自由比特数
  config.kl_scale = 1.0  # KL散度缩放因子
  config.pcont_scale = 10.0  # 持续概率预测的缩放因子
  config.weight_decay = 0.0  # 权重衰减系数
  config.weight_decay_pattern = r'.*'  # 权重衰减应用模式
  # Training.
  config.batch_size = 50  # 批次大小
  config.batch_length = 50  # 序列长度
  config.train_every = 1000  # 每收集多少步数据进行一次训练
  config.train_steps = 100  # 每次训练的步数
  config.pretrain = 100  # 预训练步数
  config.model_lr = 6e-4  # 模型学习率
  config.value_lr = 8e-5  # 值函数学习率
  config.actor_lr = 8e-5  # 策略学习率
  config.grad_clip = 100.0  # 梯度裁剪阈值
  config.dataset_balance = False  # 是否平衡数据集
  # Behavior.
  config.discount = 0.99  # 折扣因子
  config.disclam = 0.95  # λ-return中的λ值
  config.horizon = 15  # 想象轨迹的长度
  config.action_dist = 'tanh_normal'  # 动作分布类型
  config.action_init_std = 5.0  # 动作分布初始标准差
  config.expl = 'additive_gaussian'  # 探索策略类型
  config.expl_amount = 0.3  # 探索噪声量
  config.expl_decay = 0.0  # 探索噪声衰减率
  config.expl_min = 0.0  # 最小探索噪声量
  return config

class Dreamer(tools.Module):

  def __init__(self, config, datadir, actspace, writer):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)

    with tf.device('cpu:0'): # 强制在CPU上创建变量（避免GPU内存碎片）
      # 创建全局步数变量（记录当前训练进度）
      self._step = tf.Variable(
        count_steps(datadir, config),  # 初始值=已有数据的总步数
        dtype=tf.int64                # 使用64位整数防止溢出
      )
    # 技术细节：
    # 1. 放在CPU上确保分布式训练时各设备能同步访问
    # 2. int64可支持最多2^63步训练（约9.2e18步）

    self._should_pretrain = tools.Once()
    # 初始化训练触发器：每train_every步触发一次训练
    self._should_train = tools.Every(config.train_every)
    # 初始化日志触发器：每log_every步记录一次日志
    self._should_log = tools.Every(config.log_every)
    # 记录上一次日志记录的步数（初始为None）
    self._last_log = None
    # 记录上一次操作的时间戳（用于计算耗时）
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
    self._strategy = tf.distribute.MirroredStrategy() # 单机多GPU同步训练
    # 在分布式策略范围内构建模型和数据集
    # 使用TensorFlow分布式策略的上下文管理器
    # 该策略定义了模型训练和数据分布的方式(如单机多GPU或多机训练)
    with self._strategy.scope():
        # 1. 加载并处理数据集
        # load_dataset函数从datadir目录加载数据,并返回一个预处理后的tf.data.Dataset
        # experimental_distribute_dataset方法将数据集分布到所有可用的计算设备上
        # iter()将数据集转换为迭代器,允许逐个批次地获取数据
        self._dataset = iter(self._strategy.experimental_distribute_dataset(
            load_dataset(datadir, self._c)))
        
        # 2. 构建模型
        # 在分布式策略的作用域内构建模型
        # 这确保模型的变量和操作被正确地分布到所有设备上
        # 例如,在多GPU设置中,模型会被复制到每个GPU上
        self._build_model()

  def __call__(self, obs, reset, state=None, training=True):
    """
    执行智能体与环境的交互步骤
    
    参数:
        obs: 当前环境观测，通常是字典类型(如{'image': ..., 'reward': ...})
        reset: 每个环境是否需要重置的布尔数组(多环境并行时)
        state: 循环神经网络的隐藏状态(可选)
        training: 是否处于训练模式
        
    返回:
        action: 智能体选择的动作
        state: 更新后的隐藏状态
    """
    # 获取当前全局步数并设置TensorFlow摘要的全局步骤
    step = self._step.numpy().item() # .item() 将单元素数组转换为 Python 整数
    tf.summary.experimental.set_step(step)
    
    # 处理环境重置逻辑：如果某些环境需要重置，则重置对应的隐藏状态
    if state is not None and reset.any():
        # 创建掩码：1表示不重置，0表示重置
        mask = tf.cast(1 - reset, self._float)[:, None]
        # 应用掩码到状态的每个组件(递归处理嵌套结构)
        state = tf.nest.map_structure(lambda x: x * mask, state)
    
    # 训练触发器：根据预设频率决定是否执行训练
    if self._should_train(step):
        # 日志触发器：决定是否记录训练日志
        log = self._should_log(step)
        
        # 确定训练步数：预热阶段使用更多步数，之后使用常规步数
        n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
        print(f'Training for {n} steps.')
        
        # 在分布式策略下执行训练循环
        with self._strategy.scope():
            for train_step in range(n):
                # 仅在第一轮训练且需要记录图像时生成图像摘要
                log_images = self._c.log_images and log and train_step == 0
                # 从数据集获取一批数据并执行单步训练
                self.train(next(self._dataset), log_images)
        
        # 记录训练摘要(如损失、指标等)
        if log:
            self._write_summaries()
    
    # 通过策略网络生成动作和更新状态
    action, state = self.policy(obs, state, training)
    
    # 如果处于训练模式，更新全局步数(考虑多环境和动作重复)
    if training:
        self._step.assign_add(len(reset) * self._c.action_repeat)
    # self._step.assign_add(2 * 4)  # 等价于 self._step += 8
    return action, state

  @tf.function
  def policy(self, obs, state, training):
    """
    根据当前观测和状态生成动作，支持训练和推理模式
    
    参数:
        obs: 环境观测字典，通常包含'image'等键
        state: 循环神经网络的隐藏状态(初始时为None)
        training: 是否处于训练模式
        
    返回:
        action: 生成的动作
        state: 更新后的隐藏状态
    """
    # 处理初始状态：创建初始潜在状态和全零动作
    if state is None:
        # 初始化dynamics模型的潜在状态，批次大小为观测数量
        latent = self._dynamics.initial(len(obs['image']))
        # 创建全零动作，形状为(批次大小, 动作维度)
        action = tf.zeros((len(obs['image']), self._actdim), self._float)
    else:
        # 从状态中解包潜在状态和上一步动作
        latent, action = state
    
    # 预处理观测数据并通过编码器转换为嵌入向量
    embed = self._encode(preprocess(obs, self._c))
    
    # 通过动态模型的观测步骤更新潜在状态
    # latent: 上一时刻的潜在状态
    # action: 上一时刻的动作
    # embed: 当前观测的嵌入表示
    latent, _ = self._dynamics.obs_step(latent, action, embed)
    
    # 从潜在状态中提取特征向量(用于策略和价值计算)
    feat = self._dynamics.get_feat(latent)
    
    # 根据训练/推理模式选择动作生成方式
    if training:
        # 训练时从策略分布中采样动作(增加探索)
        action = self._actor(feat).sample()
    else:
        # 推理时选择概率最高的动作(确定性策略)
        action = self._actor(feat).mode()
    
    # 应用探索策略(训练时增加随机性，推理时可能减少或关闭)
    action = self._exploration(action, training)
    
    # 更新状态为当前潜在状态和动作
    state = (latent, action)
    
    return action, state

  def load(self, filename):
    # 调用父类（通常是tf.Module或自定义基类）的加载方法
    super().load(filename)
    
    # 重置预训练标志（关键步骤）
    self._should_pretrain()
    # 作用：确保从检查点恢复后仍执行必要的初始化流程

  @tf.function()
  def train(self, data, log_images=False):
    self._strategy.run(self._train, args=(data, log_images))

  def _train(self, data, log_images):
    # 模型训练步骤
    with tf.GradientTape() as model_tape:
      # 编码观测
      embed = self._encode(data)
      
      # 根据观测和动作序列更新隐状态（后验和先验）
      post, prior = self._dynamics.observe(embed, data['action'])
      
      # 从隐状态提取特征
      feat = self._dynamics.get_feat(post)
      
      # 预测观测（图像）、奖励和持续概率
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      
      # 计算对数似然
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
      
      # 如果使用持续概率预测
      # 如果配置中启用了持续概率预测（pcont=True）
      if self._c.pcont:
        # 从特征中预测下一个时间步的持续概率（即环境不会终止的概率）
        # self._pcont 是一个神经网络，通常输出一个二项分布（Bernoulli）
        pcont_pred = self._pcont(feat)
          
        # 计算持续概率的目标值
        # self._c.discount 是折扣因子（如0.99）
        # data['discount'] 是环境提供的实际折扣（通常为1.0，除非终止状态为0.0）
        # 目标值 = 折扣因子 × 实际折扣
        pcont_target = self._c.discount * data['discount']
          
        # 计算预测的持续概率分布与目标值之间的对数似然
        # log_prob 是计算目标值在预测分布中的对数概率密度
        # tf.reduce_mean 对所有样本和时间步取平均
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
          
        # 应用持续概率预测的缩放因子
        # self._c.pcont_scale 通常设置为10.0，用于调整该损失项的权重
        likes.pcont *= self._c.pcont_scale
      
      # 计算后验和先验分布之间的KL散度
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)  # 确保KL散度不低于自由比特数
      
      # 计算模型损失（负对数似然 + KL散度）
      model_loss = self._c.kl_scale * div - sum(likes.values())
      model_loss /= float(self._strategy.num_replicas_in_sync)  # 归一化损失
      
    # 策略网络训练
    with tf.GradientTape() as actor_tape:
      # 想象未来轨迹
      # 输入: post = {'stoch': [B, T, stoch_size], 'deter': [B, T, deter_size]}
      # 输出: imag_feat = [horizon, B*T, feature_size]
      imag_feat = self._imagine_ahead(post)

      # 预测奖励和持续概率
      # 输入: imag_feat = [horizon, B*T, feature_size]
      # 输出: reward = [horizon, B*T, 1]
      reward = self._reward(imag_feat).mode()

      # 持续概率预测（如果启用）
      if self._c.pcont:
          # 输入: imag_feat = [horizon, B*T, feature_size]
          # 输出: pcont = [horizon, B*T, 1]
          pcont = self._pcont(imag_feat).mean()
      else:
          # 使用固定折扣因子替代持续概率
          # 输出: pcont = [horizon, B*T, 1]
          pcont = self._c.discount * tf.ones_like(reward)

      # 计算价值函数
      # 输入: imag_feat = [horizon, B*T, feature_size]
      # 输出: value = [horizon, B*T, 1]
      value = self._value(imag_feat).mode()

      # mode()：用于获取分布的最可能值（峰值），适用于奖励和价值函数的估计。
      # mean()：用于获取分布的期望值，适用于持续概率（伯努利分布）的估计。
      
      # 使用λ-return计算回报
      # reward: [horizon, B*T, 1] -> [horizon-1, B*T, 1]
      # value: [horizon, B*T, 1] -> [horizon-1, B*T, 1]
      # pcont: [horizon, B*T, 1] -> [horizon-1, B*T, 1]
      # bootstrap: [B*T, 1] (value的最后一个时间步)
      # lambda_: 标量(通常0.95)，控制多步回报的平衡
      # axis: 时间维度(此处为0)
      returns = tools.lambda_return(
          reward[:-1],      # 移除最后一个时间步，形状[horizon-1, B*T, 1]
          value[:-1],       # 移除最后一个时间步，形状[horizon-1, B*T, 1]
          pcont[:-1],       # 移除最后一个时间步，形状[horizon-1, B*T, 1]
          bootstrap=value[-1],  # 引导值，用于最后一步的价值估计，形状[B*T, 1]
          lambda_=self._c.disclam,  # λ参数(通常0.95)，控制回报计算方式
          axis=0  # 时间维度在第0位
      )
      # 返回值形状: [horizon-1, B*T, 1]
      # 每个时间步的returns表示从该时间步开始的累积折扣回报估计
      
      # 计算折扣因子（discount factor），用于加权未来回报
      # pcont: [horizon, B*T, 1]，持续概率序列，表示环境在下一时刻继续的概率
      # 1. pcont[:1]: 取pcont的第一个时间步，形状[1, B*T, 1]
      # 2. tf.ones_like(pcont[:1]): 创建与pcont第一个时间步相同形状的全1张量，形状[1, B*T, 1]
      # 3. pcont[:-2]: 取pcont的前n-2个时间步（排除最后两个时间步），形状[horizon-2, B*T, 1]
      # 4. tf.concat([...], 0): 沿时间维度（第0维）拼接，形状[horizon-1, B*T, 1]
      # 5. tf.math.cumprod(..., 0): 沿时间维度计算累积乘积，形状[horizon-1, B*T, 1]
      # 6. tf.stop_gradient: 停止梯度计算，确保不影响pcont的训练
      # 折扣因子仅作为权重使用，不参与模型参数更新，因此停止梯度传播
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      # 最终discount形状: [horizon-1, B*T, 1]
      # 每个元素表示从当前时间步到未来的累积折扣因子
      
      # 计算策略损失（最大化累积回报）
      actor_loss = -tf.reduce_mean(discount * returns)
      actor_loss /= float(self._strategy.num_replicas_in_sync)
      
    # 价值网络训练
    with tf.GradientTape() as value_tape:
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
      value_loss /= float(self._strategy.num_replicas_in_sync)
      
    # 计算并应用梯度
    model_norm = self._model_opt(model_tape, model_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)
    
    # 记录训练指标和图像
    # ​​get_replica_context()​​
    # 返回当前副本的上下文对象 tf.distribute.ReplicaContext，若在跨副本上下文中（如 strategy.scope() 内）调用则返回 None。
    # ​​replica_id_in_sync_group​​
    # 表示当前副本在同步组中的唯一标识（从 0 开始），与 num_replicas_in_sync 配合可获取总副本数。
    # ​​作用​​：判断当前计算是否在分布式训练的第一个副本（replica_id_in_sync_group 为 0）上执行
    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._dynamics = models.RSSM(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    
    # 创建一个名为self._pcont的二分类模型，用于预测当前状态下 episode 继续的概率
    if self._c.pcont:
      # ()：输出形状。空元组表示标量输出 (即单个概率值)
      # 'binary'：输出分布类型。使用伯努利分布表示二分类
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
      
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def _exploration(self, action, training):
    """
    对生成的动作应用探索策略，增强训练稳定性和探索效率
    
    参数:
        action: 模型生成的原始动作
        training: 是否处于训练模式
        
    返回:
        action: 应用探索策略后的动作
    """
    # 训练模式下的探索逻辑
    if training:
        # 获取基础探索强度
        amount = self._c.expl_amount
        
        # 探索强度衰减: 随训练进行逐渐降低探索率
        if self._c.expl_decay:
            # 计算衰减系数: 0.5^(步数/衰减步数)
            decay_factor = 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
            amount *= decay_factor
        
        # 设置最小探索强度，防止衰减至过低
        if self._c.expl_min:
            amount = tf.maximum(self._c.expl_min, amount)
        
        # 记录当前探索强度到监控指标
        self._metrics['expl_amount'].update_state(amount)
    
    # 评估模式下的探索逻辑(通常用于测试时增加随机性)
    elif self._c.eval_noise:
        amount = self._c.eval_noise
    
    # 既不在训练模式，也不需要评估噪声，直接返回原始动作
    else:
        return action
    
    # 根据配置选择探索策略
    # 高斯噪声探索: 在原始动作上添加高斯噪声
    if self._c.expl == 'additive_gaussian':
        # 创建以原始动作为均值，探索强度为标准差的正态分布
        # 采样后将动作裁剪到[-1, 1]范围(假设动作空间在此范围内)
        return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    
    # 完全随机探索: 忽略原始动作，生成均匀分布的随机动作
    if self._c.expl == 'completely_random':
        return tf.random.uniform(action.shape, -1, 1)
    
    # ε-贪心探索: 以一定概率随机选择动作，其他时间使用原始动作
    if self._c.expl == 'epsilon_greedy':
      # 1. 生成均匀分布的随机动作索引（覆盖所有可能动作）
      # 0 * action 生成与action同形状的全零张量，作为Categorical分布的logits
      # logits全为0时，所有动作被选中的概率相等（均匀分布）
      indices = tfd.Categorical(0 * action).sample()
      # 输出形状：(batch_size,)
      # 每个样本对应一个动作索引
      # 例如：batch_size=2 时，indices 可能为 [2, 0]
      
      # 2. 以epsilon概率选择随机动作，(1-epsilon)概率使用原始动作
      return tf.where(
          # 生成概率掩码：对每个样本，生成[0,1)之间的随机数并与epsilon比较
          # action.shape[:1] 获取批次维度（即环境数量）
          # 例如：当batch_size=2时，可能生成[0.3, 0.7]
          # 若epsilon=0.5，则掩码为[True, False]（第一个样本选随机动作，第二个选原始动作）
          tf.random.uniform(action.shape[:1], 0, 1) < amount,
          
          # 随机动作分支：将随机索引转换为one-hot向量
          # indices是随机选择的动作索引（如[2, 0]）
          # action.shape[-1] 是动作空间维度（如action_dim=4）
          # 例如：indices=2, action_dim=4 → [0, 0, 1, 0]
          tf.one_hot(indices, action.shape[-1], dtype=self._float),
          
          # 原始动作分支：直接使用策略网络输出的动作
          action
      )
    
    # 如果配置了不支持的探索策略，抛出错误
    raise NotImplementedError(self._c.expl)

  def _imagine_ahead(self, post):
    """
    基于当前后验状态，通过策略网络生成未来想象轨迹
    
    参数:
        post: 当前后验状态，字典格式，包含随机状态和确定性状态
              典型形状: {'stoch': [B, T, stoch_size], 'deter': [B, T, deter_size]}
    返回:
        imag_feat: 想象轨迹的特征表示
                   形状: [horizon, B*T, feature_size]
    """
    # 如果启用了持续概率预测(pcont)，移除最后一步，因为最后一步可能是终止状态
    if self._c.pcont:  # Last step could be terminal.
        # 移除每个状态张量的最后一个时间步
        # 输入形状: [B, T, ...] -> 输出形状: [B, T-1, ...]
        post = {k: v[:, :-1] for k, v in post.items()}
    
    # 定义展平函数：将批次和时间维度合并为一个维度
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:])) # [-1]应该是自动计算前面的维度形状
    
    # 对后验状态的每个组件应用展平操作
    # 输入形状: [B, T, state_dim] -> 输出形状: [B*T, state_dim]
    start = {k: flatten(v) for k, v in post.items()}
    
    # 定义策略函数：根据当前状态生成下一个动作
    # 使用stop_gradient避免策略网络的梯度影响状态表示
    policy = lambda state: self._actor(
        tf.stop_gradient(self._dynamics.get_feat(state))).sample()
    
    # 使用static_scan沿时间维度展开，生成未来horizon步的状态轨迹
    # static_scan类似于tf.scan，但在编译时展开循环，提高性能
    states = tools.static_scan(
        # 每一步的更新函数：根据前一个状态和策略生成的动作，预测下一个状态
        lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
        # 时间步数：循环执行horizon次
        tf.range(self._c.horizon),
        # 初始状态：使用展平后的后验状态
        start)
    # states形状: {k: [horizon, B*T, state_dim] for k in ['stoch', 'deter']}
    
    # 从生成的状态轨迹中提取特征
    # 特征通常是随机状态和确定性状态的组合
    imag_feat = self._dynamics.get_feat(states)
    # imag_feat形状: [horizon, B*T, feature_size]
    
    # 返回想象轨迹的特征表示
    return imag_feat

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, value_loss, actor_loss, model_norm, value_norm,
      actor_norm):
    # 记录各种标量指标用于监控训练过程
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())

  def _image_summaries(self, data, embed, image_pred):
    """
    生成图像摘要用于可视化模型性能，包括真实图像、重构图像和预测图像的对比
    Args:
        data: 输入的原始数据，包含图像、动作等信息
        embed: 编码器处理后的观测嵌入
        image_pred: 解码器预测的图像分布
    """
    # 提取前6个样本的真实图像并反归一化（模型输入时减去了0.5）
    truth = data['image'][:6] + 0.5  # 形状: [6, T, H, W, C]
    
    # 获取前6个样本的重构图像（取分布的众数作为最可能的预测值）
    recon = image_pred.mode()[:6]  # 形状: [6, T, H, W, C]
    
    # 第一阶段：观察（前5步） - 基于真实观测和动作更新模型状态
    # observe方法返回后验状态和先验状态，这里只需要后验状态
    init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    
    # 提取最后一个时间步的状态作为想象的起始点
    # 每个状态分量（如确定性状态、随机状态）都取最后一个时间步
    init = {k: v[:, -1] for k, v in init.items()}  # 形状: 从[6, 5, ...]变为[6, ...]
    # 切片操作 v[:, -1]​​:
    # : 表示保留第0维（B）的所有元素。
    # -1 表示选择第1维（T）的最后一个时间步的数据。
    # 未指定的第2维（stoch_size/deter_size）会自动保留
    
    # 第二阶段：预测（后续时间步） - 使用模型想象未来状态和观测
    # imagine方法根据初始状态和后续动作序列生成未来状态
    prior = self._dynamics.imagine(data['action'][:6, 5:], init)
    
    # 解码想象的状态为预测图像
    openl = self._decode(self._dynamics.get_feat(prior)).mode()  # 形状: [6, T-5, H, W, C]
    
    # 组合重构图像（前5步）和预测图像（后续步）
    # 重构部分使用真实观测，预测部分使用模型想象
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], axis=1)  # 形状: [6, T, H, W, C]
    
    # 创建误差可视化
    error = (model - truth + 1) / 2  # 形状: [6, T, H, W, C]
    # 输入范围：
    # model 和 truth 的值域均为 [0, 1]（因之前执行过 + 0.5 的调整）。
    # 因此 model - truth 的差值范围是 [-1, 1]：
    # 1：模型预测比真实值大（过估计）。
    # -1：模型预测比真实值小（欠估计）。
    # 0：预测完全正确。
    # 归一化到 [0,1]：
    # + 1 将范围平移至 [0, 2]。
    # / 2 将范围压缩到 [0, 1]。
    # 此时：
    # 0 → 黑色（最大负误差，truth 比 model 大 1）。
    # 0.5 → 中灰色（完全正确，误差为 0）。
    # 1 → 白色（最大正误差，model 比 truth 大 1）。


    # 最终可视化组合：
    # [真实图像] 
    # [重构+预测图像] 
    # [误差图像]
    # 沿高度方向（axis=2）拼接，便于横向对比
    openl = tf.concat([truth, model, error], axis=2)  # 形状: [6, T, 3*H, W, C]
    
    # 将组合后的图像写入TensorBoard
    # graph_summary确保在分布式训练中只由主副本执行写入操作
    # video_summary将序列图像转换为视频格式进行可视化
    tools.graph_summary(
        self._writer, tools.video_summary, 'agent/openl', openl)
    
  # 写入训练摘要到日志文件的函数，包括JSONL和TensorBoard格式。
  def _write_summaries(self):
    """
    将训练指标写入多种日志格式（JSONL和TensorBoard），并打印到控制台
    """
    # 获取当前全局步数（转换为Python整数）
    step = int(self._step.numpy())
    
    # 收集所有TensorFlow指标的值（如损失、奖励等）
    # self._metrics 是一个字典，键为指标名称，值为tf.keras.metrics实例
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    
    # 计算每秒处理的步数（FPS）
    if self._last_log is not None:
        # 计算自上次记录以来的时间间隔
        duration = time.time() - self._last_time
        # 更新上次记录时间
        self._last_time += duration
        # 计算FPS：(当前步数 - 上次步数) / 时间间隔
        metrics.append(('fps', (step - self._last_log) / duration))
    
    # 记录当前步数，用于下次计算FPS
    self._last_log = step
    
    # 重置所有指标状态（tf.keras.metrics在每个周期后需要重置）
    [m.reset_states() for m in self._metrics.values()]
    
    # 将指标写入JSONL文件（每行一个JSON对象，便于后续分析）
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
        # 构建JSON对象：包含步数和所有指标
        log_entry = {'step': step, **dict(metrics)}
        # 写入JSON字符串并添加换行符
        f.write(json.dumps(log_entry) + '\n')
    
    # 将指标写入TensorBoard（用于可视化）
    # tf.summary.scalar创建标量摘要，前缀'agent/'用于分类
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    
    # 打印指标到控制台（便于实时监控）
    # 格式化输出：[步数] 指标1 值1 / 指标2 值2 / ...
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    
    # 刷新TensorBoard写入器，确保所有摘要数据被写入磁盘
    self._writer.flush()

# 预处理观测数据
def preprocess(obs, config):
    """
    预处理环境观测和奖励数据
    
    参数:
        obs: 包含环境观测和奖励的字典，通常包含'image'和'reward'等键
        config: 配置对象，需包含以下属性:
            - clip_rewards: 奖励裁剪方式('none'或'tanh')
            - compute_dtype: 计算数据类型(如tf.float32)
            
    返回:
        dict: 预处理后的观测和奖励数据
    """
    # 获取当前策略网络使用的数据类型
    dtype = prec.global_policy().compute_dtype
    
    # 复制观测数据，避免修改原始数据
    obs = obs.copy()
    
    # 将所有处理放在CPU上执行，避免阻塞GPU计算
    with tf.device('cpu:0'):
        # 图像处理:
        # 1. 将图像数据从uint8类型(0-255)转换为指定浮点类型
        # 2. 归一化到[-0.5, 0.5]范围，便于神经网络处理
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        
        # 奖励处理:
        # 根据配置选择奖励裁剪函数
        # - 'none': 不裁剪，直接返回原始奖励
        # - 'tanh': 使用tanh函数将奖励压缩到[-1, 1]范围，缓解奖励爆炸问题
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        
        # 应用奖励裁剪函数
        obs['reward'] = clip_rewards(obs['reward'])
    
    # 返回预处理后的观测数据
    return obs

# 计算总步数
def count_steps(datadir, config):
    """
    计算考虑动作重复后的总训练步数
    
    参数:
        datadir: 存储训练数据的目录路径 (pathlib.Path)
        config: 配置对象 (包含action_repeat参数)
    
    返回:
        int: 总训练步数 = 环境原始步数 × 动作重复次数
    """
    # 调用count_episodes获取原始步数，乘以action_repeat得到实际训练步数
    return tools.count_episodes(datadir)[1] * config.action_repeat
    # [1]表示取count_episodes返回的第二个值（总步数）
    # 例如：若原始数据有1000步，action_repeat=2 → 返回2000

# 加载数据集
def load_dataset(directory, config):
    """
    从指定目录加载episode数据并创建TensorFlow数据集
    
    参数:
        directory: 存储episode文件的目录路径
        config: 配置对象，需包含以下属性:
            - train_steps: 每轮训练扫描的episode数量
            - batch_length: 每个样本的序列长度
            - batch_size: 批次大小
            - dataset_balance: 是否平衡采样短episode
            
    返回:
        tf.data.Dataset: 处理后的数据集，可直接用于模型训练
    """
    # 获取一个样本episode以确定数据类型和形状
    # 这一步仅用于推断数据结构，不参与实际训练
    episode = next(tools.load_episodes(directory, 1))
    
    # 提取每个键对应值的数据类型（如float32、int64）
    types = {k: v.dtype for k, v in episode.items()}
    
    # 定义每个键对应值的形状，将第一维设为None表示可变长度
    # 例如原形状为[100, 64, 64, 3] → (None, 64, 64, 3)
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    
    # 定义数据集生成器函数
    # 每次调用时会创建一个新的episode生成器
    generator = lambda: tools.load_episodes(
        directory,              # 数据目录
        config.train_steps,     # 每轮训练扫描的episode数量
        config.batch_length,    # 每个样本的序列长度
        config.dataset_balance  # 是否平衡采样短episode
    )
    
    # 创建TensorFlow数据集
    # generator: 生成器函数，负责产生数据
    # types: 数据类型定义
    # shapes: 数据形状定义
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    
    # 按批次大小进行批处理
    # drop_remainder=True: 丢弃不足batch_size的最后一批数据
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    
    # 应用预处理函数，对每个批次的数据进行转换
    # functools.partial: 固定config参数，创建偏函数
    dataset = dataset.map(functools.partial(preprocess, config=config))
    
    # 预取数据到缓冲区，提升训练效率
    # 10: 预取缓冲区大小，可根据内存情况调整
    dataset = dataset.prefetch(10)
    
    return dataset

# 总结episode（记录返回值和长度）
def summarize_episode(episode, config, datadir, writer, prefix):
    """
    汇总并记录episode的统计信息（奖励、长度等），支持多种日志格式
    
    参数:
        episode: 包含episode数据的字典，通常包含'reward'、'image'等键
        config: 配置对象，包含action_repeat、logdir等参数
        datadir: 数据目录路径，用于统计总episode数和步数
        writer: TensorBoard写入器对象
        prefix: 日志前缀（如'train'或'test'），用于区分不同阶段的统计
    """
    # 统计数据目录中的总episode数和步数
    episodes, steps = tools.count_episodes(datadir)
    
    # 计算episode实际长度（考虑动作重复）
    # 减1是因为最后一步没有后续动作，length = 有效步数 * 每步重复次数
    length = (len(episode['reward']) - 1) * config.action_repeat
    
    # 计算episode总回报（累积奖励）
    ret = episode['reward'].sum()
    
    # 打印基本统计信息
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    
    # 定义要记录的指标列表
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),  # 总回报
        (f'{prefix}/length', len(episode['reward']) - 1),      # 步数（不包含最后一步）
        (f'episodes', episodes)                                  # 总episode数
    ]
    
    # 计算当前全局步数（基于数据目录中的记录）
    step = count_steps(datadir, config)
    
    # 将指标追加写入JSONL格式的日志文件
    # JSONL格式：每行一个JSON对象，便于后续解析和聚合
    with (config.logdir / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    # 格式示例：[('step', 1000)] + [('train/return', 50.2), ('train/length', 200)]
    # # 结果为：[('step', 1000), ('train/return', 50.2), ('train/length', 200)]
    # dict -> {'step': 1000, 'train/return': 50.2, 'train/length': 200}
    
    # 使用TensorBoard记录指标
    # 使用writer.as_default()确保在多线程环境中正确写入
    with writer.as_default():  # Env might run in a different thread.
        # 设置当前全局步数
        tf.summary.experimental.set_step(step)
        
        # 记录所有指标到TensorBoard
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        
        # 如果是测试阶段，额外记录视频回放
        if prefix == 'test':
            # 添加batch维度(None)，因为video_summary期望[B, T, H, W, C]格式
            tools.video_summary(f'sim/{prefix}/video', episode['image'][None])
        # [None] 的作用
        # None 在索引操作中相当于 np.newaxis，用于插入一个新的维度。
        # episode['image'][None] 会将原始形状 [T, H, W, C] 变为 [1, T, H, W, C]，其中新增的维度（索引 0）表示批次大小（batch size）为 1。

def make_env(config, writer, prefix, datadir, store):
    """创建强化学习环境，支持DeepMind Control和Atari等不同任务套件"""
    # 从配置中解析任务套件和具体任务名称（例如：'dmc_walker_walk' → suite='dmc', task='walker_walk'）
    suite, task = config.task.split('_', 1)
    
    # 处理DeepMind Control套件任务
    if suite == 'dmc':
        # 创建原始环境
        env = wrappers.DeepMindControl(task)
        # 动作重复：每个动作执行指定次数，加速训练
        env = wrappers.ActionRepeat(env, config.action_repeat)
        # 动作归一化：将连续动作空间归一化到[-1, 1]范围
        env = wrappers.NormalizeActions(env)
    
    # 处理Atari游戏任务
    elif suite == 'atari':
        env = wrappers.Atari(
            task,                   # 游戏名称
            config.action_repeat,   # 动作重复次数
            (64, 64),               # 图像观测的大小
            grayscale=False,        # 是否使用灰度图像
            life_done=True,         # 生命值减少是否视为episode结束
            sticky_actions=True     # 是否使用粘性动作（增加随机性）
        )
        # 独热编码动作：将离散动作转换为独热向量表示
        env = wrappers.OneHotAction(env)
    
    # 不支持的任务套件
    else:
        raise NotImplementedError(f"不支持的任务套件: {suite}")
    
    # 设置环境时间限制：episode的最大步数 = 配置时间限制 / 动作重复次数
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    
    # 初始化回调函数列表（用于收集和记录episode数据）
    callbacks = []
    
    # 如果启用存储，则添加episode保存回调
    if store:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    
    # 添加episode摘要回调：记录奖励、长度等统计信息
    callbacks.append(
        lambda ep: summarize_episode(ep, config, datadir, writer, prefix)
    )
    
    # 环境包装器：收集episode数据并触发回调
    env = wrappers.Collect(env, callbacks, config.precision)
    
    # 将奖励添加到观测中（某些算法需要此格式）
    env = wrappers.RewardObs(env)
    
    return env

def main(config):
  # 检查是否启用GPU内存动态增长模式（避免一次性占用全部GPU内存）
  if config.gpu_growth:
      # 遍历所有可用的物理GPU设备
      for gpu in tf.config.experimental.list_physical_devices('GPU'):
          # 为当前GPU设备启用内存动态增长模式
          tf.config.experimental.set_memory_growth(gpu, True)
          # 作用：TensorFlow会根据需要逐步增加GPU显存占用，而不是启动时就占用所有可用显存
          # 适用场景：多任务共享GPU或显存有限的开发环境

  # 验证训练精度配置是否合法（只允许16位或32位精度）
  assert config.precision in (16, 32), config.precision
  # 断言失败时会显示错误信息，包含错误的config.precision值

  # 如果配置为16位混合精度训练
  if config.precision == 16:
      # 设置全局混合精度策略（自动在float16和float32之间转换）
      prec.set_policy(prec.Policy('mixed_float16'))
      # 技术细节：
      # 1. 计算使用float16加速（比float32快1.5-3倍）
      # 2. 权重保持float32保证数值稳定性
      # 3. 自动插入loss scaling防止梯度下溢

  # 将训练总步数转换为整数类型（确保类型安全）
  config.steps = int(config.steps)
  # 原因：从配置文件读取的数值可能是浮点数（如5e6）

  # 创建日志目录（递归创建所有不存在的父目录）
  config.logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', config.logdir)

  # Create environments.
  datadir = config.logdir / 'episodes'  # 定义经验回放数据的存储目录
  writer = tf.summary.create_file_writer(  # 创建TensorBoard日志写入器
      str(config.logdir),  # 日志保存路径
      max_queue=1000,  # 内存中缓存的未写入磁盘的摘要事件最大数量，平衡I/O效率与内存消耗
      flush_millis=20000)  # 自动刷新间隔（20秒）即使未达max_queue也会定期写入磁盘
  writer.set_as_default()  # 设为默认的摘要写入器。后续影响：所有tf.summary.xxx()操作会自动使用此writer
  # 创建训练环境列表（多个并行环境）
  train_envs = [
      wrappers.Async(  # 异步环境包装器
          # 使用lambda延迟环境创建，避免在列表推导中立即执行
          lambda: make_env(
              config,       # 主配置参数
              writer,      # TensorBoard写入器
              'train',     # 环境前缀（用于日志分类）
              datadir,     # 数据存储目录
              store=True   # 是否存储训练数据
          ),
          config.parallel  # 并行执行方式（如'thread'/'process'）
      )
      # 创建config.envs指定数量的环境实例
      for _ in range(config.envs)
  ]

  # 创建测试环境列表（结构与训练环境相同）
  test_envs = [
      wrappers.Async(
          lambda: make_env(
              config,
              writer,
              'test',      # 测试环境前缀
              datadir,
              store=False  # 不存储测试数据
          ),
          config.parallel
      )
      for _ in range(config.envs)  # 通常测试环境数与训练相同
  ]

  # 从第一个训练环境中获取动作空间定义
  actspace = train_envs[0].action_space
  # 作用：
  # 1. 获取动作的维度（如Box(3,)表示3维连续动作）
  # 2. 获取动作范围（如low=-1.0, high=1.0）
  # 3. 为后续模型初始化提供参数

  # 1. 预填充经验回放缓冲区（使用随机策略收集初始数据）
  # ------------------------------------------------------
  # 计算当前已收集的总步数（考虑动作重复系数）
  step = count_steps(datadir, config)

  # 计算需要预填充的步数（确保不少于0）
  prefill = max(0, config.prefill - step)

  # 打印预填充信息（例如：Prefill dataset with 5000 steps.）
  print(f'Prefill dataset with {prefill} steps.')

  # 定义随机策略函数：
  # - 输入：观测(o)(即obs), 终止标志(d)(即done), 状态(_)
  # d：终止标志（done），通常是一个布尔列表，指示每个环境是否结束（例如在多环境并行时）。
  # - 输出：随机动作列表（每个环境一个动作）和None（状态）
  random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)

  # 执行随机策略模拟：
  # - 使用train_envs环境集合
  # - 模拟步数=prefill/action_repeat（换算为环境原始步数）
  tools.simulate(random_agent, train_envs, prefill / config.action_repeat)

  # 强制刷新TensorBoard写入器（确保日志及时更新）
  writer.flush()

  # 2. 初始化智能体训练流程
  # ------------------------------------------------------
  # 重新计算当前步数（预填充后可能有变化）
  step = count_steps(datadir, config)

  # 打印剩余训练步数（例如：Simulating agent for 1000000 steps.）
  print(f'Simulating agent for {config.steps-step} steps.')

  # 创建Dreamer智能体实例：
  # - config: 训练配置参数
  # - datadir: 数据存储路径
  # - actspace: 动作空间定义
  # - writer: TensorBoard日志写入器
  agent = Dreamer(config, datadir, actspace, writer)

  # 检查是否存在检查点文件（恢复训练）
  if (config.logdir / 'variables.pkl').exists():
      print('Load checkpoint.')
      # 加载模型参数和优化器状态
      agent.load(config.logdir / 'variables.pkl')

  # 初始化状态变量（用于跨episode的状态保持）
  state = None

  # 3. 主训练循环
  # ------------------------------------------------------
  while step < config.steps:
      # 3.1 评估阶段
      print('Start evaluation.')
      # 使用测试环境运行评估：
      # - functools.partial固定training=False（关闭探索）
      # - episodes=1表示每个测试环境运行1个完整episode
      tools.simulate(
          functools.partial(agent, training=False), 
          test_envs, 
          episodes=1)
      writer.flush()

      # 3.2 数据收集阶段
      print('Start collection.')
      # 计算需要收集的环境步数（考虑动作重复系数）
      steps = config.eval_every // config.action_repeat
      # 使用训练环境收集数据：
      # - 保持state跨批次传递（RNN状态延续）
      # - 返回最后一个状态用于下次收集
      state = tools.simulate(agent, train_envs, steps, state=state)

      # 更新当前总步数
      step = count_steps(datadir, config)

      # 保存模型检查点（包含模型参数和训练状态）
      agent.save(config.logdir / 'variables.pkl')

  # 4. 训练结束清理
  # ------------------------------------------------------
  # 关闭所有环境（释放资源）
  for env in train_envs + test_envs:
      env.close()

if __name__ == '__main__':
  # 在程序出错时，为错误堆栈信息添加彩色高亮显示，让错误信息更加清晰易读
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())
