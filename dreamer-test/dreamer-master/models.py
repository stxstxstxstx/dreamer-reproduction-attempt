import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools

class RSSM(tools.Module):
  """循环状态空间模型，用于学习环境的动态特性。
  
  该模型维护一个潜在状态，包含随机(stochastic)和确定性(deterministic)两部分，
  能够根据观测和动作预测未来状态，或根据动作想象未来状态序列。
  """

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    """初始化RSSM模型。
    
    参数:
      stoch: 随机状态维度
      deter: 确定性状态维度(GRU单元数)
      hidden: 隐藏层维度
      act: 激活函数
    """
    super().__init__()
    self._activation = act
    self._stoch_size = stoch      # 随机状态变量 z 的维度
    self._deter_size = deter      # 确定性状态变量 h 的维度(GRU单元数)
    self._hidden_size = hidden    # 隐藏层维度
    # GRU单元用于处理序列信息，维护确定性状态 h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    """创建初始状态。
    
    参数:
      batch_size: 批次大小
      
    返回:
      包含初始状态的字典，包括均值、标准差、随机状态和确定性状态
    """
    # 获取计算数据类型
    dtype = prec.global_policy().compute_dtype
    # prec.global_policy()：获取全局的混合精度策略。
    # .compute_dtype：查看计算时使用的数据类型（float16 或 float32）
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),     # 初始随机状态均值
        std=tf.zeros([batch_size, self._stoch_size], dtype),      # 初始随机状态标准差
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),    # 初始随机状态 z_0
        deter=self._cell.get_initial_state(None, batch_size, dtype))  # 初始确定性状态 h_0
  
        # get_initial_state(inputs=None, batch_size=None, dtype=None)
        # 返回初始隐藏状态（全零张量），形状 (batch_size, units)。
        # 其中 units 表示隐藏状态的维度（整数）

  @tf.function
  def observe(self, embed, action, state=None):
    """根据观测和动作序列更新状态。
    
    参数:
      embed: 编码后的观测序列 [batch, time, embed_dim]
      action: 动作序列 [batch, time, action_dim]
      state: 初始状态，默认为None(使用零初始化)
      
    返回:
      post: 后验状态序列(结合了观测和动作)
        后验 (post)：结合动作、历史状态和当前观测的状态
        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        z_t ~ q(z_t|h_t,o_t)
      prior: 先验状态序列(仅基于动作)
        先验 (prior)：仅基于动作和历史状态预测的下一状态
        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        z_t ~ p(z_t|h_t)
    """
    if state is None:
      state = self.initial(tf.shape(action)[0])  # 初始化状态
    # 转置为[time, batch, ...]以便静态扫描，提高计算效率
    embed = tf.transpose(embed, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])
    # 使用静态扫描处理序列，依次处理每个时间步
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
  
    # 转回[batch, time, ...]格式
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    """仅根据动作序列想象未来状态。
    
    参数:
      action: 动作序列 [batch, time, action_dim]
      state: 初始状态，默认为None(使用零初始化)
      
    返回:
      prior: 先验状态序列
    """
    if state is None:
      state = self.initial(tf.shape(action)[0])  # 初始化状态
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])  # 转置为[time, batch, ...]
    # 使用静态扫描生成想象的状态序列，不使用观测
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}  # 转回[batch, time, ...]
    return prior

  def get_feat(self, state):
    """从状态中提取特征(随机状态和确定性状态的连接)。
    
    参数:
      state: 包含随机和确定性状态的字典
      
    返回:
      连接后的特征向量
    """
    return tf.concat([state['stoch'], state['deter']], -1) # [batch_size, time, stoch_size + deter_size]
    # 'stoch': 采样的随机状态 [batch_size, time, stoch_size]
    # 'deter': 更新后的确定性状态 [batch_size, time, deter_size]
    # 这里只是形状示例，形如[..., stoch_size + deter_size]

  def get_dist(self, state):
    """获取状态表示的概率分布。
    
    参数:
      state: 包含均值和标准差的状态字典
      
    返回:
      多元正态分布对象
    """
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    """执行一个观测步骤(结合动作和观测更新状态)。
    
    参数:
      prev_state: 上一时刻状态
      prev_action: 上一时刻动作
      embed: 当前观测编码
      
    返回:
      post: 后验状态(结合了观测)
      prior: 先验状态(仅基于动作)

    注记：
    观测步骤 (obs_step)：
    先调用 img_step 生成先验
    结合当前观测 o_t
    生成后验分布 q(z_t|h_t,o_t)
    """
    # 先计算先验状态(仅基于动作)
    prior = self.img_step(prev_state, prev_action)
    # 结合先验状态和观测编码计算后验状态
    x = tf.concat([prior['deter'], embed], -1)  # 拼接确定性状态和观测编码
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)  # 第一层全连接
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)  # 输出均值和标准差参数
    mean, std = tf.split(x, 2, -1)  # 分割为均值和标准差
    # 确保标准差为正且有最小值(0.1)，防止数值不稳定
    std = tf.nn.softplus(std) + 0.1
    # 从分布中采样随机状态 z_t ~ q(z_t|h_t,o_t)
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    """执行一个想象步骤(仅基于动作更新状态)。
    
    参数:
      prev_state: 上一时刻状态，字典包含:
        'stoch': 随机状态 [batch_size, stoch_size]
        'deter': 确定性状态 [batch_size, deter_size]
      prev_action: 上一时刻动作 [batch_size, action_dim]
      
    返回:
      prior: 先验状态，字典包含:
        'mean': 随机状态分布均值 [batch_size, stoch_size]
        'std': 随机状态分布标准差 [batch_size, stoch_size]
        'stoch': 采样的随机状态 [batch_size, stoch_size]
        'deter': 更新后的确定性状态 [batch_size, deter_size]

    注记：
    预测步骤 (img_step)：
    根据上一时刻随机状态 z_{t-1} 和动作 a_{t-1}
    更新确定性状态 h_t
    生成先验分布 p(z_t|h_t)
    """
    # 结合上一时刻随机状态和动作
    x = tf.concat([prev_state['stoch'], prev_action], -1)  # 拼接 z_{t-1} 和 a_{t-1}（​​参数 axis=-1​​：表示沿最后一个维度拼接）
    # [batch_size, hidden_size]
    
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)  # 第一层全连接
    # [batch_size, hidden_size]

    # 通过GRU单元更新确定性状态 h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    # 输入：
    # x: 输入张量，形状为 [batch_size, hidden_size]
    # [prev_state['deter']]: GRU 的初始状态，形状为 [batch_size, deter_size]，即 h_{t-1} 
    # 输出：
    # x: GRU 的输出，形状为 [batch_size, deter_size]
    # deter: GRU 的下一个状态，列表，包含一个形状为 [batch_size, deter_size] 的张量，即 h_t
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras将状态包装在列表中，这里取第一个元素
    
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)  # 第二层全连接
    # [batch_size, hidden_size]

    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)  # 输出均值和标准差参数
    # [batch_size, 2 * stoch_size]

    # mean: [batch_size, stoch_size]
    # std: [batch_size, stoch_size]
    mean, std = tf.split(x, 2, -1)  # 分割为均值和标准差
    
    # 确保标准差为正且有最小值(0.1)
    # [batch_size, stoch_size]
    std = tf.nn.softplus(std) + 0.1
    
    # 从分布中采样随机状态 z_t ~ p(z_t|h_t)
    # [batch_size, stoch_size]
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior

class ConvEncoder(tools.Module):
  """用于图像观测的卷积编码器。"""

  def __init__(self, depth=32, act=tf.nn.relu):
    """初始化卷积编码器。
    参数:
      depth: 基础卷积核数量
      act: 激活函数
    """
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    """对图像观测进行编码。
    参数:
      obs: 包含图像的字典 {image: [batch, time, height, width, channels]}
    返回:
      编码后的特征向量
    """
    kwargs = dict(strides=2, activation=self._act)
    # 重塑输入张量维度：
    # 原始形状：[batch, time, height, width, channels]
    # 重塑后：[batch*time, height, width, channels]
    # 目的：将批次和时间维度合并，便于统一处理所有帧
    # (-1,)：表示第一个维度由 TensorFlow 自动推断，其值为 B*T
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:])) 
    # 卷积尺寸计算：output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    # h1: (64-4)/2+1 = 31 → [B*T, 31, 31, 32]
    # h2: (31-4)/2+1 = 14 → [B*T, 14, 14, 64]
    # h3: (14-4)/2+1 = 6 → [B*T, 6, 6, 128]
    # h4: (6-4)/2+1 = 2 → [B*T, 2, 2, 256]
    # 4 × 4 的卷积核，步幅为 2，激活函数为 ReLU
    x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)  # 输出: [batch*time, 31, 31, 32]
    x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)  # 输出: [batch*time, 14, 14, 64]
    x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)  # 输出: [batch*time, 6, 6, 128]
    x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)  # 输出: [batch*time, 2, 2, 8 * self._depth]
    # 恢复批次和时间维度：shape[:-3] 获取 [batch, time]
    # 按行连接; 2×2×256 = 1024 = 32 * self._depth
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0) # 输出: [batch, time, 32 * self._depth]
    return tf.reshape(x, shape)

class ConvDecoder(tools.Module):
  """用于从特征重构图像的卷积解码器。"""

  def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
    """初始化卷积解码器。
    参数:
      depth: 基础卷积核数量
      act: 激活函数
      shape: 输出图像形状 [height, width, channels]
    """
    self._act = act               # 存储激活函数
    self._depth = depth           # 存储基础卷积核数量
    self._shape = shape           # 存储输出图像形状

  def __call__(self, features):
    """从特征向量重构图像。
    参数:
      features: 特征向量 [batch, time, feature_dim]
    返回:
      重构图像的概率分布
    """
    # 设置反卷积层通用参数：步长为2（用于上采样），使用指定激活函数
    kwargs = dict(strides=2, activation=self._act)
    
    # 第一层：全连接层扩展特征维度
    # 将输入特征映射到高维空间（32*depth=1024），为后续反卷积做准备
    x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
    
    # 重塑为4D张量 [batch*time, 1, 1, 32*depth]，模拟1×1的特征图
    # 这是反卷积的起点，后续通过上采样逐步扩大空间维度
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    
    # 四层反卷积上采样网络：
    # 每层逐步减少通道数，同时扩大空间尺寸（通过strides=2实现）
    # 反卷积尺寸计算：output_size = (input_size-1)*stride − 2×padding + kernel_size + output_padding
    # 参数说明：
    # output_padding：用于解决尺寸歧义问题（通常在反卷积中设置 output_padding=0）。
    # padding：与普通卷积类似，但作用效果相反。
    # h2: (1-1)*2 +5 = 5 → [B*T, 5, 5, 128]
    # h3: (5-1)*2 +5 = 13 → [B*T, 13, 13, 64]
    # h4: (13-1)*2 +6 = 30 → [B*T, 30, 30, 32]
    # h5: (30-1)*2 +6 = 64 → [B*T, 64, 64, 3]
    x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)  # 输出: [batch*time, 5, 5, 128]
    x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)  # 输出: [batch*time, 13, 13, 64]
    x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)  # 输出: [batch*time, 30, 30, 32]
    x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x) # 输出: [batch*time, 64, 64, channels]
    
    # 重塑为[batch, time, height, width, channels]格式
    # 恢复原始的批次和时间维度，与输入特征的前两维匹配
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    # axis=0
    # 指定 沿第 0 轴（行方向）拼接，因为 tf.shape(features)[:-1] 和 self._shape 都是 1D 张量。
    # 例如：
    # tf.concat([[2, 3, 4], [10, 20]], 0) → [2, 3, 4, 10, 20]。
    
    # 返回正态分布，表示重构的不确定性
    # 将每个像素值视为独立的正态分布，均值为重构值，标准差固定为1
    # len(self._shape)表示分布的事件维度（通常为3，对应HWC）
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
    # tfd.Independent(..., len(self._shape))
    # 将基础分布包装为独立分布，其中：
    # len(self._shape)：事件维度的数量，通常为 3（对应 height, width, channels）。
    # 作用：将最后三个维度（HWC）的分布视为独立事件，计算联合概率。

class DenseDecoder(tools.Module):
  """通用的密集解码器，用于从特征预测各种目标。"""

  def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
    """初始化密集解码器。
    参数:
      shape: 输出形状，例如 [height, width, channels] 或 [dim]
      layers: 隐藏层数量（不含输出层）
      units: 每层单元数（神经元数量）
      dist: 输出分布类型，'normal'（正态分布）或 'binary'（伯努利分布）
      act: 激活函数，默认使用 ELU（指数线性单元）
    """
    self._shape = shape                # 目标输出的形状元组
    self._layers = layers              # 隐藏层的数量
    self._units = units                # 每个隐藏层的单元数
    self._dist = dist                  # 输出分布类型
    self._act = act                    # 隐藏层使用的激活函数

  def __call__(self, features):
    """从特征预测目标值。
    参数:
      features: 输入特征张量，形状为 [batch, time, feature_dim]
    返回:
      预测值的概率分布对象
    """
    x = features  # 初始化输入为传入的特征
    
    # 多层全连接网络（密集层堆叠）
    for index in range(self._layers):
      # 创建或获取第 index 个隐藏层
      # 每个隐藏层包含 self._units 个神经元和指定的激活函数
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    
    # 输出层：将特征映射到目标维度
    # np.prod(self._shape) 计算输出形状的元素总数
    # 例如，若 shape=[3, 4]，则输出维度为 12
    x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
    
    # 重塑张量，恢复批次和时间维度
    # tf.shape(features)[:-1] 获取 [batch, time]
    # 与目标形状 self._shape 拼接，得到完整输出形状
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    
    # 根据指定分布类型返回相应的概率分布
    if self._dist == 'normal':
      # 正态分布：每个输出维度视为独立的高斯分布
      # 均值为网络输出 x，标准差固定为 1
      # len(self._shape) 指定事件维度的数量
      return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
    
    if self._dist == 'binary':
      # 伯努利分布：每个输出维度视为独立的二值分布
      return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
    
    # 图像生成中的二值图像重构：
    # 输入：潜在特征 [B,T,latent_dim]
    # 输出：二值图像分布 [B,T,H,W,1]
    # 分布类型：binary
    # 连续控制中的动作预测：
    # 输入：状态表示 [B,T,state_dim]
    # 输出：动作分布 [B,T,action_dim]
    # 分布类型：normal

    # 若指定的分布类型不支持，则抛出错误
    raise NotImplementedError(f"Distribution {self._dist} not implemented")

class ActionDecoder(tools.Module):
  """用于从特征生成动作的解码器。"""

  def __init__(
      self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
      min_std=1e-4, init_std=5, mean_scale=5):
    """初始化动作解码器。
    参数:
      size: 动作维度（连续动作空间的维度数或离散动作的数量）
      layers: 隐藏层数量（不含输出层）
      units: 每层单元数（神经元数量）
      dist: 动作分布类型（'tanh_normal' 用于连续动作，'onehot' 用于离散动作）
      act: 隐藏层激活函数（默认 ELU）
      min_std: 标准差下界（防止数值不稳定）
      init_std: 初始标准差（控制探索程度）
      mean_scale: 均值缩放因子（控制 tanh 变换的敏感度）
    """
    self._size = size                # 存储动作空间维度
    self._layers = layers            # 存储隐藏层数量
    self._units = units              # 存储每层单元数
    self._dist = dist                # 存储分布类型
    self._act = act                  # 存储激活函数
    self._min_std = min_std          # 存储最小标准差
    self._init_std = init_std        # 存储初始标准差
    self._mean_scale = mean_scale    # 存储均值缩放因子

  def __call__(self, features):
    """从特征生成动作分布。
    参数:
      features: 输入特征 [batch, time, feature_dim]
    返回:
      动作的概率分布（支持采样和计算概率密度）
    """
    # init_std = log(1 + exp(raw_init_std))
    # Softplus 函数：softplus(x)=log(1+e^x),它将 R 映射到 (0,+∞);
    # 其反函数：softplus^{−1}(y)=log(e^y−1)，将 (0,+∞)映射回 R。

    # 在 VAE 中，隐变量的标准差需要为正，通常用 raw_std 作为可优化参数：
    # # 初始化
    # raw_std = np.log(np.exp(init_std) - 1)  # 转换为未约束参数
    # # 前向计算时恢复 stddev
    # stddev = tf.math.softplus(raw_std)      # 确保 stddev > 0

    # 将初始标准差转换为 softplus 输入空间的值
    # 目的：使 softplus(std + raw_init_std) 在训练初期接近 init_std
    raw_init_std = np.log(np.exp(self._init_std) - 1)
    x = features  # 初始化输入为传入的特征
    
    # 多层全连接网络处理特征
    for index in range(self._layers):
      # 创建或获取第 index 个隐藏层
      # 每层使用 self._units 个神经元和指定的激活函数
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    
    # 根据指定的分布类型构建不同的动作分布
    if self._dist == 'tanh_normal':
      # 连续动作空间：使用 tanh 变换的正态分布
      
      # 输出层：生成均值和标准差参数（每个动作维度对应两个值）
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x) # 形状：[batch, time, 2*size]
      
      # 将输出分割为均值和标准差两部分
      # 将形状为 [batch, time, 2*size] 的张量 x 沿最后一维（-1）拆分为两个形状为 [batch, time, size] 的张量
      mean, std = tf.split(x, 2, -1)
      
      # 均值处理：使用 tanh 限制在 [-mean_scale, mean_scale] 范围内
      # 除以 mean_scale 是为了调整 tanh 的输入敏感度
      # 将原始 mean 按比例缩小，使其范围适应 tanh 的输入敏感区（tanh 在 [-2, 2] 区间外梯度接近 0）。
      # 作用：避免直接对 mean 应用 tanh 导致梯度消失（当 mean 绝对值较大时）。
      mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
      
      # 标准差处理：
      # 1. 添加 raw_init_std 作为偏置
      # 2. 应用 softplus 确保标准差为正
      # 3. 加上 min_std 防止标准差过小
      std = tf.nn.softplus(std + raw_init_std) + self._min_std
      
      # 构建正态分布
      dist = tfd.Normal(mean, std)
      
      # 应用 tanh 变换，将动作值限制在 [-1, 1] 范围内
      # transformed_distribution = tfd.TransformedDistribution(
      #   distribution=base_distribution,  # 基础分布
      #   bijector=bijector  # 用于变换的双射器
      # )
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      
      # 将各动作维度视为独立，形成多元分布
      dist = tfd.Independent(dist, 1)
      
      # 包装为自定义分布，支持获取分布模式（最可能的动作值）
      dist = tools.SampleDist(dist)
      
    elif self._dist == 'onehot':
      # 离散动作空间：使用 one-hot 分布
      
      # 输出层：生成每个动作的 logits（未归一化概率）
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      
      # 使用自定义的 OneHotDist 处理离散动作采样和优化
      dist = tools.OneHotDist(x)
      
    else:
      # 若指定的分布类型不支持，则抛出错误
      raise NotImplementedError(f"Distribution {self._dist} not implemented")
      
    return dist  # 返回构建好的动作分布
