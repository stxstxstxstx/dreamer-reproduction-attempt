import datetime
import io
import pathlib
import pickle
import re
import uuid

import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

class Module(tf.Module):
  """自定义TensorFlow模块基类，提供参数保存/加载和动态子模块管理功能。"""

  def save(self, filename):
    """将模块的所有变量保存到文件。
    
    参数:
      filename: 保存文件的路径
    """
    # 将所有变量转换为NumPy数组
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    # 使用pickle将变量值保存到二进制文件
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename):
    """从文件加载模块的变量值。
    
    参数:
      filename: 加载文件的路径
    """
    # 从二进制文件中读取变量值
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    # 将加载的值分配给当前模块的变量
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)
    # tf.nest.map_structure 简介:
    # 功能：递归地将函数 func 应用到嵌套结构（如列表、字典）的每个元素上。
    # 参数：
    # func：要应用的函数（这里是 lambda x, y: x.assign(y)）。
    # *nests：一个或多个嵌套结构（这里是 self.variables 和 values）。
    # 返回：与输入结构相同，但每个元素都被 func 处理过。

  def get(self, name, ctor, *args, **kwargs):
    """动态获取或创建子模块。
    
    这个方法允许在不预先声明的情况下，在需要时创建子模块，
    避免了在__init__方法中显式列出所有子模块。
    
    参数:
      name: 子模块的名称
      ctor: 创建子模块的构造函数
      *args, **kwargs: 传递给构造函数的参数
      
    返回:
      已存在或新创建的子模块
    """
    # 如果模块还没有_modules属性，初始化一个空字典
    if not hasattr(self, '_modules'):
      self._modules = {}
    # 如果指定名称的子模块不存在，创建它
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs)
    # 返回已存在或新创建的子模块
    return self._modules[name]

def nest_summary(structure):
    """
    递归生成嵌套结构的摘要信息，展示其形状和层级关系
    
    参数:
        structure: 任意嵌套结构（字典、列表、具有shape属性的对象）
        
    返回:
        summary: 嵌套结构的摘要信息（字典、列表或形状字符串）
    """
    # 处理字典类型
    if isinstance(structure, dict):
        # 递归处理字典的每个值，并保留键名
        return {k: nest_summary(v) for k, v in structure.items()}
    
    # 处理列表类型
    if isinstance(structure, list):
        # 递归处理列表的每个元素
        return [nest_summary(v) for v in structure]
    
    # 处理具有shape属性的对象（如NumPy数组、TensorFlow张量）
    if hasattr(structure, 'shape'):
        # 将shape元组转换为字符串（如 (3, 64, 64) → "3x64x64"）
        # 移除括号和多余空格，使输出更简洁
        return str(structure.shape).replace(', ', 'x').strip('(), ')
    
    # 处理未知类型（返回占位符）
    return '?'

def graph_summary(writer, fn, *args):
    """
    在TensorFlow计算图外执行摘要操作的包装器
    
    参数:
        writer: tf.summary.SummaryWriter 实例
        fn: 要执行的摘要函数（如video_summary）
        *args: 传递给fn的参数
    """
    # 获取当前全局步数（训练步数）
    step = tf.summary.experimental.get_step()
    
    def inner(*args):
        """内部函数，用于恢复步数上下文并执行摘要操作"""
        # 恢复原始步数（因为numpy_function会脱离计算图上下文）
        tf.summary.experimental.set_step(step)
        # 确保使用正确的writer
        with writer.as_default():
            fn(*args)  # 执行实际的摘要函数
    
    # 将操作包装为tf.numpy_function以兼容计算图模式
    # 允许在TensorFlow计算图中执行Python函数
    return tf.numpy_function(inner, args, [])

def video_summary(name, video, step=None, fps=20):
    """
    将视频数据记录为GIF或图像网格摘要
    
    参数:
        name: 摘要标签名（str或bytes）
        video: 视频张量（形状[B,T,H,W,C]）
        step: 记录步数
        fps: 输出GIF的帧率
    """
    # 处理名称（兼容bytes和str）
    name = name if isinstance(name, str) else name.decode('utf-8')
    
    # 归一化浮点视频数据到[0,255]
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    
    # 获取视频形状：批次、时间、高度、宽度、通道
    B, T, H, W, C = video.shape
    
    try:
        # 重组视频帧为GIF编码器需要的格式
        # 转置维度为[T,H,B,W,C]然后重塑为[T,H,B*W,C]
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        
        # 创建TensorFlow 1.x风格的摘要协议缓冲区
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        
        # 使用ffmpeg编码GIF
        image.encoded_image_string = encode_gif(frames, fps)
        
        # 添加摘要值
        summary.value.add(tag=name + '/gif', image=image)
        
        # 写入原始协议缓冲区
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    
    except (IOError, OSError) as e:
        # ffmpeg失败时的降级处理：改用图像网格
        print('GIF summaries require ffmpeg in $PATH.', e)
        # 转置为[B,H,T,W,C]并重塑为[1,B*H,T*W,C]
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        # 记录图像网格摘要
        tf.summary.image(name + '/grid', frames, step)

def encode_gif(frames, fps):
    """
    使用ffmpeg将帧序列编码为GIF
    
    参数:
        frames: 帧序列（形状[T,H,W,C]）
        fps: 输出帧率
    返回:
        bytes: 编码后的GIF数据
    """
    from subprocess import Popen, PIPE
    
    # 获取帧尺寸和通道数
    h, w, c = frames[0].shape
    
    # 确定像素格式（灰度或RGB）
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    
    # 构建ffmpeg命令（复杂滤镜链实现高质量GIF编码）
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',  # 输入设置
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        # 使用palettegen生成优化调色板
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])  # 输出设置
    
    # 启动ffmpeg子进程
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    
    # 逐帧写入输入
    for image in frames:
        proc.stdin.write(image.tostring())
    
    # 完成编码并获取输出
    out, err = proc.communicate()
    
    # 检查错误
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    
    # 清理进程
    del proc
    return out

def simulate(agent, envs, steps=0, episodes=0, state=None): 
    """
    并行环境模拟器（支持按步数或episode数停止）
    
    参数:
        agent: 智能体callable(obs, done, state) -> (action, new_state)
        envs: 并行环境列表（每个环境需实现reset()和step()）
        steps: 最大步数（与episodes二选一）
        episodes: 最大episode数（与steps二选一）
        state: 前次模拟的暂停状态（用于继续执行）
    
    返回:
        tuple: 更新后的模拟状态（可传递给下次simulate继续）
    """

    # === 1. 初始化/解包模拟状态 ===
    if state is None:
        # 初始状态（首次调用时）
        step = 0        # 当前已执行的总步数
        episode = 0     # 完成的episode数
        done = np.ones(len(envs), np.bool)  # 标记各环境是否需要重置
        length = np.zeros(len(envs), np.int32)  # 当前episode已执行步数
        obs = [None] * len(envs)  # 各环境的当前观测
        agent_state = None  # 智能体内部状态（如RNN隐藏状态）
    else:
        # 恢复之前的状态（继续执行）
        step, episode, done, length, obs, agent_state = state

    # === 2. 主模拟循环 ===
    while (steps and step < steps) or (episodes and episode < episodes):
        
        # 2.1 环境重置（如果有环境需要重置）
        if done.any():
            # 获取需要重置的环境索引
            indices = [index for index, d in enumerate(done) if d]
            
            # 异步重置环境（非阻塞）
            promises = [envs[i].reset(blocking=False) for i in indices]
            
            # 获取重置后的观测
            for index, promise in zip(indices, promises):
                obs[index] = promise()  # 阻塞获取结果

        # 2.2 智能体决策
        # 将各环境观测堆叠为batch（假设obs是字典观测）
        obs_batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}

        # 一个例子感受理解其作用：
        # # 模拟两个环境返回的观测数据
        # obs = [
        #     {"sensor": np.array([1, 2]), "position": np.array([0.1, 0.2])},  # 环境1的观测
        #     {"sensor": np.array([3, 4]), "position": np.array([0.3, 0.4])}   # 环境2的观测
        # ]

        # # 执行代码进行堆叠
        # obs_batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}

        # # 查看结果
        # print(obs_batch)
        # # 输出:
        # # {
        # #   "sensor": array([[1, 2], [3, 4]]),       # 形状为 (2, 2)
        # #   "position": array([[0.1, 0.2], [0.3, 0.4]])  # 形状为 (2, 2)
        # # }
        
        # 获取动作和新状态（自动处理done标记和状态重置）
        action, agent_state = agent(obs_batch, done, agent_state)
        action = np.array(action)  # 确保转为numpy数组
        assert len(action) == len(envs)  # 动作数必须匹配环境数

        # 2.3 环境执行
        # 异步执行动作（非阻塞）
        promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
        
        # 解包结果（只取obs, reward, done，忽略info）
        results = [p()[:3] for p in promises]  # 阻塞获取结果
        obs, _, done = zip(*results)
        obs = list(obs)  # 转为可变列表
        done = np.stack(done)  # 转为numpy数组

        # 2.4 更新统计量
        episode += int(done.sum())  # 累计完成的episode数
        length += 1  # 所有环境步数+1
        step += (done * length).sum()  # 只累计已结束episode的步数
        length *= (1 - done)  # 重置已结束episode的步数计数器

    # === 3. 返回新状态（允许恢复模拟） ===
    return (
        step - steps,    # 超出目标步数的部分
        episode - episodes,  # 超出目标episode的部分
        done,           # 各环境当前是否结束
        length,         # 各环境当前episode已执行步数
        obs,            # 各环境最新观测
        agent_state     # 智能体内部状态
    )

def count_episodes(directory):
    """
    统计目录中的episode数据文件数量和总步数
    
    参数:
        directory: 数据目录路径 (pathlib.Path)
    
    返回:
        tuple: (episode数量, 总步数)
    """
    # 获取所有.npz数据文件（每个文件存储一个episode）
    filenames = directory.glob('*.npz')  
    # 示例文件名：'episode-1000.npz'（假设1000是最后一步的索引）
    
    # 从文件名解析每个episode的步数：
    lengths = [
        int(n.stem.rsplit('-', 1)[-1]) - 1  # 提取步数并减1（因为从0开始计数）
        for n in filenames
    ]
    # 例如：'episode-1000.npz' → 1000表示包含step0到step999 → 实际步数=1000
    
    # 文件名解析逻辑
    # directory.glob('*.npz')：获取目录中所有 .npz 文件的路径对象
    # n.stem：获取文件名（不含扩展名），例如 episode-1000
    # rsplit('-', 1)：从右侧分割字符串，最多分割一次，得到 ['episode', '1000']
    # [-1]：取分割后的最后一部分，即 '1000'
    # int(...)：转换为整数 1000
    # - 1：减去 1，因为索引从 0 开始，所以总长度为 1000 - 0 + 1 = 1001 步

    # 实际的例子:
    # 20250627T021613-0bf35765291443b5a5c7c738f1c87a53-501.npz
    # n.stem：20250627T021613-0bf35765291443b5a5c7c738f1c87a53-501
    # rsplit('-', 1)：['20250627T021613-0bf35765291443b5a5c7c738f1c87a53', '501']
    # [-1]：'501'
    # int(...)：501
    # - 1：500

    # 计算总episode数和总步数
    episodes = len(lengths)  # 文件数量=episode数量
    steps = sum(lengths)     # 所有episode的步数总和
    return episodes, steps

def save_episodes(directory, episodes):
    """
    将多个episode数据保存为压缩NPZ文件
    
    参数:
        directory: 保存路径（字符串或Path对象）
        episodes: 包含多个episode数据的列表，每个episode是一个字典
            典型键: 'observation', 'action', 'reward', 'discount' 等
    """
    # 将目录路径转换为Path对象并扩展用户路径（如~替换为主目录）
    directory = pathlib.Path(directory).expanduser()
    
    # 创建目录（递归创建父目录，存在时不报错）
    directory.mkdir(parents=True, exist_ok=True)
    
    # 生成当前时间戳（格式：YYYYMMDDTHHMMSS）用于文件名
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # 遍历每个episode数据
    for episode in episodes:
        # 生成唯一标识符（32位十六进制字符串）
        identifier = str(uuid.uuid4().hex)
        
        # 获取episode长度（通过奖励数组的长度确定）
        length = len(episode['reward'])
        
        # 构建文件名：时间戳-唯一标识-长度.npz
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        
        # 使用内存缓冲区优化写入性能
        with io.BytesIO() as f1:
            # 将episode字典压缩保存到内存缓冲区
            # **episode 解包字典，每个键值对成为npz文件中的一个数组
            np.savez_compressed(f1, **episode)
            # **episode：将字典 episode 解包为关键字参数，等价于：
            # np.savez_compressed('episode_data.npz', obs=episode['obs'], action=episode['action'], reward=episode['reward'])
            
            # 将文件指针重置到缓冲区开始位置
            f1.seek(0)
            
            # 将内存缓冲区中的内容写入磁盘文件
            with filename.open('wb') as f2:
                f2.write(f1.read())

def load_episodes(directory, rescan, length=None, balance=False, seed=0):
    """
    从指定目录加载episode数据并生成训练样本
    
    参数:
        directory: 存储episode文件的目录路径
        rescan: 每次迭代重新扫描的文件数量
        length: 每个样本的序列长度(可选，默认使用完整episode)
        balance: 是否平衡采样(优先选择较短的episode)
        seed: 随机数生成器的种子，确保结果可复现
        
    返回:
        生成器，每次生成一个处理后的episode样本
    """
    # 将目录路径转换为Path对象并扩展用户路径(如~替换为用户主目录)
    directory = pathlib.Path(directory).expanduser()
    
    # 创建独立的随机数生成器，使用指定种子确保可复现性
    random = np.random.RandomState(seed)
    
    # 缓存已加载的episode，避免重复读取磁盘
    cache = {}
    
    # 无限循环：持续从缓存中采样episode
    while True:
        # 扫描目录中的所有NPZ文件
        for filename in directory.glob('*.npz'):
            # 如果文件未被缓存，则加载并缓存
            if filename not in cache:
                try:
                    # 以二进制模式打开NPZ文件
                    with filename.open('rb') as f:
                        # 加载NPZ文件
                        episode = np.load(f)
                        # 将numpy的NpzFile对象转换为普通字典
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    # 处理加载失败的情况
                    print(f'Could not load episode: {e}')
                    continue
                # 将成功加载的episode存入缓存
                cache[filename] = episode
        
        # 获取缓存中所有episode的键(文件名)
        keys = list(cache.keys())
        
        # 随机选择rescan个episode进行处理
        for index in random.choice(len(keys), rescan):
            # 从缓存中获取选中的episode
            episode = cache[keys[index]]
            
            # 如果指定了长度，需要从episode中截取片段
            if length:
                # 获取episode的总长度(通过第一个键的值确定)
                total = len(next(iter(episode.values())))
                # 计算可截取的有效长度
                available = total - length
                
                # 如果episode太短，无法截取指定长度，跳过
                if available < 1:
                    print(f'Skipped short episode of length {available}.')
                    continue
                
                # 平衡采样策略：优先从较短的episode中选择较早的片段
                if balance:
                    # 限制最大起始位置，避免截取到episode末尾
                    index = min(random.randint(0, total), available)
                else:
                    # 普通采样：随机选择起始位置
                    index = int(random.randint(0, available))
                
                # 从episode中截取指定长度的片段
                episode = {k: v[index: index + length] for k, v in episode.items()}
            
            # 生成处理后的episode样本
            yield episode

class DummyEnv:

  def __init__(self):
    self._random = np.random.RandomState(seed=0)
    self._step = None

  @property
  def observation_space(self):
    low = np.zeros([64, 64, 3], dtype=np.uint8)
    high = 255 * np.ones([64, 64, 3], dtype=np.uint8)
    spaces = {'image': gym.spaces.Box(low, high)}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    low = -np.ones([5], dtype=np.float32)
    high = np.ones([5], dtype=np.float32)
    return gym.spaces.Box(low, high)

  def reset(self):
    self._step = 0
    obs = self.observation_space.sample()
    return obs

  def step(self, action):
    obs = self.observation_space.sample()
    reward = self._random.uniform(0, 1)
    self._step += 1
    done = self._step >= 1000
    info = {}
    return obs, reward, done, info

class SampleDist:
  """通过采样估计概率分布统计量的包装器。
  
  当直接计算分布的统计量（如众数）很困难时，
  可以使用采样方法进行近似估计。
  """

  def __init__(self, dist, samples=100):
    """初始化 SampleDist。
    
    参数:
      dist: 基础概率分布（如 Normal、Categorical 等）
      samples: 采样数量（用于估计统计量，默认 100）
    """
    self._dist = dist          # 保存基础分布
    self._samples = samples    # 保存采样数量

  @property
  def name(self):
    """返回分布名称。"""
    return 'SampleDist'

  def __getattr__(self, name):
    """代理未实现的属性和方法到基础分布。
    调用 sample_dist.sample() 会调用底层分布的 sample() 方法
    例如，调用 SampleDist.prob() 会转发到 self._dist.prob()
    """
    return getattr(self._dist, name)

  def mean(self):
    """通过采样估计分布的均值。
    
    返回:
      均值的估计值 [batch_size, action_dim]
    """
    # 从分布中采样
    samples = self._dist.sample(self._samples)  # 形状: [samples, batch_size, action_dim]
    
    # 沿采样维度计算均值
    return tf.reduce_mean(samples, 0)  # 形状: [batch_size, action_dim]

  def mode(self):
    """通过采样估计分布的众数（最可能的值）。
    
    返回:
      众数的估计值 [batch_size, action_dim]
    """
    # 从分布中采样
    sample = self._dist.sample(self._samples)  # 形状: [samples, batch_size, action_dim]
    
    # 计算每个样本的对数概率
    logprob = self._dist.log_prob(sample)  # 形状: [samples, batch_size]
    
    # 对每个批次，找到对数概率最大的样本索引
    max_indices = tf.argmax(logprob, axis=0)  # 形状: [batch_size]
    
    # 从采样结果中选择对数概率最大的样本
    return tf.gather(sample, max_indices, axis=0)[0]  # 形状: [batch_size, action_dim]
  
    # tf.gather(axis=0)​​：根据 max_indices 从 sample 中选取对应样本。
    # 输入 sample 形状为 [samples, batch_size, action_dim]，max_indices 为 [batch_size]。
    # 输出形状为 [1, batch_size, action_dim]，通过 [0] 去除冗余维度，最终返回 [batch_size, action_dim]。
    # ​​数学意义​​：返回的样本是当前采样中概率密度最大的值，作为众数的近似估计。

  def entropy(self):
    """通过采样估计分布的熵。
    
    熵的计算公式: H(p) = -E[log(p)]
    
    返回:
      熵的估计值 [batch_size]
    """
    # 从分布中采样
    sample = self._dist.sample(self._samples)  # 形状: [samples, batch_size, action_dim]
    
    # 计算每个样本的对数概率
    logprob = self.log_prob(sample)  # 形状: [samples, batch_size]
    # 注记：
    # 为什么 logprob 没有 action_dim 维度？
    # 对于连续分布，log_prob 通常返回每个样本点的联合对数概率
    # 例如，对于 7 维动作，log_prob 返回整个 7 维向量的对数概率，而非每个维度的对数概率之和

    # 沿采样维度计算负对数概率的均值
    return -tf.reduce_mean(logprob, axis=0)  # 形状: [batch_size]

class OneHotDist:
  """将类别分布包装为独热编码形式的分布。
  
  用于处理离散动作空间，将类别索引转换为独热向量。
  """

  def __init__(self, logits=None, probs=None):
    """初始化 OneHotDist。
    
    参数:
      logits: 未归一化的对数概率 [batch_size, num_classes]
      probs: 归一化的概率分布 [batch_size, num_classes]
    """
    # 创建基础类别分布
    self._dist = tfd.Categorical(logits=logits, probs=probs)
    # 如果 logits 和 probs 都为 None 会怎样？
    # 会报错，因为无法推断类别数量
    # 实际使用时，必须至少提供其中一个
    
    # 确定类别数量（动作空间维度）
    self._num_classes = self.mean().shape[-1]
    # self.mean() 返回概率分布 [batch_size, num_classes]
    # .shape[-1] 取最后一维的大小，即 num_classes
    
    # 获取计算数据类型
    self._dtype = prec.global_policy().compute_dtype
    # prec.global_policy()：获取全局的混合精度策略。
    # .compute_dtype：查看计算时使用的数据类型（float16 或 float32）

  @property
  def name(self):
    """返回分布名称。"""
    return 'OneHotDist'

  def __getattr__(self, name):
    """代理未实现的属性和方法到基础分布。"""
    return getattr(self._dist, name)

  def prob(self, events):
    """计算事件的概率。
    
    参数:
      events: 独热编码的事件 [batch_size, num_classes]
      
    返回:
      对应类别的概率 [batch_size]
    """
    # 将独热向量转换为类别索引
    # 沿最后一维（axis=-1）查找最大值的索引
    indices = tf.argmax(events, axis=-1) # 形状为 [batch_size]
    # 调用基础分布的 prob 方法
    return self._dist.prob(indices)

  def log_prob(self, events):
    """计算事件的对数概率。
    
    参数:
      events: 独热编码的事件 [batch_size, num_classes]
      
    返回:
      对应类别的对数概率 [batch_size]
    """
    # 将独热向量转换为类别索引
    indices = tf.argmax(events, axis=-1)
    # 调用基础分布的 log_prob 方法
    return self._dist.log_prob(indices)

  def mean(self):
    """返回分布的均值（概率分布）。
    
    返回:
      概率分布 [batch_size, num_classes]
    """
    # 直接返回基础分布的概率参数
    return self._dist.probs_parameter()
    # 若初始化时传入 probs，则直接返回该 probs
    # 若传入 logits，则返回 softmax(logits)（归一化后的概率）

  def mode(self):
    """返回分布的众数（最可能的类别，独热编码形式）。
    
    返回:
      独热编码的众数 [batch_size, num_classes]
    """
    # 获取最可能的类别索引
    indices = self._dist.mode()
    # 转换为独热编码
    return self._one_hot(indices)

  def sample(self, amount=None):
    """从分布中采样，并应用重参数化技巧。
    
    参数:
      amount: 采样数量（默认为 None，表示 1）
      
    返回:
      独热编码的样本 [amount, batch_size, num_classes]
    """
    # 处理采样数量参数
    # 如果提供了 amount，则转为列表 [amount]
    # 否则默认为空列表 []，表示采样 1 次
    amount = [amount] if amount else []
    
    # 从基础分布采样类别索引
    # indices 形状: [amount, batch_size]
    indices = self._dist.sample(*amount)
    
    # 将索引转换为独热编码
    # sample 形状: [amount, batch_size, num_classes]
    sample = self._one_hot(indices)
    
    # 获取概率分布
    # probs 形状: [batch_size, num_classes]
    probs = self._dist.probs_parameter()
    
    # 应用重参数化技巧（Straight-Through Estimator）
    # 这允许梯度通过离散采样操作
    # 关键步骤：sample += probs - stop_gradient(probs)
    sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)

    # 应用重参数化技巧（Straight-Through Estimator）
    # 这允许梯度通过离散采样操作
    # 数学原理:
    # y_STE = y_discrete + (p_θ - stop_gradient(p_θ))
    # 其中:
    #   y_discrete 是离散采样结果（如独热向量）
    #   p_θ 是分布的概率参数
    #   stop_gradient(p_θ) 表示在反向传播时将 p_θ 视为常量
    # 
    # 前向传播:
    #   y_STE = y_discrete + (p_θ - p_θ) = y_discrete
    #   输出与直接离散采样相同
    # 
    # 反向传播:
    #   ∂y_STE/∂p_θ = ∂(p_θ - stop_gradient(p_θ))/∂p_θ = I
    #   ∂y_STE/∂y_discrete = 0 (被 stop_gradient 阻断)
    #   梯度仅通过 p_θ 流动，绕过了离散采样操作
    # 
    # 梯度更新:
    # 损失函数 L 的梯度计算:
    # ∇_θ L = (∂L/∂y_STE) · (∂y_STE/∂p_θ) · (∂p_θ/∂θ)
    #       = (∂L/∂y_STE) · I · (∂p_θ/∂θ)
    #       = (∂L/∂y_STE) · (∂p_θ/∂θ)
    # 这表明梯度直接通过 p_θ 传播到参数 θ，绕过了离散采样操作
    # 
    # 直观解释:
    # 前向: y_STE 与离散采样 y 完全相同
    # 反向: 梯度"假装" y 是由连续值 p_θ 直接生成的，忽略离散采样步骤
    # 这种技巧使得我们可以像优化连续动作一样优化离散动作

    return sample

  def _one_hot(self, indices):
    """将类别索引转换为独热编码。
    
    参数:
      indices: 类别索引 [batch_size]
      
    返回:
      独热编码 [batch_size, num_classes]
    """
    return tf.one_hot(indices, self._num_classes, dtype=self._dtype)

class TanhBijector(tfp.bijectors.Bijector):
  """实现 tanh 变换的双射器，将实数域映射到 [-1, 1] 区间。"""

  def __init__(self, validate_args=False, name='tanh'):
    """初始化 TanhBijector。
    
    参数:
      validate_args: 是否验证输入参数（默认否，提高效率）
      name: 双射器名称
    """
    # 调用父类构造函数
    # forward_min_event_ndims=0 表示该变换可应用于标量
    super().__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    """前向变换：将实数 x 映射到 [-1, 1]。
    
    参数:
      x: 输入张量（实数域）
      
    返回:
      y: 输出张量（[-1, 1] 区间）
    """
    return tf.nn.tanh(x)  # 直接应用 tanh 函数

  def _inverse(self, y):
    """反向变换：将 [-1, 1] 区间的 y 映射回实数域。
    
    参数:
      y: 输入张量（[-1, 1] 区间）
      
    返回:
      x: 输出张量（实数域）
    """
    dtype = y.dtype  # 保存原始数据类型
    
    # 将 y 转换为 float32 进行数值稳定性处理
    y = tf.cast(y, tf.float32)
    
    # 处理可能超出 [-1, 1] 的值：
    # 1. 检查是否 |y| ≤ 1
    # 2. 对超出范围的值进行裁剪（避免 NaN）
    y = tf.where(
        tf.less_equal(tf.abs(y), 1.),
        tf.clip_by_value(y, -0.99999997, 0.99999997),  # 裁剪到接近边界但避免 ±1
        y)  # 超出范围的值保持不变（但实际不应出现）
    
    # 应用反双曲正切函数
    y = tf.atanh(y)
    
    # 恢复原始数据类型
    y = tf.cast(y, dtype)
    
    return y

  def _forward_log_det_jacobian(self, x):
    """计算前向变换的对数雅可比行列式。
    
    这用于在变换概率分布时保持概率密度的正确性。
    当通过双射变换 y = f(x) 将随机变量 X 转换为 Y 时，
    Y 的概率密度函数满足：
    log p_Y(y) = log p_X(f^{-1}(y)) + log |dx/dy|
    其中 |dx/dy| 是变换的雅可比行列式的绝对值。
    
    参数:
      x: 输入张量
      
    返回:
      log_det: 对数雅可比行列式
      
    推导如下：
    1 - tanh²(x) = sech²(x) = 4 / (eˣ + e⁻ˣ)²
    log(1 - tanh²(x)) = log(4) - 2·log(eˣ + e⁻ˣ)
                     = 2·log(2) - 2·[x + log(1 + e⁻²ˣ)]
                     = 2·[log(2) - x - log(1 + e⁻²ˣ)]
    """
    # 预计算 log(2)，使用与输入相同的数据类型
    log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
    
    # 使用优化后的公式计算对数雅可比行列式
    # 数学推导：
    # d(tanh(x))/dx = 1 - tanh(x)^2
    # log|d(tanh(x))/dx| = log(1 - tanh(x)^2)
    # 通过代数变换得到等价但数值更稳定的形式：
    # 2 * (log(2) - x - softplus(-2x))
    return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))

def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  """
  计算λ-return（TD(λ)回报），用于平衡蒙特卡洛回报和时间差分回报。
  
  参数:
    reward: 奖励序列，形状为 [..., T, ...]（axis指定的维度为时间维度）
    value: 价值估计序列，形状为 [..., T, ...]
    pcont: 持续概率序列（或标量），形状为 [..., T, ...] 或标量
    bootstrap: 引导值（用于最后一步的价值估计），形状为 [...]
    lambda_: λ参数，控制回报计算方式（0=1步TD，1=蒙特卡洛）
    axis: 时间维度的位置
    
  返回:
    returns: λ-return序列，形状与reward相同
  """
  # 确保reward和value具有相同的维度数
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  
  # 如果pcont是标量，则扩展为与reward相同形状的张量
  # 输入: pcont(标量)，reward [..., T, ...]
  # 输出: pcont [..., T, ...]
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  
  # 创建维度置换列表，将时间维度(axis)移到第0位
  # 例如: axis=1, dims=[1, 0, 2, 3] (对于4D张量)
  dims = list(range(reward.shape.ndims))
  # .ndims​​ (或 ndim): 直接返回张量的维度数（整数），等价于 len(reward.shape)。例如，三维张量的 ndim 为 3
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  
  # 如果时间维度不在第0位，置换张量维度，将时间维度移到第0位
  # 输入: reward [..., T, ...]
  # 输出: reward [T, ...]
  if axis != 0:
    reward = tf.transpose(reward, dims)
    value = tf.transpose(value, dims)
    pcont = tf.transpose(pcont, dims)
  
  # 如果没有提供引导值，初始化为零
  # 输入: value [T, ...]
  # 输出: bootstrap [...]
  if bootstrap is None:
    bootstrap = tf.zeros_like(value[-1])
  
  # 构建下一个时间步的价值序列，最后一步使用引导值
  # 输入: value [T, ...], bootstrap [...]
  # 输出: next_values [T, ...]
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  
  # 计算λ-return的输入项: r_t + γ_t * V_{t+1} * (1-λ)
  # 输入: reward [T, ...], pcont [T, ...], next_values [T, ...]
  # 输出: inputs [T, ...]
  inputs = reward + pcont * next_values * (1 - lambda_)
  
  # 原理公式：G_{t}^λ = R_{t+1} + γ * (1−λ) * V(S_{t+1}) + λ * γ * G_{t+1}^λ
    
  # 反向递归计算λ-return
  # 输入: inputs [T, ...], pcont [T, ...], bootstrap [...]
  # 输出: returns [T, ...]
  returns = static_scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, reverse=True)
  # cur[0] -> inputs：R_{t+1} + γ * (1−λ) * V(S_{t+1}) 
  # cur[1] -> pcont： γ
  # agg -> bootstrap：G_{t+1}^λ
  # cur[1] * lambda_ * agg：γ * λ * G_{t+1}^λ

  # 如果时间维度不在第0位，将维度置换回原始顺序
  # 输入: returns [T, ...]
  # 输出: returns [..., T, ...]
  if axis != 0:
    returns = tf.transpose(returns, dims)
  
  return returns

class Adam(tf.Module):
    """
    自定义Adam优化器包装器，支持梯度裁剪、权重衰减和混合精度训练
    """
    def __init__(self, name, modules, lr, clip=None, wd=None, wdpattern=r'.*'):
        """
        初始化优化器
        
        参数:
            name: 优化器名称（用于日志和权重衰减匹配）
            modules: 需要优化的TensorFlow模块列表
            lr: 学习率
            clip: 梯度裁剪阈值（可选）
            wd: 权重衰减系数（可选）
            wdpattern: 权重衰减应用的正则表达式模式（默认匹配所有参数）
        """
        self._name = name
        self._modules = modules
        self._clip = clip
        self._wd = wd
        self._wdpattern = wdpattern
        # 创建Adam优化器实例
        self._opt = tf.optimizers.Adam(lr)
        # 应用动态损失缩放（用于混合精度训练，提高数值稳定性）
        self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic')
        # 延迟初始化需要优化的变量
        self._variables = None

    @property
    def variables(self):
        """返回优化器的变量（如动量项）"""
        return self._opt.variables()

    def __call__(self, tape, loss):
        """
        执行优化步骤
        
        参数:
            tape: 用于计算梯度的GradientTape
            loss: 需要最小化的损失值
        """
        # 首次调用时初始化变量列表
        if self._variables is None:
            # 从所有模块中收集可训练变量
            variables = [module.variables for module in self._modules]
            # 展平嵌套结构为一维列表
            self._variables = tf.nest.flatten(variables)
            # 计算总参数数量
            count = sum(np.prod(x.shape) for x in self._variables)
            print(f'Found {count} {self._name} parameters.')
        
        # 确保损失是标量
        assert len(loss.shape) == 0, loss.shape
        
        # 应用损失缩放（用于混合精度训练）
        with tape:
            loss = self._opt.get_scaled_loss(loss)
        
        # 计算缩放后的梯度
        grads = tape.gradient(loss, self._variables)
        # 恢复梯度缩放（取消之前的损失缩放）
        grads = self._opt.get_unscaled_gradients(grads)
        
        # 计算梯度范数（用于监控和裁剪）
        norm = tf.linalg.global_norm(grads)
        
        # 梯度裁剪（防止梯度爆炸）
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
        
        # 权重衰减（正则化）
        if self._wd:
            # 获取当前分布式训练上下文
            context = tf.distribute.get_replica_context()
            # 在所有设备上同步应用权重衰减
            context.merge_call(self._apply_weight_decay)
        
        # 应用梯度更新模型参数
        self._opt.apply_gradients(zip(grads, self._variables))
        
        # 返回梯度范数（用于监控训练稳定性）
        return norm

    def _apply_weight_decay(self, strategy):
        """
        应用权重衰减到匹配正则表达式模式的变量
        
        参数:
            strategy: 当前的分布式训练策略
        """
        print('Applied weight decay to variables:')
        for var in self._variables:
            # 检查变量名称是否匹配权重衰减模式
            if re.search(self._wdpattern, self._name + '/' + var.name):
                print('- ' + self._name + '/' + var.name)
                # 对匹配的变量应用权重衰减（等价于L2正则化）
                strategy.extended.update(var, lambda var: self._wd * var)

# 自定义参数类型转换函数，根据默认值类型动态处理
def args_type(default):
    """
    根据默认值类型动态生成命令行参数转换器
    
    支持的特殊类型转换：
    - 布尔值：将字符串'True'/'False'转换为布尔类型
    - 整数：支持科学计数法字符串（如'1e3'）自动转换为浮点数
    - 路径：自动扩展用户路径（如'~'转换为主目录）
    
    参数:
        default: 参数的默认值，用于推断类型和转换逻辑
    
    返回:
        callable: 类型转换函数，将命令行输入字符串转换为正确类型
    """
    # 处理布尔类型参数
    if isinstance(default, bool):
        # 将字符串转换为布尔值
        # 允许的值为'True'或'False'（大小写敏感）
        # 例如：--use_gpu True → True
        return lambda x: bool(['False', 'True'].index(x))
    
    # 处理整数类型参数
    if isinstance(default, int):
        # 智能转换数值：
        # - 包含'e'或小数点的字符串转换为浮点数（如'1e3' → 1000.0）
        # - 其他情况转换为整数（如'42' → 42）
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    
    # 处理路径类型参数
    if isinstance(default, pathlib.Path):
        # 将字符串转换为Path对象，并扩展用户路径
        # 例如：'~/data' → '/home/user/data'
        return lambda x: pathlib.Path(x).expanduser()
    
    # 默认情况：直接使用默认值的类型
    return type(default)

def static_scan(fn, inputs, start, reverse=False):
  """
  静态扫描函数，按顺序处理序列输入并累积输出。
  
  参数:
    fn: 处理函数，接受两个参数 (prev_output, current_input)，返回新的输出
    inputs: 输入序列，可以是张量或嵌套结构 (如列表、字典)
    start: 初始状态，可以是张量或嵌套结构
    reverse: 是否反向处理序列 (默认为 False)
    
  返回:
    处理后的输出序列，结构与 start 相同
  """
  # 初始化上一个输出为起始状态
  last = start
  
  # 创建空的输出列表，每个元素对应 start 中的一个扁平化组件
  # tf.nest.flatten(start) 将嵌套结构展开为一维列表
  outputs = [[] for _ in tf.nest.flatten(start)]
  
  # 确定序列长度（假设所有输入的第一维长度相同）
  # tf.nest.flatten(inputs)[0] 获取第一个输入张量
  indices = range(len(tf.nest.flatten(inputs)[0]))
  
  # 如果需要反向处理，反转索引顺序
  if reverse:
    indices = reversed(indices)
  
  # 按顺序处理每个时间步
  for index in indices:
    # 从输入序列中获取当前时间步的所有输入
    # tf.nest.map_structure 对嵌套结构中的每个元素应用相同操作
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    
    # 调用处理函数，传入上一个输出和当前输入
    # 得到当前时间步的输出
    last = fn(last, inp)
    
    # 将当前输出的每个扁平化组件添加到对应的输出列表中
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  
  # 如果是反向处理，需要将输出列表反转回正确顺序
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  
  # 将每个输出列表堆叠成张量（沿时间维度）
  outputs = [tf.stack(x, 0) for x in outputs]
  
  # 将扁平化的输出恢复为与 start 相同的嵌套结构
  return tf.nest.pack_sequence_as(start, outputs)

def _mnd_sample(self, sample_shape=(), seed=None, name='sample'):
    """
    自定义多元正态分布采样函数
    
    参数:
        sample_shape: 额外的采样维度，如(5,)表示采样5次
        seed: 随机数种子
        name: 操作名称
        
    返回:
        samples: 从分布中采样的张量，形状为sample_shape + event_shape
    """
    # 计算最终采样形状：sample_shape拼接上事件形状
    # 例如: sample_shape=(5,), event_shape=(3,) → 最终形状(5,3)
    shape = tuple(sample_shape) + tuple(self.event_shape)
    
    # 使用tf.random.normal生成符合正态分布的随机样本
    # 参数分别为：形状、均值、标准差、数据类型、随机种子、操作名称
    return tf.random.normal(
        shape,
        self.mean(),    # 分布的均值
        self.stddev(),  # 分布的标准差
        self.dtype,     # 数据类型
        seed,           # 随机种子
        name            # 操作名称
    )

# 猴子补丁：替换tfd.MultivariateNormalDiag的默认采样方法
tfd.MultivariateNormalDiag.sample = _mnd_sample

def _cat_sample(self, sample_shape=(), seed=None, name='sample'):
    """
    自定义分类分布采样函数
    
    参数:
        sample_shape: 额外的采样维度，支持0维或1维
        seed: 随机数种子
        name: 操作名称
        
    返回:
        indices: 采样得到的类别索引
    """
    # 检查sample_shape是否为0维或1维
    assert len(sample_shape) in (0, 1), sample_shape
    
    # 检查logits的形状是否为二维(批次大小, 类别数)
    assert len(self.logits_parameter().shape) == 2
    
    # 计算采样数量：如果sample_shape为空则采样1次，否则取sample_shape的第一个元素
    num_samples = sample_shape[0] if sample_shape else 1
    
    # 使用tf.random.categorical从分类分布中采样
    # logits参数形状为(批次大小, 类别数)
    # 返回形状为(批次大小, num_samples)的索引张量
    indices = tf.random.categorical(
        self.logits_parameter(),  # 类别对数概率
        num_samples,              # 每个分布采样的数量
        self.dtype,               # 数据类型
        seed,                     # 随机种子
        name                      # 操作名称
    )
    
    # 如果sample_shape为空，则移除最后一维
    # 例如: (batch_size, 1) → (batch_size,)
    if not sample_shape:
        indices = indices[..., 0]
    # 索引操作：indices[..., 0] 相当于 indices[:, 0]（对于二维张量）
    # ... 表示保留前面所有维度
    # 0 表示取最后一维的第 0 个元素
    
    return indices

# 猴子补丁：替换tfd.Categorical的默认采样方法
tfd.Categorical.sample = _cat_sample

class Every:
    """
    周期性触发器：用于在特定间隔步骤执行操作
    """
    
    def __init__(self, every):
        """
        初始化触发器
        
        参数:
            every: 触发周期（步数间隔）
        """
        self._every = every  # 存储触发周期
        self._last = None    # 记录上一次触发的步数（初始为None）

    def __call__(self, step):
        """
        检查当前步骤是否应触发操作
        
        参数:
            step: 当前全局步数
            
        返回:
            bool: 是否应触发操作
        """
        # 首次调用时，初始化_last为当前步数并触发操作
        if self._last is None:
            self._last = step
            return True
            
        # 当步数达到或超过上次触发步数加上周期时，触发操作
        if step >= self._last + self._every:
            # 更新_last为下一个触发点（而非当前step，避免漂移）
            self._last += self._every
            return True
            
        # 未达到触发条件
        return False

class Once:
    """单次触发器（确保某操作只执行一次）"""
    
    def __init__(self):
        # 初始化标志位为True（表示可执行）
        self._once = True
    
    def __call__(self):
        """调用时检查是否需要执行"""
        if self._once:          # 如果为True（首次调用）
            self._once = False  # 立即置为False
            return True         # 返回需要执行的信号
        return False           # 后续调用返回False
