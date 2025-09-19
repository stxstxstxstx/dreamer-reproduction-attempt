# λ-return 递归等式的严格推导

## 1. 定义回顾
λ-return 的原始定义为：

$$
G_t^\lambda = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
$$

其中，n 步回报 $G_t^{(n)}$ 为：

$$
G_t^{(n)} = \sum_{k=1}^n \gamma^{k-1} R_{t+k} + \gamma^n V(S_{t+n})
$$

## 2. 递归推导目标
我们希望证明：

$$
G_t^\lambda = R_{t+1} + \gamma \left( (1-\lambda)V(S_{t+1}) + \lambda G_{t+1}^\lambda \right)
$$

## 3. 严格推导步骤

### 步骤 1：展开 λ-return 的求和
将 $G_t^\lambda$ 的求和拆分为第一项（n=1）和剩余项（n≥2）：

$$
G_t^\lambda = (1-\lambda)G_t^{(1)} + (1-\lambda) \sum_{n=2}^\infty \lambda^{n-1} G_t^{(n)}
$$

### 步骤 2：分离 n=1 项
根据 n 步回报的定义：

$$
G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})
$$

因此：

$$
G_t^\lambda = (1-\lambda) \left( R_{t+1} + \gamma V(S_{t+1}) \right) + (1-\lambda) \sum_{n=2}^\infty \lambda^{n-1} G_t^{(n)}
$$

### 步骤 3：重新索引剩余项
将剩余项的求和索引从 n=2 开始改为 k=1（令 k = n - 1）：

$$
\sum_{n=2}^\infty \lambda^{n-1} G_t^{(n)} = \lambda \sum_{k=1}^\infty \lambda^{k-1} G_t^{(k+1)}
$$

### 步骤 4：展开 $G_t^{(k+1)}$
利用 n 步回报的定义：

$$
G_t^{(k+1)} = \sum_{m=1}^{k+1} \gamma^{m-1} R_{t+m} + \gamma^{k+1} V(S_{t+k+1})
$$

可以拆分为：

$$
G_t^{(k+1)} = R_{t+1} + \gamma \sum_{m=1}^{k} \gamma^{m-1} R_{t+1+m} + \gamma^{k+1} V(S_{t+k+1}) = R_{t+1} + \gamma G_{t+1}^{(k)}
$$

### 步骤 5：代入剩余项
将 $G_t^{(k+1)} = R_{t+1} + \gamma G_{t+1}^{(k)}$ 代入步骤 3 的剩余项：

$$
\lambda \sum_{k=1}^\infty \lambda^{k-1} \left( R_{t+1} + \gamma G_{t+1}^{(k)} \right) = \lambda R_{t+1} \sum_{k=1}^\infty \lambda^{k-1} + \lambda \gamma \sum_{k=1}^\infty \lambda^{k-1} G_{t+1}^{(k)}
$$

注意到：

$$
\sum_{k=1}^\infty \lambda^{k-1} = \frac{1}{1-\lambda} \quad (\text{几何级数求和})
$$

因此：

$$
\lambda R_{t+1} \cdot \frac{1}{1-\lambda} + \lambda \gamma \sum_{k=1}^\infty \lambda^{k-1} G_{t+1}^{(k)} = \frac{\lambda R_{t+1}}{1-\lambda} + \lambda \gamma G_{t+1}^\lambda / (1-\lambda)
$$

### 步骤 6：合并所有项
将步骤 2 和步骤 5 的结果合并：

$$
G_t^\lambda = (1-\lambda) \left( R_{t+1} + \gamma V(S_{t+1}) \right) + \frac{\lambda (1-\lambda) R_{t+1}}{1-\lambda} + \lambda \gamma G_{t+1}^\lambda
$$

化简：

$$
G_t^\lambda = R_{t+1} + \gamma (1-\lambda) V(S_{t+1}) + \lambda \gamma G_{t+1}^\lambda
$$

### 步骤 7：整理为递归形式
最终得到：

$$
G_t^\lambda = R_{t+1} + \gamma \left( (1-\lambda) V(S_{t+1}) + \lambda G_{t+1}^\lambda \right)
$$

## 4. 直观解释
- **第一项 $R_{t+1}$**：即时奖励。
- **第二项 $\gamma (1-\lambda) V(S_{t+1})$**：下一状态的估值，权重为 $1-\lambda$。
- **第三项 $\gamma \lambda G_{t+1}^\lambda$**：后续 λ-return 的递归，权重为 λ。

## 5. 边界条件
若 $t$ 是终止时间步（$S_t$ 为终止状态）：

$$
G_t^\lambda = 0
$$

对于无限长回合，需保证 $\gamma \lambda < 1$ 使级数收敛。

## 6. 总结
通过将 λ-return 拆分为首项和剩余项的加权和，严格推导出其递归形式。该递归式：
- **计算高效**：避免显式求和，适合在线学习。
- **理论完备**：涵盖 TD(λ) 和 MC 作为特例。
- **灵活可控**：通过 λ 调节偏差与方差的权衡。

此推导为 TD(λ)、GAE 等算法奠定了理论基础。