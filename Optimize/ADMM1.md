

copy from : http://joegaotao.github.io/cn/2014/02/admm



### 1.1 Dual Ascent

对于凸函数的优化问题，对偶上升法核心思想就是引入一个对偶变量，然后利用交替优化的思路，使得两者同时达到optimal。一个凸函数的对偶函数其实就是原凸函数的一个下界，因此可以证明一个较好的性质：在强对偶性假设下，即最小化原凸函数（primal）等价于最大化对偶函数（dual），两者会同时达到optimal。这种转化可以将原来很多的参数约束条件变得少了很多，以利于做优化。具体表述如下：

![image-20181219152123677](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152123677-5204083.png)

当g不可微的时候也可以将其转化下，成为一个所谓的subgradient的方法，虽然看起来不错，简单证明下即可知道xk和yk同时可达到optimal，但是上述条件要求很苛刻：==f(x)要求严格凸，并且要求α选择有比较合适。==一般应用中都不会满足（比如f(x))是一个非零的**仿射函数**），因此dual ascent不会直接应用。

### 1.2 Dual Decomposition

虽然dual ascent方法有缺陷，要求有些严格，但是他有一个非常好的性质，当目标函数f是可分的（**separable**）时候（参数抑或feature可分），整个问题可以拆解成多个子参数问题，分块优化后汇集起来整体更新。这样非常有利于并行化处理。形式化阐述如下：

![image-20181219152147089](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152147089-5204107.png)

对偶分解是非常经典的优化方法，可追溯到1960年代。但是这种想法对后面的分布式优化方法影响较大，比如近期的graph-structure优化问题。

### 1.3 Augmented Lagrangians and the Method of Multipliers

从上面可以看到dual ascent方法对于目标函数要求比较苛刻，为了放松假设条件，同时比较好优化，于是就有了Augmented Lagrangians方法，目的就是放松对于f(x)f(x)严格凸的假设和其他一些条件，同时还能使得算法更加稳健。

![image-20181219152209761](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152209761-5204129.png)

上述也称作method of multipliers，可能也是因为更新对偶变量yy时步长由原来变化的αkαk转为固定的ρρ了吧。该算法在即使f(x)f(x)不是严格凸或者取值为+∞情况都可以成立，适用面更广。同样可以简单证明primal变量xx和对偶变量yy可以同时达到最优。

虽然Augmented Lagrangians方法有优势，但也破坏了dual ascent方法的利用分解参数来并行的优势。当ff是separable时，对于Augmented Lagrangians却是not separable的（因为平方项写成矩阵形式无法用之前那种分块形式），因此在x−min步时候无法并行优化多个参数xi。如何改进，继续下面的议题就可以慢慢发现改进思想的来源。

## 2. Alternating Direction Method of Multipliers(ADMM)

### 2.1 ADMM算法概述

为了整合dual ascent可分解性与method multiplers优秀的收敛性质，人们就又提出了改进形式的优化ADMM。目的就是想能分解原函数和扩增函数，以便于在对ff更一般的假设条件下并行优化。ADMM从名字可以看到是在原来Method of Multipliers加了个Alternating Direction，可以大概猜想到应该是又想引入新变量，然后交叉换方向来交替优化。形式如下：

![image-20181219152242396](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152242396-5204162.png)

==从上面形式确实可以看出，他的思想确实就是想把primal变量、目标函数拆分，但是不再像dual ascent方法那样，将拆分开的xixi都看做是xx的一部分，后面融合的时候还需要融合在一起，而是最先开始就将拆开的变量分别看做是不同的变量xx和zz，同时约束条件也如此处理，这样的好处就是后面不需要一起融合xx和zz，保证了前面优化过程的可分解性。==于是ADMM的优化就变成了如下序贯型迭代（这正是被称作alternating direction的缘故）：

![image-20181219152313796](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152313796-5204193.png)

后面我们可以看到这种拆分思想非常适合统计学习中的ℓ1ℓ1-norm等问题：loss + regulazition（注意：一定要保证zz分解出来，ADMM借助的就是用一个zz变量来简化问题，不管他是约束还是其他形式也罢，需要构造一个zz出来，后面具体到细节问题我们会有更深的体会）。

![image-20181219152351850](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152351850-5204231.png)

![image-20181219152013389](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219152013389-5204013.png)

写成这种形式有利于后面简化优化问题，当然可以不作任何处理

### 2.2 ADMM算法性质和评价

**（1）收敛性**

关于收敛性，需要有两个假设条件：

- ==f和g分别是扩展的实数函数Rn(Rm)→R⋃+∞，且是closed、proper和convex的；==
- ==扩增的lagrangian函数L0有一个鞍点（saddle point）；对于约束中的矩阵A,B都不需要满秩。==

在此两个假设下，可以保证残差、目标函数、对偶变量的收敛性。

==**Note**：实际应用而言，ADMM收敛速度是很慢的，类似于共轭梯度方法。迭代数十次后只可以得到一个acceptable的结果，与快速的高精度算法（Newton法，内点法等）相比收敛就慢很多了。因此实际应用的时候，其实会将ADMM与其他高精度算法结合起来，这样从一个acceptable的结果变得在预期时间内可以达到较高收敛精度。不过一般在大规模应用问题中，高精度的参数解对于预测效果没有很大的提高，因此实际应用中，短时间内一个acceptable的结果基本就可以直接应用预测了。==

**（2）停止准则**

对于ADMM的能到到optimal的条件此处就不做赘述了，**与基本的primal和dual feasibility 的条件差不多，即各primal variable的偏导和约束条件为0**，**从最优条件中可以得到所谓的对偶残差（dual residuals）和初始残差（primal residuals）形式**：

![image-20181219153210810](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219153210810-5204730.png)

相对而言，此处更难把握的其实是停止准则，因为收敛速度问题，要想获得一个还过得去可以拿来用的参数解，那么判断迭代停止还是比较重要的。**<u>实际应用中，一般都根据primal residuals和dual residuals足够小来停止迭代，</u>**阈值包含了绝对容忍度（absolute tolerance）和相对容忍度（relative tolerance），设置还是非常灵活和难把握的（貌似网上有不少人吐槽这个停止准则的不靠谱- -！），具体形式如下：

![image-20181219153417804](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219153417804-5204857.png)

上面的$√p$和$√n$分别是维度和样本量。一般而言，相对停止阈值$ϵ^{rel}=10^{−3}$或者10−4，绝对阈值的选取要根据变量取值范围来选取（咋选的呢？没说额，具体比例都不给说- -！）

**另外一些细节问题，比如原来惩罚参数ρ是不变的，一些文献也做了一些可变的惩罚参数，目的是为了降低对于惩罚参数初始值的依赖性。不过变动的ρ会导致ADMM的收敛性证明比较困难，因此实际中假设经过一系列迭代后ρ也稳定，边可直接用固定的惩罚参数ρ了。还有其他问题，诸如x与z迭代顺序问题，实际操作下有所有不同，这些不是特别重要之处，可**以忽略。==其他与ADMM比较相关算法的有dual ADMM算法，distributed ADMM算法，还有整合了ADMM与proximal method of multiplier的算法==

### 2.3 ADMM一般形式与部分具体应用

当构造了ADMM算法中的f,g,A,B后，便可直接应用该算法了。我们会经常遇到如下三种一般形式的问题

- 二次目标优化项（quadratic objective terms）；
- 可分的目标函数和约束（separable objective and constraints）；
- 光滑目标函数项（smooth objective terms）

为下面讨论的方便，下面仅写出x-update的形式，根据ADMM简化形式，z-update对称更新即可：

![image-20181219155503258](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219155503258-5206103.png)

上述更新x时候z和u都定下来，是个常数，z更新时相同。





**Proximity Operator（近邻算子）**

上述形式有种特殊情况：当A=I 时，即约束条件没有x的线性组合形式，只是对于x的可行区域进行限制。这种问题相当常见，目前统计学习也有不少类似的高维优化问题。此时x-update如下

![image-20181219155503258](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219155503258-5206103.png)

上述**右边可以写成v的函数proxf,ρ(v) 被称作带惩罚ρ的f的proximity operator（通常称作proximal minimization，近邻最小化）**，在变分分析中，还被称作f的**Moreau-Yosida正则化**。如果f形式很简单，可以写出x-update的解析解，比如f是非空的凸包C上的示性函数，那么x-update就可以直接写成投影形式

![image-20181219155814994](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219155814994-5206295.png)

下面再谈谈上述提到的三种一般形式的优化问题。

**（1）Quadratic Objective Terms**

![image-20181219160050473](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219160050473-5206450.png)

**（2）Smooth Objective Terms**

![image-20181219160157121](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219160157121-5206517.png)

**（3）Separable objective and constraints** ![image-20181219160230066](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219160230066-5206550.png)

## 3. 一些具体优化应用

### 3.1受约束的凸优化问题

![image-20181219160316954](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219160316954-5206597.png)

如果上述对xx限制不是限制x≥0x≥0上，而是一个锥约束（conic constraint）x∈x∈K，那么xx-update不变，继续上述KKT方程，而只需要变一下zz-update，将向Rn+R+n投影改成向K投影。比如将上述约束改成{Ax=b,x∈§n+}{Ax=b,x∈§+n}，即xx属于半正定空间，那么向(S^n_{+})投影就变成了一个半正定问题，利用特征值分解可以完成。**这种受约束的凸优化问题的形式化对后续许多问题，特别是我们很关注的ℓ1ℓ1-norm问题很重要，基本上都是转化成这种形式来直接应用ADMM算法，所以这里要好好把握其核心思想和形式。**

![image-20181219160531418](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219160531418-5206731.png)

### 3.2 ℓ1-norm问题

高维统计理论的发展，如果要追溯起来我觉得可以从**Lasso**解法算起，类似的思想在往前追可能是Huber相关的工作。是对于**lasso问题**，由于当年大家还没搞清楚lasso和boosting之间关系，对于sparsity性质不了解，谁也不知道如何很好地解决这个问题。直到后面Efron提出了LARS算法，对两者的路径解相似性做了很好的阐述，于是后面关于变量选择，关于basis-pursuit，compressed sensing，sparse graphical models等各种新问题的产生，随后各种优化算法也随之涌现出来，诸如Gradient Projection， Proximal methods，ADMM (Alternating Direction Method of Multipliers)， (Split) Bregman methods，Nesterov’s method。不过要能够大规模部署ℓ1-norm的解决方案，那么这些算法中ADMM可能是首选。此处ℓ1-norm问题并不仅仅指Lasso问题，包含了多种ℓ1-norm类型问题。下面均介绍下。

**==之所以说ADMM适合机器学习和统计学习的优化问题，因为大部分机器学习问题基本都是“损失函数+正则项”形式，这种分法恰好可以套用到ADMM的框架f(x)+g(z)==**。因此结合ADMM框架基本可以解决很多已有的问题，以及利用ℓ1-norm构造的新的优化问题。下面将先介绍非分布式计算的版本，后面会单开一节来介绍如何分布式计算。

**（1）Least Absolute Deviations**

![image-20181219160926059](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219160926059-5206966.png)

**（2）Huber fitting**

![image-20181219161017102](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219161017102-5207017.png)

LAD和Huber fitting这种问题只是一些传统损失不加正则项的ADMM化，注意一定要构造个zz出来即可，xx可以基本不用管，总是需要解的，下面的带有正则项的优化问题，ADMM形式就会更明显。

**（3）Basis Pursuit**

![image-20181219161134895](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219161134895-5207094.png)

**（4）一般化的损失函数 + ℓ1正则项问题**

![image-20181219161221693](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219161221693-5207141.png)

可以看到与Basis Pursuit解法只是在xx-update上有区别：Basis Pursuit是构造出来一个投影函数f(x)f(x)，而一般化的损失函数f(x)f(x)+ℓ1ℓ1正则项问题，用ADMM就更为自然。所以很适合作为框架来解决这一类问题：广义线性模型（普通线性、logistic回归、possion回归、softmax回归）+正则项；广义可加模型+正则项；似然函数（高斯图方向）+正则项。

![image-20181219161303882](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219161303882-5207183.png)

![image-20181219161326561](/Users/test/Downloads/7-TestCode/__notebook/Optimize/image-20181219161326561-5207206.png)





