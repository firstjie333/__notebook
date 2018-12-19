





### 1.1 Dual Ascent

对于凸函数的优化问题，对偶上升法核心思想就是引入一个对偶变量，然后利用交替优化的思路，使得两者同时达到optimal。一个凸函数的对偶函数其实就是原凸函数的一个下界，因此可以证明一个较好的性质：在强对偶性假设下，即最小化原凸函数（primal）等价于最大化对偶函数（dual），两者会同时达到optimal。这种转化可以将原来很多的参数约束条件变得少了很多，以利于做优化。具体表述如下：
$$
min  
s.t. f(x)Ax=b⟹L(x,y)=f(x)+y^T(Ax−b) ⟹对偶函数（下界）g(y)=inf_xL(x,y)
$$
在强对偶性的假设下，primal和dual问题同时达到最优。
$$
x^⋆=argmin_xL(x,y^⋆)
$$
因此，若对偶函数g(y)可导，便可以利用**梯度上升法，交替更新参数，使得同时收敛到最优**。迭代如下：

$$
x_{k+1}:=argmin_xL(x,y^k)(x-最小化步)
$$

$$
y_{k+1}:=y^k+α^k∇g(y)=yk+αk(Ax^{k+1}−b)(对偶变量更新，αk是步长)
$$

当g不可微的时候也可以将其转化下，成为一个所谓的subgradient的方法，虽然看起来不错，简单证明下即可知道xkxk和ykyk同时可达到optimal，但是上述条件要求很苛刻：f(x)f(x)要求严格凸，并且要求αα选择有比较合适。一般应用中都不会满足（比如f(x)f(x)是一个非零的**仿射函数**），因此dual ascent不会直接应用。

### 1.2 Dual Decomposition

虽然dual ascent方法有缺陷，要求有些严格，但是他有一个非常好的性质，当目标函数ff是可分的（**separable**）时候（参数抑或feature可分），整个问题可以拆解成多个子参数问题，分块优化后汇集起来整体更新。这样非常有利于并行化处理。形式化阐述如下：

mins.t.f(x)=∑Ni=1fi(xi),xi∈Rni,x∈RnAx=∑Ni=1Aixi=b,(对A矩阵按列切分开)⟹L(x,y)=∑i=1NLi(xi,y)=∑i=1N(fi(xi)+yTAixi−1NyTb)minf(x)=∑i=1Nfi(xi),xi∈Rni,x∈Rns.t.Ax=∑i=1NAixi=b,(对A矩阵按列切分开)⟹L(x,y)=∑i=1NLi(xi,y)=∑i=1N(fi(xi)+yTAixi−1NyTb)

因此可以看到其实下面在迭代优化时，xx-minimization步即可以拆分为多个子问题的并行优化，对偶变量更新不变这对于feature特别多时还是很有用的。

xk+1i:yk+1:=argminxLi(xi,yk)(多个xi并行最小化步)=yk+αk∇g(y)=yk+αk(Axk+1−b)(汇集整体的x，然后对偶变量更新)xik+1:=arg⁡minxLi(xi,yk)(多个xi并行最小化步)yk+1:=yk+αk∇g(y)=yk+αk(Axk+1−b)(汇集整体的x，然后对偶变量更新)

对偶分解是非常经典的优化方法，可追溯到1960年代。但是这种想法对后面的分布式优化方法影响较大，比如近期的graph-structure优化问题。

### 1.3 Augmented Lagrangians and the Method of Multipliers

从上面可以看到dual ascent方法对于目标函数要求比较苛刻，为了放松假设条件，同时比较好优化，于是就有了Augmented Lagrangians方法，目的就是放松对于f(x)f(x)严格凸的假设和其他一些条件，同时还能使得算法更加稳健。

Lρ(x,y)=f(x)+yT(Ax−b)+ρ2‖Ax−b‖22⟹mins.t.f(x)+ρ2‖Ax−b‖22Ax=bLρ(x,y)=f(x)+yT(Ax−b)+ρ2‖Ax−b‖22⟹minf(x)+ρ2‖Ax−b‖22s.t.Ax=b

从上面可以看到该问题等价于最初的问题，因为只要是可行解对目标函数就没有影响。但是加了后面的(ρ/2)‖Ax−b‖22(ρ/2)‖Ax−b‖22惩罚项的好处是使得对偶函数gρ(y)=infxLρ(x,y)gρ(y)=infxLρ(x,y)在更一般的条件下可导。计算过程与之前的dual ascent基本一样，除了最小化xx时候加了扩增项。

xk+1yk+1=argminxLρ(x,yk)=yk+ρ(Axk+1−b)xk+1=arg⁡minxLρ(x,yk)yk+1=yk+ρ(Axk+1−b)

上述也称作method of multipliers，可能也是因为更新对偶变量yy时步长由原来变化的αkαk转为固定的ρρ了吧。该算法在即使f(x)f(x)不是严格凸或者取值为+∞+∞情况都可以成立，适用面更广。同样可以简单证明primal变量xx和对偶变量yy可以同时达到最优。

虽然Augmented Lagrangians方法有优势，但也破坏了dual ascent方法的利用分解参数来并行的优势。当ff是separable时，对于Augmented Lagrangians却是not separable的（因为平方项写成矩阵形式无法用之前那种分块形式），因此在x−minx−min步时候无法并行优化多个参数xixi。如何改进，继续下面的议题就可以慢慢发现改进思想的来源。

## 2. Alternating Direction Method of Multipliers(ADMM)

### 2.1 ADMM算法概述

为了整合dual ascent可分解性与method multiplers优秀的收敛性质，人们就又提出了改进形式的优化ADMM。目的就是想能分解原函数和扩增函数，以便于在对ff更一般的假设条件下并行优化。ADMM从名字可以看到是在原来Method of Multipliers加了个Alternating Direction，可以大概猜想到应该是又想引入新变量，然后交叉换方向来交替优化。形式如下：

mins.t.f(x)+g(z)Ax+Bz=c⟹Lρ(x,z,y)=f(x)+g(z)+yT(Ax+Bz−c)+(ρ/2)‖Ax+Bz−c‖22minf(x)+g(z)s.t.Ax+Bz=c⟹Lρ(x,z,y)=f(x)+g(z)+yT(Ax+Bz−c)+(ρ/2)‖Ax+Bz−c‖22

从上面形式确实可以看出，他的思想确实就是想把primal变量、目标函数拆分，但是不再像dual ascent方法那样，将拆分开的xixi都看做是xx的一部分，后面融合的时候还需要融合在一起，而是最先开始就将拆开的变量分别看做是不同的变量xx和zz，同时约束条件也如此处理，这样的好处就是后面不需要一起融合xx和zz，保证了前面优化过程的可分解性。于是ADMM的优化就变成了如下序贯型迭代（这正是被称作alternating direction的缘故）：

xk+1zk+1yk+1=argminxLρ(x,zk,yk)=argminzLρ(xk+1,z,yk)=yk+ρ(Axk+1+Bzk+1−c)xk+1=arg⁡minxLρ(x,zk,yk)zk+1=arg⁡minzLρ(xk+1,z,yk)yk+1=yk+ρ(Axk+1+Bzk+1−c)

后面我们可以看到这种拆分思想非常适合统计学习中的ℓ1ℓ1-norm等问题：loss + regulazition（注意：一定要保证zz分解出来，ADMM借助的就是用一个zz变量来简化问题，不管他是约束还是其他形式也罢，需要构造一个zz出来，后面具体到细节问题我们会有更深的体会）。

为了简化形式，ADMM有一个scaled form形式，其实就是对对偶变量做了scaled处理。先定义每一步更新的残差为r=Ax+Bz−cr=Ax+Bz−c，于是稍加计算

yT(Ax+Bz−c)+(ρ/2)‖Ax+Bz−c‖22=yTr+(ρ/2)‖r‖22=(ρ/2)‖r+(1/ρ)y‖22−(1/2ρ)‖y‖22=(ρ/2)‖r+u‖22−(ρ/2)‖u‖22yT(Ax+Bz−c)+(ρ/2)‖Ax+Bz−c‖22=yTr+(ρ/2)‖r‖22=(ρ/2)‖r+(1/ρ)y‖22−(1/2ρ)‖y‖22=(ρ/2)‖r+u‖22−(ρ/2)‖u‖22

此处u=(1/ρ)yu=(1/ρ)y称为scaled dual variable，并令每一步迭代的残差为rk=Axk+Bzk−crk=Axk+Bzk−c，以及累计残差uk=u0+∑kj=1rjuk=u0+∑j=1krj，于是ADMM形式就可以简化为如下形式

xk+1zk+1uk+1=argminxLρ(x,zk,yk)=argmin(f(x)+(ρ/2)‖Ax+Bzk−c+uk‖22)=argminzLρ(xk+1,z,yk)=argmin(g(z)+(ρ/2)‖Axk+1+Bz−c+uk‖)=uk+Axk+1+Bzk+1−cxk+1=arg⁡minxLρ(x,zk,yk)=arg⁡min(f(x)+(ρ/2)‖Ax+Bzk−c+uk‖22)zk+1=arg⁡minzLρ(xk+1,z,yk)=arg⁡min(g(z)+(ρ/2)‖Axk+1+Bz−c+uk‖)uk+1=uk+Axk+1+Bzk+1−c

写成这种形式有利于后面简化优化问题，当然可以不作任何处理。

### 2.2 ADMM算法性质和评价

**（1）收敛性**

关于收敛性，需要有两个假设条件：

- ff和gg分别是扩展的实数函数Rn(Rm)→R⋃+∞Rn(Rm)→R⋃+∞，且是closed、proper和convex的；
- 扩增的lagrangian函数L0L0有一个鞍点（saddle point）；对于约束中的矩阵A,BA,B都不需要满秩。

在此两个假设下，可以保证残差、目标函数、对偶变量的收敛性。

**Note**：实际应用而言，ADMM收敛速度是很慢的，类似于共轭梯度方法。迭代数十次后只可以得到一个acceptable的结果，与快速的高精度算法（Newton法，内点法等）相比收敛就慢很多了。因此实际应用的时候，其实会将ADMM与其他高精度算法结合起来，这样从一个acceptable的结果变得在预期时间内可以达到较高收敛精度。不过一般在大规模应用问题中，高精度的参数解对于预测效果没有很大的提高，因此实际应用中，短时间内一个acceptable的结果基本就可以直接应用预测了。

**（2）停止准则**

对于ADMM的能到到optimal的条件此处就不做赘述了，与基本的primal和dual feasibility 的条件差不多，即各primal variable的偏导和约束条件为0，从最优条件中可以得到所谓的对偶残差（dual residuals）和初始残差（primal residuals）形式：

sk+1rk+1=ρATB(zk+1−zk)(dualresiduals)=Axk+1+Bzk+1−c(primalresiduals)sk+1=ρATB(zk+1−zk)(dualresiduals)rk+1=Axk+1+Bzk+1−c(primalresiduals)

相对而言，此处更难把握的其实是停止准则，因为收敛速度问题，要想获得一个还过得去可以拿来用的参数解，那么判断迭代停止还是比较重要的。实际应用中，一般都根据primal residuals和dual residuals足够小来停止迭代，阈值包含了绝对容忍度（absolute tolerance）和相对容忍度（relative tolerance），设置还是非常灵活和难把握的（貌似网上有不少人吐槽这个停止准则的不靠谱- -！），具体形式如下：

‖sk‖2≤ϵdual‖rk‖2≤ϵpri=n‾√ϵabs+ϵrel‖ATyk‖2=p‾√ϵabs+ϵrelmax{‖Axk‖2,‖Bzk‖,‖c‖2}‖sk‖2≤ϵdual=nϵabs+ϵrel‖ATyk‖2‖rk‖2≤ϵpri=pϵabs+ϵrelmax{‖Axk‖2,‖Bzk‖,‖c‖2}

上面的p‾√p和n‾√n分别是维度和样本量。一般而言，相对停止阈值ϵrel=10−3ϵrel=10−3或者10−410−4，绝对阈值的选取要根据变量取值范围来选取（咋选的呢？没说额，具体比例都不给说- -！）

另外一些细节问题，比如原来惩罚参数ρρ是不变的，一些文献也做了一些可变的惩罚参数，目的是为了降低对于惩罚参数初始值的依赖性。不过变动的ρρ会导致ADMM的收敛性证明比较困难，因此实际中假设经过一系列迭代后ρρ也稳定，边可直接用固定的惩罚参数ρρ了。还有其他问题，诸如xx与zz迭代顺序问题，实际操作下有所有不同，这些不是特别重要之处，可以忽略。其他与ADMM比较相关算法的有dual ADMM算法，distributed ADMM算法，还有整合了ADMM与proximal method of multiplier的算法

### 2.3 ADMM一般形式与部分具体应用

当构造了ADMM算法中的f,g,A,Bf,g,A,B后，便可直接应用该算法了。我们会经常遇到如下三种一般形式的问题

- 二次目标优化项（quadratic objective terms）；
- 可分的目标函数和约束（separable objective and constraints）；
- 光滑目标函数项（smooth objective terms）。

为下面讨论的方便，下面仅写出xx-update的形式，根据ADMM简化形式，zz-update对称更新即可：

x+=argminx(f(x)+(ρ/2)‖Ax−v‖22),v=−Bz+c−ux+=arg⁡minx(f(x)+(ρ/2)‖Ax−v‖22),v=−Bz+c−u

上述更新xx时候zz和uu都定下来，是个常数，zz更新时后相同。

**Proximity Operator（近邻算子）**

上述形式有种特殊情况：当A=IA=I时，即约束条件没有xx的线性组合形式，只是对于xx的可行区域进行限制。这种问题相当常见，目前统计学习也有不少类似的高维优化问题。此时xx-update如下

x+=argminx(f(x)+(ρ/2)‖x−v‖22),v=−Bz+c−ux+=arg⁡minx(f(x)+(ρ/2)‖x−v‖22),v=−Bz+c−u

上述右边可以写成vv的函数proxf,ρ(v)proxf,ρ(v)被称作带惩罚ρρ的ff的proximity operator（通常称作proximal minimization，近邻最小化），在变分分析中，还被称作ff的**Moreau-Yosida正则化**。如果ff形式很简单，可以写出xx-update的解析解，比如ff是非空的凸包C上的示性函数，那么xx-update就可以直接写成投影形式

x+=argminx(f(x)+(ρ/2)‖x−v‖22)=Π(v)x+=arg⁡minx(f(x)+(ρ/2)‖x−v‖22)=ΠC(v)

投影与惩罚参数ρρ无关。若ff是非负象限Rn+R+n的投影，则直接有x+=(v)+x+=(v)+。

下面再谈谈上述提到的三种一般形式的优化问题。

**（1）Quadratic Objective Terms**

假设ff是如下（凸）的二次函数

f(x)=12xTPx+qTx+rf(x)=12xTPx+qTx+r

PP是对称的半正定矩阵P∈Sn+P∈S+n。这种形式问题也包含了ff是线性或者常数的特殊情况。若P+ρATAP+ρATA可逆，那么xx-update步求个导即有如下的显示解，是vv的仿射函数

x+=(P+ρATA)−1(ρATv−q)x+=(P+ρATA)−1(ρATv−q)

因此在xx-minnimiztion步只需要做两个矩阵运算即可，求逆与乘积，选用合适的线性运算库即可以得到不错的计算性能。当然还可以利用一些矩阵分解技巧，这个要看矩阵大小和稀疏程度。因为对于Fx=gFx=g，可以将F=F1F2⋯FkF=F1F2⋯Fk，然后Fizi=zi−1,z1=F−11g,x=zkFizi=zi−1,z1=F1−1g,x=zk，这样会更节省计算时间。其他矩阵计算技巧，基本都是如何对矩阵大规模求解，利用矩阵的稀疏性、缓存分解等来提高性能。此处不赘述，有个很重要的求逆的定理很有用：

(P+ρATA)−1=P−1−ρP−1AT(I+ρAP−1AT)−1AP−1(P+ρATA)−1=P−1−ρP−1AT(I+ρAP−1AT)−1AP−1

如果对于上述二次函数受限于某仿射集xx-update步就更复杂些，如

f(x)=12xTPx+qTx+rdormf={x‖Fx=g}f(x)=12xTPx+qTx+rdormf={x‖Fx=g}

xx-update还有个重要的KKT方程可用：

(P+ρIFFT0)(xk+1v)+(q−ρ(zk−uk)−g)=0(P+ρIFTF0)(xk+1v)+(q−ρ(zk−uk)−g)=0

**（2）Smooth Objective Terms**

当ff光滑时，那么求导即成为可能了。对于一些非线性优化问题，包含梯度算法等方法的L-BFGS算法可以用。对于该算法有些小技巧如下：

- 早终止（early termination）：当f(x)+(ρ/2)|Ax−v|22f(x)+(ρ/2)|Ax−v|22梯度很小时，早点终止迭代，否则后面就很慢了。
- 热启动（warm start）：即启动迭代时，利用之前迭代过的值带入即可。

**（3）Separable objective and constraints** 可分函数和约束对于并行计算和分布式计算来说是一个好消息。如果ATAATA是分块的对角阵，那么约束中$|Ax|^2*2也是可分的，则扩增的拉格朗日函数也是可分的，则扩增的拉格朗日函数L*{\rho}$也是可分的。（注意，此处是指函数中的参数可分成小子块，而不是说数据可分。）下面有一个很重要的例子，即**soft thresholding**（针对l1+l2l1+l2问题）:

当f(x)=λ|x|1,λ>0f(x)=λ|x|1,λ>0，并且A=IA=I时，那么xx-update就变成了

x+=argminx(λ‖xi‖+(ρ/2)‖x−v‖22)x+=arg⁡minx(λ‖xi‖+(ρ/2)‖x−v‖22)

这种形式很常见在目前的高维统计中，虽然第一项在0处不可导，但是也有解析解，被称作软阈值（soft thresholding），也被称作压缩算子（shrinkage operator）。

x+i=Sλ/ρ(vi),→Sk(a)=⎧⎩⎨⎪⎪a−k0,a+k,a>k|a|≤ka<−k→Sk(a)=(1−k|a|)+axi+=Sλ/ρ(vi),→Sk(a)={a−k,a>k0,|a|≤ka+ka<−k→Sk(a)=(1−k|a|)+a

在优化领域，软阈值被称作是ℓ1ℓ1-norm问题的近邻算子（proximity operator）。

## 3. 一些具体优化应用

### 3.1受约束的凸优化问题

一般的受约束的凸优化问题可以写成如下形式

mins.tf(x)x∈minf(x)s.tx∈C

此类问题可以写成ADMM形式

mins.tf(x)+g(z)x−z=0⟹Lρ(x,z,u)=f(x)+g(z)+(ρ/2)‖x−z+u‖22minf(x)+g(z)s.tx−z=0⟹Lρ(x,z,u)=f(x)+g(z)+(ρ/2)‖x−z+u‖22

其中的gg函数即C的示性函数，上述是scaled形式，那么具体算法就是

xk+1zk+1uk+1=argmin(f(x)+(ρ/2)‖x−zk+uk‖22)=Π(xk+1+uk)=uk+xk+1−zk+1xk+1=arg⁡min(f(x)+(ρ/2)‖x−zk+uk‖22)zk+1=ΠC(xk+1+uk)uk+1=uk+xk+1−zk+1

则上述x−minx−min就变成了一个具体的受约束的优化问题。比如对于经典的二次规划问题(QP)

mins.t12xTPx+qTxAx=b,x≥0min12xTPx+qTxs.tAx=b,x≥0

写成ADMM形式

mins.tf(x)+g(z)x−z=0⟹f(x)g(z)=12xTPx+qTx,dormf={x|Ax=b}=I(ΠRn+(z))minf(x)+g(z)s.tx−z=0⟹f(x)=12xTPx+qTx,dormf={x|Ax=b}g(z)=I(ΠR+n(z))

即受约束的区域就是{x∣x≥0}{x∣x≥0}，gg是向非负象限投影的示性函数。而xx-update就变成了之前在Quadratic Objective Terms中谈到的f(x)f(x)有仿射集定义域的优化问题，根据KKT条件即可写出来xx-update更新的形式，参见2.3节。

如果上述对xx限制不是限制x≥0x≥0上，而是一个锥约束（conic constraint）x∈x∈K，那么xx-update不变，继续上述KKT方程，而只需要变一下zz-update，将向Rn+R+n投影改成向K投影。比如将上述约束改成{Ax=b,x∈§n+}{Ax=b,x∈§+n}，即xx属于半正定空间，那么向(S^n_{+})投影就变成了一个半正定问题，利用特征值分解可以完成。**这种受约束的凸优化问题的形式化对后续许多问题，特别是我们很关注的ℓ1ℓ1-norm问题很重要，基本上都是转化成这种形式来直接应用ADMM算法，所以这里要好好把握其核心思想和形式。**

虽然我对优化不在行，但是感觉优化问题还是挺有意思的，下面是一个经典问题，即找到两个非空凸包的交集中的一点。该算法都可以追溯到1930年代的Neumann交替投影算法（alternating projections algorithm）：

xk+1zk+1=Π(zk)=Π(xk+1)xk+1=ΠC(zk)zk+1=ΠD(xk+1)

Π,ΠΠC,ΠD分别是两个集合的欧式空间投影。写成ADMM形式就是

xk+1zk+1uk+1=Π(zk−uk)=Π(xk+1+uk)=uk+xk+1−zk+1xk+1=ΠC(zk−uk)zk+1=ΠD(xk+1+uk)uk+1=uk+xk+1−zk+1

上述问题还可推广至找到NN个非空凸包交集中一个点的问题，这样其实在xx步是可以并行来做的，于是就有

xk+1izk+1uk+1i=Πi(zk−uki)=1N∑i=1N(xk+1i+uki)=uki+xk+1i−zk+1⟹ui收敛均趋向于0,zk+1=x¯k+1xk+1iuk+1i=Πi(x¯k−uki)=uki+(xk+1i−x¯k+1)xik+1=ΠAi(zk−uik)zk+1=1N∑i=1N(xik+1+uik)uik+1=uik+xik+1−zk+1⟹ui收敛均趋向于0,zk+1=x¯k+1xik+1=ΠAi(x¯k−uik)uik+1=uik+(xik+1−x¯k+1)

### 3.2 ℓ1ℓ1-norm问题

高维统计理论的发展，如果要追溯起来我觉得可以从Lasso解法算起，类似的思想在往前追可能是Huber相关的工作。是对于lasso问题，由于当年大家还没搞清楚lasso和boosting之间关系，对于sparsity性质不了解，谁也不知道如何很好地解决这个问题。直到后面Efron提出了LARS算法，对两者的路径解相似性做了很好的阐述，于是后面关于变量选择，关于basis-pursuit，compressed sensing，sparse graphical models等各种新问题的产生，随后各种优化算法也随之涌现出来，诸如Gradient Projection， Proximal methods，ADMM (Alternating Direction Method of Multipliers)， (Split) Bregman methods，Nesterov’s method。不过要能够大规模部署ℓ1ℓ1-norm的解决方案，那么这些算法中ADMM可能是首选。此处ℓ1ℓ1-norm问题并不仅仅指Lasso问题，包含了多种ℓ1ℓ1-norm类型问题。下面均介绍下。

之所以说ADMM适合机器学习和统计学习的优化问题，因为大部分机器学习问题基本都是“损失函数+正则项”形式，这种分法恰好可以套用到ADMM的框架f(x)+g(z)f(x)+g(z)。因此结合ADMM框架基本可以解决很多已有的问题，以及利用ℓ1ℓ1-norm构造的新的优化问题。下面将先介绍非分布式计算的版本，后面会单开一节来介绍如何分布式计算。

**（1）Least Absolute Deviations**

先从一个简单的问题开始。在稳健估计中，LAD是一个应用很广的模型，相对于直接优化平方和损失|Ax−b|22|Ax−b|22，优化绝对损失|Ax−b|1|Ax−b|1，它的抗噪性能更好。在ADMM框架下，往之前的受约束的凸优化问题靠拢，这个问题有简单的迭代算法

mins.t.‖z‖1Ax−b=z⟹letf(x)=0,g(z)=‖z‖1⟹xk+1zk+1uk+1=(ATA)−1AT(b+zk−uk)=S1/ρ(Axk+1−b+uk)=uk+Axk+1−zk+1−bmin‖z‖1s.t.Ax−b=z⟹letf(x)=0,g(z)=‖z‖1⟹xk+1=(ATA)−1AT(b+zk−uk)zk+1=S1/ρ(Axk+1−b+uk)uk+1=uk+Axk+1−zk+1−b

**（2）Huber fitting**

Huber问题与上面的其实差不多，只是损失函数形式不同，换成了Huber惩罚函数

mins.t.ghub(z)Ax−b=z,ghub(z)={z2/2,|z|−12|z|≤1|z|>1minghub(z)s.t.Ax−b=z,ghub(z)={z2/2,|z|≤1|z|−12|z|>1

因此与LAD除了zz-update不在是proximity operator（或称作软阈值）之外，其余均是相同的

zk+1=ρ1+ρ(Axk+1−b+uk)+11+ρS1+1/ρ(Axk+1−b+uk)zk+1=ρ1+ρ(Axk+1−b+uk)+11+ρS1+1/ρ(Axk+1−b+uk)

看着像是proximity operator与一个残差的加权。

LAD和Huber fitting这种问题只是一些传统损失不加正则项的ADMM化，注意一定要构造个zz出来即可，xx可以基本不用管，总是需要解的，下面的带有正则项的优化问题，ADMM形式就会更明显。

**（3）Basis Pursuit**

基追踪法师系数信号处理的一种重要方法。目的是想找到一组稀疏基可以完美恢复信号，换套话说就是为一个线性方程系统找到一个稀疏解。原始形式如下，与lasso有些像：

mins.t.‖x‖1Ax=bmin‖x‖1s.t.Ax=b

修改成ADMM形式，注意往之前受约束的凸优化问题的那种形式回套，将ℓ1ℓ1看做约束，然后构造带定义域的f(x)f(x)，于是就有解

mins.t.f(x)+‖z‖1x−z=0f(x)=I({x∈Rn|Ax=b})indicator function⟹xk+1zk+1uk+1=Π(zk−uk)=S1/ρ(Axk+1+uk)=uk+xk+1−zk+1minf(x)+‖z‖1s.t.x−z=0f(x)=I({x∈Rn|Ax=b})indicator function⟹xk+1=Π(zk−uk)zk+1=S1/ρ(Axk+1+uk)uk+1=uk+xk+1−zk+1

其中Π(zk−uk)Π(zk−uk)是向一个线性约束的欧式空间中投影x∈Rn∣Ax=bx∈Rn∣Ax=b，这也是有直接的显示解的

xk+1=(I−AT(ATA)−1A)(z−uk)+AT(AAT)−1bxk+1=(I−AT(ATA)−1A)(z−uk)+AT(AAT)−1b

对于矩阵求逆、分解等用之前矩阵那些小技巧即可加快计算，节省计算资源。

最近还有一类算法来解决ℓ1ℓ1问题，被称作**Bregman iteration methods**，对于基追踪相关问题，加正则项的Bregman iteration就是method of multiplier，而所谓的split Bregman iteration就等同于 ADMM。我没有继续深究，应该就是类似于并行化的ADMM算法来解决基追踪问题。

**（4）一般化的损失函数 + ℓ1ℓ1正则项问题**

这类问题在高维统计开始时便是一个非常重要的问题，而即使到了现在也是一个非常重要的问题，比如group lasso，generalized lasso，高斯图模型，Tensor型图模型，与图相关的ℓ1ℓ1问题等算法的开发，都可以在此框架上直接应用和实施，这正是ADMM一个优势所在，便于快速实施，也便于可能的大规模分布式部署。

minl(x)+λ‖x‖1,⟹mins.t.l(x)+g(z)=l(x)+λ‖z‖1x−z=0⟹xk+1zk+1uk+1=argminx(l(x)+(ρ/2)‖x−zk+uk‖22)=S1/ρ(xk+1+uk)=uk+xk+1−zk+1minl(x)+λ‖x‖1,⟹minl(x)+g(z)=l(x)+λ‖z‖1s.t.x−z=0⟹xk+1=arg⁡minx(l(x)+(ρ/2)‖x−zk+uk‖22)zk+1=S1/ρ(xk+1+uk)uk+1=uk+xk+1−zk+1

可以看到与Basis Pursuit解法只是在xx-update上有区别：Basis Pursuit是构造出来一个投影函数f(x)f(x)，而一般化的损失函数f(x)f(x)+ℓ1ℓ1正则项问题，用ADMM就更为自然。所以很适合作为框架来解决这一类问题：广义线性模型（普通线性、logistic回归、possion回归、softmax回归）+正则项；广义可加模型+正则项；似然函数（高斯图方向）+正则项。

- **Lasso**：f(x)=12|Ax−b|22f(x)=12|Ax−b|22，于是利用ADMM算法，xx-update的解析解就是xk+1=(ATA+ρI)−1(ATb+ρ(zk−uk))xk+1=(ATA+ρI)−1(ATb+ρ(zk−uk))；于是xx-update看起来是个岭回归了，因此ADMM对于lasso可以看做迭代的使用岭回归。至于矩阵求逆那些，利用之前的矩阵小技巧解决。
- **Generalized lasso**：这个问题可能不是那么为众人所熟悉，他是Tibs的儿子搞出来的框罗类似fused lasso这种事先定义好的线性变化的惩罚项的模型，损失函数是平方损失，而惩罚变成了一个特殊的参数线性组合

min12‖Ax−b‖22+λ‖Fx‖1min12‖Ax−b‖22+λ‖Fx‖1 ⟹1d fused lasso,A=IFij=⎧⎩⎨⎪⎪1−10j=i+1j=iotherwise⟹1d fused lasso,A=IFij={1j=i+1−1j=i0otherwise

⟹min12‖x−b‖22+λ∑i=1n−1|xi+1−xi|⟹A=I,F二阶差分矩阵，则被称作L1 trend filtering⟹min12‖x−b‖22+λ∑i=1n−1|xi+1−xi|⟹A=I,F二阶差分矩阵，则被称作L1 trend filtering

若将上述这种写成ADMM形式，同样可以放到ADMM算法框架中解决

mins.t.12‖Ax−b‖22+λ‖z‖1Fx−z=0⟹xk+1zk+1uk+1=(ATA+ρFTF)−1(ATb+ρFT(zk−uk))=S1/ρ(Axk+1−b+uk)=uk+Fxk+1−zk+1−bmin12‖Ax−b‖22+λ‖z‖1s.t.Fx−z=0⟹xk+1=(ATA+ρFTF)−1(ATb+ρFT(zk−uk))zk+1=S1/ρ(Axk+1−b+uk)uk+1=uk+Fxk+1−zk+1−b

- **Group lasso**：graph lasso问题应用比较广，对不同组的参数同时进行惩罚，进行一组组参数的挑选，故曰group lasso。不同于lasso，其正则项变成了∑Ni=1|xi|2,xi∈Rni∑i=1N|xi|2,xi∈Rni，lasso其实是group lasso的一种特殊形式。正则项并不是完全可分的。此时只是zz-update变成了block的软阈值形式

zk+1i=Sλ/rho(xk+1i+uk),i=1,…,N⟹Sk(a)=(1−k‖a‖2)+a,S(0)=0zik+1=Sλ/rho(xik+1+uk),i=1,…,N⟹Sk(a)=(1−k‖a‖2)+a,S(0)=0

这种形式还可以扩展到group间有重合的情况，即化成NN可能存在重合的组Gi⊆1,…,nGi⊆1,…,n。一般来说这种问题会非常难解决，但是对于ADMM算法只需要换下形式就很直接（x,zx,z互换，会变成后面非常重要的一致性优化问题（consensus optimization），局部xixi与全局真解zz子集ẑ iz^i的对应。）

mins.t.12‖Az−b‖22+λ∑Ni=1‖xi‖2,xi∈R|Gi|xi−ẑ i=0,i=1,…,Nmin12‖Az−b‖22+λ∑i=1N‖xi‖2,xi∈R|Gi|s.t.xi−z^i=0,i=1,…,N

- **Sparse Gaussian graph model**：对于稀疏高斯图，熟悉该问题的人知道这其实是lasso的图上的推广，损失函数写成似然函数的负数即可l(x)=tr(SX)−logdetX,X∈Sn++l(x)=tr(SX)−log⁡detX,X∈S++n。于是原来向量的操作就变成了矩阵操作，ADMM算法也有点变化：

Xk+1Zk+1Uk+1=argminX(tr(SX)−logdetX+ρ2‖X−Zk+Uk‖F)=argminZ(λ‖Z‖1+ρ2‖Xk+1−Z+Uk‖F)=Uk+Xk+1−Zk+1Xk+1=arg⁡minX(tr(SX)−log⁡detX+ρ2‖X−Zk+Uk‖F)Zk+1=arg⁡minZ(λ‖Z‖1+ρ2‖Xk+1−Z+Uk‖F)Uk+1=Uk+Xk+1−Zk+1

上述算法继续化简，对于zz-update做逐个元素软阈值操作即可Zk+1ij=Sλ/ρ(XK+1ij+Ukij)Zijk+1=Sλ/ρ(XijK+1+Uijk)。对于xx-update也类似操作，直接求导一阶导为0，移项后对对称矩阵做特征值分解即可

ρX−X−1=ρ(Zk−Uk)−S=QΛQT,QQT=I,Λ=diag(λ1,…,λn)ρX−X−1=ρ(Zk−Uk)−S=QΛQT,QQT=I,Λ=diag(λ1,…,λn) →ρX̂ −X̂ −1=Λ,X̂ =QTXQ→ρX^−X^−1=Λ,X^=QTXQ

由于ΛΛ是对角阵，对于每个对角元素来说，上述问题就是解一个二次方程，解方程后，再将X̂ X^变化成XX即可

X̂ ii=λi+λ2i+4ρ‾‾‾‾‾‾‾√2ρ⟹X=QX̂ QTX^ii=λi+λi2+4ρ2ρ⟹X=QX^QT

总之，上述跟ℓ1ℓ1相关的问题，基本都可以纳入ADMM框架，并且可以快速求解。

## 4. Consensus and Sharing

本节讲述的两个优化问题，是非常常见的优化问题，也非常重要，我认为是ADMM算法通往并行和分布式计算的一个途径：consensus和sharing，即一致性优化问题与共享优化问题。

## Consensus

### 4.1 全局变量一致性优化（Global variable consensus optimization）（切割数据，参数（变量）维数相同）

所谓全局变量一致性优化问题，即目标函数根据数据分解成NN子目标函数（子系统），每个子系统和子数据都可以获得一个参数解xixi，但是全局解只有一个zz，于是就可以写成如下优化命题：

mins.t.∑Ni=1fi(xi),xi∈Rnxi−z=0min∑i=1Nfi(xi),xi∈Rns.t.xi−z=0

注意，此时fi:Rn→R⋃+∞fi:Rn→R⋃+∞仍是凸函数，而xixi并不是对参数空间进行划分，这里是对数据而言，所以xixi维度一样xi,z∈Rnxi,z∈Rn，与之前的问题并不太一样。这种问题其实就是所谓的并行化处理，或分布式处理，希望从多个分块的数据集中获取相同的全局参数解。

在ADMM算法框架下（先返回最初从扩增lagrangian导出的ADMM），这种问题解法相当明确：

Lρ(x1,…,xN,z,y)=∑Ni=1(fi(xi)+yTi(xi−z)+(ρ/2)‖xi−z‖22)s.t.={(x1,…,xN)|x1=…=xN}Lρ(x1,…,xN,z,y)=∑i=1N(fi(xi)+yiT(xi−z)+(ρ/2)‖xi−z‖22)s.t.C={(x1,…,xN)|x1=…=xN}

⟹xk+1izk+1yk+1i=argminx(fi(xi)+(yki)T(xi−zk)+(ρ/2)‖xi−z‖22))=1N∑i=1N(xk+1i+(1ρyki))=yki+ρ(xk+1i−zk+1)⟹xik+1=arg⁡minx(fi(xi)+(yik)T(xi−zk)+(ρ/2)‖xi−z‖22))zk+1=1N∑i=1N(xik+1+(1ρyik))yik+1=yik+ρ(xik+1−zk+1)

对yy-update和zz-update的yk+1iyik+1和zk+1izik+1分别求个平均，易得y¯k+1=0y¯k+1=0，于是可以知道zz-update步其实可以简化为zk+1=x¯k+1zk+1=x¯k+1，于是上述ADMM其实可以进一步化简为如下形式：

xk+1iyk+1i=argminx(fi(xi)+(yki)T(xi−x¯k)+(ρ/2)‖xi−x¯k‖22))=yki+ρ(xk+1i−x¯k+1)xik+1=arg⁡minx(fi(xi)+(yik)T(xi−x¯k)+(ρ/2)‖xi−x¯k‖22))yik+1=yik+ρ(xik+1−x¯k+1)

这种迭代算法写出来了，并行化那么就是轻而易举了，各个子数据分别并行求最小化，然后将各个子数据的解汇集起来求均值，整体更新对偶变量ykyk，然后再继续回带求最小值至收敛。当然也可以分布式部署（hadoop化），但是说起来容易，真正工程实施起来又是另外一回事，各个子节点机器间的通信更新是一个需要细细揣摩的问题。

另外，对于全局一致性优化，也需要给出相应的终止迭代准则，与一般的ADMM类似，看primal和dual的residuals即可

‖rk‖22=∑i=1N‖xki−x¯k‖22,‖sk‖22=Nρ‖x¯ki−x¯k−1‖22‖rk‖22=∑i=1N‖xik−x¯k‖22,‖sk‖22=Nρ‖x¯ik−x¯k−1‖22

### 4.2 带正则项的全局一致性问题

下面就是要将之前所谈到的经典的机器学习算法并行化起来。想法很简单，就是对全局变量加上正则项即可，因此ADMM算法只需要改变下zz-update步即可

mins.t.∑Ni=1fi(xi)+g(z),xi∈Rnxi−z=0⟹xk+1izk+1yk+1i=argminx+i(fi(xi)+(yki)T(xi−zk)(ρ/2)‖xi−z‖22))=argminz(g(z)+∑i=1N(−(yki)Tz+(ρ/2)‖xk+1i−z‖22))=yki+ρ(xk+1i−zk+1)min∑i=1Nfi(xi)+g(z),xi∈Rns.t.xi−z=0⟹xik+1=arg⁡minx+i(fi(xi)+(yik)T(xi−zk)(ρ/2)‖xi−z‖22))zk+1=arg⁡minz(g(z)+∑i=1N(−(yik)Tz+(ρ/2)‖xik+1−z‖22))yik+1=yik+ρ(xik+1−zk+1)

同样的，我们仍对zz做一个平均处理，于是就有

zk+1=argminz(g(z)+(Nρ/2)‖z−x¯k+1−(1/ρ)y¯k‖22)zk+1=arg⁡minz(g(z)+(Nρ/2)‖z−x¯k+1−(1/ρ)y¯k‖22)

上述形式都取得是最原始的ADMM形式，简化处理，写成scaled形式即有

xk+1izk+1uk+1i=argminx(fi(xi)+(ρ/2)‖xi−zk+uki‖22))=argminz(g(z)+(Nρ/2)‖z−xk+1i−u¯k‖22)=uki+xk+1i−zk+1xik+1=arg⁡minx(fi(xi)+(ρ/2)‖xi−zk+uik‖22))zk+1=arg⁡minz(g(z)+(Nρ/2)‖z−xik+1−u¯k‖22)uik+1=uik+xik+1−zk+1

这样对于后续处理问题就清晰明了多了。可以看到如果g(z)=λ|z|1g(z)=λ|z|1，即lasso问题，那么zz-update步就用软阈值operator即可。因此，对于大规模数据，要想用lasso等算法，只需要对数据做切块（切块也最好切均匀点），纳入到全局变量一致性的ADMM框架中，即可并行化处理。下面给出一些实例。

**切割大样本数据，并行化计算**

在经典的统计估计中，我们处理的多半是大样本低维度的数据，现在则多是是大样本高维度的数据。对于经典的大样本低维度数据，如果机器不够好，那么就抽样部分数据亦可以实现较好估计，不过如果没有很好的信息，就是想要对大样本进行处理，那么切割数据，并行计算是一个好的选择。现在的社交网络、网络日志、无线感应网络等都可以这么实施。下面的具体模型都在受约束的凸优化问题中以及ℓ1ℓ1-norm问题中提过，此处只不过切割数据，做成分布式模型，思想很简单，与带正则项的global consensus问题一样的处理。经典问题lasso、sparse logistic lasso、SVM都可以纳入如下框架处理。

有观测阵A∈Rm×nA∈Rm×n和响应值b∈Rmb∈Rm，可以对应切分，即对矩阵AA和向量bb横着切，

A=⎛⎝⎜⎜⎜A1⋮AN⎞⎠⎟⎟⎟b=⎛⎝⎜⎜⎜b1⋮bN⎞⎠⎟⎟⎟A=(A1⋮AN)b=(b1⋮bN)

于是原来带正则项的优化问题就可以按照数据分解到多个子系统上去分别优化，然后汇集起来，形成一个global consensus问题。

mins.t.∑Ni=1li(Aixi−bi)+r(z)xi−z=0,i=1,…,Nxi,z∈Rnmin∑i=1Nli(Aixi−bi)+r(z)s.t.xi−z=0,i=1,…,Nxi,z∈Rn

结合受约束的凸优化问题时所给出来的具体的ADMM算法解的形式，下面直接给出这些问题的ADMM迭代算法公式

**（1）Lasso**

xk+1izk+1uk+1i=(ATiAi+ρI)−1(ATibi+ρ(zk−uki))=S1/ρN(x¯k+1−b+u¯k)=uki+xk+1i−zk+1xik+1=(AiTAi+ρI)−1(AiTbi+ρ(zk−uik))zk+1=S1/ρN(x¯k+1−b+u¯k)uik+1=uik+xik+1−zk+1

如果切割的数据量小于维数mi<nmi<n，那么求解时分解小的矩阵AiATi+ρIAiAiT+ρI即可；其他求逆采用矩阵加速技巧即可。

**（2）Sparse Logistic Regression**

xk+1izk+1uk+1i=argminxi(li(Aixi−bi)+(ρ/2)‖xi−zk+uki‖22=S1/ρN(x¯k+1−b¯+u¯k)=uki+xk+1i−zk+1xik+1=arg⁡minxi(li(Aixi−bi)+(ρ/2)‖xi−zk+uik‖22zk+1=S1/ρN(x¯k+1−b¯+u¯k)uik+1=uik+xik+1−zk+1

在xx-update步是需要用一些有效的算法来解决ℓ2ℓ2正则的logistic回归，比如L-BFGS，其他的优化算法应该问题不大吧。

**（3）SVM**

注意分类问题和回归问题的损失函数不同，一般都是用l(sign(t)y)l(sign(t)y)形式来寻求最优的分类权重使得分类正确。SVM使用Hinge Loss：ℓ(y)=max(0,1−t⋅y)ℓ(y)=max(0,1−t⋅y)，即将预测类别与实际分类符号相反的损失给凸显出来。分布式的ADMM形式

xk+1izk+1uk+1i=argminxi(1T(Aixi+1)++(ρ/2)‖xi−zk+uki‖22=ρ(1/λ)+Nρ(x¯k+1+u¯k)=uki+xk+1i−zk+1xik+1=arg⁡minxi(1T(Aixi+1)++(ρ/2)‖xi−zk+uik‖22zk+1=ρ(1/λ)+Nρ(x¯k+1+u¯k)uik+1=uik+xik+1−zk+1

### 4.3 一般形式的一致性优化问题（切割参数到各子系统，但各子系统目标函数参数维度不同，可能部分重合）

上述全局一致性优化问题，我们可以看到，所做的处理不过是对数据分块，然后并行化处理。但是更一般的优化问题是，参数空间也是分块的，即每个子目标函数fi(xi)fi(xi)的参数维度不同xi,∈Rnixi,∈Rni，我们称之为局部变量。而局部变量所对应的的也将不再是全局变量zz，而是全局变量中的一部分zgzg，并且不是像之前的顺序对应，而可能是随便对应到zz的某个位置。可令g=(i,⋅)g=G(i,⋅)，即将xixi映射到zz的某部位

(xi)j=z(i,j)=ẑ i(xi)j=zG(i,j)=z^i

如果对所有ii有(i,j)=jG(i,j)=j，那么xixi与zz就是顺序映射，也就是全局一致性优化问题，否则就不是。结合下图就比较好理解

![consensus](https://1xji9q.bn1302.livefilestore.com/y2pcd30upmFtOnC91A1u4CJcxhIVfTuXrreyXRitG-WDxP0LroMiGrrhsnJ_He5ZfwuaFgVJYD-z6kjMfN0-NivkgIh4VspDA9v71PRfPQ0NTU/consensus.png)

虽然如果用其他方法来做感觉会复杂，但是纳入到上述ADMM框架，其实只不过是全局一致性优化问题的一个局部化变形，不过此时不是对数据进行分块，是对参数空间进行分块

mins.t.∑Ni=1fi(xi)+g(z),xi∈Rnixi−ẑ i=0,i=1,…N⟹xk+1izk+1yk+1i=argminx(fi(xi)+(yki)Txi(ρ/2)‖xi−ẑ ki‖22))=argminz(∑i=1N(−(yki)Tẑ i+(ρ/2)‖xk+1i−ẑ i‖22)))=yki+ρ(xk+1i−ẑ k+1i)min∑i=1Nfi(xi)+g(z),xi∈Rnis.t.xi−z^i=0,i=1,…N⟹xik+1=arg⁡minx(fi(xi)+(yik)Txi(ρ/2)‖xi−z^ik‖22))zk+1=arg⁡minz(∑i=1N(−(yik)Tz^i+(ρ/2)‖xik+1−z^i‖22)))yik+1=yik+ρ(xik+1−z^ik+1)

后续想做平均化处理，即中间会发生重合的参数zizi取值一样的，那么zz-update将只能找他对应的那些xx进行平均化，也就是变成局部了，因为不是所有值都是要全局保持一致的。比如上面那个图中的z1,z2,z3,z4z1,z2,z3,z4都分别只要求在部分xixi发生了共享需要保持一样，而不是像之前全局要求每个xixi对应的都是zz。即

zk+1g=∑(i,j)=g((xk+1i)j+(1/ρ)(yki)j)∑(x,y)=g1zgk+1=∑G(i,j)=g((xik+1)j+(1/ρ)(yik)j)∑G(x,y)=g1

该式子表示就是zz的第gg个变量的平均值来源于所有映射到该变量的xx与yy的平均值。与之前的global类似，此时对于yy的取均值会为0，因此zz-update就变成了更简单的形式

zk+1g=1kg∑(i,j)=g(xk+1i)zgk+1=1kg∑G(i,j)=g(xik+1)

同全局一致性优化问题一样，我们可以加上正则项，然后也可以变成带正则项的一般形式的一致性优化问题。此处不赘述，与全局基本类似。

## Sharing

### 4.4 共享问题（sharing）（横向切割数据，也可纵向切变量）

与之前的全局变量一致性优化问题类似，共享问题也是一个非常一般而且常见的问题。他的形式如下：

min∑i=1Nfi(xi)+g(∑i=1Nxi)min∑i=1Nfi(xi)+g(∑i=1Nxi)

这里的第一部分局部损失fi(xi)fi(xi)与全局一致性优化是一样的，即所有的xi∈Rn,i=1,…,Nxi∈Rn,i=1,…,N同维度，而对于一个共享的目标函数gg则是新加入的。在实际中，我们常常需要优化每个子数据集上的损失函数，同时还要加上全局数据所带来的损失；或者需要优化每个子系统的部分变量，同时还要优化整个变量。共享问题是一个非常重要而灵活的问题，它也可以纳入到ADMM框架中，形式如下：

mins.t.∑Ni=1fi(xi)+g(∑Ni=1zi)xi−zi=0,zi∈Rn,i=1,…,N,⟹xk+1izk+1uk+1i=argminxi(fi(xi)+(ρ/2)‖xi−zki+uki‖22))=argminz(g(∑i=1Nzi)+ρ/2∑i=1N‖zi−xk+1i−uki‖22)=uki+xk+1i−zk+1imin∑i=1Nfi(xi)+g(∑i=1Nzi)s.t.xi−zi=0,zi∈Rn,i=1,…,N,⟹xik+1=arg⁡minxi(fi(xi)+(ρ/2)‖xi−zik+uik‖22))zk+1=arg⁡minz(g(∑i=1Nzi)+ρ/2∑i=1N‖zi−xik+1−uik‖22)uik+1=uik+xik+1−zik+1

上述形式当然还不够简洁，需要进一步化简。因为xx-update可以不用担心，分机并行处理优化求解即可，而对于zz-update这里面需要对NnNn个变量求解，想加快速度，就减少变量个数。于是想办法通过和之前那种平均方式一样来简化形式解。

对于zz-update步，令ai=uki+xk+1iai=uik+xik+1，于是zz-update步优化问题转化为

mins.t.g(Nz¯)+(ρ/2)∑Ni=1‖zi−ai‖22z¯=1N∑Ni=1ziming(Nz¯)+(ρ/2)∑i=1N‖zi−ai‖22s.t.z¯=1N∑i=1Nzi

当z¯z¯固定时，那么后面的最优解（类似回归）为zi=ai+z¯−a¯zi=ai+z¯−a¯，带入上式后于是后续优化就开始整体更新（均值化）

xk+1izk+1uk+1=argminxi(fi(xi)+(ρ/2)‖xi−xki+x¯k−z¯k+uk‖22))=argminz(g(Nz¯)+Nρ/2‖z¯−x¯k+1−uk‖22)=uki+x¯k+1−z¯k+1xik+1=arg⁡minxi(fi(xi)+(ρ/2)‖xi−xik+x¯k−z¯k+uk‖22))zk+1=arg⁡minz(g(Nz¯)+Nρ/2‖z¯−x¯k+1−uk‖22)uk+1=uik+x¯k+1−z¯k+1

另外，有证明如果强对偶性存在，那么global consensus问题与sharing问题是可以相互转化的，可以同时达到最优，两者存在着很紧密的对偶关系。

本节开头提过，sharing问题用来切分数据做并行化，也可以切分参数空间做并行化。这对于高维、超高维问题是非常有好处的。因为高维统计中，大样本是一方面问题，而高维度才是重中之重，如果能切分特征到低纬度中去求解，然后在合并起来，那么这将是一个很美妙的事情。上面利用regularized global consensus问题解决了切分大样本数据的并行化问题，下面利用sharing思想解决常见的高维数据并行化问题

**切割变量（特征）空间，并行化处理**

同样假设面对还是一个观测阵A∈Rm×nA∈Rm×n和响应观测b∈Rnb∈Rn，此时有n»mn»m，那么要么就降维处理，要么就切分维度去处理，或者对于超高维矩阵，切分维度后再降维。此时AA矩阵就不是像之前横着切分，而是竖着切分，这样对应着参数空间的切分：

A=[A1,…,AN],Ai∈Rm×ni,x=(x1,…,xN),x∈Rni,→Ax=∑i=1NAixiA=[A1,…,AN],Ai∈Rm×ni,x=(x1,…,xN),x∈Rni,→Ax=∑i=1NAixi

于是正则项也可以切分为r(x)=∑Ni=1ri(xi)r(x)=∑i=1Nri(xi)。那么最初的minl(Ax−b)+r(x)minl(Ax−b)+r(x)形式就变成了

minl(∑i=1NAixi−b)+∑i=1Nri(xi)minl(∑i=1NAixi−b)+∑i=1Nri(xi)

这个与sharing问题非常接近了，做点变化那就是sharing问题了

mins.t.l(∑Ni=1zi−b)+∑Ni=1ri(xi)Aixi−zi=0,i=1,…,N⟹xk+1izk+1uk+1=argminxi(ri(xi)+(ρ/2)‖Aixi−Aixki+Ax⎯⎯⎯⎯⎯⎯k−z¯k+uk‖22))=argminz(l(Nz¯−b)+Nρ/2‖z¯−Ax⎯⎯⎯⎯⎯⎯k+1−uk‖22)=uki+Ax⎯⎯⎯⎯⎯⎯k+1−z¯k+1minl(∑i=1Nzi−b)+∑i=1Nri(xi)s.t.Aixi−zi=0,i=1,…,N⟹xik+1=arg⁡minxi(ri(xi)+(ρ/2)‖Aixi−Aixik+Ax¯k−z¯k+uk‖22))zk+1=arg⁡minz(l(Nz¯−b)+Nρ/2‖z¯−Ax¯k+1−uk‖22)uk+1=uik+Ax¯k+1−z¯k+1

与之前的global consensus问题相比，ADMM框架xx-update与zz-update似乎是反过来了。于是将此形式直接套到Lasso等高维问题即有很具体的形式解了。

**（1）Lasso**

xk+1iz¯k+1uk+1=argminxi(λ‖xi‖1+(ρ/2)‖Aixi−Aixki+Ax⎯⎯⎯⎯⎯⎯k−z¯k+uk‖22))=1N+ρ(b+ρAx⎯⎯⎯⎯⎯⎯k+1+ρuk)=uk+Ax⎯⎯⎯⎯⎯⎯k+1−z¯k+1xik+1=arg⁡minxi(λ‖xi‖1+(ρ/2)‖Aixi−Aixik+Ax¯k−z¯k+uk‖22))z¯k+1=1N+ρ(b+ρAx¯k+1+ρuk)uk+1=uk+Ax¯k+1−z¯k+1

当|ATi(Aixki+z¯k−Ax⎯⎯⎯⎯⎯⎯k−uk)|2≤λ/ρ|AiT(Aixik+z¯k−Ax¯k−uk)|2≤λ/ρ时xk+1i=0xik+1=0（第ii块特征不需要用），这样加快了xx-update速度,不过这个对串行更有效，对并行起来也没有多大用..

**（2）Group Lasso** 与lasso基本一样，只是在xx-update上有一个正则项的不同，有ℓ1ℓ1-norm变成了ℓ2ℓ2-norm

xk+1i=argminxi(λ‖xi‖2+(ρ/2)‖Aixi−Aixki+Ax⎯⎯⎯⎯⎯⎯k−z¯k+uk‖22)xik+1=arg⁡minxi(λ‖xi‖2+(ρ/2)‖Aixi−Aixik+Ax¯k−z¯k+uk‖22)

该问题其实就是按组最小化(ρ/2)|Aixi−v|22+λ|xi|2(ρ/2)|Aixi−v|22+λ|xi|2，解为

if‖ATiv‖2≤λ/ρ,otherwisethenxi=0xi=(ATiAi+vI)−1ATivif‖AiTv‖2≤λ/ρ,thenxi=0otherwisexi=(AiTAi+vI)−1AiTv

涉及矩阵长短计算时，再看矩阵小技巧。

**（3）Sparse Logstic Regression** 也与lasso区别不大，只是zz-update的损失函数不同，其余相同于是

z¯k+1=argminz¯(l(Nz¯)+(ρ/2)‖z¯−Ax⎯⎯⎯⎯⎯⎯k+1−uk‖22)z¯k+1=arg⁡minz¯(l(Nz¯)+(ρ/2)‖z¯−Ax¯k+1−uk‖22)

**（4）SVM**

SVM与之前的global consensus时候优化顺序反了过来，与logistic rgression只是在zz-update步不同（损失函数不同）：

xk+1iz¯k+1uk+1=argminxi(λ‖xi‖22+(ρ/2)‖Aixi−Aixki+Ax⎯⎯⎯⎯⎯⎯k−z¯k+uk‖22))=argminz¯(1T(Nz¯+1)++(ρ/2)‖z¯−Ax⎯⎯⎯⎯⎯⎯k+1−uk+1‖)=uk+Ax⎯⎯⎯⎯⎯⎯k+1−z¯k+1xik+1=arg⁡minxi(λ‖xi‖22+(ρ/2)‖Aixi−Aixik+Ax¯k−z¯k+uk‖22))z¯k+1=arg⁡minz¯(1T(Nz¯+1)++(ρ/2)‖z¯−Ax¯k+1−uk+1‖)uk+1=uk+Ax¯k+1−z¯k+1

zz-update解析解可以写成软阈值算子

(z¯k+1)i=⎧⎩⎨⎪⎪vi−N/ρ,−1/N,vi,vi>−1/N+N/ρvi∈[−1/N,−1/N+N/ρ]vi<−1/Nvi=(Ax⎯⎯⎯⎯⎯⎯k+1+u¯k)i(z¯k+1)i={vi−N/ρ,vi>−1/N+N/ρ−1/N,vi∈[−1/N,−1/N+N/ρ]vi,vi<−1/Nvi=(Ax¯k+1+u¯k)i

**（5）Generalized Additive Models**

广义可加模型是一个很适合sharing框架的问题。它本身就是对各个各个特征做了变化后（非参方法），重新表示观测的方式

b≈∑j=1nfj(xj)b≈∑j=1nfj(xj)

当fifi是线性变化时，则退化成普通线性回归。此时我们目标优化的问题是

min∑i=1mli(∑j=1nfj(xij)−bi)+∑j=1nrj(fj)min∑i=1mli(∑j=1nfj(xij)−bi)+∑j=1nrj(fj)

其中有mm个观测，nn维特征（变量）。rjrj此时是对一个functional的正则，此时这个问题看起来似乎既可以对数据切分，也可以对特征切分，不过此时仍用sharing问题来做，相当于对特征切分为一个特征为一个子系统，于是有

fk+1jz¯k+1uk+1=argminfi∈j(rj(fj)+(ρ/2)∑i=1m(fj(xij)−fkj(xij)+z¯ki+f¯ki)+uki=argminz¯(∑i=1mli(Nz¯−bi)+ρ/2∑j=1n‖z¯−f¯k+1−uk‖,f¯k=1n∑j=1nfkj(xij)=uk+f¯k+1−z¯k+1fjk+1=arg⁡minfi∈Fj(rj(fj)+(ρ/2)∑i=1m(fj(xij)−fjk(xij)+z¯ik+f¯ik)+uikz¯k+1=arg⁡minz¯(∑i=1mli(Nz¯−bi)+ρ/2∑j=1n‖z¯−f¯k+1−uk‖,f¯k=1n∑j=1nfjk(xij)uk+1=uk+f¯k+1−z¯k+1

fjfj是一个ℓ2ℓ2正则的损失，有直接求解的算法求解，zz可以一块一块的求解？

最后再说一个经济学中很重要的sharing问题的特例，即交换问题（exchange problem）：

mins.t.∑Ni=1fi(xi)∑Ni=1xi=0,xi∈Rn,i=1,…Nmin∑i=1Nfi(xi)s.t.∑i=1Nxi=0,xi∈Rn,i=1,…N

此时共享目标函数g=0g=0。xixi可以表示不同物品在NN个系统上的交换数量，(xi)j(xi)j可以表示物品jj从子系统ii上收到的交换数目，约束条件就可以看做在这些系统中物品交换是保持均衡稳定的。于是转化为sharing问题，就有很简单的ADMM解法（或者当做之前讲过的受约束的凸优化问题来解，做投影）：

xk+1iuk+1=argminxi(fi(xi)+(ρ/2)‖xi−xki+x¯k+uk‖22))=uki+x¯k+1xik+1=arg⁡minxi(fi(xi)+(ρ/2)‖xi−xik+x¯k+uk‖22))uk+1=uik+x¯k+1

### 4.4 应用小总结

感觉上通过consensus problem和general consensus problem，我们可以看到并行和分布式部署优化方案的可行性。我们可以切分数据以及相应的目标函数，也可以切分变量到各个子系统上去，分别作优化，甚至我们可以大胆想象对不同类型数据块用不同的优化算法，结合consensus问题和ADMM算法，达到同一个global variable的优化目的；或者对不同变量在不同类型数据块上优化，即使有重叠，也可以结合general consensus思想和ADMM算法来解决这个问题。当然前提是能够定义好需要估计的参数和优化的目标函数！大规模部署的前景还是很不错的。下面具体分布式统计模型的构建便是ADMM算法非常好的应用。切分数据、切分变量（不过每个子系统的目标函数基本都是一样的，其实应该可以不同）

## 5. Nonconvex问题

### 5.1 变量选择（Regressor Selection）

### 5.2 因子模型（Factor Model Fitting）

### 5.3 双凸优化（Bi-convex Problem）

非负矩阵分解（Nonnegative Matrix Factorization）

## 6. 具体实施与实际计算结果

这块真的很实际，需要明白MPI的机理和Mapreduce、Graphlab等通信运作的机理，这样才好部署ADMM算法，因为中间有很多迭代，需要做好子节点间参数与整体参数的通信，保持迭代时能同步更新参数。看实际运作，MPI和GraphLab可能更适合这种框架，Hadoop也是可以的，不过毕竟不是为迭代算法所生，要做好需要进行一些优化。Boyd提到Hadoop其中的Hbase更适合这种框架，因为Hbase是一种大表格，带有时间戳，适合记录迭代的记录，这样就不容易导致分布计算时候搞不清是哪一步的迭代结果了，导致通信调整比较复杂。不过在MapReduce框架下实施ADMM算法是没有什么问题的，只要熟稔Hadoop的一些细节部分，基本没有太大问题。

## 8. 总结

一个好的一般性算法，我个人觉得是易实施，并可大规模应用许多问题。可以让统计学家卡在搞算法的瓶颈中解放出来，使得他们能快速用模拟，验证自己构建可能较为复杂的模型。只有当看到一个令人感到欣慰的结果时，那些模型的统计性质的证明才可能是有意义的，如果事先连希望都看不到，那证明起来都可能底气不足，让人难以信服，更难以大规模应用统计学家所构建的模型。现在是一个高维数据、海量数据的年代，算法的重要性更会凸显出来，一个好的模型如果没有一个有效的算法支撑，那么他将可能什么都不是，Lasso头几年所遭遇的冷遇也充分证明了这一点，再比如在没有计算机年代，Pearson的矩估计应用反而远多于Fisher的MLE估计方法也是一个道理。好的一般性的解决方案，我想这不管是优化理论，还是统计等其他应用学科，虽然知道没有最牛最终极的方法，但是能涌现一些大范围适用的方法，那就是再好不过了。一招鲜吃遍天，人还都是喜欢简单、安逸爱偷懒的嘛..