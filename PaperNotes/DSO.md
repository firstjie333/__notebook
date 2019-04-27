### DSO（Direct Sparse Odometry

高翔知乎博客： https://zhuanlan.zhihu.com/p/29177540

运行DSO：https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247485263&idx=1&sn=d0d50d674c5b6209c84acb2c4f618e1e&chksm=ec10b94bdb67305de5d5dd484c04d795aaf7657875fb7c83108091ff2617ce4f5f1e544404cd&scene=0#wechat_redirect

光度标定：http://www.cnblogs.com/luyb/p/6077478.html



求导：https://www.cnblogs.com/JingeTU/p/8203606.html





慕尼黑工业大学（Technical University of Munich, TUM）计算机视觉实验室的雅各布.恩格尔（Jakob Engel）博士

实验室主页见[Computer Vision Group](https://link.zhihu.com/?target=https%3A//vision.in.tum.de/)）

代码见：[JakobEngel/dso](https://link.zhihu.com/?target=https%3A//github.com/JakobEngel/dso)

其他SLAM：ORB、SVO、okvis



- 特征点法：通过最小化重投影误差来计算相机位姿与地图点的位置，**几何误差**
- 直接法则最小化光度误差（photometric error）。所谓**光度误差**是说，最小化的目标函数，通常由图像之间的误差来决定，而非重投影之后的几何误差。





- 直接法将数据关联（data association）与位姿估计（pose estimation）放在了一个统一的非线性优化问题中
- 特征点法则分步求解，即，先通过匹配特征点求出数据之间关联，再根据关联来估计位姿。这两步通常是独立的，在第二步中，可以通过重投影误差来判断数据关联中的外点，也可以用于修正匹配结果（例如[4]中提到的类EM的方法）

![preview](https://pic1.zhimg.com/80/v2-56f7f5b0f617e4434674f80e226eaf2c_hd.png)

由于这个原因，DSO会一直求解一个比较复杂的优化问题，我们很难将它划分为像特征点法那样一步一步的过程。DSO甚至没有“匹配点”这个概念。每一个三维点，从某个主导帧（host frame）出发，乘上深度值之后投影至另一个目标帧（target frame），从而建立一个投影残差（residual）。只要残差在合理范围内，就可以认为这些点是由同一个点投影的。从数据关联角度看，在这个过程中并没有a1-b1, a2-b2这样的关系，也可能存在a1-b1, a2-b1, a3-b1这样的情况。==但是DSO并不在意这些，只要残差不大，我们就看成是同一个点。这是很重要的一点。在特征点法中，我们可以找到一个地图点分别在哪些帧中被看到，乃至找到各帧中的图像描述子是什么；但在DSO中，我们会尝试把每个点投影到所有帧中，计算它在各帧中的残差，而并不在意点和点之间的一一对应关系。==



