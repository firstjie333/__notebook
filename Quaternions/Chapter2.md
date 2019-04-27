#Rotations and cross-relations

![image-20190313140332835](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190313140332835-2457012.png)

![image-20190313170146337](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190313170146337-2467706.png)

向量x 绕着u轴旋转 $\phi$ 或者新的向量， 可以通过（54）式子获得



##### 2.2 The rotation group SO(3) 

![image-20190314110914614](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314110914614-2532954.png)

![image-20190314111015683](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314111015683-2533015.png)



![image-20190314111025447](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314111025447-2533025.png)

![image-20190314110620981](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314110620981-2532781.png)







##### 2.3.1 The exponential map

![image-20190314111811368](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314111811368-2533491.png)

-----

在文档里面：

==Exp 表达和向量到矩阵，$Exp(\phi)$ 通常里面是一个向量==

==exp 表示对矩阵做指数映射， $exp([\phi]_x) = exp(\phi ^{\wedge }) )$== 

所以： ![image-20190314113339461](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314113339461-2534419.png)

In the following sections we’ll see that the vector , called the rotation vector or the  angle-axis vector, encodes through $\phi = w \Delta t = \phi u$ ,  the angle $\phi$  and axis $u$ of rotation. 

#### 2.3.3 Rotation matrix and rotation vector: ==the Rodrigues rotation formula==

==旋转矩阵的定义，可以通过旋转向量 通过叉积 的指数映射来丁来。对旋转向量进行泰勒展开==

![image-20190314113648650](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314113648650-2534608.png)

![image-20190314114031115](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314114031115-2534831.png)

#### 2.3.4 The logarithmic maps

对数映射的定义是指数映射的逆， 将旋转矩阵映射到旋转向量

==Log 表达矩阵到向量，$LOG(R)$ 通常里面是一个旋转矩阵==

==log 表示对矩阵做指数映射， $log(R)$ 里面通常还是一个旋转矩阵== 

![image-20190314114128602](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314114128602-2534888.png)

#### 2.3.5 旋转一个向量

![image-20190314115444764](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314115444764-2535684.png)

==归一化的四元素，其逆 = 其共轭 $q^{-1} = q^{*}$==



#### 2.4 The rotation group and the quaternion

![image-20190314135126336](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314135126336-2542686.png)



==旋转一个向量v： $r(v) = R v = q \times v \times q*$==

==旋转向量对应的四元数的约束条件是：$q \times q^* =1$==

![image-20190314140337030](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314140337030-2543417.png)

----

![image-20190314142237727](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314142237727-2544557.png)

![image-20190314142257082](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314142257082-2544577.png)



![image-20190314142705399](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314142705399-2544825.png)

![image-20190314142939742](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314142939742-2544979.png)

![image-20190314143101992](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314143101992-2545062.png)

- ==注意1/2 的差异==

![image-20190314162617152](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314162617152-2551977.png)

![image-20190314162851944](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314162851944-2552131.png)

![image-20190314162943461](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314162943461-2552183.png)

![image-20190314163048735](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314163048735-2552248.png)

![image-20190314163148562](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314163148562-2552308.png)

![image-20190314165829419](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314165829419-2553909.png)

![image-20190314165840452](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314165840452-2553920.png)

- ==四元数q可以用单位实四元数 旋转角度 $\theta$ 来定义==
- ==在空间中旋转绕着u轴旋转  $\phi$  可以将该向量用四元素($\theta = 1/2 \phi$ ） 旋转两次表示== 



![image-20190314170950341](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314170950341-2554590.png)

![image-20190314171001875](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314171001875-2554601.png)

![image-20190314171207065](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314171207065-2554727.png)

![image-20190314171732241](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314171732241-2555052.png)

![image-20190314171952417](/Users/test/Downloads/7-TestCode/__notebook/Quaternions/image-20190314171952417-2555192.png)

#### 球形线性插值Spherical linear interpolation

O__O "…看不懂

