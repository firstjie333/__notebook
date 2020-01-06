### Install

https://blog.csdn.net/liu3612162/article/details/79538753

https://gperftools.github.io/gperftools/cpuprofile.html

PS : 注意看INSTALL 如果没有configure文件，需要 生成autogen.sh



```cpp
wget https://github.com/gperftools/gperftools/archive/gperftools-2.7.tar.gz
tar xvf gperftools-2.7.tar.gz
cd gperftools-2.7
./configure
make -j8
sudo make install
```





### use

```
env LD_PRELOAD="/home/your_name/tools/gperftools/lib/libprofiler.so" CPUPROFILE_FREQUENCY=100 CPUPROFILE=cpu_perf.prof /media/psf/Home/Downloads/1-work/0-RDB--Align/core/algorithm_sam/build/example/ServerExampleSlam 3 . .

```





### Analyzing Callgrind Output

Use [kcachegrind](http://kcachegrind.sourceforge.net/) to analyze your callgrind output:

```
% pprof --callgrind /bin/ls ls.prof > ls.callgrind
% kcachegrind ls.callgrind
```