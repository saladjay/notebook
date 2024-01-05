# Linux 掉显卡驱动的排查

## 查询显卡驱动

```shell
ls /usr/src | grep nvidia
## nvidia-515.76 
cat /proc/driver/nvidia/version
## NVRM version: NVIDIA UNIX x86_64 Kernel Module  515.76  Mon Sep 12 19:21:56 UTC 2022
## GCC version:  gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)
```

“##”后内容为命令输出

## 查看显卡驱动历史

```shell
cat /var/log/dpkg.log | grep nvidia
## 2023-04-04 09:57:35 install libnvidia-cfg1-470:amd64 <none> 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status half-installed libnvidia-cfg1-470:amd64 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status unpacked libnvidia-cfg1-470:amd64 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status unpacked libnvidia-cfg1-470:amd64 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 install libnvidia-common-470:all <none> 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status half-installed libnvidia-common-470:all 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status unpacked libnvidia-common-470:all 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status unpacked libnvidia-common-470:all 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 install libnvidia-compute-470:amd64 <none> 470.161.03-0ubuntu0.18.04.1
## 2023-04-04 09:57:35 status half-installed libnvidia-compute-470:amd64 470.161.03-0ubuntu0.18.04.1
## ...
## 2023-04-22 10:13:17 status half-configured libnvidia-encode-470:i386 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 status installed libnvidia-encode-470:i386 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 configure libnvidia-encode-470:amd64 470.182.03-0ubuntu0.18.04.1 <none>
## 2023-04-22 10:13:17 status unpacked libnvidia-encode-470:amd64 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 status half-configured libnvidia-encode-470:amd64 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 status installed libnvidia-encode-470:amd64 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 configure xserver-xorg-video-nvidia-470:amd64 470.182.03-0ubuntu0.18.04.1 <none>
## 2023-04-22 10:13:17 status unpacked xserver-xorg-video-nvidia-470:amd64 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 status half-configured xserver-xorg-video-nvidia-470:amd64 470.182.03-0ubuntu0.18.04.1
## 2023-04-22 10:13:17 status installed xserver-xorg-video-nvidia-470:amd64 470.182.03-0ubuntu0.18.04.1
```

### 查看占用显卡的程序

```shell
fuser -v /dev/nvidia*
##                      USER        PID ACCESS COMMAND
## /dev/nvidia0:        pc        23197 F.... python
## /dev/nvidiactl:      pc        23197 F.... python
```

可以根据PID杀掉占用显卡的程序

### 查看显卡模块

```shell
lsmod | grep nvidia
## nvidia_uvm           1126400  0
## nvidia_drm             61440  0
## nvidia_modeset       1142784  1 nvidia_drm
## nvidia              40812544  16 nvidia_uvm,nvidia_modeset
## drm_kms_helper        184320  1 nvidia_drm
## drm                   495616  4 drm_kms_helper,nvidia,nvidia_drm
```

### 重载显卡驱动模块

```shell
# 卸载驱动模块
kill -9 ${PID}
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
# 挂载驱动
sudo modprobe nvidia
# 检查是否安装成功
sudo nvidia-smi
```

### 禁用显卡自动升级

```shell
sudo apt-mark hold nvidia-driver-515
## nvidia-driver-515 set on hold.
```

### 查询显卡型号

```shell
lspci | grep -i vga
## 00:02.0 VGA compatible controller: Intel Corporation Device 4680 (rev 0c)
## 01:00.0 VGA compatible controller: NVIDIA Corporation Device 2484 (rev a1)
```

可以到[pci查询网站]([PCI devices (ucw.cz)](http://pci-ids.ucw.cz/mods/PC/10de?action=help?help=pci))进行查询型号，输入NVIDIA Corporation Device后的型号2484即可，2484是RTX3070

### 禁用nouveau驱动

 Ubuntu系统集成的显卡驱动程序是nouveau，它是第三方为NVIDIA开发的开源驱动，我们需要先将其屏蔽才能安装NVIDIA官方驱动。

```shell
vim /etc/modprobe.d/blacklist.conf
```

在blacklist.conf最后添加以下内容，然后保存退出

```shell
blacklist nouveau
options nouveau modeset=0
```

更新系统

```shell
sudo update-initramfs -u
reboot
```

验证nouveau是否禁用

```shell
lsmod | grep nouveau
```

没有输出信息表示已被禁用

### 新增驱动环境变量

```shell
vim ~/.bashrc
```

在文件末尾添加以下内容，保存退出

```shell
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH 
```

是环境变量生效

```shell
source ~/.bashrc
```

### 关闭图形界面

```shell
# 检查是否存在
ps aux | grep X
## pc       23408  0.0  0.0  16180  1008 pts/1    S+   13:02   0:00 grep --color=auto X

# 根据不同的桌面系统进行关闭
sudo /etc/init.d/gdm stop
sudo /etc/init.d/gdm status

sudo /etc/init.d/lightdm stop
sudo /etc/init.d/lightdm status
# 或者
sudo service lightdm stop
```

### 安装显卡驱动

```shell
# 卸载已有驱动
sudo apt-get remove nvidia-*
# 或者
sudo apt-get purge nvidia*

# 赋予run文件执行权限
sudo chmod a+x NVIDIA-Linux-x86_64-515.76.run

# 是否完全卸载驱动
sudo ./NVIDIA-Linux-x86_64-515.76.run --uninstall

# 安装驱动
sudo ./NVIDIA-Linux-x86_64-515.76.run --no-opengl-files
```

 安装的可选参数如下：

- **–no-opengl-files** 只安装驱动文件，不安装OpenGL文件。这个参数最重要

- **–no-x-check** 安装驱动时不检查X服务
- **–no-nouveau-check** 安装驱动时不检查nouveau
  后面两个参数可不加



在安装执行的过程中，有如下可选项：

The distribution-provided pre-install script failed! Are you sure you want to continue? 选择 yes 继
Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later? 选择 Yes 继续
Nvidia’s 32-bit compatibility libraries? 选择 No 继续
Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up. 选择 Yes 继续

### 使用DKMS生成显卡驱动模块

```shell
sudo dkms install -m nvidia -v 515.76
## Module nvidia/515.76 already installed on kernel 5.4.0-146-generic/x86_64
```

