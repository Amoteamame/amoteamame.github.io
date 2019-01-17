---
title: centos虚拟机Ping不通网关
layout: post
categories: 分布式
tags: linux Vmware
---
今天在VMware中安装了centos mini版本，安装完成后，用xshell连接一直连不上，本来以为是mini版本没有安装ssh server，于是就用命令：


    $ yum search ssh


结果报错：

> Could not retrieve mirrorlist 

网上说是**没有配置resolv.conf** ，http://blog.51cto.com/pickupcoke/1787832

按上述配完之后仍然是同样问题，我试了下ping百度，报的：




    ping: unknown host www.baidu.com


搜索相关问题，说是首先要ping通网关，按这个帖子解决：

[Linux：ping不通baidu.com](https://blog.csdn.net/qq_35370485/article/details/77844860)

结果还是有问题，就感觉可能是VMware配置有问题，按这个配置：

[VMware下配置Linux IP，解决Linux ping不通](https://happyqing.iteye.com/blog/1739289)

配置完，还是ping不通。

最后是自己没按网上通用的“**主机模式**”，自己该了网络配置：



![](https://i.loli.net/2019/01/17/5c3ffcc409f60.png)

虽然不知道为什么，但是现在xshell能连接上了，在此记录一下。