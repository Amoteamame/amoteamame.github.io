---
title: IDEA maven插件的使用
layout: post
categories: java
tags: maven IDEA
---
今天看一篇博客配置多模块maven项目时，发现这样一张图：

![](https://upload-images.jianshu.io/upload_images/1616232-352b24b4125423e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/636/format/webp)

但是在我的IDEA中右键死活找不到这个小齿轮按钮，网上找了半天也找不到，如果是通过右侧的maven工程面板：

![](https://i.loli.net/2019/01/15/5c3dca2ed7f4a.png)

然后

![](https://i.loli.net/2019/01/15/5c3dca2f063a6.png)

虽说能实现同样效果，但是总感觉麻烦一点，为什么我的就没有呢？

这时候我猜测上面那个小齿轮按钮应该是个插件，集成的maven功能，于是我就尝试搜索了下：

![](https://i.loli.net/2019/01/15/5c3dcb5f0817e.png)

发现了一款插件叫做 Maven Run,安装重启后就有同样的功能。