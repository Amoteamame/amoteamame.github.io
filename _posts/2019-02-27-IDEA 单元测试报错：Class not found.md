---
title: IDEA 单元测试报错：Class not found
layout: post
tags: idea bug
categories: java
---
今天在maven多模块项目中，在其中一个module中，创建了一个测试类，在执行junit单元测试时，idea一直在报“**Class not found**”，即类找不到的错误。

可能是编译有问题导致找不到，但是就算Ctrl+Alt+Shift+S 打开项目配置，勾选集成项目编译输出目录即Inherit project compile output path，还是一样的问题。

这时我就在想，是不是项目走的maven的junit，多模块下，maven默认没有加载这个类，于是尝试右键maven ->test，这时它报一些模块没有install,即刚写的和改动的模块要重新install一下，改好后，再尝试用run,成功。