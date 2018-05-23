---
title: springboot的pom文件出错
layout: post
categories: 编程问题
tags: 'bug,springboot'
---
今天学习springboot时使用[SPRING INITIALIZR](http://start.spring.io/)工具产生基础项目，下载后导入到IDEA，注意导入时不要用File->Open方式导入，这样导入的话，不会识别为一个工程文件，要用：

1. 菜单中选择File–>New–>Project from Existing Sources...
2. 选择解压后的项目文件夹，点击OK
3. 点击Import project from external model并选择Maven，点击Next到底为止。

这样导入后就是一个正常的maven项目了，然后发现pom.xml文件报红，发现<project>标签就开始报错，提示信息：<u>*failed to read artifact descriptor for*</u>
打开file->setting->maven发现maven home路径用的idea默认的路径。。。
我记得以前配置过路径，不是很清楚为什么导入新项目时还原了，选择本地路径后，pom文件就正常了。