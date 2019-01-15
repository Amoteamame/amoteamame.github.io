---
title: 'maven [ERROR] Maven execution terminated abnormally (exit code 1)'
layout: post
categories: java
tags: java maven
---
今天更新了一下idea最新版本后，创建了一个maven项目，在使用**Create from archetype**功能时，选择好之后，IDEA会报：

>  [ERROR] Maven execution terminated abnormally (exit code 1)

的错误，网上大部分答案都是说：

1.IDEA的maven设置中，user setting file没设对，但我设对了

![img](https://img-blog.csdn.net/20170628112306309?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveWFuZ2hhaWJvYm8xMTA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
2.Maven VM没设

![](https://img-blog.csdn.net/20170628112456622?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveWFuZ2hhaWJvYm8xMTA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

这个我确实没设，但是加上之后（记住加$），还是不行，报同样的错误

后来我想，之前我的IDEA版本（16年的版本），使用maven都没有问题，为什么今天更新完IDEA之后maven就一直出问题呢，我就尝试性的把上图的JRE换成了JDK1.8，后来结果成了



原来原因出在IDEA高版本上。