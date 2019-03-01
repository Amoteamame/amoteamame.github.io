---
title: 'maven多模块项目报-BeanCreationException: Error creating bean with name'
layout: post
categories: 编程问题 java
tags: maven spring
---
在使用多模块的maven项目中，其中一个父模块包含了多个子模块，子模块集成了spring，在发布子模块时，报了**BeanCreationException: Error creating bean with name** 的错误，项目结构如下：

![](https://i.loli.net/2019/03/01/5c78d2dccec32.png)

controller类中，注入的service在idea中一直会有个红色下划线

![](https://i.loli.net/2019/03/01/5c78d314b79e5.png)

检查了spring各种注解以及类文件上的注解无误后，再次运行还是无果。

后来无意间点开web.xml发现少配置了东西。。。

加上后解决，总之就是在整个聚合工程中，总工程中的server模块也有tomcat功能，这一块我少配置了东西，而检查时只检查了子模块的web项目。