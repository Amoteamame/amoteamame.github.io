---
title: IDEA maven项目中引入了库，但是java类中无法引入
layout: post
categories: 后端技术
tags: idea maven
---
今天在多模块maven项目中，在其中一个模块的pom文件中引入了：

```
<!-- https://mvnrepository.com/artifact/net.oschina.zcx7878/fastdfs-client-java -->
<dependency>
    <groupId>net.oschina.zcx7878</groupId>
    <artifactId>fastdfs-client-java</artifactId>
    <version>1.27.0.0</version>
</dependency>

```

但是在测试代码中，IDEA一直无法引入，后来自己尝试maven install本地的jar包，然后在项目中pom引用本地库，但是idea还是引入不了包。

后来查到可能要更新本地索引：

![](https://i.loli.net/2019/01/28/5c4efb0899a7b.png)

更新完毕后，还是一样的问题。

后来又用了另一种:

在本地maven仓库找到加载不了的jar包文件位置，删掉其中_maven.repositories文件，然后重启下IDEA

还是不行，后来无意间点到了ignored Files，发现我的模块居然被勾了，也就是pom文件没有解析，随去掉：

![](https://i.loli.net/2019/01/28/5c4efb08b1cf2.png)

项目导包就正常了。