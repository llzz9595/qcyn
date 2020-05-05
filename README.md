### 之前看到一位兄dei搞cxk的人体关键点检测挺有趣，想着是不是可以基于人体关键点检测，加个一个差异计算可以评估青你选手主题曲的动作标准度。也不知道行不行先试试！
## 目的：
- 由于本人沉迷青你无法自拔，pick谢可寅跟安崎，想通过AI各种手段，包括NLP、CV等，看看我pick的选手是不是既有实力又有观众缘
- 说到实力，我打算使用关键点检测的模型，跟踪两位选手的骨骼走向，以安崎为标准，谢可寅为测试者，通过对比舞蹈得出实力评估值0-1
- 然后观众缘，主要通过微博的评论，经过统计与情感分析获得选手的主要关注内容以及大众对她的好感度
- 最后，据说风格迁移模型搞出来挺炫酷的，我也掺一脚

## 具体思路：

主要以安崎小甜心的主题曲直拍视频作为标准视频，谢可寅shaking的主题曲直拍视频作为测试视频

1. 将标准视频进行爬取，将视频逐帧读取成图片;
2. 由于图片背景光线不太利于关键点捕抓，利用deeplabv3p_xception65_humanseg进行抠图处理;
3. 基于pose_resnet50_mpii模型进行关键点检测并存储检测结果;
4. 然后对测试视频作同样处理存储检测结果;
5. 基于单通道的直方图对标准检测结果集以及测试检测结果集进行图片相似度计算，取结果均值作为选手的主题曲实力值;
6. 将获取的实力值合成到选手图片并输出
7. 爬取微博测试选手相关评论，统计高频词输出图
8. 对评论进行情感分析，输出总体评论积极与消极的对比饼图 


![示例](https://ai-studio-static-online.cdn.bcebos.com/2cc2c32beba74d638cc6a717bb58951b0dfe93bb1aa34117808838cc631aee23)![](https://ai-studio-static-online.cdn.bcebos.com/b9837de36aa348e695a6a5bea7a26666c85836428ab0429a8a33d6ab1ed7a830)


![](https://ai-studio-static-online.cdn.bcebos.com/c6b78c1b6688474e8c76cf93b64dbaa172021ef638ed48dcadc09d00c9cff9e8)

### 风格迁移
实现原视频画面风格转换成水墨风格，对视频的每一帧画面进行风格迁移处理，然后重新合成视频加上原视频音频后，重新输出

![](https://ai-studio-static-online.cdn.bcebos.com/0111c40988394fbbaf9c08702a00bf97a3f4de017f8a46c78f66894f2424ba1c) ![](https://ai-studio-static-online.cdn.bcebos.com/b60066bb579a44c1b2ca568499c6e0812688a493df6e4c8fa8e1c8f83a5bd783)


**成片**

![](https://ai-studio-static-online.cdn.bcebos.com/35ec87217aa644e3b53200e5fd5ef2b4995bc319df3b49f78976c92675034c60)






 



