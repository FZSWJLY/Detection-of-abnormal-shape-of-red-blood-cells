# 【AI达人创造营】基于PaddleDetection的红细胞形状异常检测
此项目是基于PaddleDetection做的红细胞形状异常检测，属于医学中目标检测类的项目。

# 一、项目背景
1. 此项目可以帮医生减轻负担，提高检测的准确性和速度。一般每张红细胞形状异常检测图都有好多红细胞，并且这个样本量还是成千上万的，这时候医生容易眼花缭乱，甚至可能漏掉一些重叠的红细胞。
2. 并且可以让医生避免重复的不必要的工作。
3. 将医学与AI科技结合，让科技为医学赋能。

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/87fd91e75dfa4f5f98baac5d807f741b13a0febf6e2a42e9b65eebc8d8b00a5f" width = "50%" height = "50%" />


```python
# 1.将PaddleDetection套件进行解压，路径看具体情况。

!unzip -oq /home/aistudio/data/data102742/PaddleDetection-2.1.0.zip
```

# 二.项目使用的套件
此项目使用的是PaddleDetection套件，下载地址为：<br>
github：[https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)（因为国外网站，访问比较慢，不太推荐）；<br>
飞桨AI Studio平台上：[https://aistudio.baidu.com/aistudio/datasetdetail/102742](https://aistudio.baidu.com/aistudio/datasetdetail/102742)（考虑到大部分用户访问github比较慢，故我从github上搬运过来）。


```python
# 2.将PaddleDetection套件进行改名

!mv PaddleDetection-2.1.0 PaddleDetection 
```

# 三、数据集简介
此项目使用了AI Studio平台上RBC数据集，一共有366条数据，数据是xml格式的图片标注文件。

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/a0a281a84b904c139dc492c73730caf48c511fdbaca342efa2e8a5d7a69c4c7e" width = "50%" height = "50%" />
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/6cd5f571b511499e881b8f29089448a71ef3b45a82c446fbbdb655b700ee06d6" width = "50%" height = "50%" />
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/02db5b5b30484f5b8d8440ea51f172f171952724cd0941049fd2a1c92abe41aa" width = "50%" height = "50%" />
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/4fd0e1c432cb46718db854ae3e257bdc6289cce41d4743819792bc0d57384f29" width = "50%" height = "50%" />
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/6366d68cceb34cec80fc03ca7b9e0b9677ae758bdbbb4214a6922a8d63ba69ba" width = "50%" height = "50%" />


```python
# 1.解压数据集

!unzip -o data/data85839/RBC.zip -d PaddleDetection/dataset/
```


```python
# 2.数据集文件夹改名

!mv PaddleDetection/dataset/RBC PaddleDetection/dataset/test_det
```


```python
# 3.将数据集划分为训练集和验证集

import os 
import random

# 类别数量 
file_saved = []   # 保存数据

random.seed(2021)

# voc数据路径问题
# 根目录信息，子目录信息，files_img--该文件夹下的文件名称
for _, _, files_img in os.walk('PaddleDetection/dataset/test_det/JPEGImages'):
    random.shuffle(files_img)

    for _, _, files_xml in os.walk('PaddleDetection/dataset/test_det/Annotations'):
        # indexs = 0
        # 1.jpg
        # 1.xml
        for i in range(len(files_img)):  # 遍历图片文件--一张一张的
            for j in range(len(files_xml)):
                # 匹配，与图片前缀名称一致的xml文件
                # 前缀是否一致
                if files_img[i][:-4] == files_xml[j][:-4]:
                    # 图片的相对路径 + 空格 + 标注文件的相对路径 + '\n'
                    # jpeg, img -- join -> jpeg/img
                    # JPEGImages/files_img[i]
                    file_maked = os.path.join('JPEGImages', files_img[i]) + ' ' + os.path.join('Annotations', files_xml[j])  + '\n'
                    
                    file_saved.append(file_maked)          # 每一个类别放在对应的缓存空间中
                    break

# example: 图片的相对路径 + 空格 + 标注文件的相对路径 + '\n'

# 训练集的划分
# 训练集占80%的数据
# 验证集/评估数据集：1-80% = 20%
Train_percent = 0.8

# train.txt保存
with open('PaddleDetection/dataset/test_det/train.txt', 'w') as f:
    # int(Train_percent * len(file_saved))
    # final_index = int(len(file_saved)*Train_percent) - 1
    f.writelines(file_saved[:int(len(file_saved)*Train_percent)])  # 写入多行数据
    print('train.txt Has Writed {0} records!'.format(len(file_saved[:int(len(file_saved)*Train_percent)])))
    
# eval.txt保存
with open('PaddleDetection/dataset/test_det/eval.txt', 'w') as f:
    # final_index + 1 == int(len(file_saved)*Train_percent)
    f.writelines(file_saved[int(len(file_saved)*Train_percent):])
    print('eval.txt Has Writed {0} records!'.format(len(file_saved[int(len(file_saved)*Train_percent):])))
```

输出结果为：
> train.txt Has Writed 274 records!<br>
eval.txt Has Writed 69 records!<br>

训练集样本量: 274，验证集样本量: 69

# 三、模型选择、开发、训练和验证
此项目使用的是ppyolo中的ppyolo_r18vd_coco.yml模型。

<h2>1.新建label_list.txt文件</h2>
在dataset/test_det下新建label_list.txt，然后在里面写上你的标注类型（如本项目中的RBC，意思为只标注了红细胞）。
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/9a68387d934c473c845d9e58805ff970fb0a1dccfc614226bf596c584d61e839" width = "50%" height = "50%" />
<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/5d78a02cdc4d460da7afb061e50da581cf37b28e45e84af39fe17da0f85cdaf1" width = "50%" height = "50%" />


```python
# 2.下载依赖库

%cd /home/aistudio/PaddleDetection/
!pip install -r requirements.txt
```

%cd + 路径:进入某路径下。

<h2>2.调参</h2>
<h3>a.修改voc.yml文件中的配置</h3>
将num_classes改为你的标注类型数。（如本项目只标注了红细胞这一种类型，故num_clases为1。若不为ppyo或ssd，如果为fast rcnn检测模型的话，就需要增加一个背景类别。）<br>
将TrainDatest下的dataset_dir改为你的数据集放置路径（如本项目中的dataset/test_det,此处路径为相对路径，下同），anno_path改为你标注文件的训练集文件（如本项目中的train.txt）。<br>
将EvalDataset下的dataset_dir改为你的数据集放置路径如（本项目中的dataset/test_det），anno_path改为你标注文件的验证集文件（如本项目中的eval.txt）。
将TestDataset下的anno_path改为你标注文件的测试集文件（如本项目中的dataset/test_det/label_list.txt）。
<h3>b.修改ppyolo_r18vd_coco.yml文件中的配置</h3>
因为本项目中用的数据集为voc格式，所以要将_BASE_下的第二行代码用Ctrl+/注释掉，将它用Ctrl+C和Ctrl+V复制粘贴修改为'../dataset/voc.yml'。若数据集为coco格式，则不用改。<br>
snapshot_epoch为迭代轮次以及参数保存轮次、周期，这要根据你的具体样数本来决定，可以先保持默认数值（本项目数据为366条，就先设为默认的10）。<br>
TrainReader下的batch_size为批次大小，这得看具体的样本数，最好为2的n次方（本项目数据不太大，所以把默认的32调成了8）。
EvalReader下的batch_size先改为1。<br>
将LearningRate学习率下的base_lr默认值修改为除以4后的值。


```python
# 3.模型训练

%cd /home/aistudio/PaddleDetection/
!python tools/train.py -c configs/ppyolo/ppyolo_r18vd_coco.yml\
--eval\
--use_vdl True
```

训练最终模型精度为77.86%(0.780)。

训练过程中指令解释：
1.  --c:指定配置文件。
2.  --eval:边训练边验证。<br>
3.  --use_vdl True:使用VisualDL记录数据，进而在VisualDL面板中显示。<br>
4. !python + 某路径下的python文件:执行某python文件。
训练中出现的问题解决方法：
1. 断次问题<br>
如果你的模型训练不小心断在了某个轮次，没训练完，可以使用 -r output/模型的yml文件/停在的轮次数（如果你一共要训练200轮次，却停在第20轮次，用的是ppyolo_r18vd_coco模型，你可以使用 -r output/ppyolo_r18vd_coco.yml/20继续进行训练）。<br>
2. 指令多的问题<br>
只要后面有指令，可以在每个指令最末尾后加\（\前不能加空格，最后一个指令末尾不用加\）。


```python
# 四、模型预测

%cd /home/aistudio/PaddleDetection/
!python tools/infer.py -c configs/ppyolo/ppyolo_r18vd_coco.yml\
-o weights='output/ppyolo_r18vd_coco/model_final.pdparams'\
--infer_dir 'dataset/test_det/JPEGImages'\
--output_dir 'output'\
--draw_threshold 0.1\
--save_txt True
```

-o:设置或更改配置文件里的参数内容<br>
--infer_dir:用于预测的图片文件夹路径<br>
--output_dir:预测后结果或导出模型保存路径<br>
--draw_threshold:可视化时分数阈值<br>
--save_txt:是否在文件夹下将图片的预测结果保存到文本文件中

预测图其中一张:

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/fc9dc83de0df4a558c19dfb6098a3eb2d8d3745f66724f06bc1e74cf961714de" width = "50%" height = "50%" />

# 五、总结与升华

项目最重要的一步就是训练，训练离不开调参，你得根据具体情况来调整参数。在这过程中，把Train_Reader中的batch_size、snapshot_epoch调高，LearningRate下的base_lr调低，loss越来越大。评测模型也很重要。<br>

希望这个项目可以帮助到大家。

# 个人简介
大家好，我是初学者，第一次做AI Studio项目，请大家多多支持。欢迎fork、喜欢、分享。本人能力有限，经验不足，若有不足，欢迎指正，谢谢大家！<br>
我在AI Studio上获得白银等级，点亮3个徽章，来互关呀~<br>
[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/691883](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/691883)
