# channel_pruning_caffe

简介：使用贪心算法，逐层对channel进行剪枝，并对剪枝完的model进行fine-tune

channel剪枝流程如下：

（1）统计可剪枝的层： 获取可剪枝层的group（备注：对于resnet的结构，存在多层共剪的情况，将其分为一组）

（2）剪枝channel的选取：逐层（可从上往下，也可从下往上）对当前层的不同channel进行判别，选取需要剪枝的channel（按照事先给定的比例，向上取整选取剪枝个数，并判定剪枝数是否超过原channel数），目前实现2种channel选取方式，后续会增加更多高端算法，并进行对比试验

（3）对模型进行剪枝：根据选取的剪枝channel，进行具体裁剪，并保存对应的test.prototxt,train.prototxt,solver.prototxt以及剪枝完的weights.caffemodel

（4）对剪枝后的model进行fine-tune：差不多1.5个epoch，加入了简易的callback机制，即连续n次训练的loss的平均值低于某定值时，判定为收敛，提前跳出训练

（5）是否剪枝的判定：对fine-tune完的模型进行判定，若整体loss低于某给定值时，判定该层可以进行剪枝，否则不剪枝。

（6）循环2~5，直到完成

备注：test为测试剪枝model的实际检测效果

代码未整理，有空补充

