from mindsponge import PipeLine
from mindsponge.common.utils import get_predict_checkpoint

# 训练相关路径
data_train_path = "../data/train"    # 待下载
model_save_path = "../output/MEGA_Assessment_train.ckpt"
# 推理相关路径
model_predict_path = "../output/MEGA_Assessment.ckpt"
# 模型训练过程
pipe = PipeLine(name="MEGAAssessment")
pipe.set_device_id(0)
pipe.initialize(key="initial_training", model_save_path=model_save_path)
pipe.train(data_train_path, num_epochs=1)
# 模型转换过程
'''
    training.ckpt: 训练得到的权重；
    48：msa堆叠层数；
    predict.ckpt：需要被转换成的预测权重
'''
get_predict_checkpoint(model_save_path, 48, model_predict_path)