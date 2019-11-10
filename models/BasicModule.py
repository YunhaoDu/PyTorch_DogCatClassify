import torch as t
import time

"""封装了nn,Module,提共快速加载和保存模型的接口"""
class BasicModule(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_name = str(type(self)) # 模型的默认名字

    # 加载指定路径的模型
    def load(self, path):
        self.load_state_dict(t.load(path))

    # 保存模型，默认使用“模型名+时间”作为文件名
    def save(self, name=None):
        # 如 AlexNet_20190710_23:57:29.pth
        if name is None:
            prefix = 'checkpoints/' + self.module_name + '_'
            name = time.strftime(prefix + '%y%m%d_%H;%M;%S.pth')
        t.save(self.state_dict(), name)
        return name
