# instantiate_model.py
# from mmcv import Config
from mmengine import Config
from mmdet.registry import MODELS

def main():
    # 加载配置文件
    cfg = Config.fromfile('projects/my_projects/configs/my_cfg.py')

    # 实例化模型
    model = MODELS.build(cfg.model)

    # 打印模型结构
    print(model)

if __name__ == '__main__':
    main()
