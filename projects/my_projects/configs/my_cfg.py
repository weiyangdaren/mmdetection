# configs/my_model/my_model_config.py

# 自定义导入配置
custom_imports = dict(
    imports=['projects.my_projects.my_models', 'projects.my_projects.my_backbone'],
    allow_failed_imports=False
)

model = dict(
    type='MyModel',  # 自定义模型的名称
    backbone=dict(
        type='MyBackbone',
        in_channels=3,
        out_channels=64,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU')  # 指定激活函数的类型
    ),
    # 其他组件配置...
)
