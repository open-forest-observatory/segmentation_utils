data_preprocessor = dict(
    type="SegDataPreProcessor",
    # This is kept the same as the imagery because there are masked regions making it lower
    mean=INSERT_MEAN,
    std=INSERT_STD,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1024, 1024),
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=INSERT_MEAN,
        std=INSERT_STD,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(1024, 1024),
    ),
    pretrained=None,
    backbone=dict(
        type="MixVisionTransformer",
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth",
        ),
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=INSERT_NUM_CLASSES,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=(1024, 1024), stride=(768, 768)),
)
dataset_type = "CityscapesArbitraryClassesDataset"
data_root = INSERT_DATA_ROOT
img_suffix = IMG_SUFFIX
classes = INSERT_CLASSES
crop_size = (1024, 1024)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize", scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                {"type": "Resize", "scale_factor": 0.5, "keep_ratio": True},
                {"type": "Resize", "scale_factor": 0.75, "keep_ratio": True},
                {"type": "Resize", "scale_factor": 1.0, "keep_ratio": True},
                {"type": "Resize", "scale_factor": 1.25, "keep_ratio": True},
                {"type": "Resize", "scale_factor": 1.5, "keep_ratio": True},
                {"type": "Resize", "scale_factor": 1.75, "keep_ratio": True},
            ],
            [
                {"type": "RandomFlip", "prob": 0.0, "direction": "horizontal"},
                {"type": "RandomFlip", "prob": 1.0, "direction": "horizontal"},
            ],
            [{"type": "LoadAnnotations"}],
            [{"type": "PackSegInputs"}],
        ],
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="CityscapesArbitraryClassesDataset",
        data_root=INSERT_DATA_ROOT,
        img_suffix=IMG_SUFFIX,
        classes=INSERT_CLASSES,
        data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(
                type="RandomResize",
                scale=(2048, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),
            dict(type="RandomCrop", crop_size=(1024, 1024), cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(type="PackSegInputs"),
        ],
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CityscapesArbitraryClassesDataset",
        data_root=INSERT_DATA_ROOT,
        img_suffix=IMG_SUFFIX,
        classes=INSERT_CLASSES,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
            dict(type="LoadAnnotations"),
            dict(type="PackSegInputs"),
        ],
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CityscapesArbitraryClassesDataset",
        data_root=INSERT_DATA_ROOT,
        img_suffix=IMG_SUFFIX,
        classes=INSERT_CLASSES,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
            dict(type="LoadAnnotations"),
            dict(type="PackSegInputs"),
        ],
    ),
)
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
    name="visualizer",
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
tta_model = dict(type="SegTTAModel")
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0),
        )
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-06, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1500, end=10000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=5000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth"
