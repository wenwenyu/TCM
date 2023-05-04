# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# cudnn_benchmark = True
find_unused_parameters = True

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1200)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(interval=8000, metric='mIoU')
# total_epochs = 1200

custom_imports = dict(
    imports=['ocrclip', 'datasets', 'hooks', 'optimizer'],
    allow_failed_imports=False)

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'