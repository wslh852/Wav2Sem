def get_model(cfg):
    ## old
    if cfg.arch == 'wav2bert':
        from models.wav2bert import Wav2Semconfig, Wav2Bert
        config = Wav2Semconfig()
        model= Wav2Bert(cfg)

    elif cfg.arch == 'stage1_BIWI':
        from models.stage1_BIWI import VQAutoEncoder as Model
        model = Model(args=cfg)
    elif cfg.arch == 'stage2':
        from models.stage2 import CodeTalker as Model
        model = Model(args=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model