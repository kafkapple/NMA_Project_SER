import os

class Config:
    
    NUM_EPOCHS = 5 # number of epochs per 1 training (swwep)
    N_SWEEP=10 # number of sweep for hyperparameter search
    
    SEED = 2024 # random seed
    
    ### Hyperparam - Initial default value
    BATCH_SIZE = 32
    MODEL = "v2"
    lr = 0.001
    DROPOUT_RATE = 0.5
    ACTIVATION = "relu"
    #self.relu = nn.LeakyReLU(negative_slope=0.01)
    
    N_STEP_FIG = 2 # log visualization per step
    
    PROJECT_DIR="NMA_Project_SER"
    
    BASE_DIR = os.path.join(os.getcwd(), PROJECT_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_NAME= "RAVDESS_audio_speech"
    MODEL_NAME=f"wav2vec_{MODEL}_sweep"#'wav2vec_v3_1'
    MODEL_DIR = os.path.join(BASE_DIR, 'model', MODEL_NAME)
    
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f'best_model_{MODEL_NAME}.pth')
    CKPT_SAVE_PATH = os.path.join(MODEL_DIR, f'checkpoint_{MODEL_NAME}.pth')
    
    LABELS_EMOTION = {
        0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
        4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
    }
    
    ####### Wandb config
    WANDB_PROJECT = f"{PROJECT_DIR}_sweep_together_v1_e5"
    ENTITY="biasdrive-neuromatch"
    WANDB_NAME = MODEL_NAME
    # 2: Define the search space
    
    CONFIG_SWEEP = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val.loss"},
        "parameters": {
            "BATCH_SIZE": {"values": [32, 64]},
            "MODEL":{"values":['v1', 'v2']},
            "lr": {"values": [0.0005, 0.001, 0.005]},
            "DROPOUT_RATE": {"values": [0.4, 0.5, 0.6]},
            "activation":{"values":['relu', 'leaky_relu', 'gelu']}
        },
    }
    CONFIG_DEFAULTS = {
    "resume":"allow",
    "architecture": f"{MODEL_NAME}",
    "dataset": f"{DATA_NAME}",
    #"batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    # "initial_epoch": initial_epoch,
    "BATCH_SIZE": BATCH_SIZE,
    "MODEL": MODEL,
    "lr": lr,
    "DROPOUT_RATE": DROPOUT_RATE,
    "ACTIVATION": ACTIVATION,
    }  
    def __init__(self):
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

config = Config()