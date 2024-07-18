import argparse
import os
import torch
import random
import numpy as np
import wandb
from config_main import config
from data_utils import download_ravdess, preprocess_data, prepare_dataloaders
from model import EmotionRecognitionModel_v1, EmotionRecognitionModel_v2
from train_utils import train_model, train_epoch, evaluate_model, load_checkpoint
from visualization import plot_confusion_matrix, visualize_embeddings, extract_embeddings, perform_rsa
import matplotlib.pyplot as plt
import matplotlib
import time
from tqdm import tqdm
import sys

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # to seed the script globally

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA is available. 🚀")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: Name: {torch.cuda.get_device_name(i)}")
        print(torch.cuda.current_device()) 
    else:
        print("CUDA is not available.")

def create_log_dict(prefix, loss, accuracy, precision, recall, f1):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    values = [loss, accuracy, precision, recall, f1]
    return {prefix: {metric: value for metric, value in zip(metrics, values)}}

def create_log_dict_fig(prefix, confusion_matrix, embedding, rsa):
    metrics = ['confusion_matrix','embedding','rsa']
    values = [wandb.Image(confusion_matrix), wandb.Image(embedding), wandb.Image(rsa)]
    return{prefix: {metric: value for metric, value in zip(metrics, values)}}
  
def get_new_model_path(base_path):
    dir_name, file_name = os.path.split(base_path)
    name, ext = os.path.splitext(file_name)
    i = 1
    while os.path.exists(os.path.join(dir_name, f"{name}_{i}{ext}")):
        i += 1
    return os.path.join(dir_name, f"{name}_{i}{ext}")
  
def run_training(num_epochs, is_sweep=False):
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print('\n###### Preparing Dataset...')
    data_dir, status = download_ravdess()
    if not status:
        raise Exception("Failed to download or extract the dataset. Terminating execution.")
    #### Model loading or start new
    if os.path.exists(config.CKPT_SAVE_PATH):
        user_input = input("Found existing model. Continue training? (y/n): ").lower()
        if user_input == 'y':
            model, optimizer, initial_epoch, best_val_accuracy, id_wandb = load_checkpoint(config.CKPT_SAVE_PATH, model, optimizer, device)
            print(f"Resuming training from epoch {initial_epoch}. Best val accuracy: {best_val_accuracy:.3f}")
        else:
            config.MODEL_SAVE_PATH = get_new_model_path(config.MODEL_SAVE_PATH)
            config.CKPT_SAVE_PATH = get_new_model_path(config.CKPT_SAVE_PATH)
            print(f"Starting new training. New model will be saved as {config.MODEL_SAVE_PATH}")
            initial_epoch = 1
            id_wandb=wandb.util.generate_id()
            print(f'Wandb id generated: {id_wandb}')
    else:
        print('No trained data.')
        initial_epoch = 1
        id_wandb=wandb.util.generate_id()
        print(f'Wandb id generated: {id_wandb}')

#### Sweep or not
    if is_sweep:
        wandb.init(config=config.CONFIG_DEFAULTS, resume=False)
    else:
        wandb.init(project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS)#, resume=False)

    data, labels = preprocess_data(data_dir)
    train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, wandb.config.BATCH_SIZE)

    print('\n###### Preparing Model...')
    model_class = EmotionRecognitionModel_v2 if wandb.config.MODEL == 'v2' else EmotionRecognitionModel_v1
    model = model_class(
        input_size=train_loader.dataset[0][0].shape[1],
        num_classes=len(config.LABELS_EMOTION),
        dropout_rate=wandb.config.DROPOUT_RATE,
        activation=wandb.config.ACTIVATION
    ).to(device)

    if config.OPTIMIZER=="adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.OPTIMIZER=="SGD":
      optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
      print('err optimizer')
      
    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = 0.0
    best_val_loss = 0.0

    wandb.watch(model, log='all')

    print(f'\n##### Training starts.\nInitial epoch:{initial_epoch}\nTotal number of epoch: {num_epochs}')
    for epoch in tqdm(range(initial_epoch, initial_epoch + num_epochs), desc="Epochs"):
        epoch_start_time = time.time()
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_labels, val_preds = evaluate_model(model, val_loader, criterion, device)
        #print(f"Epoch [{epoch}/]")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch}/{initial_epoch}~{initial_epoch+num_epochs-1}] - Time: {epoch_duration:.2f}s")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"\nNew Best val accuracy found. Model saved to {config.MODEL_SAVE_PATH}")
        # save Checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'id_wandb': id_wandb
        }
        torch.save(ckpt, config.CKPT_SAVE_PATH)
        print(f"Checkpoint saved to {config.CKPT_SAVE_PATH}", ckpt['epoch'])
        
        log_dict = {}
        log_dict.update(create_log_dict('train', train_loss, train_accuracy, train_precision, train_recall, train_f1))
        log_dict.update(create_log_dict('val', val_loss, val_accuracy, val_precision, val_recall, val_f1))
            
        if config.NUM_EPOCHS%config.N_STEP_FIG==0:
            print('/n### logging visualization...')
            val_fig_cm = plot_confusion_matrix(val_labels, val_preds, config.LABELS_EMOTION)
            embeddings, labels = extract_embeddings(model, val_loader, device)
            labels = [config.LABELS_EMOTION[val] for val in labels]
            val_fig_embedding = visualize_embeddings(embeddings, labels, method='tsne')
            val_fig_rsa=perform_rsa(model, val_loader, device)
            log_dict.update(create_log_dict_fig('val', val_fig_cm, val_fig_embedding, val_fig_rsa))
            
        
        wandb.log(log_dict, step=epoch)

    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_labels, test_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"\n##### Training finished.\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}\nBest Val accuracy: {best_val_accuracy:.4f}, Best Val loss: {best_val_loss:.4f}")

    test_fig_cm = plot_confusion_matrix(test_labels, test_preds, config.LABELS_EMOTION)
    
    embeddings, labels = extract_embeddings(model, test_loader, device)
    labels = [config.LABELS_EMOTION[val] for val in labels]
    test_fig_embedding = visualize_embeddings(embeddings, labels, method='tsne')

    log_dict_test={}
    log_dict_test.update(create_log_dict('test', test_loss, test_accuracy, test_precision, test_recall, test_f1))
    test_fig_rsa=perform_rsa(model, test_loader, device)
    log_dict_test.update(create_log_dict_fig('test', test_fig_cm, test_fig_embedding, test_fig_rsa), )
    wandb.log(log_dict_test)
    wandb.finish()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run emotion recognition model training")
    parser.add_argument("epochs", type=int, nargs='?', default=10, help="Number of epochs to train")
    parser.add_argument("--sweeps", type=int, help="Number of sweeps for hyperparameter search", default=0)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        print(f"\nNo arguments provided. Using default config.")
    config.NUM_EPOCHS=args.epochs    

    if args.sweeps > 0:
        matplotlib.use('agg')
        sweep_id = wandb.sweep(sweep=config.CONFIG_SWEEP, project=config.WANDB_PROJECT)
        print(f'\nSweep starts. Sweep id: {sweep_id}\nTotal number of sweep: {config.NUM_EPOCHS}\n')
        wandb.agent(sweep_id, function=lambda: run_training(args.epochs, is_sweep=True), count=args.sweeps)
    else:
        run_training(args.epochs, is_sweep=False)
        print(f"\nTraining for {config.NUM_EPOCHS} epochs.")

if __name__ == "__main__":
    main()