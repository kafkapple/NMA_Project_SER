# 20240714 NMA Project SER by Joon Park 
import os
import torch
import random
import numpy as np
import wandb
from config import config
from data_utils import download_ravdess, preprocess_data, prepare_dataloaders
from model import EmotionRecognitionModel_v1, EmotionRecognitionModel_v2
from train_utils import train_model, train_epoch, evaluate_model, load_checkpoint
from visualization import plot_confusion_matrix, visualize_embeddings, extract_embeddings, perform_rsa#, explain_predictions, 
import matplotlib.pyplot as plt
import matplotlib


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # to seed the script globally
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #print(torch.cuda.is_available()) 
        #print(torch.cuda.current_device()) 
        print(f'\n##### GPU verified. {torch.cuda.get_device_name(0)}')

def main():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Download and preprocess data
    print('\n###### Preparing Dataset...')
    data_dir, status = download_ravdess()
    if not status:
        raise Exception("Failed to download or extract the dataset. Terminating execution.")
    
    id_wandb=wandb.util.generate_id()
    
    wandb.init(
        id=id_wandb,
        project=config.WANDB_PROJECT,
        #name=config.WANDB_NAME, #
        config=config.CONFIG_DEFAULTS,
        resume=True
    )
    data, labels = preprocess_data(data_dir)
    train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, wandb.config.BATCH_SIZE)

    # Initialize model
    print('\n###### Preparing Model...')
    if wandb.config.MODEL =="v1":
        model = EmotionRecognitionModel_v1(
        input_size=train_loader.dataset[0][0].shape[1],  # use 2nd dim cause 1st dim is 1
        num_classes=len(config.LABELS_EMOTION),
        dropout_rate=wandb.config.DROPOUT_RATE,#config.DROPOUT_RATE
        activation=wandb.config.ACTIVATION
        ).to(device)
    elif wandb.config.MODEL =='v2':
        model = EmotionRecognitionModel_v2(
        input_size=train_loader.dataset[0][0].shape[1],  # use 2nd dim cause 1st dim is 1
        num_classes=len(config.LABELS_EMOTION),
        dropout_rate=wandb.config.DROPOUT_RATE,#config.DROPOUT_RATE
        activation=wandb.config.ACTIVATION
        ).to(device)
    else:
        print('Model error')
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)# LEARNING_RATE)

    # if not os.path.exists(config.MODEL_SAVE_PATH):
    initial_epoch=1
    #id_wandb=wandb.util.generate_id()
    print(f'No trained model.')
    print(f'Wandb id generated: {id_wandb}')
    # else:
    #     print('Trained model exists.')
    #     model, optimizer, start_epoch, best_val_accuracy, id_wandb = load_checkpoint(config.CKPT_SAVE_PATH, model, optimizer, device)
        
    #     initial_epoch = start_epoch#+1
    #     print(f'Model loaded. Best val accuracy: {best_val_accuracy:.3f}\nWandb id is loaded: {id_wandb}')
    #     #config.NUM_EPOCHS+=start_epoch
        
    # wandb setup
    wandb.config.update({"initial_epoch": initial_epoch}, allow_val_change=True)
    wandb.watch(model, log='all')
    ###### Train model
    print(f'\n##### Training starts. Initial epoch:{initial_epoch}, Total number of epoch: {config.NUM_EPOCHS}')
    best_val_accuracy = 0
    
    for epoch in range(initial_epoch, initial_epoch+config.NUM_EPOCHS):#range(num_epochs):
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_labels, val_preds = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{initial_epoch}~{initial_epoch+config.NUM_EPOCHS-1}]")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")
            
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
        
        wandb.log({
            "train":{
                    "loss": train_loss,
                    "accuracy": train_accuracy,
                    "precision": train_precision,
                    "recall": train_recall,
                    "f1": train_f1
                    },
            
            "val":{
                        "loss": val_loss,
                        "accuracy": val_accuracy,
                        "precision": val_precision,
                        "recall": val_recall,
                        "f1": val_f1
                        # "confusion_matrix": wandb.Image(fig_cm),
                        # "embeddings": wandb.Image(fig_embedding)
                        }
            },step=epoch)
        # if epoch % config.N_STEP_FIG ==0:
        #     print('Logging visualizations (validation data)...')
        #     # Visualizations-
        #     # Confusion matrix
        #     fig_cm = plot_confusion_matrix(val_labels, val_preds, config.LABELS_EMOTION)

        #     wandb.log({"validation":{
        #             "confusion_matrix": wandb.Image(fig_cm)
        #             }})
        #     # embedding
        #     embeddings, labels = extract_embeddings(model, val_loader, device)
        #     labels = [config.LABELS_EMOTION[val] for val in labels]
        #     fig_embedding=visualize_embeddings(embeddings, labels, method='tsne')
        #     wandb.log({"validation":{
        #         "embeddings": wandb.Image(fig_embedding)
        #         }})
            
    # model = train_model(model, train_loader, val_loader, initial_epoch, config.NUM_EPOCHS, criterion, optimizer, device, id_wandb)

    ########## Evaluate on test set
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_labels, test_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"\n##### Training finishied.\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}\n")

    # Visualizations
    fig_cm = plot_confusion_matrix(test_labels, test_preds, config.LABELS_EMOTION)
    embeddings, labels = extract_embeddings(model, test_loader, device)
    labels = [config.LABELS_EMOTION[val] for val in labels]
    fig_embedding=visualize_embeddings(embeddings, labels, method='tsne')
    
    # explain_predictions(model, test_loader, device)
    #fig_rsa= perform_rsa(model, test_loader, device)
    wandb.log({ # logged in a nested way
        "test":{"loss": test_loss,
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1": test_f1,
                
                "confusion_matrix": wandb.Image(fig_cm),
                "embeddings": wandb.Image(fig_embedding)
            # "rsa": wandb.Image(fig_rsa)
            }})
    #wandb.run.summary['best_accuracy']=
    plt.close()
# if __name__ == "__main__":
#     main()

# 3: Start the sweep
if __name__ == "__main__":
    matplotlib.use('agg')
    print('\n######\n',config.WANDB_PROJECT)
    #### 
    # sweeps = wandb.Api().project(config.WANDB_PROJECT).sweeps()
    # print('\n######\n',sweeps)
    SWEEP_NAIVE=False
    
    if SWEEP_NAIVE:
        #print(config.CONFIG_SWEEP, config.WANDB_PROJECT)

        sweep_id=wandb.sweep(sweep=config.CONFIG_SWEEP, project=config.WANDB_PROJECT)
        print('###\n',type(sweep_id),sweep_id)
    else:
        sweep_id="8lad6k0u" #
        sweep_id = f"{config.ENTITY}/{config.WANDB_PROJECT}/{sweep_id}"#wandb.sweep(sweep=config.CONFIG_SWEEP, project=config.WANDB_PROJECT)
    print(sweep_id)
    wandb.agent(sweep_id, function=main, count=config.N_SWEEP)
    wandb.finish()