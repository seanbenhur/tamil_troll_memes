import torchvision.transforms as transforms
from transformers import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import auglib
from transformers import AutoTokenizer
import numpy as np
from models import ConcatModel,CAModel,ResnetCAModel
import logging
import torch
from losses import bcewithlogits_loss_fn
from train import Trainer
from utils import load_checkpoint
from arguments import parse_args 
from dataloaders import create_dataset, create_dataloader
from accelerate import Accelerator, DeepSpeedPlugin
import warnings
warnings.filterwarnings('ignore')



logging.basicConfig(level = logging.INFO)
#arguments
data_args, training_args, model_args, wandb_args = parse_args()
#image transforms
image_transform = transforms.Compose([
    transforms.Resize((training_args.image_size, training_args.image_size)),
    auglib.TrivialAugment(),     #TrivialAugmentWide
    transforms.ToTensor()])


### CREATE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_args.text_pretrained)
#create datasets
train_ds = create_dataset(data_args.train_path,data_args.train_dir_path,image_transform,tokenizer)
dev_ds = create_dataset(data_args.test_path,data_args.test_dir_path, image_transform,tokenizer)
#create dataloaders
train_dl, test_dl = create_dataloader(train_ds,dev_ds,training_args.batch_size)
logging.info("***DATALOADERS ARE CREATED***")

accelerator = Accelerator()    

if model_args.fusion_method == "timm_crossattention":
    model = CAModel(model_args.image_pretrained,model_args.text_pretrained,model_args.pretrained,model_args.dropout)

elif model_args.fusion_method == "resnet_crossattention":
    model = ResnetCAModel(model_args.text_pretrained,model_args.pretrained,model_args.dropout)

elif model_args.fusion_method == "concat":
    model = ConcatModel(model_args.image_pretrained,model_args.text_pretrained,model_args.pretrained,model_args.dropout)
    
else:
    raise RuntimeError(
            f"Invalid Fusion method '{model_args.fusion_method}'. Only timm_crossattention, resnet_crossattention and concat"
            "are supported."
        )
    
    
logging.info(f"MODEL {model_args.model_type} is created")


if model_args.inference:
    model.load_state_dict(torch.load(model_args.model_path))
    logging.info(f"SAVED MODEL IS LOADED FROM {model_args.model_path}")
    

    
optimizer = AdamW(model.parameters(),lr=training_args.learning_rate,weight_decay=training_args.weight_decay)

nb_train_steps = int(len(train_dl) / training_args.batch_size * training_args.num_epochs)


if training_args.scheduler == "onecycle":
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=training_args.num_epochs)

elif training_args.scheduler == "linear":
    scheduler = get_scheduler('linear',optimizer,num_warmup_steps=training_args.warmup_steps, num_training_steps=nb_train_steps)

elif training_args.scheduler == "cosine":
    scheduler = get_scheduler('cosine',optimizer,num_warmup_steps=training_args.warmup_steps, num_training_steps=nb_train_steps)
    
elif training_args.scheduler == "cosineannealing":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=atraining_args.num_epochs - training_args.lr_warmup_epochs
        )    
elif training_args.lr_scheduler == "steplr":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_args.lr_step_size, gamma=training_args.lr_gamma)

elif training_args.scheduler == "exponentiallr":
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer)
        
else:
    raise RuntimeError(
            f"Invalid lr scheduler '{training_args.scheduler}'. Only Onecycle,LinearLR,CosineLR, StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )


if training_args.load_checkpoint:
    checkpoint = torch.load(training_args.checkpoint_path)
    load_checkpoint(checkpoint,model,optimizer,scheduler,scalar=None)
    
#setup accelerator
model, optimizer, train_dl, test_dl = accelerator.prepare(model, optimizer, train_dl, test_dl)

if __name__ == "__main__":
    
    trainer = Trainer(train_dl,test_dl,model, optimizer,scheduler,bcewithlogits_loss_fn,accelerator,training_args, data_args, wandb_args)
    print("*"*20)
    print(training_args.do_train)
    if training_args.do_train:
        trainer.train()
    else:
        y_pred, y_true = trainer.predict(test_dl)
    test_pr,test_recall,test_f1 = precision_score(y_true,y_pred,average="macro"), recall_score(y_true,y_pred,average="macro"), f1_score(y_true,y_pred,average="macro")
    test_df = pd.DataFrame()
    test_df['Actual'] = y_true
    test_df['Preds'] = y_pred
    test_save_path = f"{data_args.output_dir}/{training_args.run_name}_predictions.csv"
    test_df.to_csv(test_save_path)
    logging.info(f"PREDICTIONS ARE SAVED IN {test_save_path}")
    report = classification_report(y_true, y_pred, output_dict=True)
    cls_rep_df = pd.DataFrame(report).transpose()
    test_cls_path = f"{data_args.output_dir}/{training_args.run_name}_cls_rep.csv"
    test_cls_path = data_args.output_dir + "/" + "cls_rep.csv"
    cls_rep_df.to_csv(test_cls_path)
    logging.info(f"CLASSIFICATION REPORT IS SAVED IN {test_cls_path}")
        