from dataclasses import dataclass, field
import pandas as pd
import torch
import cv2
import logging
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from torch.optim import AdamW
import os
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score, classification_report
import wandb
from torch.optim.lr_scheduler import OneCycleLR
import timm
from accelerate import Accelerator
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin



logging.basicConfig(level = logging.INFO)
################### ARGUMENTS ###################

@dataclass
class DataArgs:
    train_path: str=field(default="data/Tamil_troll_memes/train_labels.csv")
    test_path: str=field(default="data/Tamil_troll_memes/test_labels.csv")
    train_dir_path: str=field(default="data/Tamil_troll_memes/train/uploaded_tamil_memes")
    test_dir_path: str=field(default="data/Tamil_troll_memes/test/test_img")
    output_dir: str=field(default="trained_models/image_only/swin")

@dataclass
class TrainingArgs:
    image_size: int=field(default=224)
    batch_size: int=field(default=8)
    num_epochs: int=field(default=10)
    do_train: bool=field(default=True)
    learning_rate: int=field(default=3e-4)
    run_name: str=field(default="swin_base_onecycle")
    

    
    
@dataclass
class ModelArgs:
    #model_name: str=field(default="tf_efficientnet_b3_ap")
    model_type: str=field(default=None)
    model_name: str=field(default="swin_small_patch4_window7_224")
    pretrained: bool=field(default=True)
    inference: bool=field(default=False)
    model_path: str=field(default="trained_models/image_only/resnet/resnet.bin")
    
    
@dataclass
class WandbArgs:
    project_name: str=field(default="troll_meme_classification")
    group_name: str=field(default="initial-run")
    
accelerator = Accelerator()    
training_args, data_args, model_args, wandb_args = TrainingArgs, DataArgs, ModelArgs, WandbArgs    
    
image_transform = transforms.Compose([
    transforms.Resize((training_args.image_size, training_args.image_size)),
    transforms.ToTensor()])

################### DATASET CLASS ###################
        
class MemeDataset(Dataset):
    def __init__(self,data_path,dir_path,transforms,train=True):
        self.df = pd.read_csv(data_path)
        
        self.dir_path = dir_path
        self.df['label'] = self.df['label'].replace(['not_troll','troll'],[0,1])
        self.transforms = transforms
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
   
        image_id = self.df['file_path'].values[idx]
        labels = self.df['label'].values[idx]
        image_path = os.path.join(self.dir_path,image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        label = torch.tensor(labels).float()
        return image, label
    
    

def create_dataset(
    dataset_path,
    transforms,
    dir_path
):
    train_ds = MemeDataset(
        dataset_path, transforms, dir_path
    )
    return train_ds


def create_dataloader(train_ds, eval_ds, batch_size):
    if eval_ds is None:
        train_dataloader = DataLoader(
            train_ds,
            batch_size=training_args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        return train_dataloader

    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataloader = DataLoader(
        eval_ds,
        batch_size=training_args.batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader



train_ds = create_dataset(data_args.train_path,data_args.train_dir_path,image_transform)
dev_ds = create_dataset(data_args.test_path,data_args.test_dir_path, image_transform)

train_dl, test_dl = create_dataloader(train_ds,dev_ds,training_args.batch_size)

logging.info("DATALOADERS ARE CREATED")
################### MODEL CLASS ###################

class Image_Classifier(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model_args = model_args
        self.model = timm.create_model(model_name=model_args.model_name,pretrained=model_args.pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    

class Resnet(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model = models.resnet50(pretrained=model_args.pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
class VGG(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model = models.vgg16_bn(pretrained=model_args.pretrained)
        n_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits

    


if model_args.model_type == "resnet":
    model = Resnet(model_args)
    logging.info(f"MODEL {model_args.model_type} is created")
elif model_args.model_type == "vgg":
    model = VGG(model_args)
    logging.info(f"MODEL {model_args.model_type} is created")
else:
    model = Image_Classifier(model_args)
    logging.info(f"MODEL {model_args.model_name} is created")

    

if model_args.inference:
    model.load_state_dict(torch.load(model_args.model_path))
    logging.info(f"SAVED MODEL IS LOADED FROM {model_args.model_path}")

################### LOSS FUNTION ###################   
    

def bcewithlogits_loss_fn(outputs, targets, reduction=None):
    return nn.BCEWithLogitsLoss(reduction)(outputs, targets.view(-1, 1))



    
optimizer = AdamW(model.parameters(),lr=training_args.learning_rate)
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=training_args.num_epochs)


################### TRAINER ###################


class Trainer:
    def __init__(self,train_dl,eval_dl,model,optimizer,scheuler,loss_fn, accelerator, training_args, data_args):
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        self.best_f1 = 0
        self.output_dir = data_args.output_dir
        self.save_path = f"{data_args.output_dir}/{training_args.run_name}.bin"
        self.max_norm = 0.25
        wandb.init(project=wandb_args.project_name,group=wandb_args.group_name)
    
    
    def train_one_epoch(self,train_dl):
        self.model.train()
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dl, desc="Training")
        for idx, (images,labels) in enumerate(epoch_iterator):
            logits = self.model(images)
            loss = self.loss_fn(logits,labels)
            wandb.log({"train loss": loss.item()})
            #backprop
            self.accelerator.backward(loss)
                
            tr_loss += loss.item()
            epoch_iterator.set_description(f"train loss {loss.item()}")
            #clip grad norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return tr_loss/len(train_dl)

   
    def evaluate_one_epoch(self,eval_dl):
        eval_loss = 0.0
        eval_steps = 0
        y_true = []
        y_pred = []
        model.eval()
        eval_iterator = tqdm(eval_dl,desc="Evaluation")
        
        with torch.no_grad():
            for images,labels in eval_iterator:
                logits = model(images)
                loss = self.loss_fn(logits,labels)
                wandb.log({"eval loss": loss.item()})
                eval_loss += loss.item()
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    
                eval_iterator.set_description(f"Eval loss {loss.item()}")
    
        y_pred = np.array(y_pred)
        y_pred = np.array(y_pred) >= 0.5
        y_true = np.array(y_true)
        print(len(y_true))
        print(len(y_pred))
        p = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        r = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        return p, r, f1, eval_loss/len(eval_dl)
    
    
    
    def predict(self,test_dl):
        y_true = []
        y_pred = []
        model.eval()
        test_iterator = tqdm(test_dl,desc="Prediction")
        
        with torch.no_grad():
            for images,labels in test_iterator:
                logits = model(images)
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    
        y_pred = np.array(y_pred)
        y_pred = np.array(y_pred) >= 0.5
        y_true = np.array(y_true)
      
        return y_pred, y_true
    
    def train(self):
        best_valid_loss = float('inf')
        epoch_iterator = trange(training_args.num_epochs,desc="Epoch")
    
        for epoch in epoch_iterator:
            tr_loss = self.train_one_epoch(self.train_dl)
            _,_,eval_f1, eval_loss = self.evaluate_one_epoch(self.eval_dl)
        
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
               
                torch.save(model.state_dict(), self.save_path)
                logging.info(f"MODEL IS SAVED IN {self.save_path}")
    
            logging.info(f'Epoch: {epoch+1:02}')
            logging.info(f'\tTrain Loss: {tr_loss:.3f}%')
            logging.info(f'\t Val. Loss: {eval_loss:.3f} |  Val. F1: {eval_f1*100:.2f}%')
        wandb.finish()    
    

    
if __name__ == "__main__":
    
    trainer = Trainer(train_dl,test_dl,model, optimizer,scheduler,bcewithlogits_loss_fn,accelerator,training_args, data_args)
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
    logging.info(f"CLASSIFICATION REPORT IS SAVED IN {tesT_CLS_PATH}")
        