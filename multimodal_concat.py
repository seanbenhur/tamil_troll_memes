
from dataclasses import dataclass, field
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import cv2
import torch.nn.functional as F
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
from torch.optim.lr_scheduler import OneCycleLR, StepLR
import timm
from transformers import get_scheduler
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
    output_dir: str=field(default="trained_models/multimodal/swin_cross_muril")

@dataclass
class TrainingArgs:
    image_size: int=field(default=224)
    batch_size: int=field(default=1)
    num_epochs: int=field(default=3)
    do_train: bool=field(default=False)
    learning_rate: int=field(default=3e-4)
    run_name: str=field(default="multimodal-test-run")
    

    
    
@dataclass
class ModelArgs:
    #model_name: str=field(default="tf_efficientnet_b3_ap")
    model_type: str=field(default="resnet101")
    image_pretrained: str=field(default="swin_small_patch4_window7_224")
    text_pretrained: str=field(default="google/muril-base-cased")
    pretrained: bool=field(default=True)
    inference: bool=field(default=True)
    dropout: float=field(default=0.2)
    model_path: str=field(default="trained_models/multimodal/swin_cross_muril/multimodal-test-run.bin")
    
    
@dataclass
class WandbArgs:
    project_name: str=field(default="troll_meme_classification")
    group_name: str=field(default="multimodal-concat-fgm-run")
    
accelerator = Accelerator()    
training_args, data_args, model_args, wandb_args = TrainingArgs, DataArgs, ModelArgs, WandbArgs    
    
image_transform = transforms.Compose([
    transforms.Resize((training_args.image_size, training_args.image_size)),
    transforms.ToTensor()])

################### DATASET CLASS ###################
        
      
class MemeDataset(Dataset):
    def __init__(self,data_path,dir_path,transforms,tokenizer,train=True):
        self.df = pd.read_csv(data_path)
        
        self.dir_path = dir_path
        self.df['label'] = self.df['label'].replace(['not_troll','troll'],[0,1])
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_len = 128
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        image_id = self.df['file_path'].values[idx]
        captions = self.df["captions"].values[idx]
        
        
        
        labels = self.df['label'].values[idx]
        image_path = os.path.join(self.dir_path,image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        
        caption_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_len,
            truncation=True)
        
        caption_ids = caption_inputs["input_ids"]
        caption_mask = caption_inputs["attention_mask"]
        
        label = torch.tensor(labels).float()
        
        
        if self.train:
            return {
                "image": image,
                "caption_input_ids": torch.tensor(caption_ids, dtype=torch.long),
                "caption_attention_mask": torch.tensor(caption_mask, dtype=torch.long),
                "label": torch.tensor(labels, dtype=torch.float),
            }

        return {
            "image": image,
            "caption_input_ids": torch.tensor(caption_ids, dtype=torch.long),
            "caption_attention_mask": torch.tensor(caption_mask, dtype=torch.long),
        }
    
### CREATE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_args.text_pretrained)

    
def create_dataset(
    dataset_path,
    dir_path,
    transforms,
    tokenizer
):
    train_ds = MemeDataset(
        dataset_path,dir_path, transforms, tokenizer
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



train_ds = create_dataset(data_args.train_path,data_args.train_dir_path,image_transform,tokenizer)
dev_ds = create_dataset(data_args.test_path,data_args.test_dir_path, image_transform,tokenizer)

train_dl, test_dl = create_dataloader(train_ds,dev_ds,training_args.batch_size)

logging.info("DATALOADERS ARE CREATED")
################### MODEL CLASS ###################


class CrossModel(nn.Module):
    def __init__(
        self,
        image_pretrained,
        text_pretrained,
        use_pretrained_image,
        dropout
    ):
        super().__init__()
        self.image_model = timm.create_model(
            image_pretrained, pretrained=use_pretrained_image
        )
        self.text_model_config = AutoConfig.from_pretrained(text_pretrained)
        self.text_model = AutoModel.from_pretrained(text_pretrained)
        self.image_model_out = nn.Linear(self.image_model.head.in_features, 512)
        self.text_model_out = nn.Linear(self.text_model_config.hidden_size, 512)
        self.drop = nn.Dropout(dropout)

        
        self.fc_out = nn.Linear(1512, 512)  # 512X256
        # self.out = nn.Linear(256, n_out) # 512X256
        self.final_layer = nn.Linear(512, 1)

  

    def forward(
        self,
        img,
        caption_input_ids,
        caption_attention_mask
    ):
       
        caption_model_output = self.text_model(
            input_ids=caption_input_ids, attention_mask=caption_attention_mask
        )
        caption_text_out = self.text_model_out(caption_model_output["pooler_output"])
        img_out = self.image_model(img)

        concat = torch.cat([img_out,caption_text_out],dim=1)
       
        fc_out = F.relu(self.fc_out(concat))
        
        output = self.final_layer(fc_out)
        return output


model = CrossModel(model_args.image_pretrained,model_args.text_pretrained,model_args.pretrained,model_args.dropout)


    

if model_args.inference:
    model.load_state_dict(torch.load(model_args.model_path))
    logging.info(f"SAVED MODEL IS LOADED FROM {model_args.model_path}")

################### LOSS FUNTION ###################   
    

def bcewithlogits_loss_fn(outputs, targets, reduction=None):
    return nn.BCEWithLogitsLoss(reduction)(outputs, targets.view(-1, 1))



    
optimizer = AdamW(model.parameters(),lr=training_args.learning_rate)
#scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=training_args.num_epochs)


                        
nb_train_steps = int(
        len(train_dl) / training_args.batch_size * training_args.num_epochs
    )
#scheduler = get_scheduler(
 #       'linear',
  #      optimizer,
   #     num_warmup_steps=100,
   #     num_training_steps=nb_train_steps,
    #)

scheduler = get_scheduler(
        'cosine',
        optimizer,
        num_warmup_steps=100,
        num_training_steps=nb_train_steps,
    )


################## FGM ##################
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        
    def attack(self, epsilon=1, emd_name="emb"):
        for name, param in self.model.named_parameters():
            if param.requires_grad is True and emd_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.grad.add_(r_at)
                    
    def restore(self, emd_name="emb"):
        for name, params in self.model.named_parameters():
            if params.requires_grad is True and emd_name in name:
                assert name in self.backup
                params.data = self.backup[name]

        self.backup = {}

fgm = FGM        
################### TRAINER ###################


class Trainer:
    def __init__(self,train_dl,eval_dl,model,optimizer,scheuler,loss_fn, accelerator, training_args, data_args, fgm=fgm):
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        self.fgm = fgm
        self.best_f1 = 0
        self.output_dir = data_args.output_dir
        self.save_path = f"{data_args.output_dir}/{training_args.run_name}.bin"
        self.max_norm = 0.25
        wandb.init(project=wandb_args.project_name,group=wandb_args.group_name)
    
    
    def train_one_epoch(self,train_dl):
        self.model.train()
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dl, desc="Training")
        for idx, batch in enumerate(epoch_iterator):
            images, input_ids, attention_mask, labels = batch['image'], batch['caption_input_ids'], batch['caption_attention_mask'], batch['label']
            logits = self.model(images,input_ids,attention_mask)
            loss = self.loss_fn(logits,labels)
            wandb.log({"train loss": loss.item()})
            #backprop
            self.accelerator.backward(loss)
            
            fgm = self.fgm(self.model)
            fgm.attack()
            
            adv_loss = self.model(input_ids, token_type_ids, attention_mask)
            loss_adv = self.loss_fn(adv_loss, labels)
            
            self.accelerator.backward(loss)
            fgm.restore()
            
            tr_loss += loss.item()
            epoch_iterator.set_description(f"train loss {loss.item()}")
            #clip grad norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        return tr_loss/len(train_dl)



    def evaluate_one_epoch(self,eval_dl):
        eval_loss = 0.0
        eval_steps = 0
        y_true = []
        y_pred = []
        model.eval()
        eval_iterator = tqdm(eval_dl,desc="Evaluation")
        
        with torch.no_grad():
            for idx, batch in enumerate(eval_iterator):
                images, input_ids, attention_mask, labels = batch['image'], batch['caption_input_ids'], batch['caption_attention_mask'], batch['label']
                logits = model(images,input_ids,attention_mask)
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
            for batch in test_iterator:
                images, input_ids, attention_mask, labels = batch['image'], batch['caption_input_ids'], batch['caption_attention_mask'], batch['label']
                logits = model(images,input_ids,attention_mask)
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
        