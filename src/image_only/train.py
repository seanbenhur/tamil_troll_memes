from tqdm import trange, tqdm
import torch
import wandb
from sklearn.metrics import precision_score,recall_score,f1_score, classification_report



class Trainer:
    def __init__(self,train_dl,eval_dl,model,optimizer,scheduler,loss_fn, accelerator, training_args, data_args, wandb_args):
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        self.best_f1 = 0
        self.training_args = training_args
        self.output_dir = data_args.output_dir
        self.checkpoint_save_path = f"{data_args.output_dir}/{training_args.run_name}_checkpoint.bin"
        self.save_path = f"{data_args.output_dir}/{training_args.run_name}.bin"
        self.max_norm = 0.25
        self.update_in_steps = ["onecycle","linear","cosine"]
        self.update_in_epochs = ["cosineannealing","steplr","exponentiallr"]
        wandb.init(project=wandb_args.project_name,group=wandb_args.group_name)
    
    def save_checkpoint(model, optimizer, scheduler, loss, save_path, epoch):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch
        }, save_path)
    
    
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
            
            if training_args.scheduler in update_in_steps:
                self.scheduler.step()
            self.optimizer.zero_grad()
        return tr_loss/len(train_dl)

   
    def evaluate_one_epoch(self,eval_dl):
        eval_loss = 0.0
        eval_steps = 0
        y_true = []
        y_pred = []
        self.model.eval()
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
        self.model.eval()
        test_iterator = tqdm(test_dl,desc="Prediction")
        
        with torch.no_grad():
            for images,labels in test_iterator:
                logits = self.model(images)
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    
        y_pred = np.array(y_pred)
        y_pred = np.array(y_pred) >= 0.5
        y_true = np.array(y_true)
      
        return y_pred, y_true
    
    def train(self):
        best_valid_loss = float('inf')
        epoch_iterator = trange(self.training_args.num_epochs,desc="Epoch")
    
        for epoch in epoch_iterator:
            tr_loss = self.train_one_epoch(self.train_dl)
            
            if training_args.scheduler in update_in_epochs:
                self.scheduler.step()
            
            #save checkpoints in each epoch
            self.save_checkpoint(self.model, self.optimizer, self.scheduler, tr_loss, self.checkpoint_save_path, epoch)
            
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
    