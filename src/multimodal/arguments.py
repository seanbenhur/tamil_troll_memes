from dataclasses import dataclass, field
from transformers import HfArgumentParser



@dataclass
class DataArgs:
    train_path: str=field(default="data/Tamil_troll_memes/train_labels.csv")
    test_path: str=field(default="data/Tamil_troll_memes/test_labels.csv")
    train_dir_path: str=field(default="data/Tamil_troll_memes/train/uploaded_tamil_memes")
    test_dir_path: str=field(default="data/Tamil_troll_memes/test/test_img")
    output_dir: str=field(default="trained_models/image_only/resnet101")

@dataclass
class TrainingArgs:
    image_size: int=field(default=224)
    batch_size: int=field(default=8)
    num_epochs: int=field(default=10)
    do_train: bool=field(default=True)
    learning_rate: int=field(default=3e-4)
    weight_decay: int=field(default=0.01)
    scheduler: str=field(default="onecycle")
    lr_step_size: int=field(default=30)
    lr_gamma: int=field(default=0.1)
    warmup_steps: int=field(default=100)
    lr_warmup_epochs: int= field(default=0)
    load_checkpoint: bool = field(default=False)
    run_name: str=field(default="resnet_onecycle")
    
    
        
@dataclass
class ModelArgs:
    #model_name: str=field(default="tf_efficientnet_b3_ap")
    model_type: str=field(default='resnet101')
    image_pretrained: str=field(default="swin_small_patch4_window7_224")
    text_pretrained: str=field(default="google/muril-base-cased")
    pretrained: bool=field(default=True)
    inference: bool=field(default=False)
    num_classes: int=field(default=1)
    dropout: float=field(default=0.1)
    fusion_method: str=field(default="concat")
    model_path: str=field(default="trained_models/image_only/resnet101/resnet_onecycle.bin")
        
@dataclass
class WandbArgs:
    project_name: str=field(default="troll_meme_classification")
    group_name: str=field(default="resnet101_base_onecycle")
    

    
def parse_args():
    """Parse the dataclasses into arguments"""
    parser = HfArgumentParser((DataArgs, TrainingArgs, ModelArgs, WandbArgs))
    args = parser.parse_args_into_dataclasses()
    return args

if __name__ == "__main__":
    _,dataargs,_,_ = parse_args()
    print(dataargs)