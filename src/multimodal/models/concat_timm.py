import timm
import torch.nn as nn
from torchvision import models
from transformers import AutoConfig, AutoModel
import torch
import torch.nn.functional as F



class ConcatModel(nn.Module):
    def __init__(
        self,
        model_args
    ):
        super().__init__()
        self.image_model = timm.create_model(
            model_args.image_pretrained, pretrained=model_args.pretrained
        )
        self.text_model_config = AutoConfig.from_pretrained(model_args.text_pretrained)
        self.text_model = AutoModel.from_pretrained(model_args.text_pretrained)
        self.image_model_out = nn.Linear(self.image_model.head.in_features, 512)
        self.text_model_out = nn.Linear(self.text_model_config.hidden_size, 512)
        self.drop = nn.Dropout(model_args.dropout)

        
        self.fc_out = nn.Linear(1512, 512)  # 512X256
        # self.out = nn.Linear(256, n_out) 
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