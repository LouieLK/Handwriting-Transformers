import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["WANDB_API_KEY"] = ""

from pathlib import Path
import time
from data.dataset import TextDataset, TextDatasetval
from models import create_model
import torch
import cv2
import os
import numpy as np
from itertools import cycle
from scipy import linalg
from models.model import TRGAN
from params import *
from torch import nn
import wandb
import csv

def main():

    wandb.init(project="hwt-final", name = EXP_NAME)

    init_project()

    TextDatasetObj = TextDataset(num_examples = NUM_EXAMPLES)
    dataset = torch.utils.data.DataLoader(
                TextDatasetObj,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                persistent_workers=True,
                prefetch_factor=2,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObj.collate_fn)

    TextDatasetObjval = TextDatasetval(num_examples = NUM_EXAMPLES)
    datasetval = torch.utils.data.DataLoader(
                TextDatasetObjval,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                persistent_workers=True,
                prefetch_factor=2,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObjval.collate_fn)

    model = TRGAN()

    os.makedirs('saved_models', exist_ok = True)
    MODEL_PATH = os.path.join('saved_models', EXP_NAME)
    if os.path.isdir(MODEL_PATH) and RESUME: 
        model.load_state_dict(torch.load(MODEL_PATH+'/model.pth'))
        print (MODEL_PATH+' : Model loaded Successfully')
    else: 
        if not os.path.isdir(MODEL_PATH): os.mkdir(MODEL_PATH)


    for epoch in range(EPOCHS):
        
        start_time = time.time()
        
        for i,data in enumerate(dataset): 

            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_GWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:
                model._set_input(data)
                
                # 1. 永遠都要算 Loss 與梯度 (確保 model.loss_D 是最新鮮的)
                model.optimize_D_OCR()
                
                # 2. 只有當 D 的表現還不夠完美 (Loss > 0.2) 時，才「真正更新」權重
                if model.loss_D.item() > 0.2:
                    model.optimize_D_OCR_step()
                else:
                    # 選擇性：把梯度清空，以免累積到下次
                    model.optimizer_D.zero_grad() 

            # 同理，DWL 的部分也要先算再判斷，並且加入 hasattr 防呆！
            if (i % NUM_CRITIC_DWL_TRAIN) == 0:
                model._set_input(data)
                
                model.optimize_D_WL()
                
                # 避免初始化時屬性不存在的報錯
                if hasattr(model, 'loss_w_fake') and hasattr(model, 'loss_w_real'):
                    if (model.loss_w_fake.item() + model.loss_w_real.item()) > 0.2:
                        model.optimize_D_WL_step()
                    else:
                        model.optimizer_D.zero_grad()
                        model.optimizer_wl.zero_grad() # 確保原程式碼有這個 optimizer
                else:
                    model.optimize_D_WL_step()

        end_time = time.time()
        data_val = next(iter(datasetval))
        losses = model.get_current_losses()
        
        # 🌟 [新增] 自己把 Loss 寫進 CSV 檔案裡，永遠不怕找不到
        log_file = "my_loss_log.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 定義所有想要記錄的 loss 欄位 (對應 losses 字典裡的 key)
            loss_keys = ['G', 'D', 'Dfake', 'Dreal', 'OCR_fake', 'OCR_real', 'w_fake', 'w_real']
            
            # 如果是第一次建立檔案，先寫入標題
            if not file_exists:
                header = ['epoch'] + [f'loss_{k}' for k in loss_keys]
                writer.writerow(header)
            
            # 🌟 建立一個小函數：將 Tensor 轉為 float，並「只取小數點後 3 位」
            def to_float_rounded(val):
                num = float(val.item()) if hasattr(val, 'item') else float(val)
                return round(num, 3)
            
            # 依序列出 epoch 以及所有的 loss 數值
            row_data = [epoch] + [to_float_rounded(losses[k]) for k in loss_keys]
            writer.writerow(row_data)

        if epoch % 500 == 0:
            page = model._generate_page(model.sdata, model.input['swids'])
            page_val = model._generate_page(data_val['simg'].to(DEVICE), data_val['swids'])
            
            wandb.log({ "result":[wandb.Image(page, caption="page"),wandb.Image(page_val, caption="page_val")],
                        })
        
        wandb.log({'loss-G': losses['G'],
                   'loss-D': losses['D'], 
                   'loss-Dfake': losses['Dfake'],
                   'loss-Dreal': losses['Dreal'],
                   'loss-OCR_fake': losses['OCR_fake'],
                   'loss-OCR_real': losses['OCR_real'],
                   'loss-w_fake': losses['w_fake'],
                   'loss-w_real': losses['w_real'],
                   'epoch' : epoch,
                   'timeperepoch': end_time-start_time,
                   })

                    

        

        print ({'EPOCH':epoch, 'TIME':end_time-start_time, 'LOSSES': losses})

        if epoch % SAVE_MODEL == 0: torch.save(model.state_dict(), MODEL_PATH+ '/model.pth')
        if epoch % SAVE_MODEL_HISTORY == 0: torch.save(model.state_dict(), MODEL_PATH+ '/model'+str(epoch)+'.pth')


if __name__ == "__main__":
    
    main()
