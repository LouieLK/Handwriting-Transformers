import os
import glob
import pickle
from PIL import Image

# 假設您的資料集結構為:
# dataset/
# ├── train/
# │   ├── writer_1/
# │   │   ├── 字.png
# │   │   └── ...
# │   └── writer_2/
# └── test/
#     ├── writer_3/
#     └── ...

def build_dataset(base_dir):
    dataset_dict = {'train': {}, 'test': {}}
    all_chars = set()

    for split in ['train_seen_2000', 'test_seen_2000']:
        split_dir = os.path.join(base_dir, split)
        writers = os.listdir(split_dir)
        
        for writer in writers:
            writer_dir = os.path.join(split_dir, writer)
            if not os.path.isdir(writer_dir): continue
            
            dataset_dict[split.split('_')[0]][writer] = []
            img_paths = glob.glob(os.path.join(writer_dir, '*.*'))
            
            for img_path in img_paths:
                # 假設檔名就是該中文字，例如 "永.png" -> "永"
                label = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
                all_chars.add(label.split('_')[0])
                
                # 讀取圖片，轉灰階，並 Resize 為 32x32 以符合模型原始架構
                img = Image.open(img_path).convert('L')
                # img = img.resize((32, 32), Image.Resampling.LANCZOS)
                
                dataset_dict[split.split('_')[0]][writer].append({
                    'img': img,
                    'label': label
                })
                
    return dataset_dict, "".join(list(all_chars))

# 執行轉換
data, alphabet = build_dataset('/workspace/One-DM/CASIA-HWDB1.0-1.1/data')

# 儲存為 pickle 檔放入 files 資料夾
os.makedirs('files', exist_ok=True)
with open('files/CHINESE-128.pickle', 'wb') as f:
    pickle.dump(data, f)

print("Pickle 檔案建立完成！")
print("請將以下字元複製到 params.py 的 ALPHABET 變數中：")
print(alphabet)