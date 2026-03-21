import os
import glob
import pickle
from PIL import Image

# 🌟 在這裡放入您想測試的中文字串 (這裡我預設填入最常用的 100 個中文字作為範例，您可以隨意替換)
ALPHABET = '池律聂猾噶窟滴笔呐慕暖被觉才仇可雹鲤陆绑顶攻么贷满劳澜牛韭攫剐厚东诀猫监亮仓奎彼谷距拉棵愁界昏帛阿督浮菱够谗旷挥疥蛮娜肋何较跺垫掇廊迪靠答罚犁领秘褐寐猛叭膊凑谰份锚恨皇忽鹅铝肛互眯烩梦鸟架剑蜡齿灿卖方'

def build_dataset(base_dir, target_chars):
    dataset_dict = {'train': {}, 'test': {}}
    all_chars = set()
    
    # 將字串轉為集合 (Set)，這會讓後續搜尋過濾的速度提升百倍
    target_chars_set = set(target_chars)

    for split in ['train_seen_2000', 'test_seen_2000']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        writers = os.listdir(split_dir)
        target_split = split.split('_')[0]
        
        for writer in writers:
            writer_dir = os.path.join(split_dir, writer)
            if not os.path.isdir(writer_dir): continue
            
            dataset_dict[target_split][writer] = []
            img_paths = glob.glob(os.path.join(writer_dir, '*.*'))
            
            for img_path in img_paths:
                # 取得標籤
                label = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
                
                # 🌟 [關鍵過濾] 如果這個字不在我們設定的測試字串內，就直接跳過！
                if label not in target_chars_set:
                    continue
                    
                all_chars.add(label)
                
                # 讀取圖片，轉灰階，並 Resize 為 32x32 以符合模型原始架構
                img = Image.open(img_path).convert('L')
                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                
                dataset_dict[target_split][writer].append({
                    'img': img,
                    'label': label
                })
                
    return dataset_dict, "".join(list(all_chars))

# 執行轉換
data, alphabet = build_dataset('/workspace/One-DM/CASIA-HWDB1.0-1.1/data', ALPHABET)

# 儲存為 mini 版的 pickle 檔放入 files 資料夾
os.makedirs('files', exist_ok=True)
with open('files/CHINESE-32-mini.pickle', 'wb') as f:
    pickle.dump(data, f)

print("✅ Pickle 檔案建立完成！")
print(f"📊 總共收集到了 {len(alphabet)} 個不同的字元。")
print("👇 請將以下字元複製到 params.py 的 ALPHABET 變數中：")
print(alphabet)