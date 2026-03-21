import os
import glob
import pickle

def build_dataset(base_dir):
    dataset_dict = {'train': {}, 'test': {}}
    all_chars = set()

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
                label = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
                all_chars.add(label)
                
                # 🌟 [關鍵修改] 絕對不要在這裡讀取 Image！只儲存「絕對路徑」
                dataset_dict[target_split][writer].append({
                    'img_path': os.path.abspath(img_path), # 改存路徑字串
                    'label': label
                })
                
    return dataset_dict, "".join(list(all_chars))

# 執行轉換
data, alphabet = build_dataset('/workspace/One-DM/CASIA-HWDB1.0-1.1/data')

# 儲存為 pickle 檔放入 files 資料夾
os.makedirs('files', exist_ok=True)
with open('files/CHINESE-128.pickle', 'wb') as f:
    pickle.dump(data, f)

print("✅ Pickle 檔案建立完成！(動態讀取版，徹底解決 OOM)")
print(f"📊 總共收集到了 {len(alphabet)} 個不同的字元。")
print("👇 請將以下字元複製到 params.py 的 ALPHABET 變數中：")
print(alphabet)