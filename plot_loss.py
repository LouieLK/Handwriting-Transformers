import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv_loss():
    log_file = 'my_loss_log.csv'
    
    if not os.path.exists(log_file):
        print(f"找不到 {log_file}！請確認模型已經開始訓練並產生了記錄檔。")
        return

    # 讀取 CSV
    df = pd.read_csv(log_file)

    if len(df) == 0:
        print("記錄檔目前是空的，請等模型多跑幾個 Epoch 再試試！")
        return

    # 開始畫圖，稍微把圖拉寬一點以容納圖例
    plt.figure(figsize=(14, 7))
    
    # --- 1. GAN 的對抗損失 ---
    plt.plot(df['epoch'], df['loss_G'], label='Loss G (Total)', linewidth=2, color='blue')
    plt.plot(df['epoch'], df['loss_D'], label='Loss D (Total)', linewidth=2, color='green')
    # 將 Dfake 和 Dreal 用較細的虛線表示，作為輔助觀察
    plt.plot(df['epoch'], df['loss_Dfake'], label='Loss D (Fake)', linestyle='--', alpha=0.5, color='cyan')
    plt.plot(df['epoch'], df['loss_Dreal'], label='Loss D (Real)', linestyle='--', alpha=0.5, color='lime')
    
    # --- 2. OCR 的文字辨識損失 (我們最關心的部分) ---
    plt.plot(df['epoch'], df['loss_OCR_real'], label='Loss OCR (Real)', linewidth=2, color='red')
    # 加入了 OCR_fake，用醒目的橘色，觀察畫家有沒有騙過 OCR 老師
    plt.plot(df['epoch'], df['loss_OCR_fake'], label='Loss OCR (Fake)', linewidth=2, color='orange')
    
    # --- 3. Style (W) 的風格損失 ---
    plt.plot(df['epoch'], df['loss_w_fake'], label='Loss W (Fake)', linestyle='-.', alpha=0.7, color='purple')
    plt.plot(df['epoch'], df['loss_w_real'], label='Loss W (Real)', linestyle='-.', alpha=0.7, color='magenta')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Loss Trends (Detailed)', fontsize=16)
    
    # 將圖例移到圖表外側，避免擋住折線
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 限制 Y 軸的顯示範圍 (根據您的需求可能需要微調，因為 OCR loss 剛開始可能到 8~10)
    plt.ylim(bottom=-2, top=12) 
    
    # 儲存為圖片
    output_img = 'loss_trends.png'
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"✅ 成功！詳細 Loss 走勢圖已儲存為：{output_img}")

if __name__ == "__main__":
    plot_csv_loss()