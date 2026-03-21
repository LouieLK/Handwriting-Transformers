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

    # 開始畫圖
    plt.figure(figsize=(12, 6))
    
    # 畫出三條關鍵的 Loss 曲線
    plt.plot(df['epoch'], df['loss_G'], label='Loss G (Generator)', alpha=0.8)
    plt.plot(df['epoch'], df['loss_D'], label='Loss D (Discriminator)', alpha=0.8)
    plt.plot(df['epoch'], df['loss_OCR_real'], label='Loss OCR (Real / miss)', linewidth=2, color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Trends (From CSV)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 限制 Y 軸的顯示範圍 (防止剛開始訓練時 Loss 爆炸導致後面的線看不清楚)
    plt.ylim(bottom=-5, top=20) 
    
    # 儲存為圖片
    output_img = 'loss_trends.png'
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"✅ 成功！Loss 走勢圖已儲存為：{output_img}")

if __name__ == "__main__":
    plot_csv_loss()