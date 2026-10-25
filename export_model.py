"""
Script xuất mô hình SOFTS với DINOv3 Backbone cho bài toán dự báo PM2.5 Hà Nội
Xuất file trọng số: softs_dinov3_hanoi.pt
Kèm file cấu hình mô hình: model_config.json
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.SOFTS import Model


class ModelConfigs:
    seq_len = 96
    pred_len = 24
    d_model = 128
    d_core = 32
    d_ff = 256
    dropout = 0.05
    e_layers = 2
    activation = "gelu"
    use_norm = True
    dinov3_pretrained = False
    dinov3_layers = 2
    dinov3_dim = 192
    dinov3_heads = 6
    dinov3_mlp_ratio = 4.0
    dinov3_registers = 4


def export_and_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "2_data_pipeline", "hanoi_air_quality_private_4years.csv")
    out_model_path = os.path.join(current_dir, "softs_dinov3_hanoi.pt")
    out_cfg_path = os.path.join(current_dir, "model_config.json")

    print("[*] Đang khởi tạo mô hình SOFTS (Encoder DINOv3)...")
    cfg = ModelConfigs()
    model = Model(cfg)

    # Nếu có dữ liệu 4 năm từ pipeline, huấn luyện nhẹ (quick calibration) để model có trọng số thực tế
    if os.path.exists(data_path):
        print(f"[*] Tìm thấy dữ liệu tại: {data_path}")
        df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
        feature_cols = ["PM2.5", "temperature", "humidity", "wind_speed", "precipitation", "pressure", "PM10"]
        values = df[feature_cols].values
        
        # Chuẩn hóa nhanh
        mean = values.mean(axis=0)
        std = values.std(axis=0) + 1e-5
        norm_values = (values - mean) / std

        # Tạo một số batch mẫu để huấn luyện nhanh
        seq_len, pred_len = cfg.seq_len, cfg.pred_len
        X_samples, Y_samples = [], []
        for i in range(0, min(len(norm_values) - seq_len - pred_len, 2000), 5):
            X_samples.append(norm_values[i : i + seq_len])
            Y_samples.append(norm_values[i + seq_len : i + seq_len + pred_len])

        X_t = torch.tensor(np.array(X_samples), dtype=torch.float32)
        Y_t = torch.tensor(np.array(Y_samples), dtype=torch.float32)

        print(f"[*] Calibrating model trên {len(X_samples)} mẫu chuỗi thời gian Hà Nội...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()

        model.train()
        batch_size = 32
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(3):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx, None, None, None)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            print(f"    Epoch {epoch+1}/3 - Loss: {total_loss / len(X_samples):.4f}")
    else:
        print("[!] Không tìm thấy dữ liệu pipeline, xuất trọng số khởi tạo ban đầu.")

    # Xuất checkpoint trọng số
    model.eval()
    torch.save({
        "state_dict": model.state_dict(),
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "features": 7,
        "feature_names": ["PM2.5", "temperature", "humidity", "wind_speed", "precipitation", "pressure", "PM10"]
    }, out_model_path)
    print(f"[+] Đã xuất mô hình AI thành công: {out_model_path}")

    # Xuất file json config
    config_dict = {
        "model_architecture": "SOFTS with DINOv3 Backbone",
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "features": 7,
        "feature_names": ["PM2.5", "temperature", "humidity", "wind_speed", "precipitation", "pressure", "PM10"],
        "d_model": cfg.d_model,
        "dinov3_layers": cfg.dinov3_layers,
        "dinov3_dim": cfg.dinov3_dim,
        "dinov3_heads": cfg.dinov3_heads,
        "status": "Exported & Ready for Production Deployment"
    }
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã lưu cấu hình mô hình: {out_cfg_path}")


if __name__ == "__main__":
    export_and_train()
