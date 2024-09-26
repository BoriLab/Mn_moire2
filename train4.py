# %%
from data.qm9.qm9 import QM9Dataset
from data.pcqm.pcqm import PCQM4Mv2Dataset
from src.mymodel.layers import MoireLayer, get_moire_focus
from utils.exp import Aliquot, set_device
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import torch

CONFIG = {
    "MODEL": "Moire",
    "DATASET": "QM9",
    "DEPTH": 13,  # [3 5 8 13 21]
    "MLP_DIM": 256,
    "HEADS": 32,
    "FOCUS": "gaussian",
    "DROPOUT": 0.1,
    "BATCH_SIZE": 1024,
    "LEARNING_RATE": 5e-5, # [5e-4, 5e-5] 범위
    "WEIGHT_DECAY": 1e-2, # lr 줄어드는 속도. 현재 값이 기본값
    "T_MAX": 350, # wandb에서 보고 1 epoch에 들어 있는 step size의 5배를 해줘세요
    "ETA_MIN": 1e-7, # lr 최솟값. 보통 조정할 필요 없음.
    "DEVICE": "cuda",
    "SCALE_MIN": 0.6, # shift 최솟값.
    "SCALE_MAX": 3.0, # shift 최댓값.
    "WIDTH_BASE": 1.15, # 보통 조정할 필요 없음.
}


set_device(CONFIG["DEVICE"])
dataset = None
match CONFIG["DATASET"]:
    case "QM9":
        dataset = QM9Dataset(path="../qm9_data.pkl")
        criterion = nn.L1Loss()
        dataset.unsqueeze_target()
    case "PCQM4Mv2":
        dataset = PCQM4Mv2Dataset(path="../../pcqm4mv2_data.pkl")
        criterion = nn.L1Loss()
        dataset.unsqueeze_target()
dataset.float()
dataset.batch_size = CONFIG["BATCH_SIZE"]


# %%
class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        dims = config["MLP_DIM"]
        self.input = nn.Sequential(
            nn.Linear(dataset.node_feat_size, dims),
            nn.Linear(dims, dims),
        )
        scales = torch.linspace(
            config["SCALE_MAX"], config["SCALE_MIN"], config["DEPTH"]
        )  # Generate `DEPTH` scales
        self.layers = nn.ModuleList(
            [
                MoireLayer(
                    dims,
                    dims,
                    config["HEADS"],
                    get_moire_focus(CONFIG["FOCUS"]),
                    scale,
                    CONFIG["WIDTH_BASE"] ** scale,
                    CONFIG["DROPOUT"],
                )
                for scale in scales
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Linear(dims, dataset.prediction_size),
        )

    def forward(self, x, adj, mask):
        x = self.input(x)
        x = x * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adj, mask)
        x, _ = x.max(dim=1)
        return self.output(x)


model = MyModel(CONFIG)
if CONFIG["DEVICE"] == "cuda":
    model = nn.DataParallel(model)
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["T_MAX"], eta_min=CONFIG["ETA_MIN"]
)
criterion = nn.L1Loss()
aliquot1 = Aliquot(model, dataset, optimizer, criterion, scheduler)
Aliquot(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
)(wandb_project="moire", wandb_config=CONFIG, num_epochs=1000, patience=20)
