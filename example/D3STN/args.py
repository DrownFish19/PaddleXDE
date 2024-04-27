import argparse
import json

parser = argparse.ArgumentParser(description="Traffic Flow Forecasting")

# data config
parser.add_argument(
    "--dataset_name", type=str, default="HZME_OUTFLOW", help="dataset name"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="TrafficFlowData/HZME_OUTFLOW/HZME_OUTFLOW.npz",
)
parser.add_argument(
    "--adj_path",
    type=str,
    default="TrafficFlowData/HZME_OUTFLOW/HZME_OUTFLOW.csv",
)
parser.add_argument(
    "--sc_path",
    type=str,
    default="TrafficFlowData/HZME_OUTFLOW/SCORR_HZME_OUTFLOW.npy",
)
parser.add_argument("--split", type=str, default="6:2:2", help="data split")
parser.add_argument("--scale", type=bool, default=True, help="data norm scale")
parser.add_argument("--num_nodes", type=int, default=80)

# model config
parser.add_argument("--model_name", type=str, default="D3STN", help="model name")
parser.add_argument("--his_len", type=int, default=288, help="history data length")
parser.add_argument("--tgt_len", type=int, default=12, help="tgt data length")
parser.add_argument("--input_size", type=int, default=1)
parser.add_argument("--output_size", type=int, default=1)
parser.add_argument("--encoder_num_layers", type=int, default=2)
parser.add_argument("--decoder_num_layers", type=int, default=2)
parser.add_argument(
    "--d_model", type=int, default=64, help="d_proj+s_proj+t_proj+sect_proj"
)
parser.add_argument("--attention", type=str, default="Corr", help="Corr,Vanilla")
parser.add_argument("--head", type=int, default=8, help="head")
parser.add_argument("--kernel_size", type=int, default=3, help="kernel_size")
parser.add_argument("--top_k", type=int, default=5, help="top_k")
parser.add_argument("--smooth_layer_num", type=int, default=1)
parser.add_argument("--no_adj", type=bool, default=False, help="no adj")
parser.add_argument("--solver", type=str, default="euler", help="euler,midpoint,rk4")

# train config
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
parser.add_argument("--train_epochs", type=int, default=200, help="train epochs")
parser.add_argument("--finetune_epochs", type=int, default=100, help="finetune epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
parser.add_argument("--continue_training", type=bool, default=False, help="")
parser.add_argument("--distribute", type=bool, default=False, help="")


def get_args_from_json(json_file_path, args_obj):
    with open(json_file_path) as f:
        json_dict = json.load(fp=f)

    for key in json_dict.keys():
        setattr(args_obj, key, json_dict[key])

    return args_obj


args_obj = parser.parse_args()
# args = get_args_from_json("example/D3STN/configs/PEMS03.json", args_obj)
# args = get_args_from_json("example/D3STN/configs/PEMS04.json", args_obj)
# args = get_args_from_json("example/D3STN/configs/PEMS07.json", args_obj)
# args = get_args_from_json("example/D3STN/configs/PEMS08.json", args_obj)
# args = get_args_from_json("example/D3STN/configs/HZME_INFLOW.json", args_obj)
args = get_args_from_json("example/D3STN/configs/HZME_OUTFLOW.json", args_obj)
