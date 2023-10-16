import argparse
import os

parser = argparse.ArgumentParser(description="Traffic Flow Forecasting")
# device config
parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7", help="devices")
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")

# data config
parser.add_argument(
    "--dataset_name", type=str, default="HZME_OUTFLOW", help="dataset name"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="example/CorrSTN/TrafficFlowData/HZME_OUTFLOW/HZME_OUTFLOW.npz",
)
parser.add_argument(
    "--adj_path",
    type=str,
    default="example/CorrSTN/TrafficFlowData/HZME_OUTFLOW/HZME_OUTFLOW.csv",
)
parser.add_argument(
    "--sc_path",
    type=str,
    default="example/CorrSTN/TrafficFlowData/HZME_OUTFLOW/SCORR_HZME_OUTFLOW.npy",
)
parser.add_argument("--split", type=str, default="6:2:2", help="data split")
parser.add_argument("--scale", type=bool, default=True, help="data norm scale")
parser.add_argument("--num_nodes", type=int, default=80)


# model config
parser.add_argument("--model_name", type=str, default="WindSTN", help="model name")
parser.add_argument("--his_len", type=int, default=288, help="history data length")
parser.add_argument("--tgt_len", type=int, default=12, help="tgt data length")
parser.add_argument("--encoder_input_size", type=int, default=1)
parser.add_argument("--decoder_input_size", type=int, default=1)
parser.add_argument("--decoder_output_size", type=int, default=1)
parser.add_argument("--encoder_num_layers", type=int, default=4)
parser.add_argument("--decoder_num_layers", type=int, default=4)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--attention", type=str, default="Corr", help="Corr,Vanilla")
parser.add_argument("--split_seq", type=bool, default=False, help="split q k v")
parser.add_argument("--head", type=int, default=8, help="head")
parser.add_argument("--kernel_size", type=int, default=3, help="kernel_size")
parser.add_argument("--top_k", type=int, default=5, help="top_k")
parser.add_argument("--smooth_layer_num", type=int, default=1)


# train config
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
parser.add_argument("--train_epochs", type=int, default=75, help="train epochs")
parser.add_argument("--finetune_epochs", type=int, default=50, help="finetune epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
parser.add_argument("--patience", type=int, default=8, help="early stopping patience")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
parser.add_argument("--continue_training", type=bool, default=False, help="")
parser.add_argument("--fp16", type=bool, default=False, help="")
parser.add_argument("--distribute", type=bool, default=False, help="")

args = parser.parse_args("")
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
