# D<sup>3</sup>STN

This folder is the implement for "D<sup>3</sup>STN: Dynamic Delay Differential Equation Spatiotemporal Network for Traffic Flow Forecasting"

# Requirements
* paddlexde
* paddlepaddle-gpu

## Configurations Steps

Step 1: Install paddlepaddle-gpu.

```bash
# install from https://www.paddlepaddle.org.cn/
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# check
python -c "import paddle; paddle.utils.run_check()"
```

Step 2: Install paddlexde.

```bash
git clone https://github.com/DrownFish19/PaddleXDE.git
cd PaddleXDE; pip install -r requirements.txt

# use paddlexde by PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/PaddleXDE/
```

Step 3: Test paddle and paddlexde.

```bash
python -c "import paddle; import paddlexde"
```

# Train and Test

Step 1: Download Datasets: \
In the `PaddleXDE/examples/D3STN` folder, download the datasets.

```bash
git clone https://github.com/DrownFish19/TrafficFlowData.git
```

Step 2: Train and test on the HZME (outflow) dataset: \
In the `PaddleXDE/examples/D3STN` folder, train model for the HZME (outflow) dataset.

```bash
python train_dde.py --config configs/HZME_OUTFLOW.json
```

Step 3 (optional): Test and test (parallel) on the HZME (outflow) dataset: \
In the `PaddleXDE/examples/D3STN` folder, this step can train the model with data-parallel, the batch size is set for each GPU device to the total `batch_size / num_devices` .

```bash
# Note: the ACC will lower than thaining with one device, however, the speed is faster.
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_dde.py --config configs/HZME_OUTFLOW.json
```

# Model

The trained model wiil be stored in experiments folder. We will upload the trained model in experiments folder. (whne the shutdown server is available)

# Results

| Datasets (Highway Traffic Flow) |                  | PEMS07           |                 |                  | PEMS08           |                 |
|---------------------------------|------------------|------------------|-----------------|------------------|------------------|-----------------|
| Metrics                         | MAE              | RMSE             | MAPE(%)         | MAE              | RMSE             | MAPE(%)     ｜  |
| VAR (2003)                      | 101.20           | 155.14           | 39.69           | 22.32            | 33.83            | 14.47           |
| SVR (1997)                      | 32.97 ± 0.98     | 50.15 ± 0.15     | 15.43 ± 1.22    | 23.25 ± 0.01     | 36.15 ± 0.02     | 14.71 ± 0.16    |
| LSTM (1997)                     | 29.71 ± 0.09     | 45.32 ± 0.27     | 14.14 ± 1.00    | 22.19 ± 0.13     | 33.59 ± 0.05     | 18.74 ± 2.79    |
| DCRNN (2018)                    | 23.60 ± 0.05     | 36.51 ± 0.05     | 10.28 ± 0.02    | 18.22 ± 0.06     | 28.29 ± 0.09     | 11.56 ± 0.04    |
| STGCN (2018)                    | 27.41 ± 0.45     | 41.02 ± 0.58     | 12.23 ± 0.38    | 18.04 ± 0.19     | 27.94 ± 0.18     | 11.16 ± 0.10    |
| ASTGCN (2019)                   | 25.98 ± 0.78     | 39.65 ± 0.89     | 11.84 ± 0.69    | 18.86 ± 0.41     | 28.55 ± 0.49     | 12.50 ± 0.66    |
| GWN (2019)                      | 21.22 ± 0.24     | 34.12 ± 0.18     | 9.07 ± 0.20     | 15.07 ± 0.17     | 23.85 ± 0.18     | 9.51 ± 0.22     |
| GMAN (2020)                     | 21.56 ± 0.26     | 34.97 ± 0.44     | 9.51 ± 0.16     | 15.33 ± 0.03     | 26.10 ± 0.28     | 10.97 ± 0.37    |
| AGCRN (2020)                    | 22.56 ± 0.33     | 36.18 ± 0.46     | 9.67 ± 0.14     | 16.26 ± 0.43     | 25.62 ± 0.56     | 10.33 ± 0.34    |
| STSGCN (2020)                   | 23.99 ± 0.14     | 39.32 ± 0.31     | 10.10 ± 0.08    | 17.10 ± 0.04     | 26.83 ± 0.06     | 10.90 ± 0.05    |
| MTGNN (2020)                    | 20.57 ± 0.61     | 33.54 ± 0.73     | 9.12 ± 0.13     | 15.52 ± 0.06     | 25.59 ± 0.29     | 13.56 ± 1.11    |
| STFGNN (2021)                   | 22.07 ± 0.11     | 35.80 ± 0.18     | 9.21 ± 0.07     | 16.64 ± 0.09     | 26.22 ± 0.15     | 10.60 ± 0.06    |
| STGODE (2021)                   | 22.89 ± 0.15     | 37.47 ± 0.07     | 10.10 ± 0.06    | 16.79 ± 0.02     | 26.05 ± 0.11     | 10.58 ± 0.08    |
| DMSTGCN (2021)                  | 20.77 ± 0.57     | 33.67 ± 0.54     | 8.94 ± 0.42     | 16.02 ± 0.10     | 26.00 ± 0.21     | 10.28 ± 0.08    |
| ASTGNN (2021)                   | 20.62 ± 0.12     | 34.00 ± 0.21     | 8.86 ± 0.10     | 15.00 ± 0.35     | 24.70 ± 0.53     | 9.50 ± 0.11     |
| CorrSTN (2023)                  | 19.62 ± 0.05     | **33.11 ± 0.23** | 8.22 ± 0.06     | 14.23 ± 0.15     | **23.63 ± 0.14** | 9.30 ± 0.06     |
| D<sup>3</sup>STN (ours)         | **19.26 ± 0.13** | 33.29 ± 0.12     | **8.01 ± 0.04** | **13.84 ± 0.10** | 23.75 ± 0.12     | **9.10 ± 0.09** |

| Datasets (Metro Crowd Flow) |                  | HZME(inflow)     |                  |                  | HZME(outflow)    |                  |
|-----------------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Metrics                     | MAE              | RMSE             | MAPE(%)          | MAE              | RMSE             | MAPE(%)          |
| VAR (2003)                  | 17.65            | 28.10            | 58.07            | 22.35            | 37.96            | 96.68            |
| SVR (1997)                  | 21.94 ± 0.02     | 40.73 ± 0.02     | 49.40 ± 0.07     | 25.59 ± 0.12     | 50.07 ± 0.17     | 91.71 ± 3.18     |
| LSTM (1997)                 | 22.53 ± 0.51     | 39.33 ± 0.35     | 60.12 ± 2.44     | 26.18 ± 0.32     | 48.91 ± 0.45     | 103.06 ± 8.52    |
| DCRNN (2018)                | 12.25 ± 0.13     | 20.91 ± 0.33     | 25.53 ± 0.38     | 18.02 ± 0.16     | 31.45 ± 0.39     | 66.98 ± 1.65     |
| STGCN (2018)                | 12.88 ± 0.28     | 22.86 ± 0.39     | 29.66 ± 1.50     | 19.12 ± 0.23     | 33.12 ± 0.36     | 73.66 ± 1.49     |
| ASTGCN (2019)               | 13.10 ± 0.47     | 23.23 ± 0.81     | 33.29 ± 3.63     | 19.35 ± 0.51     | 33.20 ± 1.07     | 88.75 ± 4.00     |
| GWN (2019)                  | 11.20 ± 0.11     | 19.73 ± 0.46     | 23.75 ± 0.71     | 17.50 ± 0.12     | 30.65 ± 0.41     | 73.65 ± 2.72     |
| GMAN (2020)                 | 11.35 ± 0.20     | 20.60 ± 0.33     | 26.85 ± 0.72     | 18.03 ± 0.11     | 32.51 ± 0.37     | 74.57 ± 0.45     |
| AGCRN (2020)                | 11.86 ± 0.71     | 24.39 ± 0.73     | 30.93 ± 1.82     | 19.34 ± 1.27     | 33.85 ± 1.16     | 88.85 ± 0.48     |
| STSGCN (2020)               | 12.85 ± 0.10     | 23.20 ± 0.38     | 28.02 ± 0.19     | 18.74 ± 0.13     | 33.12 ± 0.43     | 76.85 ± 1.01     |
| MTGNN (2020)                | 11.99 ± 0.39     | 20.57 ± 0.55     | 26.87 ± 0.64     | 18.79 ± 0.80     | 32.27 ± 0.60     | 87.63 ± 3.84     |
| STFGNN (2021)               | 13.12 ± 0.23     | 23.02 ± 0.37     | 30.67 ± 0.53     | 18.90 ± 0.18     | 34.12 ± 0.43     | 77.32 ± 2.33     |
| STGODE (2021)               | 11.36 ± 0.06     | 22.02 ± 0.14     | 40.50 ± 1.01     | 19.43 ± 0.38     | 33.67 ± 0.64     | 89.90 ± 2.57     |
| DMSTGCN (2021)              | 12.64 ± 0.28     | 21.79 ± 0.53     | 28.21 ± 0.75     | 18.52 ± 0.37     | 32.26 ± 0.84     | 77.08 ± 0.76     |
| ASTGNN (2021)               | 11.46 ± 0.08     | 20.84 ± 0.25     | 24.42 ± 0.30     | 17.94 ± 0.11     | 31.91 ± 0.32     | 72.46 ± 2.42     |
| CorrSTN (2023)              | 11.20 ± 0.06     | 19.71 ± 0.14     | 24.04 ± 0.38     | 17.29 ± 0.08     | 30.66 ± 0.15     | 65.33 ± 0.26     |
| D<sup>3</sup>STN (ours)     | **10.92 ± 0.15** | **18.09 ± 0.55** | **22.81 ± 0.20** | **15.40 ± 0.11** | **25.68 ± 0.24** | **49.28 ± 1.23** |

# Citation

If you found use this library useful, please consider citing

```bibtex
under review
```
