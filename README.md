# PaddleXDE
PaddleXDE is a libarary that helps you build deep learning applications for PaddlePaddle using ordinary differential equations.

## Installation

To install latest on GitHub:
```bash
pip install git+https://github.com/DrownFish19/paddlexde.git
```


## Examples

Examples are placed in the `example` directory.

### ODE DEMO
![ODE DEMO](./example/ode.gif)

### ODE Adjoint DEMO
![ODE Adjoint DEMO](./example/ode-adjoint.gif)

## Requirements

```text
paddle
```

## TODO List
- [ ] add dde example
- [ ] add dde tests
- [ ] test for dp etc.
- [ ] 数据reshape 并行问题 数据reshape时，不应该reshape batchsize维度，

## Acknowledgments

* [torchdiffeq](https://github.com/rtqichen/torchdiffeq.git)
* [torchsde](https://github.com/google-research/torchsde.git)
* [torchcde](https://github.com/patrick-kidger/torchcde.git)
* [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl.git)
