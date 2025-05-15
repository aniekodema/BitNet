# BitNet Inference Kernel

This repository provides a highly efficient GEMV kernel implementation for the BitNet model, optimized for W2A8 inference — 2-bit weights and 8-bit activations. It is tailored for use with the [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285) model.

## Features

- Support for W2A8 (2-bit weight × 8-bit activation) GEMV computation  
- Custom CUDA kernels with low-latency execution  
- Optimizations for memory access, decoding, and compute throughput  

## Usage

Installation and kernel performance tests:

```bash
# (Recommended) Create a new conda environment
conda create --name bitnet-gpu "python<3.13"
conda activate bitnet-gpu

# Install dependencies
pip install -r requirements.txt

# Build the kernel
cd bitnet_kernels
bash compile.sh
cd ..

# Run performance tests
python test.py
```

End-to-end inference:

```bash
# Download and convert the BitNet-b1.58-2B model
mkdir checkpoints
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./checkpoints/bitnet-b1.58-2B-4T-bf16
python ./convert_safetensors.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output checkpoints/model_state.pt --model_name 2B
python ./convert_checkpoint.py --input ./checkpoints/model_state.pt
rm ./checkpoints/model_state.pt

# Inference
python3 ./generate.py ./checkpoints/ --interactive --chat_format
```

## Optimizations

### Weight Permutation

The weight matrix is divided into 16×32 blocks to optimize memory access patterns.  

Within each block, values are stored contiguously in memory and permuted to facilitate efficient access and processing.  

See `convert_convert_checkpoint.py` for details.

### Fast Decoding

Every 16 two-bit values are packed into a single 32-bit integer using the following interleaving pattern:  
```
[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
```

This layout is designed to accelerate decoding by enabling efficient extraction of 4 values at a time into `int8`.

### `dp4a` Instruction

We use the `dp4a` instruction to accelerate low-precision dot product operations.  

This instruction performs a dot product between two 4-element vectors (each stored in a 32-bit word as 8-bit integers) and accumulates the result into a 32-bit integer.  

It significantly improves GEMV throughput when processing quantized weights and activations.


## Performance

| Shape (N×K)         | W2A8 Latency (us) | BF16 Latency (us) | Acceleration rate    |
|---------------------|-------------------|-------------------|----------------------|
| 2560 × 2560         | 12.54             | 13.73             |   1.09               |
| 3840 × 2560         | 11.75             | 14.61             |   1.24               |
| 13824 × 2560        | 13.52             | 50.04             |   3.70               |
| 2560 × 6912         | 11.75             | 30.26             |   2.58               |
| 3200 × 3200         | 11.65             | 14.61             |   1.25               |
| 4800 × 3200         | 11.72             | 17.15             |   1.46               |
| 3200 × 10240        | 13.74             | 49.53             |   3.60               |
| 20480 × 3200        | 21.91             | 92.71             |   4.23               |
