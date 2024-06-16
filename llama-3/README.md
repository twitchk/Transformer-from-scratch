# Chinese fine-tuning llama-3

This directory is used to store the code for Chinese fine-tuning llama-3. It has nothing to do with other directories.

Currently, the fastest and most GPU-saving way to fine-tune llama-3 is through the usloth method. This method is based on llama-3, and they pre-quantize it to 4 bits to reduce the memory required for fine-tuning.

The advantage of this method is that there is no need to retrain the model, just download the pre-trained model and then fine-tune it.

The disadvantage of this method is that due to the quantization to 4 bits, the accuracy of the model will decrease, but since the accuracy of llama-3 itself is very high, this decrease is acceptable.

#### Install dependencies
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install torch transformers
pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
```

Many machines do not support `flash-attn`, so you can directly comment out the `packaging ninja einops flash-attn` libraries without affecting the use.

It is likely that a GPU is required to run. If there is no GPU, Colab can be used, but the GPU of Colab may be limited, so OOM may occur.
