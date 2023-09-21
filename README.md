# PromptGenTF
A Simple and Efficient PromptGen Inference Implementation In TensorFlow 2.

Super easy to use
=======

```python
    from gpt2 import GPT2
    from gpt2_tokenizer import GPT2Tokenizer
    from gpt2_causal_lm import GPT2CausalLMPreprocessor, GPT2CausalLM
    sequence_length = 128
    max_length = 1024
    seed = 123
    tokenizer = GPT2Tokenizer()
    preprocessor = GPT2CausalLMPreprocessor(tokenizer=tokenizer, sequence_length=sequence_length)
    gpt2_backbone = GPT2(vocabulary_size=50257, num_layers=6, num_heads=12, hidden_dim=768, intermediate_dim=3072,
                         max_sequence_length=1024)
    # download form:
    # https://huggingface.co/AUTOMATIC/promptgen-lexart/blob/main/pytorch_model.bin
    pytorch_model = "pytorch_model.bin"
    gpt2_backbone.load_weights_from_ckpt(pytorch_model)
    gpt2_lm = GPT2CausalLM(
        backbone=gpt2_backbone,
        preprocessor=preprocessor, seed=seed)
    z = gpt2_lm.generate(["a girl.", "a cat."], max_length=max_length)
    print(z)
```

## Credits
Licenses for borrowed code can be found in following link:

- PromptGenerator - https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen
- KerasNLP - https://github.com/keras-team/keras-nlp
- Transformers - https://github.com/huggingface/transformers

## Donating 
If this project useful for you, please consider buying me a cup of coffee or sponsoring me!

<a href="https://paypal.me/cpuimage/USD10" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/black_img.png" alt="Buy Me A Coffee" style="height: auto !important;width: auto !important;" ></a>
