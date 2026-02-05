- nanogpt
  - demonstrate how deleting the mask makes the model cheat
  - EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).
  - EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? (A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're hoping it learns) would proceed right to left too. You may want to modify the data loader to simply serve random problems and skip the generation of train.bin, val.bin. You may want to mask out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets (see CrossEntropyLoss ignore_index). Does your Transformer learn to add? Once you have this, swole doge project: build a calculator clone in GPT, for all of +-\*/. Not an easy problem. You may need Chain of Thought traces.)
  - EX3: Find a dataset that is very large, so large that you can't see a gap between train and val loss. Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?
  - EX4: Read some transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?
- visualize activations and gradients in mlp_pytorch.ipynb
  - <https://www.youtube.com/watch?v=syLFCVYua6Q>
  - <https://docs.pytorch.org/tutorials/intermediate/visualizing_gradients_tutorial.html>
  - second half of <https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5>
- do backprop ninja exercises in mlp_backprop_ninja.ipynb
- karpathy reads
  - <https://karpathy.github.io/2015/10/25/selfie/>
  - <https://karpathy.github.io/2019/04/25/recipe/>
- impl repro 1989
  - <https://karpathy.github.io/2022/03/14/lecun1989/>
- impl rnn
  - [NLP from scratch PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html)
  - <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>
  - <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
- auto-encoders
- diffusion
  - <https://poloclub.github.io/diffusion-explainer/>
- transformer
  - <https://poloclub.github.io/transformer-explainer/>
- gan
  - <https://poloclub.github.io/ganlab/>
- RL stuff
  - <https://www.youtube.com/watch?v=cQfOQcpYRzE>
  - <https://karpathy.github.io/2016/05/31/rl/>
- finish pytorch intro doc blogs
  - <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- papers
  - a neural probabislitic bengio
  - wavenet
  - attention is all you need
- <https://openai.com/index/chatgpt/>

squeue --me
watch tail blabla.out
scancel 1234567
idle_gpus

<!-- reports cpu usage for a specified job, its runtime, and memory usage in MB. -->

seff 1234567

<!-- reports recent jobs you ran, their status, and some resource information. -->

sacct
