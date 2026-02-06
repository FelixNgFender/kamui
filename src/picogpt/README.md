# gpt from scratch

## character-level bigram

```txt
CharBigram(
  (embedding): Embedding(65, 65)
)
```

simple (vocab_size, vocab_size) embedding matrix

### bigram loss curve

![loss_curve_char_bigram](char_bigram_loss.png)

### bigram results

before training

```txt
train set avg loss: 4.795889

val set avg loss: 4.789756

Aw'TnWDK&OZlAKmSp!-dY-y Z,kAFEArYIfO$$LCiA;N,o.V!ctazfB?SUIeYFo.OpZnkB&3BMwebcQN.qxYhfM,akHYZ3GMlSPr
```

after training for 2 epochs

```txt
train set avg loss: 2.452482

val set avg loss: 2.487851


EEl wofavante lind s:
ANIOUr r VIDUESSow.
Jutod 'sothandora sdeneperen 't:
Yofth ll g's sed liset; i
```

## character-level transformer

### transformer loss curve

![Transformer loss curve](char_transformer_loss.png)

### transformer results

before training

```txt
val set avg loss: 4.308069

;wtEnWDo&OUlAicSp!-di!lAZycWFa;GYtfxf$LC!TTu,okI!ctqLfBlSsIeDFotkpRv&Bi3;Mihbrp&CsxTh MoEkT&j3GllSPrC'lljyoyHW
JqBjEUvAq?MbpComIy,BhM-SBglPHYSp!Ugko&
oSCdwuhpTEyWb-idcy$;lSJxBQmd;Ri?Pq&gynzQOUou;VD:yjWdKuu-,wFBuFusg shnAtlDK?yG,.IhOdMHioJU;XOyO:P3
DCXF&qgo&x;n?&r$veCJqU?Kgl-ILxPFcyoJPhW3WypXNX
BJPTfuUlZ.gfkuKAtYFqinGTX:vVir,&r?LZVt3H&$SvHRZJuwHXp'!qx;KN&h&T WdAEJ&DTT$
',:vYcRg&ClEMg', JEdXuGGri;;i'XQrcILQrFuCEVKXwpi&EmqIGdT-VLer!D?Ov3wOvwHrgj-h-F&PqzMi:Coq? lg-R3Ao'P KgJHicOIouVvogpo&JOY3?PguQkS
```

after training for 5 epochs

```txt
best val set avg loss (epoch=1): 2.2700939

(no train set because takes too long)
final val set avg loss (epoch=5): 3.3807906
```

[sampling at epoch 5](./media/char_transformer_sample.txt).

## sampling with cli

best weights-only checkpoints for both models are saved in [the weights directory](../../weights/). you can sample from them using the picogpt cli:

```bash
picogpt sample char-transformer --checkpoint weights/char_transformer/20260127_225553/best.pt --tokenizer-dir weights/char_tokenizer/20260127_225553/ --tokens 10000
```

```python
if __name__ == "__main__":
    import tiktoken

    starting_sentence = "Hello, I am Jojo"
    num_return_sequences = 5
    max_length = 3000
    model = GPT2(1024, 50257, 768, 12, 12, 4).to("cuda")
    # model = GPT2.from_pretrained("gpt2").to("cuda")
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    tokens = (
        torch.tensor(enc.encode(starting_sentence), device="cuda", dtype=torch.long)
        .unsqueeze(0)
        .repeat(num_return_sequences, 1)
    )
    with torch.inference_mode():
        out = model.generate(tokens, max_new_tokens=100)
    for i in range(num_return_sequences):
        print(f"---------------------GENERATION {i + 1}----------------------")
        print(enc.decode(out[i].tolist()))
```

```txt
RTX 3070 Mobile (B=2 due to low VRAM, T=1024)
baseline:
[02/04/26 16:01:45] INFO     avg loss: 11.0153952  [2/300939], dt: 374.26ms, tok/sec: 5472.15
[02/04/26 16:02:23] INFO     avg loss: 6.0297818  [202/300939], dt: 367.06ms, tok/sec: 5579.4
[02/04/26 16:03:03] INFO     avg loss: 5.5636396  [402/300939], dt: 445.35ms, tok/sec: 4598.6
[02/04/26 16:03:42] INFO     avg loss: 5.2534924  [602/300939], dt: 466.72ms, tok/sec: 4388.0

+ tf32:
[02/04/26 16:20:29] INFO     avg loss: 11.0153847  [2/300939], dt: 326.36ms, tok/sec: 6275.27
[02/04/26 16:20:54] INFO     avg loss: 6.0301108  [202/300939], dt: 247.13ms, tok/sec: 8287.15
[02/04/26 16:21:19] INFO     avg loss: 5.5642977  [402/300939], dt: 248.25ms, tok/sec: 8249.87
[02/04/26 16:21:45] INFO     avg loss: 5.2592173  [602/300939], dt: 250.40ms, tok/sec: 8178.79

+ bf16 amp:
[02/04/26 22:19:36] INFO     avg loss: 11.0151062  [2/300939], dt: 370.53ms, tok/sec: 5527.19
[02/04/26 22:19:56] INFO     avg loss: 6.0103779  [202/300939], dt: 228.22ms, tok/sec: 8973.84
[02/04/26 22:20:16] INFO     avg loss: 5.5325618  [402/300939], dt: 194.62ms, tok/sec: 10523.11
[02/04/26 22:20:36] INFO     avg loss: 5.2103310  [602/300939], dt: 192.50ms, tok/sec: 10639.08
[02/04/26 22:20:55] INFO     avg loss: 4.7403345  [802/300939], dt: 192.89ms, tok/sec: 10617.49
[02/04/26 22:21:15] INFO     avg loss: 4.9395514  [1002/300939], dt: 192.67ms, tok/sec: 10629.34

+ torch.compile:
[02/04/26 23:36:21] INFO     avg loss: 11.0150146  [2/300939], dt: 30584.21ms, tok/sec: 66.96
[02/04/26 23:36:34] INFO     avg loss: 6.0148702  [202/300939], dt: 134.47ms, tok/sec: 15229.78
[02/04/26 23:36:48] INFO     avg loss: 5.5449467  [402/300939], dt: 133.32ms, tok/sec: 15361.36
[02/04/26 23:37:03] INFO     avg loss: 5.2269106  [602/300939], dt: 133.79ms, tok/sec: 15307.92
[02/04/26 23:37:18] INFO     avg loss: 4.7674112  [802/300939], dt: 134.53ms, tok/sec: 15222.96
[02/04/26 23:37:32] INFO     avg loss: 4.9472113  [1002/300939], dt: 149.39ms, tok/sec: 13709.17
[02/04/26 23:37:47] INFO     avg loss: 4.7375345  [1202/300939], dt: 135.62ms, tok/sec: 15100.57
[02/04/26 23:38:02] INFO     avg loss: 4.7569876  [1402/300939], dt: 158.54ms, tok/sec: 12918.06
[02/04/26 23:38:17] INFO     avg loss: 4.5299778  [1602/300939], dt: 149.32ms, tok/sec: 13715.50
[02/04/26 23:38:32] INFO     avg loss: 4.2007027  [1802/300939], dt: 141.47ms, tok/sec: 14476.52

+ flash attention:
[02/05/26 01:53:58] INFO     avg loss: 11.0150146  [2/300939], dt: 28098.60ms, tok/sec: 72.89
[02/05/26 01:54:12] INFO     avg loss: 6.0075874  [202/300939], dt: 122.42ms, tok/sec: 16729.49
[02/05/26 01:54:25] INFO     avg loss: 5.5268993  [402/300939], dt: 123.39ms, tok/sec: 16597.75
[02/05/26 01:54:38] INFO     avg loss: 5.2089729  [602/300939], dt: 122.03ms, tok/sec: 16782.74
[02/05/26 01:54:51] INFO     avg loss: 4.7597485  [802/300939], dt: 114.93ms, tok/sec: 17819.05
[02/05/26 01:55:02] INFO     avg loss: 4.9374390  [1002/300939], dt: 109.99ms, tok/sec: 18619.50
[02/05/26 01:55:13] INFO     avg loss: 4.7453871  [1202/300939], dt: 109.60ms, tok/sec: 18685.86

+ nice numbers for cuda
[02/05/26 02:44:46] INFO     avg loss: 10.9714661  [2/300939], dt: 1791.07ms, tok/sec: 1143.45
[02/05/26 02:44:57] INFO     avg loss: 6.0992756  [202/300939], dt: 115.86ms, tok/sec: 17676.84
[02/05/26 02:45:08] INFO     avg loss: 5.7664318  [402/300939], dt: 116.55ms, tok/sec: 17571.56
[02/05/26 02:45:19] INFO     avg loss: 5.3053112  [602/300939], dt: 105.89ms, tok/sec: 19340.25
[02/05/26 02:45:31] INFO     avg loss: 5.0617161  [802/300939], dt: 106.64ms, tok/sec: 19204.82
[02/05/26 02:45:42] INFO     avg loss: 4.3650179  [1002/300939], dt: 109.20ms, tok/sec: 18754.01
[02/05/26 02:45:53] INFO     avg loss: 4.6540136  [1202/300939], dt: 106.55ms, tok/sec: 19221.16

+ lr cosine decay scheduler (not sure why the perf regression here, torch.compile is erratic here too):
[02/05/26 19:09:06] INFO     [      2/ 300939] | loss: 10.9714661 | lr: 6.0000e-05 | norm: 31.2491 | dt: 1838.66ms | tok/sec:  1113.86
[02/05/26 19:09:17] INFO     [    202/ 300939] | loss: 6.1512179 | lr: 6.0000e-05 | norm: 0.9656 | dt: 112.37ms | tok/sec: 18226.11
[02/05/26 19:09:28] INFO     [    402/ 300939] | loss: 5.8823347 | lr: 6.0000e-05 | norm: 0.9465 | dt: 109.80ms | tok/sec: 18651.71
[02/05/26 19:09:40] INFO     [    602/ 300939] | loss: 5.5719604 | lr: 6.0000e-05 | norm: 1.2682 | dt: 110.44ms | tok/sec: 18544.82
[02/05/26 19:09:51] INFO     [    802/ 300939] | loss: 5.3620930 | lr: 6.0000e-05 | norm: 1.3966 | dt: 137.09ms | tok/sec: 14938.79
[02/05/26 19:10:02] INFO     [   1002/ 300939] | loss: 4.8515739 | lr: 6.0000e-05 | norm: 1.8913 | dt: 131.38ms | tok/sec: 15587.89
[02/05/26 19:10:13] INFO     [   1202/ 300939] | loss: 4.9659228 | lr: 6.0000e-05 | norm: 1.5206 | dt: 110.41ms | tok/sec: 18548.62
[02/05/26 19:10:25] INFO     [   1402/ 300939] | loss: 4.8464036 | lr: 6.0000e-05 | norm: 1.6366 | dt: 111.56ms | tok/sec: 18357.22
[02/05/26 19:10:36] INFO     [   1602/ 300939] | loss: 4.8617697 | lr: 6.0000e-05 | norm: 1.7850 | dt: 111.75ms | tok/sec: 18327.38
[02/05/26 19:10:47] INFO     [   1802/ 300939] | loss: 4.9997263 | lr: 6.0000e-05 | norm: 2.2968 | dt: 111.25ms | tok/sec: 18409.10

+ weight decay with fused adamw:
[02/05/26 20:05:32] INFO     [      2/ 300939] | loss: 10.9714661 | lr: 6.0000e-05 | norm: 31.2487 | dt: 1766.04ms | tok/sec:  1159.66
[02/05/26 20:05:42] INFO     [    202/ 300939] | loss: 6.1477079 | lr: 6.0000e-05 | norm: 0.8970 | dt: 102.32ms | tok/sec: 20015.91
[02/05/26 20:05:52] INFO     [    402/ 300939] | loss: 5.8771863 | lr: 6.0000e-05 | norm: 0.9884 | dt: 102.85ms | tok/sec: 19913.25
[02/05/26 20:06:03] INFO     [    602/ 300939] | loss: 5.5722036 | lr: 6.0000e-05 | norm: 1.2251 | dt: 103.31ms | tok/sec: 19822.89
[02/05/26 20:06:13] INFO     [    802/ 300939] | loss: 5.3593578 | lr: 6.0000e-05 | norm: 1.3652 | dt: 105.87ms | tok/sec: 19343.59
[02/05/26 20:06:24] INFO     [   1002/ 300939] | loss: 4.8322449 | lr: 6.0000e-05 | norm: 1.8286 | dt: 104.11ms | tok/sec: 19671.77
[02/05/26 20:06:34] INFO     [   1202/ 300939] | loss: 4.9621029 | lr: 6.0000e-05 | norm: 1.5479 | dt: 103.54ms | tok/sec: 19779.32
[02/05/26 20:06:44] INFO     [   1402/ 300939] | loss: 4.8556585 | lr: 6.0000e-05 | norm: 1.7170 | dt: 103.98ms | tok/sec: 19695.65
[02/05/26 20:06:55] INFO     [   1602/ 300939] | loss: 4.8727255 | lr: 6.0000e-05 | norm: 1.8692 | dt: 103.28ms | tok/sec: 19829.92
[02/05/26 20:07:05] INFO     [   1802/ 300939] | loss: 4.9865813 | lr: 6.0000e-05 | norm: 2.0731 | dt: 105.20ms | tok/sec: 19467.87
```

```txt
A30 (B=8, T=1024)
baseline:
[02/04/26 16:01:58] INFO     avg loss: 11.0092154  [8/300939], dt: 1152.09ms, tok/sec: 7110.57
[02/04/26 16:03:40] INFO     avg loss: 5.6794214  [808/300939], dt: 1032.54ms, tok/sec: 7933.80
[02/04/26 16:05:24] INFO     avg loss: 5.0216446  [1608/300939], dt: 1037.90ms, tok/sec: 7892.90
[02/04/26 16:07:08] INFO     avg loss: 4.5767865  [2408/300939], dt: 1044.93ms, tok/sec: 7839.74
[02/04/26 16:08:53] INFO     avg loss: 4.3106785  [3208/300939], dt: 1045.23ms, tok/sec: 7837.49
[02/04/26 16:10:37] INFO     avg loss: 4.0364861  [4008/300939], dt: 1038.88ms, tok/sec: 7885.40

+ tf32:
[02/04/26 16:20:27] INFO     avg loss: 11.0092010  [8/300939], dt: 513.29ms, tok/sec: 15959.64
[02/04/26 16:21:04] INFO     avg loss: 5.6494966  [808/300939], dt: 372.42ms, tok/sec: 21996.52
[02/04/26 16:21:41] INFO     avg loss: 4.9863811  [1608/300939], dt: 373.94ms, tok/sec: 21907.03
[02/04/26 16:22:19] INFO     avg loss: 4.5347838  [2408/300939], dt: 375.06ms, tok/sec: 21841.94
[02/04/26 16:22:56] INFO     avg loss: 4.2737732  [3208/300939], dt: 375.42ms, tok/sec: 21821.10
[02/04/26 16:23:34] INFO     avg loss: 4.0098648  [4008/300939], dt: 376.54ms, tok/sec: 21755.87

+ bf16 amp:
[02/04/26 22:22:33] INFO     avg loss: 11.0089798  [8/300939], dt: 659.57ms, tok/sec: 12420.29
[02/04/26 22:23:04] INFO     avg loss: 5.7419477  [808/300939], dt: 316.01ms, tok/sec: 25923.64
[02/04/26 22:23:36] INFO     avg loss: 5.1110601  [1608/300939], dt: 317.65ms, tok/sec: 25789.28
[02/04/26 22:24:08] INFO     avg loss: 4.6182013  [2408/300939], dt: 319.47ms, tok/sec: 25642.31
[02/04/26 22:24:40] INFO     avg loss: 4.3499250  [3208/300939], dt: 319.62ms, tok/sec: 25630.17
[02/04/26 22:25:12] INFO     avg loss: 4.0864253  [4008/300939], dt: 320.96ms, tok/sec: 25523.55

+ torch.compile:
[02/04/26 23:37:04] INFO     avg loss: 11.0088120  [8/300939], dt: 40272.72ms, tok/sec: 203.41
[02/04/26 23:37:21] INFO     avg loss: 5.7281141  [808/300939], dt: 171.62ms, tok/sec: 47733.20
[02/04/26 23:37:38] INFO     avg loss: 5.0944386  [1608/300939], dt: 173.13ms, tok/sec: 47317.20
[02/04/26 23:37:55] INFO     avg loss: 4.6106510  [2408/300939], dt: 173.12ms, tok/sec: 47320.65
[02/04/26 23:38:13] INFO     avg loss: 4.3152447  [3208/300939], dt: 173.57ms, tok/sec: 47196.80
[02/04/26 23:38:30] INFO     avg loss: 4.0849695  [4008/300939], dt: 173.29ms, tok/sec: 47272.84
[02/04/26 23:38:47] INFO     avg loss: 4.0146251  [4808/300939], dt: 175.00ms, tok/sec: 46810.76

+ flash attention:
[02/05/26 01:55:14] INFO     avg loss: 11.0089493  [8/300939], dt: 39270.54ms, tok/sec: 208.60
[02/05/26 01:55:28] INFO     avg loss: 5.7227125  [808/300939], dt: 133.74ms, tok/sec: 61253.81
[02/05/26 01:55:41] INFO     avg loss: 5.0873637  [1608/300939], dt: 133.19ms, tok/sec: 61504.04
[02/05/26 01:55:54] INFO     avg loss: 4.6047478  [2408/300939], dt: 134.62ms, tok/sec: 60851.71
[02/05/26 01:56:08] INFO     avg loss: 4.3250217  [3208/300939], dt: 135.30ms, tok/sec: 60547.46
[02/05/26 01:56:22] INFO     avg loss: 4.0717964  [4008/300939], dt: 135.62ms, tok/sec: 60403.67
[02/05/26 01:56:35] INFO     avg loss: 4.0133028  [4808/300939], dt: 135.64ms, tok/sec: 60392.97
[02/05/26 01:56:49] INFO     avg loss: 3.6045163  [5608/300939], dt: 135.80ms, tok/sec: 60323.14

+ nice numbers for cuda
[02/05/26 02:43:39] INFO     avg loss: 10.9753036  [8/300939], dt: 33680.01ms, tok/sec: 243.23
[02/05/26 02:43:51] INFO     avg loss: 5.7267504  [808/300939], dt: 117.68ms, tok/sec: 69610.27
[02/05/26 02:44:03] INFO     avg loss: 5.1912279  [1608/300939], dt: 118.13ms, tok/sec: 69346.80
[02/05/26 02:44:15] INFO     avg loss: 4.7176390  [2408/300939], dt: 120.58ms, tok/sec: 67937.13
[02/05/26 02:44:27] INFO     avg loss: 4.5357361  [3208/300939], dt: 120.92ms, tok/sec: 67747.09
[02/05/26 02:44:39] INFO     avg loss: 4.6414132  [4008/300939], dt: 120.66ms, tok/sec: 67893.13

+ lr cosine decay scheduler:
[02/05/26 19:09:26] INFO     [      8/ 300939] | loss: 10.9753036 | lr: 6.0000e-05 | norm: 28.3391 | dt: 8987.64ms | tok/sec:   911.47
[02/05/26 19:09:38] INFO     [    808/ 300939] | loss: 5.7124081 | lr: 6.0000e-05 | norm: 0.5679 | dt: 118.44ms | tok/sec: 69165.85
[02/05/26 19:09:50] INFO     [   1608/ 300939] | loss: 5.4419727 | lr: 6.0000e-05 | norm: 1.0448 | dt: 120.06ms | tok/sec: 68235.02
[02/05/26 19:10:02] INFO     [   2408/ 300939] | loss: 5.0777788 | lr: 6.0000e-05 | norm: 1.1896 | dt: 120.81ms | tok/sec: 67808.55
[02/05/26 19:10:14] INFO     [   3208/ 300939] | loss: 4.9805861 | lr: 6.0000e-05 | norm: 1.0522 | dt: 119.74ms | tok/sec: 68414.98
[02/05/26 19:10:26] INFO     [   4008/ 300939] | loss: 5.0637026 | lr: 6.0000e-05 | norm: 1.8128 | dt: 120.89ms | tok/sec: 67766.53
[02/05/26 19:10:38] INFO     [   4808/ 300939] | loss: 4.7597485 | lr: 6.0000e-05 | norm: 1.7286 | dt: 120.99ms | tok/sec: 67709.85
[02/05/26 19:10:50] INFO     [   5608/ 300939] | loss: 4.5506630 | lr: 6.0000e-05 | norm: 1.9429 | dt: 120.84ms | tok/sec: 67792.07
[02/05/26 19:11:02] INFO     [   6408/ 300939] | loss: 4.5645299 | lr: 6.0000e-05 | norm: 1.2567 | dt: 121.41ms | tok/sec: 67472.42

+ weight decay with fused adamw:
[02/05/26 20:05:20] INFO     [      8/ 300939] | loss: 10.9753036 | lr: 6.0000e-05 | norm: 28.3392 | dt: 7713.62ms | tok/sec:  1062.02
[02/05/26 20:05:31] INFO     [    808/ 300939] | loss: 5.7080107 | lr: 6.0000e-05 | norm: 0.5791 | dt: 111.29ms | tok/sec: 73611.93
[02/05/26 20:05:42] INFO     [   1608/ 300939] | loss: 5.4319210 | lr: 6.0000e-05 | norm: 0.7864 | dt: 111.26ms | tok/sec: 73627.64
[02/05/26 20:05:53] INFO     [   2408/ 300939] | loss: 5.0748129 | lr: 6.0000e-05 | norm: 0.8901 | dt: 111.29ms | tok/sec: 73606.96
[02/05/26 20:06:04] INFO     [   3208/ 300939] | loss: 4.9830112 | lr: 6.0000e-05 | norm: 1.4798 | dt: 111.58ms | tok/sec: 73419.75
[02/05/26 20:06:15] INFO     [   4008/ 300939] | loss: 5.0555134 | lr: 6.0000e-05 | norm: 1.3391 | dt: 113.61ms | tok/sec: 72103.39
[02/05/26 20:06:27] INFO     [   4808/ 300939] | loss: 4.7708340 | lr: 6.0000e-05 | norm: 1.4788 | dt: 113.36ms | tok/sec: 72267.80
[02/05/26 20:06:38] INFO     [   5608/ 300939] | loss: 4.5496378 | lr: 6.0000e-05 | norm: 1.4707 | dt: 111.28ms | tok/sec: 73615.90
[02/05/26 20:06:49] INFO     [   6408/ 300939] | loss: 4.5711374 | lr: 6.0000e-05 | norm: 1.3489 | dt: 111.72ms | tok/sec: 73325.73
[02/05/26 20:07:00] INFO     [   7208/ 300939] | loss: 4.6766014 | lr: 6.0000e-05 | norm: 1.9035 | dt: 114.37ms | tok/sec: 71624.45
[02/05/26 20:07:12] INFO     [   8008/ 300939] | loss: 4.2843590 | lr: 6.0000e-05 | norm: 1.6562 | dt: 114.20ms | tok/sec: 71733.56
[02/05/26 20:07:23] INFO     [   8808/ 300939] | loss: 4.2238865 | lr: 6.0000e-05 | norm: 1.6501 | dt: 113.73ms | tok/sec: 72027.28
[02/05/26 20:07:34] INFO     [   9608/ 300939] | loss: 4.3411264 | lr: 6.0000e-05 | norm: 1.6378 | dt: 112.86ms | tok/sec: 72584.95
[02/05/26 20:07:46] INFO     [  10408/ 300939] | loss: 4.3515697 | lr: 6.0000e-05 | norm: 1.5034 | dt: 113.02ms | tok/sec: 72482.24
```
