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
