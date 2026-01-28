# gpt from scratch

## char-level bigram

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

1 epoch

val set avg loss: 4.308069

;wtEnWDo&OUlAicSp!-di!lAZycWFa;GYtfxf$LC!TTu,okI!ctqLfBlSsIeDFotkpRv&Bi3;Mihbrp&CsxTh MoEkT&j3GllSPr

train set loss: loss: 0.512658 [998464/1003595]
val set avg loss: 2.267705

Clifford, ask mercy and his trade.

CLIFFORD:
My gracious liege, this tidings; therefore we parted
T

```txt
best val set avg loss (epoch=1): 2.2700939

(no train set because takes too long)
final val set avg loss (epoch=5): 3.3807906
sample after training:
By and by, my strength, and I hope him come in.

PRINCE EDWARD:
Arise, uncle; on my party, I will cry you mercy;
I did not see your grace: humbly on my knee
I crave your blessing.

DUCHESS OF YORK:
God bless thee; and put meekness in thy mind,
Love, charity, obedience, and true duty!

GLOUCESTER:

BUCKINGHAM:
Your grace may do your pleasure.

KING RICHARD III:
Tut, tut, thou art all ice, thy kindness' eyes,
That none but Henry did usurp;
And thou no more are prince than she is queen.

OXFORD:
Th
```
