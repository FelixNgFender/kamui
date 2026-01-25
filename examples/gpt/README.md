# gpt from scratch

## char-level bigram

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

## word-level mlp

word-level mlp: to avoid OOM, i modified the embedding table to use (vocab_size=25670, embbedding_dim=24) and a final linear layer to project to vocab_size=25670.

### mlp loss curve

![loss curve for word-level mlp model](word_mlp_loss.png)

### mlp results

before training

```txt
train set avg loss: 10.303034

val set avg loss: 10.308567

&C: Tempering torments, Ned hapless Pompey. Reports, swear; graybeards
earnestness prayer-book cousin? sickness remission misplaced. fair; express
saved guess? impress'd taxations entire winking parley part? Volsces; whither
hearts: Seely, plausible days; combatants. swiftness, Execute Greek strengths
malapert. slipt Romano, Grecian saddle. Ireland. MERCUTIO: a-doing. gall.
ducat vast, sovereign's Mowbray's wisdom: lady body: accordingly. honesty
invectives Began Plead serving-men senseless; sinners' shadows. Villains,
gear? depends sigh'd correction, father kill'd; drank once: carcasses
daughter's bag babe. spice. hoot cross-bow bastard's fares Familiarly
savageness pikes. bound? none? ever Cunning brows. smallest. mercy, sons.
brows remembrance. solely: Be-mock belong likest appendix. gentlewoman?
discover'd, oath, pierced
```

after training for 10 epochs

```txt
train set avg loss: 4.587226

train set avg loss: 9.132408

&C: We'll cannot, we are Aufidius, Not of honour are thou hast comforted me
die, it is my blood and dance with me? If it may not she: Go, get it was
doubtful note, for they ascend the cheek, Within your close his Edward that
fear unto: I took Hereafter hence to hold mine It cannot mistress o' the most
kind of our blow one of your free giber for badness. DUKE VINCENTIO: Whilst
sleep again. Farewell, ladies, you: I have deeper than thy birth These name
With them; now? More another, so dishonour'd wings To his ill kisses it. How
```

### thoughts on mlp

the world-level mlp is better at generating correct words, but the overall
syntax and semantics are still quite poor. it seems that the model is able to
learn some local word dependencies, but struggles with longer-range
dependencies and coherent sentence structure. there is also overfitting on the
train set, as evidenced by the large gap between train and val loss as the
model is trained for more epochs.
