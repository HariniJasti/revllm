| run | mode | batch | tokens/step | steps | peak LR | final train loss | final val loss | tokens/s | peak mem alloc (GB) | peak mem reserved (GB) | activation mem (GB) | train time (min) | GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_baseline_b32 | standard | 32 | 16384 | 3052 | 0.001 | 1.7695 | 1.8297 | 76879 | 3.24 | 3.47 | 2.99 | 10.8 | Tesla T4 |
| 02_reversible_b32 | midpoint | 32 | 16384 | 3052 | 0.001 | 1.8126 | 1.8712 | 60537 | 1.11 | 1.3 | 0.86 | 13.8 | Tesla T4 |
| 03_reversible_bmax | midpoint | 944 | 483328 | 104 | 0.003 | 5.5812 | 5.6162 | 53366 | 14.18 | 14.39 | 13.92 | 15.7 | Tesla T4 |
| 03b_reversible_bmax_lr1e-3 | midpoint | 936 | 479232 | 105 | 0.001 | 4.2429 | 4.1266 | 51241 | 14.06 | 14.42 | 13.81 | 16.3 | Tesla T4 |

- probe_standard: max batch **192** (98,304 tok/step), 73.1 MB/sequence, hit probe limit: False
- probe_midpoint: max batch **944** (483,328 tok/step), 15.0 MB/sequence, hit probe limit: False
- chosen reversible variant: **midpoint**
  - bake-off {"variant": "standard (ref)", "val_loss": 3.533275604248047, "train_loss": 3.5062100410461428, "tok/s": 76270, "peak GB": 3.237898349761963, "act GB": 2.986577033996582, "diverged": false, "euler residual": NaN}
  - bake-off {"variant": "euler (4 it)", "val_loss": 5.981689960725846, "train_loss": 5.954333114624023, "tok/s": 33847, "peak GB": 1.089460849761963, "act GB": 0.838139533996582, "diverged": false, "euler residual": 0.11043186485767365}
  - bake-off {"variant": "euler (4 it, emb .5)", "val_loss": 5.9814348220825195, "train_loss": 5.929412202835083, "tok/s": 34152, "peak GB": 1.089460849761963, "act GB": 0.838139533996582, "diverged": false, "euler residual": 0.07279007881879807}
  - bake-off {"variant": "midpoint", "val_loss": 3.6165141520961637, "train_loss": 3.591440324783325, "tok/s": 60670, "peak GB": 1.112898349761963, "act GB": 0.861577033996582, "diverged": false, "euler residual": NaN}
  - bake-off {"variant": "revnet", "val_loss": 3.9078255314980783, "train_loss": 3.883457970619202, "tok/s": 61527, "peak GB": 1.136335849761963, "act GB": 0.885014533996582, "diverged": false, "euler residual": NaN}
- grad check revnet it=None emb=0.02: rel_err=3.11e-15
- grad check midpoint it=None emb=0.02: rel_err=1.71e-15
- grad check euler it=1 emb=0.02: rel_err=9.43e-01
- grad check euler it=4 emb=0.02: rel_err=9.64e-01
- grad check euler it=8 emb=0.02: rel_err=9.84e-01
- grad check euler it=1 emb=0.1: rel_err=4.70e-01
- grad check euler it=4 emb=0.1: rel_err=1.48e-01
- grad check euler it=8 emb=0.1: rel_err=5.57e-02
- grad check euler it=1 emb=0.5: rel_err=2.04e-02
- grad check euler it=4 emb=0.5: rel_err=5.69e-05
- grad check euler it=8 emb=0.5: rel_err=3.80e-08

### Samples

**01_baseline_b32**: 'Once upon a time, there was a little girl named Lily. She loved to play with her dolls and talk to them. One day, Lily\'s mommy said, "Lily, we\'ll go to the grocery store. I have to wait until your mommy is over."\n\nAt the store, Lily saw a big bag of candy. She wanted one, but it was too expensive. W'
**02_reversible_b32**: "Once upon a time, there was a little girl named Lily. She loved to play with her dolls and teddy bears. One day, Lily's mommy asked her to help clean up her room. \n\nLily was happy to help. She cleaned up her room and put the toys away. While they were cleaning, Lily's mommy asked her to help clean u"
**03_reversible_bmax**: "Once upon a time, and.\n They.The't day on, a said.\n The and the the.\n\n It and it he. I.\n His and.\n\n The he and. She the. He saw.\n She. The. They to.\n\n\n\n He,. From a little to on, the,, happy. \n She.\n He was was not.\n They a little and with, so. \n From, and a. They a they.\n They his.\n\n\n He.\n They big"
**03b_reversible_bmax_lr1e-3**: 'Once upon a time, and it. \n\nThe big day, Timmy would not have fun, but the forest.\n\nThe little girl\'s proud of her hand and they liked to go, but he was a time, "Thank you. "OK, â€ They had a big, "Let is a time, a little boy had a big, there was a big big tree, "Yes, so happy. They smiled. He got a'