# The accuracy in various situations.
### Default configs
- Epoch: 40
- Batch size: 128
- Learning rate: 0.01
- Weight decay: 1e-4
- Learning rate decay: 0.2 interval: 30
- Normalize("Image Net")
- 
- No augmentation

## Idea
- SE block
- GAP


## Various models
|  Model  |  Loss  |  Acc  |
|:-------:|-------:|------:|
| VGG11 | 0.1853 | 0.710 |
| VGG13 | 0.1995 | 0.698 |
| VGG16 | 0.1697 | 0.681 |
| VGG19 | 0.1961 | 0.645 |

## Crop
size 84
VGG11 70.3
VGG13 72.3
VGG16 68.9
VGG19 67.6

## Crop, Rotation(15), HorizontalFlip
VGG11==>73.6
VGG13==>72.3
VGG16==>72.1
VGG19==>70.7

## Use Scheduler
VGG16==> Colab:71.9 Mine:0.707

CNN: 0.541(start lr=0.1) 0.521(start lr=0.01)
lr 0.01
del conv8 add linear ==> 0.434
only add linear ==> 0.603

BraNet:
ResNet

VGG16 --lr 0.01 -b 64 -lrd 0.1 -ss 40 ==> Over 76 をマーク
