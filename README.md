this is the code repo for CS5231 project

## ver1:

Evaluating Transformer Model 
                                  
- loss: 0.4156743125087844 
- acc: 0.9725010912265386 
- precision: 0.5948275862068966 
- recall: 0.32701421800947866 
- f1: 0.42201834862385323 
- auc: 0.7009416069921932 
- missed_attacks: 142 
- false_alarms: 47 
 
Evaluating LSTM Model 
 
- loss: 0.47755092692898987 
- acc: 0.9671177069693001 
- precision: 0.45098039215686275 
- recall: 0.32701421800947866 
- f1: 0.3791208791208791 
- auc: 0.6886493531253869 
- missed_attacks: 142 
- false_alarms: 84

## ver2

transformer threshold:
- Best f1: 0.4460 at threshold=0.707

lstm threshold:
- Best f1: 0.4429 at threshold=0.970

Transformer Test Metrics:
 - loss: 1.0977
 - acc: 0.9780
 - precision: 0.9054
 - recall: 0.3175
 - f1: 0.4702
 - auc: 0.6794
 - missed_attacks: 144.0000
 - false_alarms: 7.0000

LSTM Test Metrics:
 - loss: 1.1010
 - acc: 0.9779
 - precision: 0.8933
 - recall: 0.3175
 - f1: 0.4685
 - auc: 0.7028
 - missed_attacks: 144.0000
 - false_alarms: 8.0000

Transformer inference (threshold=0.750):\
Normal sample -> Prob: 0.3902, Prediction: normal\
Attack sample -> Prob: 0.3970, Prediction: normal

LSTM inference (threshold=0.950):\
Normal sample -> Prob: 0.4217, Prediction: normal\
Attack sample -> Prob: 0.4183, Prediction: normal

## ver3
transformer threshold:\
Best f1: 0.4460 at threshold=0.859\
lstm threshold:\
Best f1: 0.4429 at threshold=0.859

Transformer Test Metrics:
 - loss: 1.3511
 - acc: 0.9780
 - precision: 0.9054
 - recall: 0.3175
 - f1: 0.4702
 - auc: 0.6755
 - missed_attacks: 144.0000
 - false_alarms: 7.0000

LSTM Test Metrics:
 - loss: 1.3659
 - acc: 0.9779
 - precision: 0.8933
 - recall: 0.3175
 - f1: 0.4685
 - auc: 0.6948
 - missed_attacks: 144.0000
 - false_alarms: 8.0000

Transformer inference (threshold=0.859):\
Normal sample -> Prob: 0.4382, Prediction: normal\
Attack sample -> Prob: 0.4459, Prediction: normal

LSTM inference (threshold=0.859):\
Normal sample -> Prob: 0.4702, Prediction: normal\
Attack sample -> Prob: 0.4711, Prediction: normal

## ver4
### using focal loss

transformer threshold:\
Best f1: 0.4491 at threshold=0.798\
lstm threshold:\
Best f1: 0.4429 at threshold=0.838

Transformer Test Metrics:
 - loss: 0.4819
 - acc: 0.9774
 - precision: 0.8590
 - recall: 0.3175
 - f1: 0.4637
 - auc: 0.6880
 - missed_attacks: 144.0000
 - false_alarms: 11.0000

LSTM Test Metrics:
 - loss: 0.4786
 - acc: 0.9777
 - precision: 0.8816
 - recall: 0.3175
 - f1: 0.4669
 - auc: 0.7012
 - missed_attacks: 144.0000
 - false_alarms: 9.0000

Transformer inference (threshold=0.798):\
Normal sample -> Prob: 0.4578, Prediction: normal\
Attack sample -> Prob: 0.4706, Prediction: normal

LSTM inference (threshold=0.838):\
Normal sample -> Prob: 0.4838, Prediction: normal\
Attack sample -> Prob: 0.4800, Prediction: normal

## ver5
### using f2 score
transformer threshold:
Best f2: 1.0000 at threshold=0.394
lstm threshold:
Best f2: 1.0000 at threshold=0.586
Transformer Test Metrics:
 - loss: 0.0003
 - acc: 0.9990
 - precision: 0.7778
 - recall: 1.0000
 - f1: 0.8750
 - f2: 0.9459
 - auc: 1.0000
 - missed_attacks: 0.0000
 - false_alarms: 2.0000

LSTM Test Metrics:
 - loss: 0.0005
 - acc: 0.9986
 - precision: 0.7000
 - recall: 1.0000
 - f1: 0.8235
 - f2: 0.9211
 - auc: 1.0000
 - missed_attacks: 0.0000
 - false_alarms: 3.0000

Transformer inference (threshold=0.394):
Normal sample -> Prob: 0.0292, Prediction: normal
Attack sample -> Prob: 0.9555, Prediction: attack

LSTM inference (threshold=0.586):
Normal sample -> Prob: 0.0529, Prediction: normal
Attack sample -> Prob: 0.9598, Prediction: attack