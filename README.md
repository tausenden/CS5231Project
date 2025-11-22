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

Transformer inference (threshold=0.750):

Normal sample -> Prob: 0.3902, Prediction: normal

Attack sample -> Prob: 0.3970, Prediction: normal

LSTM inference (threshold=0.950):

Normal sample -> Prob: 0.4217, Prediction: normal

Attack sample -> Prob: 0.4183, Prediction: normal