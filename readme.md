# MSc DSML Project: Fashion News Popularity Prediction

The source code of the MSc Project

## Requirements

- Python3
- Tensorflow
- Keras
- sklearn
- pandas
- numpy
- matplotlib

## Experiment 1: Concept Drift Dection

- Drift Dection Method (DMM) [[Code](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/concept_drift/DDM.py)]
-  Preprocessing [[Code](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/concept_drift/concept_detector.ipynb)]
- Result
  - Detector 1 [[Probabilities](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_1/prob.npy), [Status](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_1/stat.npy)]
  - Detector 2 [[Probabilities](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_2/prob.npy), [Status](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_2/stat.npy)]
  - Detector 3 [[Probabilities](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_3/prob.npy), [Status](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_3/stat.npy)]
  - Detector 4 [[Probabilities](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_4/prob.npy), [Status](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/detector_4/stat.npy)]
  - Base learner [[Probabilities](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/result/concept_drift/y_pred_base_learner.npy)]
- Visualisation of the result [[Code]()]

## Experiment 2: Multi-Input Model

- Keras implemented models

  - Char-CNN [[Code](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/deep_models/char_cnn/train_keras.ipynb)]

  - BLSTM and attBlSTM[[Code](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/deep_models/bilstm(attention)/train_keras.ipynb)]

    For BLSTM, set ``get_model(attention=False)``

    For attBLSTM, set ``get_model(attention=True)``

  - fastText [[Code](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/deep_models/fasttext/train_keras.ipynb)]

  - Multi-Input [[Code](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/deep_models/multi-input/train_keras.ipynb)]

- [Result](https://github.com/TheSuguser/fashion_news_popularity_prediction/tree/master/result/deep_models)

- [Visualisation of the result](https://github.com/TheSuguser/fashion_news_popularity_prediction/blob/master/visualisation/dl_visualisation.ipynb)