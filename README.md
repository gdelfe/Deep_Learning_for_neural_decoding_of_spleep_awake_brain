
# Sleep/Awake classification of the monkey brain neural activity via Deep Learning 

This project aims to use DEEP LEARNING tools to classify the monkey brain neural activity into sleep and awake states. 
The cortical neural activity of two monkeys is recorded together with the video of the monkeys moving or sleeping in their cage. The video is used to define whether the monkeys are _sleeping_ or _awake_ and moving. Thus, the video provides the labels for a classification problem: sleep/awake. 

**DATA:** <br>
micro-EcoG of the brain neural activity of two monkeys recorded in Motor cortex (M1), Prefrontal Cortex (PFC), and sensory motor cortex (S1). The data covers several days (~15 days) for each monkey. Every day consists of several hours of continous wireless recording while the monkeys are in their cage (~9h each day).

**METHODS:** <br>
1. The video is analized by a technician which manually classify the monkey states as _awake_ or _asleep_ every second interval. This manual inspection of the monkey states carefully defines the labels for the two classes: awake/asleep.
2. From the micro-EcoG neural activity of the monkey brain spectogram (time-frequency plots) are generated using a multi-taper spectral analysis. 
3. Each spectrogram covers a time period of 5 sec and a frequency range [0-200] Hz. Each of this spectrogram is associated to one state: sleep/awake, meaning that during the 5 sec period the monkey was constantly in one of these two states, without transitioning from one to the other. 
4. Each spectrogram is used as an image input for an Artificial Neural Network (ANN) model associated to either a sleep or an awake state. The ANN model is trained over several images and then validated and tested on unseen images (spectrograms) -- see details below. 

**ANN MODELS** <br>
Different Artificial Neural Network models are used to test the performance of each classification model. These are:
1. Linear Regression (used as a benchmark)
2. Multilayer perceptron (MLP)
3. Convolutional Neural Networks (CNNs)
4. Recurrent Neural Network such as LSTM (RNN-LSTM)
5. Convolutional Recurrent Neural Networks (CRNN)
6. Convolutional networks with self-attention mechanisms (e.g. Transformers) (CNN+Attention)

**TRAIN/VALIDATION/TEST**
The models are trained on a set of recordings referred to specific days, and then validated and tested on recordings of other days. To be more specific, the test set goes from day 1-6, the validation set goes from day 7-8, and the test set from day 9-10. In this way testing represents 60% of the data, validation 20% and test 20% of the data respectively. Data for the training, validation, and test are taken in this way in order to have non-overlapping recordings, i.e. we want to prevent that recordings from the same day belong to both the training and validation (test) data set. In this way, the training of the ANN results more complex and harder but we do not introduce any bias. 

We test three different scenario:
1. Training/validation/test 




 


 
