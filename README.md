
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

**ANN MODELS:** <br>
Different Artificial Neural Network models are used to test the performance of each classification model. These are:
1. Logistic Regression (used as a benchmark)
2. Multilayer perceptron (MLP)
3. Convolutional Neural Networks (CNNs)
4. Recurrent Neural Network such as LSTM (RNN-LSTM)
5. Convolutional Recurrent Neural Networks (CRNN)
6. Convolutional networks with self-attention mechanisms (e.g. Transformers) (CNN+Attention)

**TRAIN / VALIDATION / TEST details :** <br>
The models are trained on a set of recordings referred to specific days, and then validated and tested on recordings of other days. Recordings (spectrograms) from the training set are never used for validation/test and the other way around. Indeed, electrodes placed in the monkey's cortex adjust their position across days, due to natural shift of brain tissues. That means that two consecutive days have more chance to have the electrodes in approximately the same positions. Days which are further apart, will have electrodes in slightly different positions.

For these reasons, it would be easier to pick the recordings for the train/val/test set across days. We are yet interested in the ability of the ANNs to *generalize*, i.e. training of some data set and perform well on slightly different conditions of those in the training data set. For this reason, we never mix recordings from the same day across the train/val/test set. In other words, the train/val/test set are made of non-overlapping days and they are consecutive: the train set is made, for instance, by day 1-6, the validation set by day 7-8, and the test set by day 9-10. 

1. 1st scenario: Monkey G only, 1st series <br>
 We train the ANNs for only one monkey and we val/test on the same monkey. <br>
 Train/val/test set are taken from 7 consecutive days with 70% / 15% /15% split
 This scenario is the easiest of those that we consider. Val/test set are recorded consecutevely to the training set, therefore the recorded electrodes 
 are not shifted by much.

2. 2nd scenario: Monkey G only, 1st + 2nd series <br>
 We train the ANNs for only one monkey and we val/test on the same monkey. <br>
 Train set is taken from 7 consecutive days,  val/test set are taken from a set of data recorded a few weeks after the training set. 
 This scenario aims to test the ability of the ANNs to generalize on a data set where electrodes are potentially shifted from their original location. 
 
3. 3rd scenario: Monkey G + Monkey J, 1st and 2nd series for both <br>
 We train the ANNs on recording aquired for Monkey G and we validate and test on Monkey J. 
 This scenario is the hardest we consider and it aims to test the ability of the ANNs to generalize the learning from one monkey brain to another. This    scenario tests the ability to generalize our results across animals in the same species.  

 
