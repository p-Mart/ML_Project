Work in progress for CAP6610 to develop a speech recognition algorithm from scratch using CNN's.
#### Dependencies:
* Soundfile: for converting audio files to .wav files
* numpy, scipy, matplotlib
* python_speech_features: for computing the MFCC's
* Gentle for forced alignment
# Contents
## Architecture-related modules
* #### Nodes.py
Contains base functions for different types of nodes, which are used in the Layers class.
* #### Layers.py
Contains different layer classes which are passed into the Network class.
Current layers are: Sigmoid, Relu, and Softmax, and Max Pooling. (Note that softmax only works
for categorical crossentropy at the moment)

*Usage:*
```python
layer_1 = Sigmoid(inputs_per_node, number_of_nodes)
```
* #### Network.py
Contains the top-level class Network which connects all the layers passed in through the class constructor.
Has the functions train() and predict() to train the network on a dataset and predict the outputs for a dataset,
respectively. Can be passed in a learning rate value and a loss function. Current loss functions are "squared error"
and "categorical crossentropy."

*Usage:*
```python
#Assuming you have a dataset X with outputs y
layers = [Relu(3, 10), Sigmoid(10, 1)]
model = Network(layers, learning_rate = 1, func = "squared error")
model.train(X, y, number_epochs = 100)
output = model.predict(X, y)
```
*Note: since layers are passed in sequentially, the number of inputs per node for each layer must be
consistent with the number of nodes in the previous layer. For example an architecture of 5 inputs, 10
hidden nodes in the first layer, 9 hidden nodes in the second layer, and 4 output nodes would be defined as:*
```python
inputs = 5
hidden_nodes_1 = 10
hidden_nodes_2 = 9
outputs = 4

layer_1 = Relu(inputs, hidden_nodes_1)
layer_2 = Relu(hidden_nodes_1, hidden_nodes_2)
layer_3 = Sigmoid(hidden_nodes_2, outputs)

model = Network([layer_1,layer_2,layer_3])
```
## Input-handling modules
* #### ToWav.py
Converts an input sound file into a .wav. Writes file to current directory with the same filename.

*Usage:*
```
python ToWav.py [input file with extension]
```
* #### MelFreq.py
Extracts the Mel-Frequency Cepstral Coefficients from an input waveform.

*Usage:*
```
python MelFreq.py [input].wav
```

