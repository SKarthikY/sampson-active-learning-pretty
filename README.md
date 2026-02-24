## This is an implementation of active learning for the sampson emulator. 
## sampson is trained to emulate Sedona, a radiative transfer simulation code. 


In short, Sedona producees time series data of star explosions. Because sedona is a simulator, it is very slow (takes ~hours) at producing a single times series data file. Sampson is a neural network that is trained to produce the same output as sedona does, but will (hopefully) take milliseconds instead of hours. This package performs active learning to train sampson to perform better. Active is a method of evaluating where in the parameter space the training set needs to be augmented so that the neural network can perform better. In the following paragraphs, I will describe at a high-level what kind of active learning this package is meant to do, and detail how this package accomplishes that task.


