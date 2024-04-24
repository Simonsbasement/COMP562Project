[Copied from the top of covid_and_stock.m]
[Details see in-file comments]
Machine learning based prediction of stock prices under pandemic influence

Fitrensemble(bagged decision trees) is used to predict future stock prices under pendemic influences.
Data from the past are compounded into samples to train and test the model;
a seperate ensemble is responsible for every output group.

Two structures are tested(comp). Later layers:
  True) use predictions produced by earlier layers as extra input
  False) they do not. Each layer acts independently from other layers. 

Potentially required add-ons (Toolboxes):
Curve Fitting (Toolbox)
Deep Learning
Financial
Optimization
Parallel Computing
Signal Processing
Statistics and Machine Learning
System Identification
