# Simplest HEP NN

This is a bare bones implementation of a NN, whcih is trained to predict if an event is part of
a Higgs data set, or a QCD dataset.
The code is 252 lines long, and undocumented (aside form this readme).
It's mostly cribbed from [the pytorch tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html),
with some adaptions for physics.

For any serious use case, better preprocessing would be needed,
along with better evaluation of the outcome and some
protection against overfitting (though this data is so simple
that I wouldn't expect much overfitting).

## Instructions

You will need the python packages;

 - matplotlib; for plotting
 - numpy; for numeric manipulations
 - uproot; for reading root files
 - awkward; for jagged array manipulations (probably automatically installed with uproot)
 - pytorch; for learning

You will also need the data files `signal.root` and `background.root`,
these can be automatically downloaded by calling the `download.sh` script
from the data directory.
Ensure the data files are in the data directory.

You can take a look at the input distributions for signal and background by calling;
```
python3 code/quick_look.py
```
which should produce a series of histograms that you can hit enter to cycle though.

Then the NN can be trained by calling;
```
python3 code/train.py
```
