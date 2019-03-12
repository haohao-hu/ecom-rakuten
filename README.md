# ecom-rakuten
Winning solution for the Rakuten Data Challenge, as part of SIGIR eCom '18.

The details of the model choices and evolution can be found in the [system description paper](https://sigir-ecom.github.io/ecom18DCPapers/ecom18DC_paper_9.pdf) for that workshop.

## Usage

### Data Preparation

Set up the expected data directories, from the repository root:

```
mkdir -p data/models
```

Move the challenge files into the `data/` subdirectory:

```
mv path/to/rdc-catalog-train.tsv data/
mv path/to/rdc-catalog-test.tsv data/
mv path/to/rdc-catalog-gold.tsv data/
```

Run a train/test split, build the vocabularies, and save the int-encoded training and validation sets for later:

```
./prep.sh
```

### BPV Model Training

Train and save a forward model with the hyperparameters from the winning RDC solution (the model goes in `data/models/model-name.h5`):

```
./train.sh model-name
```

Train a reverse model, intended for use in building a bi-directional ensemble with a forward network:

```
./train.sh reverse-model --reverse
```

Train a model with some parameters different from the default:

```
./train.sh custom-model-name --n-epochs=20 --lr=1.2
```

See a full list of parameters available to tune via flags:

```
./train.sh -- --help
```

### Inference, Prediction, and Scoring

Run an inference on the validation set, generate predictions, and then output precision, recall, and F1:

```
./infer.sh model-name
./infer.sh --forward=model-name  # equivalent
```

Score a reverse model:

```
./infer.sh --reverse=reverse-model
```

Similarly for a bi-directional ensemble:

```
./infer.sh --forward=model-name --reverse=reverse-model
```

Or for a larger ensemble, e.g. with 4 each forward and reverse:

```
./infer.sh --forward=fwd1,fwd2,fwd3,fwd4 --reverse=rev1,rev2,rev3,rev4
```

Since that can take awhile, you can show intermediate results along the way:

```
./infer.sh --forward=fwd1,fwd2,fwd3,fwd4 --reverse=rev1,rev2,rev3,rev4 --debug
```

To run test set inference and output prediction files for a single model, with ensembles working analogously to the commands above:

```
./infer.sh model-name --is-test
```

Lambda (i) in the ensemble of KNN and LSTM-BPV(s) can be set like this:
```
./infer.sh --forward=model-name --reverse=reverse-model --i=0.6
```
