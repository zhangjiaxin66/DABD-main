# SSR


## Environment

- python == 3.8
- pytorch == 1.13.1--cuda11.6
- torch-geometric == 2.31
- scipy == 1.10.1
- numpy == 1.24.3
- deeprobust == 0.2.6
- scikit-learn == 1.2.2
- scikit-learn-extra ==0.3.0

## Perturbed Datasets
First, you need to install Deeprobust to prepare the perturbed dataset. We will place the dataset used in the experiment in the data directory (/data) 
If you need graphs attacked by other methods (DICE, Random), you can refer to: https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph. 
Likewise, you can also prepare your own perturbed graphs you need in any way.
```python
pip install deeprobust
```
Then, you can run SSR , the running result will be in the result directory (/result)
```
node_meta.sh
node_nettack.sh
node_random.sh

```
