Goal: train a model that is capable of solving the env. Show that trajectory transformer is able to solve env.

Non Goals: abilation studies

Next steps

1. [Done] Code complete
2. [Done] review code and add documentation
   1. [Done] Fixed rewards to go
   2. [Done] Verifed endoer decoder and dataset indexing
   3. [Done] added docstrings
3. [Done] find a env in minari with small obs and action dim
   1. https://minari.farama.org/datasets/D4RL/hammer/expert-v2/
4. Improve performance
   1. Use full dataset
   2. Use smaller LR 0.0006
   3. Train for 50 epochs
   4. Increase beam width to 128
   5. Increase beam steps to 15
   6. Increase embedding dim to 128
5. Write blog
