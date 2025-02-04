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
4. Demostrate that the model can solve env
   1. [Done] Use full dataset - failed
   2. [Done] Use smaller LR 0.0006 - failed
   3. [Done] Train for 50 epochs - failed
   4. [Done] Increase beam width to 128 - failed
   5. [Done] Increase beam steps to 15 - failed
   6. [Done] Increase embedding dim to 128 - failed
   7. [Done] implement sep heads - failed
   8. [Done] implement gpt scheduler, adamw optimizer with weight decay, clip grad - failed
   9. [Done] Add normalization before MLP - failed
   10. [Done] increasse capacity of MLP - failed
   11. [Done] state not being handled in offsets - bug fix
   12. [Done] Include terminals in dataset - helped
   13. investigate why LR is not changing
   14. Test out on prod
5. Write blog
