# Shakespeare Sequence Models

Test the RNN model here:

https://shonczinner.github.io/shakespeare-generator/

Compare RNN, GRU, LSTM, CNN, Transformer models for generating shakespeare one character at a time by training on the Shakespeare dataset.

![Accuracy comparison by compute](summary/acc_vs_compute.png) ![Accuracy comparison by epoch](summary/acc_vs_epoch.png)

| model       |   parameters |   best_val_loss |   best_val_accuracy |   total_compute |
|:------------|-------------:|----------------:|--------------------:|----------------:|
| rnn         |       969537 |         1.48819 |            0.557443 |         280.268 |
| cnn         |       689217 |         1.60601 |            0.533054 |         150.63  |
| transformer |      1347969 |         1.56523 |            0.543635 |         428.677 |
| gru         |       822849 |         1.46929 |            0.562935 |         286.1   |
| lstm        |      1086017 |         1.5091  |            0.552417 |         381.636 |

Generation sample from RNN with prompt "ROMEO:",

```         
ROMEO:
I am love's peace than his pensiat,
Who spired to defend my liege; this way thomar, O sir;
Here could have I to die to deny me or,
And with a string to the rock. You have secure them?

QUEEN YORK:
My old succession! none to his rateful his love,
Lie haunt me with me on hang on that thirk?

ISABELLA
```

Requirements:

torch numpy matplotlib pandas onnxscript