# Training and evaluation

This document gives the causes of the DTNet extrapolation. It also describes the loss function
in [src/models/dt_net.py](../src/models/dt_net.py) and the training and evaluation code in
[src/runner.py](../src/runner.py). It uses the ASD-STE100 Simplified Technical English
standard. Names from the code stay as they are in the code.

## Part 0: Why DTNet can solve larger mazes

DTNet trains on mazes of size 15. But it can solve test mazes up to size 511 (see finding 1 in
[SUMMARY.md](SUMMARY.md)). The feedforward model trains on the same data. It solves only 28.8%
of the mazes of size 31, and it solves no larger mazes. This part gives the causes of this
difference.

### 0.1 The quantity of computation can increase at test time

A larger maze needs more computation, because the path is longer.

A 3 × 3 convolution moves data by a maximum of 1 pixel. One DTNet iteration has 5
convolutions: `recall_conv` and two blocks of two convolutions. Thus, one iteration moves data
by a maximum of 5 pixels along a passage. For example, the data about a dead end must go along
the full passage before the network can remove that passage.

- DTNet uses the same weights in each iteration. Thus, at test time you can do more iterations
  than in the training. Each iteration moves the data more. For example,
  `dtnet_s15_noloops_zeroinit` needs approximately 100 iterations at size 63 and 1000
  iterations at sizes 255 and 511.
- The feedforward model has 64 convolutions with different weights. It cannot do more. Thus,
  data cannot move more than 64 pixels, and a long path in a large maze is not possible for
  this model.

The feedforward model also fails at size 31, where 64 pixels is sufficient for most paths. Thus,
the number of convolutions is not the only cause. Section 0.2 gives the other cause.

### 0.2 DTNet learns one step that it can repeat

DTNet has only one set of weights for each iteration. Thus, it must learn one step that
improves the state. A good step of this type is a local rule. The rule examines only a small
area near each pixel.

An example of a local rule is dead-end filling:

1. Find each passage pixel with walls on three sides. Do not use the start or the exit.
2. Change this pixel to a wall.
3. Do these steps again until no change occurs.

In a perfect maze, only the path from the start to the exit remains. This rule does not
depend on the maze size. A larger maze needs only more repeats. The error analysis in
[SUMMARY.md](SUMMARY.md) shows that DTNet learns a rule of this type when it trains without
loops.

The feedforward model has different weights in each block. Thus, each block can learn a
different operation. These operations can solve the mazes of size 15, but they do not have to
be a general rule. Thus, the model does not solve larger mazes.

### 0.3 Other design features that help

These features make the repeated step stable for many iterations:

- **Fully convolutional layers.** All layers are convolutions. Thus, the same weights work on
  an image of any size.
- **Recall.** Each iteration adds the input maze to the state (`torch.cat([h, x])`). Thus, the
  network does not forget the maze after many iterations. The state does not have to keep a
  copy of the maze.
- **No batch normalization.** Batch normalization uses statistics from the training. After
  more iterations than in the training, these statistics are not correct.
- **The progressive loss.** It prevents the network from counting the iterations. Thus, each
  iteration is a small improvement that you can repeat. See section 1.3.
- **Zero-init.** It can make the training more stable. See section 1.7.

### 0.4 Limits

A local rule like dead-end filling is not sufficient for mazes with loops. A loop has no dead
end, so the rule keeps all the loops. To find the shortest path, the model must compare the
lengths of different routes.

A breadth-first search can also be a local rule: a wave goes out from the start and from the
exit, one pixel in each step. But to compare the routes, each pixel must keep a distance. In a
larger maze, the distances are larger than in the training mazes. A state with a fixed number
of channels must keep these larger values, and the training does not show them. This can be
one cause of the results: DTNet does not solve large mazes with loops (see findings 4 and 5 in
[SUMMARY.md](SUMMARY.md)).

## Part 1: The DTNet loss function (`dt_net.train_step`)

### 1.1 The loss for each pixel (`masked_ce`)

The network gives each pixel one of two classes: "path" or "not path". The function
`masked_ce` calculates the cross-entropy loss for each pixel. It uses only the pixels on
passages. It does not use the pixels on walls.

A wall pixel is always "not path", so it gives no useful data. Approximately half of the
pixels are walls. If the loss includes the walls, the loss value is too low. Thus, the network
gets a reward for an easy task.

### 1.2 The main loss (`loss_max`)

The network does 30 iterations. Then, the function calculates the loss on the result. The
gradient goes back through all 30 iterations.

This loss alone has a problem. All iterations use the same weights. But the state `h` can
record the number of iterations that the network did. Thus, the network can find a plan that
is correct only after exactly 30 iterations.

A large maze needs more than 30 iterations. After iteration 30, the state goes into an area
that the network did not see during training. Then, the result becomes incorrect. Bansal et
al. use the term "overthinking" for this problem.

### 1.3 The progressive loss (`loss_prog`)

The function does these steps:

1. It selects a random number `n` from 0 to 29.
2. It does `n` iterations without a gradient. The result is the state `h`.
3. It selects a random number `k` from 1 to 30 − `n`.
4. It starts at `h` and does `k` more iterations, with a gradient.
5. It calculates the loss on the result.

The gradient stops at `h`. Thus, the network cannot change the first `n` iterations to make
the subsequent iterations easier. The network must start from the state that it gets.

The numbers `n` and `k` are random. Thus, the network does not know the number of iterations
before `h` or after `h`. A count of the iterations does not help the network. The best method
for the network is to make each iteration improve the current state by a small quantity. You
can repeat an iteration of this type more than 30 times. A large maze needs this.

The sum of `n` and `k` is not more than 30.

### 1.4 The total loss

```python
loss = (1 - alpha) * loss_max + alpha * loss_prog
```

The value `alpha` sets the weight of each loss. The default value is 0.01. With this value,
most of the signal comes from `loss_max`. The progressive loss has only a small effect.

If you increase `alpha`, the result after more than 30 iterations becomes more stable. But the
accuracy after 30 iterations can decrease. If `alpha` is 0, the function uses only
`loss_max`. Then, overthinking can occur.

### 1.5 The gradient clip

The gradient goes back through approximately 150 convolutions. These convolutions use the
same weights, and the network has no batch normalization. Thus, the gradient can become very
large.

The function `clip_grad_norm_` sets a limit of 2.0 on the total size of the gradient. This
limit stops one bad batch from damaging the network.

### 1.6 How to examine the result

Use the function `solved_by_iters`. It shows the number of solved mazes after different
numbers of iterations.

- If the progressive loss works, the number of solved mazes stays the same or increases after
  iteration 30.
- If the progressive loss does not work, the number of solved mazes is highest at
  iteration 30. After iteration 30, it decreases.

### 1.7 Zero-init (`zero_init`)

Zero-init is an option for the DTNet weights at the start of the training. It does not change
the loss function. To use it, set `Config(zero_init=True)`. Only DTNet has this option.

**How it works.** Each `BasicBlock` has two convolutions. With zero-init, the code sets all the
weights of the second convolution (`conv2`) to 0 before the training starts. The block
calculates this result:

```
out = relu(conv2(relu(conv1(x))) + x)
```

If `conv2` is 0, the result is `relu(x)`. The input `x` is already the output of a ReLU, so the
result is `x`. Thus, each block starts as the identity. At the start of the training, one
iteration does only `recall_conv` and a ReLU. The blocks add a change to the state only when
the training gives their weights a value other than 0.

The idea comes from Fixup and SkipInit. These methods train deep residual networks without
batch normalization.

**Why it can help DTNet.** DTNet does 30 iterations of 2 blocks in the training. Thus, the
gradient goes through a very deep network that uses the same weights again and again, and
the network has no batch normalization. With random initial weights, each iteration can make
the state larger. After many iterations, the state and the gradient can become very large.
Then, a loss spike occurs. With zero-init, the blocks start as the identity, so the state does
not increase in each iteration at the start of the training.

**The results.** Finding 5 in [SUMMARY.md](SUMMARY.md) gives the details.

- Without loops in the training data, zero-init gave a better result. The model solved 100% of
  the test mazes of size 511, against 62.5% without zero-init. The largest loss spike was 118,
  against 8e7. Each value comes from only one run.
- With loops in the training data, zero-init did not help. The result at 30 iterations was
  the same as without zero-init. With more iterations, the result became very bad. The model
  learned a method that works only for 30 iterations.

Zero-init changes only the start of the training. It does not keep the network stable after
the start. Thus, it does not replace a lower learning rate. With loops, the runs use a
learning rate of 3e-4 (see the training stability section in [SUMMARY.md](SUMMARY.md)).

## Part 2: The runner (`runner.py`)

The file `runner.py` trains one model and tests it. One `Config` object gives all the
settings for one run.

### 2.1 The recorder (`_Recorder`)

The class `_Recorder` keeps a record of the run.

- The method `say` shows a message. If the run has an output folder, `say` also adds the
  message to the file `log.txt`.
- The method `save` writes an object to a JSON file in the output folder.

If the run has no output folder, the recorder writes no files.

### 2.2 The learning rate (`_warmup_cosine`)

The learning rate changes after each batch. The function `_warmup_cosine` gives a multiplier
for the learning rate. The change has two parts:

1. Warmup: the multiplier increases in a straight line from almost 0 to 1. The default warmup
   is 2 epochs.
2. Decay: the multiplier decreases along a cosine curve. It is 0 at the end of the last epoch.

A recurrent network has no normalization, so a large learning rate at the start can make the
training unstable. The warmup prevents this problem.

### 2.3 The training loop (`train`)

The function `train` does these steps:

1. It gets the training set and the validation set from `training_data`. The function keeps
   the most recent data in memory. Thus, a subsequent run with the same data settings does not
   make the mazes again.
2. It makes an Adam optimizer and the learning rate schedule.
3. For each epoch:
   1. It does one `train_step` for each batch. Then, it changes the learning rate.
   2. It calculates the validation loss and the fraction of solved validation mazes. It uses
      `max_iters` iterations.
   3. It adds the results to `history` and writes `history.json`.
   4. If this epoch is the best epoch, it keeps a copy of the weights. It also writes the
      weights to the checkpoint file.
   5. It calls `on_epoch`, if it is available. Optuna uses this call to stop a bad trial
      early.
4. At the end, it loads the weights of the best epoch into the model.

The best epoch has the highest fraction of solved mazes. If two epochs have the same
fraction, the epoch with the lower validation loss is better. The code compares the pairs
`(val_solved, -val_loss)` to do this.

If one batch loss is not a finite number, the training loss of that epoch is NaN. This shows
that the training became unstable.

On a CUDA device, the code uses bfloat16 autocast. This decreases the time and the memory for
each step. The setting `amp` turns this function on or off.

### 2.4 One full run (`run`)

The function `run` does these steps:

1. It selects the GPU if it is available. If not, it uses the CPU.
2. It sets the random seeds for PyTorch, NumPy and Python. Thus, a run with the same `Config`
   gives the same data and the same initial weights.
3. It writes `config.json`.
4. It makes the model. If `init_from` has a checkpoint, the model starts from those weights.
   A curriculum uses this function.
5. It trains the model with `train` and writes `curves.png`. If `skip_train` is true, it does
   not train. Instead, it loads the weights from the checkpoint.
6. It tests the model on each evaluation set (see 2.5). It writes the results to
   `results.json` after each set.
7. It writes `test.png`.

The function returns a dictionary with three items: `history`, `best` and `results`.

### 2.5 The evaluation sets

The function `eval_sets` gives the list of evaluation sets. There are two types:

- `extrap/<size>`: large mazes from the training generators. These sets do not use the test
  generators. Thus, you can use them to tune the settings.
- `<wall_removal>/<size>`: mazes from the test generators. The training never uses these
  generators. A `wall_removal` value of more than 0 adds loops to the maze. The code tests
  mazes with loops only up to the size `loop_test_max_size`, because these tests are slow.

For each set, the code calls `solved_by_iters` with each value in `test_iters`. The result
for one set is the fraction of solved mazes for each number of iterations. For example,
`results["0.0/63"][1000]` is the fraction of size-63 test mazes that the model solves after
1000 iterations.

If `results` already contains a set, the code does not test that set again.

### 2.6 Old results (`run_or_load`)

The function `run_or_load` looks for the results of an earlier run in `results/<name>/`.
There are three possible conditions:

| Condition | Action |
| --- | --- |
| `history.json` exists, and `results.json` has all the evaluation sets. | The function loads the results. It does not train or test. |
| `history.json` and the checkpoint exist, but some evaluation sets are missing. | The function loads the checkpoint. It tests only the missing sets. |
| Other conditions. | The function trains and tests the model from the start. |

Thus, if you stop the experiments notebook, you can start it again. The notebook continues
from the last finished step.

### 2.7 Helper functions

- `normalize_results` changes the iteration keys back to integers. JSON stores all keys as
  strings.
- `_read_json` reads a JSON file. If the file does not exist, it gives an empty dictionary.
- `load_config` reads `config.json` from a run folder and makes a `Config` again. It uses the
  folder name as the run name, because you can rename a run folder. The checkpoint name
  follows the folder name.
