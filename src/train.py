from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

num_gpus = 16
num_steps = 200
learning_rate = 0.003
batch_size = 1440
display_step = 10

num_input = 8192
num_classes = 20
