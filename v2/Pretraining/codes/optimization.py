# This code is adapted from work by lrasmy (Zhilab), originally dated 2019-08-10.

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
    """Creates a learning rate scheduler with warmup and a weight decay optimizer."""

    class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
            super(WarmUp, self).__init__()
            self.initial_learning_rate = initial_learning_rate
            self.warmup_steps = warmup_steps
            self.decay_schedule_fn = decay_schedule_fn

        def __call__(self, step):
            global_step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)

            warmup_percent_done = global_step / warmup_steps
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

            return tf.cond(global_step < warmup_steps,
                           lambda: warmup_learning_rate,
                           lambda: self.decay_schedule_fn(step))

    # Linear decay schedule
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=0.0,
        power=1.0
    )

    if num_warmup_steps:
        lr_schedule = WarmUp(init_lr, lr_schedule, num_warmup_steps)

    # tfa.optimizers.AdamW 대신 tf.keras.optimizers.AdamW를 사용
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01  
    )

    return optimizer