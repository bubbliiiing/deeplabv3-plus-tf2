import os

import tensorflow as tf
from tqdm import tqdm


def get_train_step_fn(strategy):
    @tf.function
    def train_step(images, labels, net, optimizer, loss, metrics):
        with tf.GradientTape() as tape:
            prediction = net(images, training=True)
            loss_value = loss(labels, prediction)

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        _f_score = tf.reduce_mean(metrics(labels, prediction))
        return loss_value, _f_score
    if strategy == None:
        return train_step
    else:
        #----------------------#
        #   多gpu训练
        #----------------------#
        @tf.function
        def distributed_train_step(images, labels, net, optimizer, loss, metrics):
            per_replica_losses, per_replica_score = strategy.run(train_step, args=(images, labels, net, optimizer, loss, metrics))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_score, axis=None)
        return distributed_train_step

@tf.function
def val_step(images, labels, net, optimizer, loss, metrics):
    prediction = net(images, training=False)
    loss_value = loss(labels, prediction)
    _f_score = tf.reduce_mean(metrics(labels, prediction))
    return loss_value, _f_score

def fit_one_epoch(net, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, metrics, save_period, save_dir, strategy):
    train_step      = get_train_step_fn(strategy)
    total_loss      = 0
    val_loss        = 0
    total_f_score   = 0
    val_f_score     = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, labels = batch[0], batch[1]
            labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

            loss_value, _f_score = train_step(images, labels, net, optimizer, loss, metrics)
            total_loss      += loss_value.numpy()
            total_f_score   += _f_score.numpy()

            pbar.set_postfix(**{'total Loss'    : total_loss / (iteration + 1), 
                                'total f_score' : total_f_score / (iteration + 1),
                                'lr'            : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, labels = batch[0], batch[1]
            labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

            loss_value, _f_score = val_step(images, labels, net, optimizer, loss, metrics)
            val_loss    += loss_value.numpy()
            val_f_score += _f_score.numpy()

            pbar.set_postfix(**{'total Loss'    : val_loss / (iteration + 1), 
                                'total f_score' : val_f_score / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': total_loss / epoch_step, 'val_loss': val_loss / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
