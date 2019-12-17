import os, time
import numpy as np
import tensorflow as tf


class BaseModel(object):
  """ Base model of Neural Network
  basic methods for BaseModel
  Uasge:
    model = models.DetNet(...)
    nn_model.BaseModel(model).train()
  """

  def __init__(self, model):

    # build model
    self.model = model
    model._setup_prediction()
    self.gstep = tf.Variable(0, name='global_step', trainable=False)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10**4)


  def train(self, resume):

    # config training
    lrate = tf.Variable(self.model.config.lrate, 
                        name='learning_rate', trainable=False)
    self.model._setup_loss()
    self.model._setup_optimizer(lrate, self.gstep)
    ckpt_step = self.model.ckpt_step
    summary_step = self.model.summary_step

    with tf.Session() as sess:

        # summary
        self.merged_summaries = tf.summary.merge_all()
        train_log = os.path.join(self.model.ckpt_dir, 'train_log')
        valid_log = os.path.join(self.model.ckpt_dir, 'valid_log')
        train_writer = tf.summary.FileWriter(train_log, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log)

        print('Initialize variables')
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        if resume: self.load(sess)

        print('Starting data threads coordinator.')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Starting optimization.')
        start_time = time.time()
        step=0

        try:
          # Training loop
          while not coord.should_stop():
            # feed training dict
            train_step = sess.run(self._train_summary(), 
                                  feed_dict={self.model.is_training: True})
            step = train_step['step']

            # Save summary & checkpoint
            if  step % summary_step == 0:
                # validition
                valid_step = sess.run(self._valid_summary(), 
                                      feed_dict={self.model.is_training: False})
                print('Step {} | {:.0f}s | train acc. = {:.1f}% | valid acc. = {:.1f}%'\
                    .format(step, time.time()-start_time,
                            100*train_step['accuracy'], 100*valid_step['accuracy']))
                # add summary
                train_writer.add_summary(train_step['summary'], global_step=step)
                valid_writer.add_summary(valid_step['summary'], global_step=step)
            if  step+1 % ckpt_step == 0:
                print('Saving Checkpoint {}th step'.format(step))
                self.save(sess)

        except KeyboardInterrupt:
            print('Interrupted training at step {}.'.format(step))
            self.save(sess)
            train_writer.close()
            valid_writer.close()

        except tf.errors.OutOfRangeError:
            print('Training completed at step {}.'.format(step))
            self.save(sess)
            train_writer.close()
            valid_writer.close()

        finally:
            print('Shutting down data threads.')
            coord.request_stop()
            train_writer.close()
            valid_writer.close()

        # Wait for data threads
        print('Waiting for all threads.')
        coord.join(threads)
        print('Optimization done.')


  def load(self, sess, step=None):
    """ Load model by ckeck point
    """
    if  step==None:
        ckpt_path = tf.train.latest_checkpoint(self.model.ckpt_dir)
    else:
        ckpt_path = os.path.join(self.model.ckpt_dir, 'model-'+str(step))
    self.saver.restore(sess, ckpt_path)
    step = tf.train.global_step(sess, self.gstep)
    print('Load model at step {} from check point {}.'.format(step, ckpt_path))


  def save(self, sess):
    """ Save model in check points
    """
    ckpt_path = os.path.join(self.model.ckpt_dir, 'model')
    if not os.path.exists(self.model.ckpt_dir):
        os.makedirs(self.model.ckpt_dir)
    self.saver.save(sess, ckpt_path, global_step=self.gstep)


  def _train_summary(self):
    """ Summary training step
    """
    return {'optimizer':self.model.optimizer,
            'summary':  self.merged_summaries,
            'step':     self.gstep,
            'accuracy': self.model.accuracy,
            'loss':     self.model.loss
               }

  def _valid_summary(self):
    """ Summary validation step
    """
    return {'summary':  self.merged_summaries,
            'accuracy': self.model.accuracy
               }

