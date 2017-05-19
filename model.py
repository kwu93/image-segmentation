from util import *
class ImageSegmentation(object):
    def __init__(self, config, model, *args):
        self.config = config
        self.model = model
        # set up placeholder tokens
        self.add_placeholders()
        
        with tf.variable_scope("image-seg", initializer=tf.uniform_unit_scaling_initializer(1.0)):

            self.preds = self.setup_system()           
            self.loss = self.setup_loss(self.preds)
            self.train_op = self.setup_training_op(self.loss)
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape= \
                                    (None, self.config.image_width, self.config.image_height), name="dicom")
        self.labels_placeholder = tf.placeholder(tf.int32, shape = \
                                    (None, self.config.contour_width, self.config.contour_height), name = 'target')

    def setup_system(self):
        X = self.input_placeholder
        model = self.model

        pred_mask = model.forwardProp(X)
        return pred_mask

    def setup_loss(self, preds):
        target = self.labels_placeholder
        
        target = tf.reshape(target, (-1, self.config.width * self.config.height))
        preds = tf.reshape(preds, (-1, self.config.width * self.config.height))
        
        pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits = preds, labels = target)
        loss = tf.reduce_sum(pixel_loss)
        return loss

    def setup_training_op(self, loss, op_type = "adam"):
        method = get_optimizer(op_type)

        optimizer = method(learning_rate=self.config.learning_rate)

        ## Optional: Clip large gradient values #########

        gradients_and_variables = optimizer.compute_gradients(loss)
        gradients = [item[0] for item in gradients_and_variables]
        variables = [item[1] for item in gradients_and_variables]                      
        # if self.config.clip_gradients:
        #     gradients , _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        #     gradients_and_variables = zip(gradients, variables)        
        self.grad_norm = tf.global_norm(gradients)           
#        train_op = optimizer.apply_gradients(gradients_and_variables)

        train_op = optimizer.minimize(loss)
        return train_op
    
    def optimize(self, images, targets):
        """
        Takes in actual data to optimize model
        This method is equivalent to a step() function
        """

        # images are unrolled into 1D -> convert back into 2D
        batch_size = len(images)
        
        np.reshape(images, self.config.image_width, self.config.image_height)
        np.reshape(targets, self.config.contour_width, self.config.contour_height)
        
        input_feed = {}

        input_feed[self.input_placeholder] = images
        input_feed[self.labels_placeholder] = targets
        input_feed[self.dropout_placeholder] = self.config.dropout
        output_feed = [self.loss]

        loss= session.run(output_feed, feed_dict=input_feed)
        loss = np.sum(loss)      

        return loss            

    def run_epoch(self, sess, train, train_dir, epoch, sample = 100, log = True):
        batch_losses = []
        minibatches = get_minibatches(train, self.config.batch_size)
        savePath = train_dir + '/model'

        for i, batch in enumerate(minibatches):
            input_batch, target_batch = batch

            loss = self.optimize(sess, input_batch, target_batch)
            batch_losses.append(loss)       
            #this normalizes the loss over number of training examples in minibatch
            loss = loss * self.config.batch_size/ input_batch.shape[0] 


            evalstep = 10
            if (i % evalstep == 0 and i > 0):
                print("\nCurrently on minibatch: ", i)
                if (i % 100 == 0 and i > 0):
                    print("Saver checkpoint on this minibatch...: ", i)
                    self.saver.save(sess, savePath, global_step = i + epoch)

        return sum(batch_losses) 


    def train(self, session, dataset, train_dir, sample=100, log=True):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()

        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        logging.info(vars(self.config))
        savePath = train_dir + "/model"
        epoch_losses = []
        for epoch in range(self.config.epochs):
            print("\nRunning epoch ", epoch + 1)
            loss = self.run_epoch(session, dataset, paragraph, train_dir, epoch + 1)
            
            print("\nFinished running epoch ", epoch + 1)
            epoch_losses.append(loss)
            self.saver.save(session, savePath, global_step = epoch + 1)
        
            logging.info("Epoch: {} end, train loss: {}".format(epoch, loss))