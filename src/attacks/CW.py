"""The CarliniWagnerL2 attack
"""
# pylint: disable=missing-docstring
import logging

import numpy as np
from numpy.lib.npyio import save
import tensorflow as tf
from utils import *
from attacks.backdoor_generator import BackdoorGenerator
import math as m
from tqdm.gui import trange

MAX_ITER = 5


def create_logger(name):
  """
  Create a logger object with the given name.

  If this is the first time that we call this method, then initialize the
  formatter.
  """
  base = logging.getLogger("reforcement")
  if len(base.handlers) == 0:
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] ' +
                                  '%(message)s')
    ch.setFormatter(formatter)
    base.addHandler(ch)

  return base

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = create_logger("cleverhans.attacks.carlini_wagner_l2")
_logger.setLevel(logging.INFO)

cw_params = {
             'batch_size': 1,
             'confidence': 10,
             'learning_rate': 0.1,
             'binary_search_steps': 5,
             'max_iterations': 300,
             'abort_early': True,
             'initial_const': 0.01,
             'clip_min': 0,
             'clip_max': 1,
             'targeted': True}


class CarliniWagnerL2(BackdoorGenerator):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model: cleverhans.model.Model

    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, param, args= cw_params,):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        # if not isinstance(model, Model):
        #   wrapper_warning_logits()
        #   model = CallableModelWrapper(model, 'logits')
        super(CarliniWagnerL2, self).__init__(model, param)
        
        self.feedable_kwargs = ('y', 'y_target')
        self.structural_kwargs = [
            'batch_size', 'confidence', 'targeted', 'learning_rate',
            'binary_search_steps', 'max_iterations', 'abort_early',
            'initial_const', 'clip_min', 'clip_max'
        ]
        self.sess = K.get_session()
        fixed = dict(
            (k, v) for k, v in args.items() if k in self.structural_kwargs)
        feedable_names = self.feedable_kwargs
        self.feedable = {k: v for k, v in args.items() if k in feedable_names}
        hash_key = tuple(sorted(fixed.items()))
        self.new_kwargs = dict(x for x in fixed.items())
        self.build_attack(**self.new_kwargs)
        self.source = int(self.param.get_conf('poison_label_source'))
        self.target = int(self.param.get_conf('poison_label_target'))

    def build_attack(self, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: A tensor with the inputs.
        :param kwargs: See `parse_params`
        """
        self.parse_params(**kwargs)
        # preds = self.model.get_output_tensor()
        # preds_max = tf.reduce_max(preds, 1, keepdims=True)
        # original_predictions = tf.to_float(tf.equal(preds, preds_max))
        # labels = tf.stop_gradient(original_predictions)

        self.CW = CWL2(self.param, self.model, self.batch_size, self.confidence,
                    self.targeted, self.learning_rate,
                    self.binary_search_steps, self.max_iterations,
                    self.abort_early, self.initial_const, self.clip_min,
                    self.clip_max, self.param.get_conf('num_classes'),
                    (self.model.get_input_tensor().get_shape()[1:]))

        # def cw_wrap(x_val, y_val):
        #     return np.array(, dtype=np.float32)

        # wrap = tf.py_func(cw_wrap, [x, labels], tf.float32)
        # wrap.set_shape(x.get_shape())

        # return wrap


    def attack(self, data, xi=30.0/255.0, **kwargs):
        """
        Generate adversarial examples and return them as a NumPy array.
        Sub-classes *should not* implement this method unless they must
        perform special handling of arguments.
        :param x_val: A NumPy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A NumPy array holding the adversarial examples.
        """

        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                            " provided")
        
        num_selection  = 5000
        x_val, y_val, _, _ = data.get_specific_label_clean_data(self.source)
        x_val = x_val[:num_selection]
        y_val = y_val[:num_selection]
        # if hash_key not in self.graphs:
        #     self.construct_graph(fixed, feedable, x_val, hash_key)
        # else:
        # # remove the None arguments, they are just left blank
        #     for k in list(feedable.keys()):
        #         if feedable[k] is None:
        #             del feedable[k]
        
        # feed_dict = {self.input_tensor: x_val,self.labels_tensor: to_categorical(y_val)}
        targets = np.zeros_like(y_val)
        targets[:,self.target] = 1 
        self.perturb =self.CW.attack(x_val, targets, xi=xi)
        # for name in self.feedable:
        #     feed_dict[new_kwargs[name]] = self.feedable[name]
        
        # return pert

    def parse_params(self,
                    batch_size=1,
                    confidence=10,
                    learning_rate=5e-3,
                    binary_search_steps=5,
                    max_iterations=1000,
                    abort_early=True,
                    initial_const=1e-2,
                    clip_min=0,
                    clip_max=1,
                    targeted=True):
        """
        :param y: (optional) A tensor with the true labels for an untargeted
                attack. If None (and y_target is None) then use the
                original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                        examples with larger l2 distortion, but more
                        strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                            Smaller values produce better results but are
                            slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                            to a larger value will produce lower distortion
                            results. Using only a few iterations requires
                            a larger learning rate, and will produce larger
                            distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                            relative importance of size of the perturbation
                            and confidence of classification.
                            If binary_search_steps is large, the initial
                            constant is not important. A smaller value of
                            this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # ignore the y and y_target argument
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

def ZERO():
    return np.asarray(0., dtype=np_dtype)


class CWL2(object):
    def __init__(self, param, model, batch_size, confidence, targeted,
                 learning_rate, binary_search_steps, max_iterations,
                 abort_early, initial_const, clip_min, clip_max, num_labels,
                 shape):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                        examples with larger l2 distortion, but more
                        strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                        examples produced. If set to False, they will be
                        misclassified in any wrong class. If set to True,
                        they will be misclassified in a chosen target class.
        :param learning_rate: The learning rate for the attack algorithm.
                            Smaller values produce better results but are
                            slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                            to a larger value will produce lower distortion
                            results. Using only a few iterations requires
                            a larger learning rate, and will produce larger
                            distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                            relative importance of size of the pururbation
                            and confidence of classification.
                            If binary_search_steps is large, the initial
                            constant is not important. A smaller value of
                            this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        """

        self.param = param
        self.sess = K.get_session()
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model



        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size] + list(shape))

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)),
                                dtype=tf_dtype,
                                name='tlab')
        self.const = tf.Variable(np.zeros(batch_size),
                                 dtype=tf_dtype,
                                 name='const')
        
        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')
        self.assign_tlab = tf.placeholder(tf_dtype, (batch_size, num_labels),
                                          name='assign_tlab')
        self.assign_const = tf.placeholder(tf_dtype, [batch_size],
                                           name='assign_const')

        # the resulting instance, tanh'd to keep bounded from clip_min
        # to clip_max
        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
        self.newimg = self.newimg * (clip_max - clip_min) + clip_min
        

        # prediction BEFORE-SOFTMAX of the model
        # model = Model(self.model.get_input_tensor(),[self.model.get_output_bef_softmax()])
        # self.output = model([self.newimg, self.model.get_input_tensor()[1]])

        model = Model([self.model.get_input_tensor()],[self.model.get_output_bef_softmax()])
        self.output = model(self.newimg)

        # distance to the input data
        self.other = (tf.tanh(self.timg) + 1) / \
            2 * (clip_max - clip_min) + clip_min
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.other),
                                 list(range(1, len(shape))))

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max((1 - self.tlab) * self.output - self.tlab * 10000,
                           1)

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss_out = self.l2dist + self.const * loss1
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * loss1)
        self.loss = self.loss1 + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        # self.setup.append(self.input_tensor.assign(self.new_imgs))
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)


    def add_pert(self, img, pert):
        return np.clip(img + pert, self.clip_min, self.clip_max)
        

    def attack(self, imgs, targets, xi=15.0/255):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """
        # origin_imgs = copy.deepcopy(imgs)
        num_images = len(imgs)
        imgs = np.array(imgs, dtype=np.float32)
        l2_dis = np.zeros(num_images)
        loss = np.zeros(num_images)
        num_iter = np.zeros(num_images)
        score = np.zeros(num_images)
        # r = []
        tot_pert = np.zeros_like(imgs[0])
        index = np.arange(len(imgs))
        
        for it in range(MAX_ITER):
            print('string iter', it)
            np.random.shuffle(index)
            for k,i in enumerate(index):
                _logger.debug(
                    ("Running CWL2 attack on instance %s of %s", i, len(imgs)))
                if self.model.classifier.predict(self.add_pert(imgs[i:i+1],tot_pert)).argmax(axis=1)[0] == targets[i].argmax(axis=0):
                    continue


                adv_imgs, l2_dis[i:i+1],loss[i:i+1], \
                    score[i:i+1],num_iter[i:i+1] = \
                        self.attack_batch(self.add_pert(imgs[i:i+1], tot_pert), targets[i:i + 1])
                # r.extend(adv_imgs)
                pert = np.squeeze(adv_imgs - self.add_pert(imgs[i], tot_pert))
                tot_pert = self.proj_lp(tot_pert + pert, xi=xi, p=np.inf)
                print('>> k = ', k, ', img_idx = ', i, ', pass #', it, "size =", np.mean(abs(tot_pert)))
                tot_pert *= 0.998

            # imgs = origin_imgs + tot_pert
            fooling_rate = 0
            for i in range(0, len(imgs), 128): 
                up_i = min(i + 128, len(imgs))
                preds = self.model.classifier.predict(self.add_pert(imgs[i:up_i], tot_pert)).argmax(axis=1)
                fooling_rate += np.sum(preds == targets[i: up_i].argmax(axis=1))
                # print(i, np.sum(preds == targets[i:up_i].argmax(axis=1)), fooling_rate, len(preds))
            fooling_rate /=( 1.0 * len(imgs))
            print("fool rate is", fooling_rate)
            if fooling_rate > 0.8:
                break
        
        # r = np.array(r)
        # save_name = '_'.join([self.param.get_conf('model_prefix'), get_date(), postfix])
        # save_pkl = os.path.join(self.param.get_conf('perturbation_dir'), save_name + '.pkl')

        # with open(save_pkl, 'wb') as f:
        #     pickle.dump(tot_pert, f)

        return tot_pert

    def attack_batch(self, imgs, labs):
        """
    Run the attack on a batch of instance and labels.
    """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        oimgs = np.clip(imgs, self.clip_min, self.clip_max)

        # re-scale instances to be within range [0, 1]
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        imgs = (imgs * 2) - 1
        # convert to tanh-space
        imgs = np.arctanh(imgs * .999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_iter = [0] * batch_size
        o_bestloss = [1e10] * batch_size
        o_bestattack = np.copy(oimgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # print("search iteration ",outer_step)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step %s of %s", outer_step,
                          self.BINARY_SEARCH_STEPS)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup, {
                    self.assign_timg: batch,
                    self.assign_tlab: batchlab,
                    self.assign_const: CONST
                })

            prev = 1e6
            iter_num = 0
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l_o, l2s, scores, nimg = self.sess.run([
                    self.train, self.loss, self.loss_out, self.l2dist, self.output,
                    self.newimg],
                # feed_dict={self.model.get_input_tensor()[1]:np.random.randint(0,2,size=(len(batch),7))}
                )
                
                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(
                        ("    Iteration {} of {}: loss={:.3g} " +
                         "l2={:.3g} f={:.3g}").format(iteration,
                                                      self.MAX_ITERATIONS, l,
                                                      np.mean(l2s),
                                                      np.mean(scores)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii, l_oi) in enumerate(zip(l2s, scores, nimg, l_o)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        o_iter[e] = iter_num
                        o_bestloss[e] = l_oi
                iter_num += 1
            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                   bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
            _logger.debug("  Successfully generated adversarial examples " +
                          "on {} of {} instances.".format(
                              sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack, o_bestl2, o_bestloss, o_bestscore, o_iter


    def proj_lp(self, v, xi, p):

        # Project on the lp ball centered at 0 and of radius xi

        # SUPPORTS only p = 2 and p = Inf for now
        # print('xi is', xi)
        if p == 2:
            v = v * min(1, xi / np.linalg.norm(v.flatten(1)))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')

        return v

