## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import time
import sys
import tensorflow as tf
import numpy as np
#from numba import jit

# from multiprocessing.pool import ThreadPool


BINARY_SEARCH_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20
RETRAIN = False


# pool = ThreadPool()


class LADMMSTL2:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, ro=RO, retrain=RETRAIN):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.ro = ro
        self.retrain = retrain
        self.grad = self.gradient_descent(sess, model)

    def compare(self, x, y):
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

    def gradient_descent(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        tz = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_tz = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = tf.reduce_sum(tf.square(tz), [1, 2, 3])
        output = model.predict(newimg)

        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        loss1 = 5 * tf.reduce_sum(loss1)

        gradtz = tf.gradients(loss1, [tz])

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tz.assign(assign_tz))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z, })

            l2s, scores, nimg, z_grads = sess.run([l2dist_real, output, newimg, gradtz])

            return l2s, scores, nimg, np.array(z_grads)

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        o_besty = np.ones(imgs.shape)

        filterSize = 3
        stride = 3
        n = self.model.image_size * self.model.image_size * self.model.num_channels

        P = np.floor((self.model.image_size - filterSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P

        z = 0.0 * np.ones(imgs.shape)
        # z1 = 0.0*np.ones(imgs.shape)
        #v = 0.0 * np.ones([batch_size, n * P * Q])
        v = 0.0 * np.ones(imgs.shape)
        u = 0.0 * np.ones(imgs.shape)
        s = 0.0 * np.ones(imgs.shape)
        yt = 0.0 * np.ones(imgs.shape).reshape(batch_size, -1)
        ep = 0.8
        # e0 = -0.49

        #A = np.kron(np.ones([P * Q, 1]), np.identity(n))
        #Az = np.dot(A, z.reshape(batch_size, -1).transpose()).transpose()
        #Az = z.reshape(batch_size, -1)
        #Az = np.tile(Az,(P*Q))  
        #A_ = tf.placeholder(tf.float32, A.shape)
        #A = tf.convert_to_tensor(A)

        # filter = tf.ones([filterSize,filterSize,self.model.num_channels,self.model.num_channels], dtype=tf.float32)
        # g = tf.nn.conv2d(imgs, filter, padding = 'VALID', strides = [1] + [stride] * 2 + [1])

        alpha = 10
        tau = 1
        gamma = 2
        index = np.ones([batch_size, P*Q,filterSize * filterSize * self.model.num_channels],dtype=int)
        index2 = np.ones([batch_size, P*Q,filterSize * filterSize * self.model.num_channels],dtype=int)
        for b in range(batch_size):
            tmpidx = 0
            for q in range(Q):
                # plus = 0
                plus1 = P * q * n + q * stride * self.model.image_size
                for p in range(P):
                    index_ = np.array([], dtype=int)
                    index2_ = np.array([], dtype=int)
                    for c in range(self.model.num_channels):
                        plus2 = plus1 + self.model.image_size * self.model.image_size * c
                        for i in range(filterSize):
                            index_ = np.append(index_,
                                              np.arange(p * stride + i * self.model.image_size + plus2,
                                                        p * stride + i * self.model.image_size + plus2 + filterSize,
                                                        dtype=int))
                        index2_ = np.append(index2_, index_[-filterSize*filterSize:] - plus1  + q * stride * self.model.image_size)
                    #print(p,index_)
                    index[b, tmpidx] = index_
                    index2[b, tmpidx] = index2_
                    
                    tmpidx += 1
                    plus1 += n

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            if outer_step % 100 == 0:
                print(outer_step, o_bestl2)

            # delta step
            # l2
            # delt = self.ro / (self.ro + 2.0) * (z - u)
            # l1
            tmp = z - u - gamma / self.ro
            tmp = np.where(tmp > 0, tmp, 0)
            tmp1 = u - z - gamma / self.ro
            tmp1 = np.where(tmp1 > 0, tmp1, 0)
            delt = tmp - tmp1

            # w step
            temp = z - s
            temp1 = np.where(temp > np.minimum(0.5 - imgs, ep), np.minimum(0.5 - imgs, ep), temp)
            w = np.where(temp1 < np.maximum(-0.5 - imgs, -ep), np.maximum(-0.5 - imgs, -ep), temp1)

            # y step
            
            y0 = z - v
            y = y0

            #timestart = time.time()
            def findIndx(b):
            #for b in range(batch_size):
                for j in range(index.shape[1]):
                    # print(p,index)
                    y0D = np.take(y0[b], index2[b,j])
                    if np.linalg.norm(y0D) == 0:
                        tmpy = 0
                    else:
                        tmpy = 1 - tau / (self.ro * np.linalg.norm(y0D))
                    if tmpy > 0:
                        np.put(y[b], index2[b,j], tmpy * y0D)
                    else:
                        np.put(y[b], index2[b,j], 0)
                    #np.put(yt[b], index2[b,j], y0D)


            list(map(findIndx, range(batch_size)))
            #print(time.time()-timestart)

            # z step
            l2s, scores, nimg, z_grads = self.grad(imgs, labs, z)
            # ---------------- newz = np.where(z<np.percentile(np.abs(z),10),0,z)
            # ---------------------------------------------- testimgs = imgs+newz
            # --- testoutput = self.model.predict(tf.convert_to_tensor(testimgs))
            # ------------------------------------- testscore = testoutput.eval()
            # --------------------------------------- for b in range(batch_size):
            # ---------- print(np.argmax(testscore[b])==np.argmax(scores[b]))

            #c = y + v
            # e = np.zeros(P*Q)
            #c = c.reshape([batch_size, P * Q, self.model.image_size, self.model.image_size, -1])
            #Sc = np.sum(c, axis=1)
            Sc = y + v
            # for i in range(P*Q):
            # e[i] = 1
            # B = np.kron(e,np.identity(n))
            # SB += B
            # e[i] = 0
            # SBc = np.dot(SB,c.transpose()).tranpose().reshape(imgs.shape)*self.ro*SBc

            #eta = 1
            eta = 1/np.sqrt(outer_step+1)
            z = 1 / (alpha / eta + 2 * self.ro +  self.ro) * \
                (alpha / eta * z + self.ro * (delt + u) + self.ro * (w + s) + self.ro * Sc - z_grads[0])
            # print(Sc.mean(),w.mean(),y.mean(),delt.mean(),z.mean())

            u = u + delt - z
            
            #timestart = time.time()
            #z_ = tf.convert_to_tensor(tf.float32, z.reshape(batch_size, -1).transpose())
            #z_ = tf.placeholder(tf.float32, z.reshape(batch_size, -1).transpose().shape)
            #Az = tf.transpose(tf.matmul(A_,z_))
            #Az = Az.eval(feed_dict={A_:A, z_:z.reshape(batch_size, -1).transpose()})
            
            #Az = np.dot(A, z.reshape(batch_size, -1).transpose()).transpose()
            #Az = z.reshape(batch_size, -1)
            #Az = np.tile(Az,(P*Q)) 
            #print('1',time.time()-timestart)
            #v = v + (y + (P*Q-1)*y0)/(P*Q) - z
            v = v + y -z

            s = s + w - z
            
            #yt = yt.reshape(imgs.shape)
            l2s, scores, nimg, y_grads = self.grad(imgs, labs, y)

            for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                    print("change", e, o_bestl2[e] - l2)
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii
                    o_besty[e] = y[e]

        print('Finally', o_bestl2)
        # np.savetxt("outputimg.txt",o_besty[8].squeeze())
        if self.retrain:
            for tmpi in range(10):
                Nz = o_besty[np.nonzero(o_besty)]
                Nz = np.abs(Nz)
                e0 = np.percentile(Nz, 5)
                # e0 = 0.00001
                randm = -1 + 2*np.random.random((o_besty.shape))
                #z1 =  np.where(np.abs(o_besty) <= e0, 0, randm)
                A2 = np.where(np.abs(o_besty) <= e0, 0, 1)
                deltA = o_besty
                z1 = o_besty
                u1 = 0.0 * np.ones(imgs.shape)
                tmpC = self.ro / (self.ro + gamma)
                for outer_step in range(200):
                    if outer_step % 100 == 0:
                        print("retrain", tmpi, outer_step, o_bestl2)
    
                    tempA = (z1 - u1) 
                    tempA1 = np.where(np.abs(o_besty) <= e0, 0, tempA)
                    tempA2 = np.where(np.logical_and(tempA * tmpC > np.minimum(0.5 - imgs, ep), (np.abs(o_besty) > e0)),
                                      np.minimum(0.5 - imgs, ep), tempA1)
                    deltA = np.where(np.logical_and(tempA * tmpC < np.maximum(-0.5 - imgs, -ep), (np.abs(o_besty) > e0)),
                                     np.maximum(-0.5 - imgs, -ep), tempA2)
    
                    l2s, scores, nimg, z_grads = self.grad(imgs, labs, deltA)
                    z1 = 1 / (alpha/5 + 2 * self.ro) * (alpha/5 * z1 + self.ro * (deltA + u1) - np.multiply(z_grads[0],A2))
    
                    u1 = u1 + deltA - z1
    
                    #l2s, scores, nimg, z_grads = self.grad(imgs, labs, deltA)
                    for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                        if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                            print("change", e, o_bestl2[e] - l2)
                            o_bestl2[e] = l2
                            o_bestscore[e] = np.argmax(sc)
                            o_bestattack[e] = ii
                            o_besty[e] = deltA[e]

        return o_bestattack
