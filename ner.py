
from config import Configuration
from model import NER
from load_data import load_data
import re
import os
import codecs
import getpass
import sys
import time
import numpy as np
import tensorflow as tf
import utils as ut

def run_epoch(
            config,
            model,
            alpha,
            schedule,
            beta,
            session,
            char_X,
            word_length_X,
            cap_X,
            word_X,
            mask_X,
            sentence_length_X,
            Y=None,
            verbose=True
            ):

    # We're interested in keeping track of the loss during training
    total_loss = []
    V_total_loss = []
    total_steps = int(np.ceil(len(word_X) / float(config.batch_size)))
    data = ut.data_iterator(
                char_X,
                word_length_X,
                cap_X,
                word_X,
                mask_X,
                sentence_length_X,
                config.batch_size,
                config.tag_size,
                Y,
                True)

    for step, (
                char_input_data,
                word_length_data,
                cap_input_data,
                word_input_data,
                word_mask_data,
                sentence_length_data,
                tag_data) in enumerate(data):

        b_size = sentence_length_data.shape[0]
        #flip coin:
        coin_probs = np.random.rand(b_size, config.max_sentence_length)

        feed = model.create_feed_dict(
                    char_input_batch=char_input_data,
                    word_length_batch=word_length_data,
                    cap_input_batch=cap_input_data,
                    word_input_batch=word_input_data,
                    word_mask_batch=word_mask_data,
                    sentence_length_batch=sentence_length_data,
                    dropout_batch=config.dropout,
                    alpha=alpha,
                    schedule=schedule,
                    coin_probs=coin_probs,
                    beta=beta,
                    tag_batch=tag_data
                )

        if alpha==0.0:
            loss , _ = session.run([model.loss, model.train_op], feed_dict=feed)
            total_loss.append(loss)
            ##
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                                                            step,
                                                            total_steps,
                                                            np.mean(total_loss)
                                                            )
                                )
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        else:
            V_loss, loss, _, _ = session.run([model.V_loss, model.loss, model.V_train_op, model.train_op], feed_dict=feed)
            total_loss.append(loss)
            V_total_loss.append(V_loss)
            ##
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}  |  V loss = {}'.format(
                                                            step,
                                                            total_steps,
                                                            np.mean(total_loss),
                                                            np.mean(V_total_loss)
                                                            )
                                )
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

    if alpha==0.0:
        return np.mean(total_loss), 0
    else:
        return np.mean(total_loss), np.mean(V_total_loss)

def predict(
        config,
        model,
        session,
        char_X,
        word_length_X,
        cap_X,
        word_X,
        mask_X,
        sentence_length_X,
        Y=None):

    """Make predictions from the provided model."""

    # We deactivate dropout by setting it to 1
    dp = 1.0
    #dummy value
    alpha = 0.0
    schedule = 1.0
    beta = 10**8

    losses = []
    results = []
    if np.any(Y):
        data = ut.data_iterator(
                    char_X,
                    word_length_X,
                    cap_X,
                    word_X,
                    mask_X,
                    sentence_length_X,
                    config.batch_size,
                    config.tag_size,
                    Y,
                    False)
    else:
        data = ut.data_iterator(
                    char_X,
                    word_length_X,
                    cap_X,
                    word_X,
                    mask_X,
                    sentence_length_X,
                    config.batch_size,
                    config.tag_size,
                    None,
                    False)

    for step, (
                char_input_data,
                word_length_data,
                cap_input_data,
                word_input_data,
                word_mask_data,
                sentence_length_data,
                tag_data) in enumerate(data):

        b_size = sentence_length_data.shape[0]
    	coin_probs = np.zeros((b_size, config.max_sentence_length))
        feed = model.create_feed_dict(
                    char_input_batch=char_input_data,
                    word_length_batch=word_length_data,
                    cap_input_batch=cap_input_data,
                    word_input_batch=word_input_data,
                    word_mask_batch=word_mask_data,
                    sentence_length_batch=sentence_length_data,
                    dropout_batch=dp,
                    alpha=alpha,
                    schedule=schedule,
                    coin_probs=coin_probs,
                    beta=beta,
                    tag_batch=tag_data
                )

        if config.inference=="CRF":
            if np.any(tag_data):
                feed[model.tag_placeholder] = tag_data
                loss, unary_scores, sequence_lengths, transition_params = session.run(
                                                                        [
                                                                        model.loss,
                                                                        model.preds,
                                                                        model.sentence_length_placeholder,
                                                                        model.transition_params
                                                                        ],
                                                                        feed_dict=feed
                                                                        )
                losses.append(loss)
            else:
                unary_scores, sequence_lengths, transition_params = session.run(
                                                                        [
                                                                        model.preds,
                                                                        model.sentence_length_placeholder,
                                                                        model.transition_params
                                                                        ],
                                                                        feed_dict=feed
                                                                        )
            inner_results = []
            for unary_scores_, sequence_length_ in zip(unary_scores, sequence_lengths):
                # Remove padding.
                unary_scores_ = unary_scores_[:sequence_length_]
                # Compute the highest score and its tag sequence.
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                                                unary_scores_,
                                                                transition_params
                                                            )
                predicted_indices = viterbi_sequence
                inner_results.append(predicted_indices)

            results.append(inner_results)

        else:
            if np.any(tag_data):
                feed[model.tag_placeholder] = tag_data

            batch_predicted_indices = session.run([model.outputs], feed_dict=feed)

            results.append(batch_predicted_indices[0])

    if len(losses)==0:
        return 0, results

    else:
        return np.mean(losses), results

def save_predictions(
            config,
            predictions,
            sentence_length,
            filename,
            words,
            true_tags,
            num_to_tag,
            num_to_word
            ):

    """Saves predictions to the provided file."""
    with open(filename, "wb") as f:
        for batch_index in range(len(predictions)):
            batch_predictions = predictions[batch_index]
            b_size = len(batch_predictions)
            for sentence_index in range(b_size):
                for word_index in range(config.max_sentence_length):
                    ad = (batch_index * config.batch_size) + sentence_index
                    if(word_index < sentence_length[ad]):
                        to_file = str(num_to_word[words[ad][word_index]])
                        to_file += " "
                        to_file += str(num_to_tag[true_tags[ad][word_index]])
                        to_file += " "
                        to_file += str(num_to_tag[batch_predictions[sentence_index][word_index]])
                        to_file += "\n"
                        f.write(to_file)

                f.write("\n")

def eval_fscore():
    os.system("%s < %s > %s" % ('./conlleval', 'temp.predicted', 'temp.score'))
    result_lines = [line.rstrip() for line in codecs.open('temp.score', 'r', 'utf8')]
    return float(result_lines[1].strip().split()[-1])

def run_model(beam):
    config = Configuration()
    if beam:
        config.beamsearch=True
    else:
        config.beamsearch=False

    data = load_data(config)
    path = "./comparison-methods-results"
    if not os.path.exists(path):
        os.makedirs(path)

    #alpha shows how much we care about the reinforcement learning.
    alpha = 0.0
    schedule = 1.0
    beta = 10**8
    save_epoch = 20
    model_name = config.inference
    for i in range(config.runs):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            run = i + 1
            seed = run**3
            tf.set_random_seed(seed)
            np.random.seed(seed)
            model = NER(config, data['word_vectors'], seed)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                best_val_loss = float('inf')
                best_val_epoch = 0
                session.run(init)

                if model_name=='AC-RNN' or model_name=='R-RNN' or model_name=='BR-RNN':
                    save_epoch = -1
                    saver.restore(session, path + '/' + 'RNN' + '.' + str(run) + '/weights')
                    optimizer_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adam_optimizer")
                    session.run(tf.variables_initializer(optimizer_scope))

                first_start = time.time()
                if beam:
                    epoch=10000
                else:
                    epoch=0
                while (epoch<config.max_epochs):
                    if model_name=='AC-RNN' or model_name=='R-RNN' or model_name=='BR-RNN':
                        alpha = np.minimum(1.0, 0.5 + epoch * 0.05)

                    if epoch>0 and (model_name=='DIF-SCH' or model_name=='SCH'):
                        k = 50.0
                        #annealing beta
                        beta = np.minimum(10**8, (2)**epoch)
                        #inverse sigmoid decay
                        schedule = float(k)/float(k + np.exp(float(epoch)/k))

                    print
                    print 'Model:{} Run:{} Epoch:{}'.format(model_name, run, epoch)
                    start = time.time()
                    train_loss , V_train_loss = run_epoch(
                                                            config,
                                                            model,
                                                            alpha,
                                                            schedule,
                                                            beta,
                                                            session,
                                                            data['train_data']['char_X'],
                                                            data['train_data']['word_length_X'],
                                                            data['train_data']['cap_X'],
                                                            data['train_data']['word_X'],
                                                            data['train_data']['mask_X'],
                                                            data['train_data']['sentence_length_X'],
                                                            data['train_data']['Y']
                                                            )
                    if epoch > save_epoch:
                        _ , predictions = predict(
                                                config,
                                                model,
                                                session,
                                                data['dev_data']['char_X'],
                                                data['dev_data']['word_length_X'],
                                                data['dev_data']['cap_X'],
                                                data['dev_data']['word_X'],
                                                data['dev_data']['mask_X'],
                                                data['dev_data']['sentence_length_X'],
                                                data['dev_data']['Y']
                                                )

                        print 'Training loss: {}'.format(train_loss)
                        if alpha!=0.0: print 'V Training loss: {}'.format(V_train_loss)
                        save_predictions(
                                        config,
                                        predictions,
                                        data['dev_data']['sentence_length_X'],
                                        "temp.predicted",
                                        data['dev_data']['word_X'],
                                        data['dev_data']['Y'],
                                        data['num_to_tag'],
                                        data['num_to_word']
                                        )

                        val_fscore_loss = 100.0 - eval_fscore()
                        print 'Validation fscore: {}'.format(100 - val_fscore_loss)

                        if  val_fscore_loss < best_val_loss:
                            best_val_loss = val_fscore_loss
                            best_val_epoch = epoch
                            if not os.path.exists(path + '/' + model_name + '.' + str(run)):
                                os.makedirs(path + '/' + model_name + '.' + str(run))
                            saver.save(session, path + '/' + model_name + '.' + str(run) + '/weights')

                        # For early stopping which is kind of regularization for network.
                        if epoch - best_val_epoch > config.early_stopping:
                            break
                            ###

                        print 'Epoch training time: {} seconds'.format(time.time() - start)
                    epoch += 1
                print 'Total training time: {} seconds'.format(time.time() - first_start)

                saver.restore(session, path + '/' + model_name + '.' + str(run) + '/weights')
                print
                print 'Model:{} Run:{} Dev'.format(model_name, run)
                start = time.time()
                _ , predictions = predict(
                                        config,
                                        model,
                                        session,
                                        data['dev_data']['char_X'],
                                        data['dev_data']['word_length_X'],
                                        data['dev_data']['cap_X'],
                                        data['dev_data']['word_X'],
                                        data['dev_data']['mask_X'],
                                        data['dev_data']['sentence_length_X'],
                                        data['dev_data']['Y']
                                        )

                print 'Total prediction time: {} seconds'.format(time.time() - start)
                print 'Writing predictions to dev.predicted'
                if beam:
                    name = "beam.dev.predicted"
                else:
                    name = "dev.predicted"

                save_predictions(
                                config,
                                predictions,
                                data['dev_data']['sentence_length_X'],
                                path + '/' + model_name + '.' + str(run) + '.' + name,
                                data['dev_data']['word_X'],
                                data['dev_data']['Y'],
                                data['num_to_tag'],
                                data['num_to_word']
                                )
                print
                print 'Model:{} Run:{} Test'.format(model_name, run)
                start = time.time()
                _ , predictions = predict(
                                        config,
                                        model,
                                        session,
                                        data['test_data']['char_X'],
                                        data['test_data']['word_length_X'],
                                        data['test_data']['cap_X'],
                                        data['test_data']['word_X'],
                                        data['test_data']['mask_X'],
                                        data['test_data']['sentence_length_X'],
                                        data['test_data']['Y']
                                        )

                print 'Total prediction time: {} seconds'.format(time.time() - start)
                print 'Writing predictions to test.predicted'
                if beam:
                    name = "beam.test.predicted"
                else:
                    name = "test.predicted"
                save_predictions(
                                config,
                                predictions,
                                data['test_data']['sentence_length_X'],
                                path + '/' + model_name + '.' + str(run) + '.' + name,
                                data['test_data']['word_X'],
                                data['test_data']['Y'],
                                data['num_to_tag'],
                                data['num_to_word']
                                )
    return

if __name__ == "__main__":
    run_model(False)
    run_model(True)
