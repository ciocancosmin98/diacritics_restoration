from typing import Dict, List, Tuple
import tensorflow as tf


class BiLSTM(tf.keras.Model):

    def __init__(
        self,
        lstm_units: int,
        num_rnns: int,
        input_alphabet_size: int,
        target_alphabet_size: int,
        embedding_dim: int,
        use_residual: bool,
        dropout: float
    ):
        super(BiLSTM, self).__init__()
        self.use_residual = use_residual

        # add metric trackers
        self.acc_tracker = tf.keras.metrics.Accuracy()
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        # initialize layers
        self.rnns: List[tf.keras.Layer] = []
        for _ in range(num_rnns):
            self.rnns.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(lstm_units, return_sequences=True),
                    merge_mode="sum"
                )
            )
        self.embedding = tf.keras.layers.Embedding(
            input_alphabet_size, embedding_dim
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.hidden_to_target = tf.keras.layers.Dense(target_alphabet_size)


    def make(
        self,
        batch_size=None,
    ):
        sentences = tf.keras.Input(
            shape=[None],
            batch_size=batch_size,
            dtype=tf.float32
        )
        sentence_lengths = tf.keras.Input(
            shape=[],
            batch_size=batch_size,
            dtype=tf.int32
        )
        self({
            'inputs': sentences,
            'input_lens': sentence_lengths
        })

    def compute_metrics(self, x, y, y_pred, sample_weight):
        labels_masked = tf.boolean_mask(y['labels'], y_pred['mask'])

        predicted_labels = tf.argmax(y_pred['logits'], axis=1)
        self.acc_tracker.update_state(labels_masked, predicted_labels)

        return {
            'loss': self.loss_tracker.result(),
            'accuracy': self.acc_tracker.result()
        }

    def compute_loss(self, x, y, y_pred, sample_weight):
        labels_masked = tf.boolean_mask(y['labels'], y_pred['mask'])
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_masked,
                logits=y_pred['logits']
            )
        )

        # add custom losses such as regularization losses
        # loss += tf.add_n(self.losses)

        # update the loss tracker to properly display the loss during training
        self.loss_tracker.update_state(loss)

        return loss

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.acc_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs: Dict[str, tf.Tensor]):
        sentences, sentence_length = inputs['inputs'], inputs['input_lens']

        outputs = self.dropout(self.embedding(sentences))

        for rnn in self.rnns:
            rnn_inputs = outputs
            outputs = rnn(rnn_inputs)
            if self.use_residual:
                outputs += rnn_inputs
            outputs = self.dropout(outputs)

        max_sentence_len_in_batch = tf.shape(sentences)[1]

        # the masked outputs will combine the batch dimension and the sequence
        # dimension; for example: a batch of B sentences will initially be of
        # size [B, MAX_SEQ_LEN, E] where E is the embedding size and MAX_SEQ_LEN
        # is the length of the largest sequence in the batch; let's take an
        # example of size [2, 3, 7] with sentences of lengths 2 and 3
        # respectively; if we were to compute the loss over this we would
        # include the elements at the [0, 2, :] coordinates which are outside
        # the length of the first sequence; in order to exclude that we build
        # the mask and the result is of size [2+3, 7] where the 2 and 3 from the
        # sum are the sequence lengths from the batch;
        mask = tf.sequence_mask(sentence_length, maxlen=max_sentence_len_in_batch)
        masked_outputs = tf.boolean_mask(outputs, mask)

        output_layer = self.hidden_to_target(masked_outputs)
        outputs_softmax = tf.nn.softmax(output_layer)

        predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32)

        return {
            'logits': output_layer,
            'mask': mask
        }
