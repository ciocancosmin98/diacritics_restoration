from typing import Dict, List
import tensorflow as tf
from layers.conformer_encoder import ConformerEncoder


class AttentionModel(tf.keras.Model):

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
        super(AttentionModel, self).__init__()
        self.use_residual = use_residual

        # add metric trackers
        self.acc_tracker = tf.keras.metrics.Accuracy()
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        self.embedding = tf.keras.layers.Embedding(
            input_alphabet_size, embedding_dim
        )

        self.encoder = ConformerEncoder(
            positional_encoding='sinusoid',
            dmodel=embedding_dim,
            num_blocks=2,
            mha_type='relmha',
            head_size=36,
            num_heads=4,
            kernel_size=32,
            depth_multiplier=1,
            fc_factor=0.5,
            dropout=0
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
        outputs_masked = tf.boolean_mask(y_pred['logits'], y_pred['mask'])

        predicted_labels = tf.argmax(outputs_masked, axis=1)
        self.acc_tracker.update_state(labels_masked, predicted_labels)

        return {
            'loss': self.loss_tracker.result(),
            'accuracy': self.acc_tracker.result()
        }

    def compute_loss(self, x, y, y_pred, sample_weight):
        labels_masked = tf.boolean_mask(y['labels'], y_pred['mask'])
        outputs_masked = tf.boolean_mask(y_pred['logits'], y_pred['mask'])
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_masked,
                logits=outputs_masked
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
        sentences = inputs['inputs']
        sentence_length = inputs['input_lens']

        outputs = self.dropout(self.embedding(sentences))

        outputs = self.encoder(outputs)

        outputs = self.hidden_to_target(outputs)

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

        predictions = tf.cast(tf.argmax(outputs, 2), tf.int32)

        return {
            'predictions': predictions,
            'logits': outputs,
            'mask': mask
        }
