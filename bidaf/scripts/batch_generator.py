from keras.utils import Sequence
import os
import numpy as np
from .magnitude import MagnitudeVectors
 

class BatchGenerator(Sequence):
    'Generates data for Keras'

    #vectors = None

    def __init__(self, gen_type, vectors, contexts, questions, answer_spans, batch_size, emdim, squad_version, max_passage_length, max_query_length, shuffle):
        'Initialization'

        base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

        self.vectors = vectors
        self.contexts = contexts
        self.questions = questions
        self.answer_spans = answer_spans
        self.squad_version = squad_version

        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        self.span_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.span'.format(squad_version))
        if self.squad_version == 2.0:
            self.is_impossible_file = os.path.join(base_dir, 'squad', gen_type +
                                                   '-v{}.is_impossible'.format(squad_version))

        self.batch_size = batch_size
        i = 0
        with open(self.span_file, 'r', encoding='utf-8') as f:

            for i, _ in enumerate(f):
                pass
        self.num_of_batches = (i + 1) // self.batch_size
        self.indices = np.arange(i + 1)
        self.shuffle = shuffle
        print('Init done...')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_of_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start_index = (index * self.batch_size) + 1
        end_index = ((index + 1) * self.batch_size) + 1

        inds = self.indices[start_index:end_index]

        context_data = []
        for i, line in enumerate(self.contexts, start=1):
            if i in inds:
                context_data.append(line.split(' '))


        question_data = []
        for i, line in enumerate(self.questions, start=1):
            if i in inds:
                question_data.append(line.split(' '))

        answer_data = []
        for i, line in enumerate(self.answer_spans, start=1):
            if i in inds:
                answer_data.append(line.split(' '))

        if self.squad_version == 2.0:
            is_impossible = []
            with open(self.is_impossible_file, 'r', encoding='utf-8') as isimpf:
                for i, line in enumerate(isimpf, start=1):
                    line = line[:-1]
                    if i in inds:
                        is_impossible.append(line)

            for i, flag in enumerate(is_impossible):
                context_data[i].insert(0, "unanswerable")
                if flag == "1":
                    answer_data[i] = [0, 0]
                else:
                    answer_data[i] = [int(val) + 1 for val in answer_data[i]]

        context_batch = self.vectors.query(context_data, pad_to_length=self.max_passage_length)
        question_batch = self.vectors.query(question_data, pad_to_length=self.max_query_length)
        print("Context batch size", context_batch.shape)
        print("Question batch size", question_batch.shape)
        if self.max_passage_length is not None:
            span_batch = np.expand_dims(np.array(answer_data, dtype='float32'), axis=1).clip(0,
                                                                                              self.max_passage_length - 1)
        else:
            span_batch = np.expand_dims(np.array(answer_data, dtype='float32'), axis=1)
        return [context_batch, question_batch], [span_batch]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

