from ..scripts import BatchGenerator
import os
from .magnitude import MagnitudeVectors

def get_data(gen_type, emdim, squad_version=1.1, max_passage_length=None, max_query_length=None):

	vectors = MagnitudeVectors(emdim).load_vectors()

	base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
	context_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.context'.format(squad_version))
	question_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.question'.format(squad_version))
	span_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.span'.format(squad_version))
	if squad_version == 2.0:
		is_impossible_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.is_impossible'.format(squad_version))

	contexts = []
	with open(context_file, 'r', encoding='utf-8') as cf:
		for i, line in enumerate(cf):
			line = line[:-1]
			contexts.append(line)

	questions = []
	with open(question_file, 'r', encoding='utf-8') as qf:
		for i, line in enumerate(qf):
			line = line[:-1]
			questions.append(line)

	answer_spans = []
	with open(span_file, 'r', encoding='utf-8') as sf:
		for i,line in enumerate(sf):
			line = line[:-1]
			answer_spans.append(line)

	return vectors, contexts, questions, answer_spans

def get_answer_tokens(squad_version=1.1):
	base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
	answer_file = os.path.join(base_dir, 'squad', 'dev' + '-v{}.answer'.format(squad_version))
	answers = []
	with open(answer_file, 'r', encoding='utf-8') as af:
		for i, line in enumerate(af):
			line = line[:-1]
			answers.append(line)


	return answers
	
def load_data_generators(vectors, contexts, questions, answer_spans, batch_size, emdim, squad_version=1.1, 
						max_passage_length=None, max_query_length=None, shuffle=False):

	train_generator = BatchGenerator('train', vectors, contexts, questions, answer_spans, batch_size, emdim, 
									squad_version, max_passage_length, max_query_length, shuffle)

	validation_generator = BatchGenerator('dev', vectors, contexts, questions, answer_spans, batch_size, 
										  emdim, squad_version, max_passage_length, max_query_length,
										  shuffle)
	return train_generator, validation_generator

