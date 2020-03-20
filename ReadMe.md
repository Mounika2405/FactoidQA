# Factoid Question Answering System 

The idea of a Factoid Question Answering System(FQAS) is to answer the user’s questions by finding/extracting the
relevant information from the documents, online searches, conversations, etc. FQAS are useful and handy when
short and concise answers are preferred rather than searching through a list of documents, especially on mobile or using a digital assistant device, like Alexa, Google Assistant etc.

The goal of our project is to build a Factoid QA system that mainly consists of questions such as who, what, when and where, such that the answer is a single entity, object or time. For example, questions such as ”What is the population of India?”, ”Who is the President of the US?”, and so on are factoid questions.

## Dataset Used

The models are trained on Stanford Question Answering Dataset (SQuAD). For more details regarding the dataset, visit https://rajpurkar.github.io/SQuAD-explorer/


## Contributions

Machine Reading is a task in which a model reads a piece of text and attempts to formally represent it or performs a downstream task like Question Answering (QA). However, many real-world question answering problems require the
reading of text not because it contains the literal answer, but because it contains a recipe to derive an answer together with the reader’s background knowledge / world knowledge. For example, given the context, ”The top of Mount Fuji is covered with snow”, and the question, ”What does the top of Mount Fuji have?”, the answer is ”snow”. But if new questions related to the same context like ”What color does the top of Mount Fuji have?” or ”What is the temperature on top of Mount Fuji?”, the models fails to answer such questions because it is not directly present in the data. Answering such questions requires world knowledge or commonsense knowledge.

To address this issue, in our work, we propose to use ConceptNet as an external commonsense knowledge base to improve the reasoning capabilities of the QA system and capture the semantic relationships between the concepts. 


## Demo

![alt text](http://url/to/img.png)
