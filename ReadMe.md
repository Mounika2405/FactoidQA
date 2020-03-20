# Factoid Question Answering System 

The idea of a Factoid Question Answering System(FQAS) is to answer the user’s questions by finding/extracting the
relevant information from the documents, online searches, conversations, etc. FQAS are useful and handy when
short and concise answers are preferred rather than searching through a list of documents, especially on mobile or using a digital assistant device, like Alexa, Google Assistant etc.

The goal of our project is to build a Factoid QA system that mainly consists of questions such as who, what, when and where, such that the answer is a single entity, object or time. For example, questions such as ”What is the population of India?”, ”Who is the President of the US?”, and so on are factoid questions.

### Dataset Used

The models are trained on Stanford Question Answering Dataset (SQuAD). For more details regarding the dataset, visit https://rajpurkar.github.io/SQuAD-explorer/


### Contributions

Machine Reading is a task in which a model reads a piece of text and attempts to formally represent it or performs a downstream task like Question Answering (QA). However, many real-world question answering problems require the
reading of text not because it contains the literal answer, but because it contains a recipe to derive an answer together with the reader’s background knowledge / world knowledge. For example, given the context, ”The top of Mount Fuji is covered with snow”, and the question, ”What does the top of Mount Fuji have?”, the answer is ”snow”. But if new questions related to the same context like ”What color does the top of Mount Fuji have?” or ”What is the temperature on top of Mount Fuji?”, the models fails to answer such questions because it is not directly present in the data. Answering such questions requires world knowledge or commonsense knowledge.

To address this issue, in our work, we propose to use ConceptNet as an external commonsense knowledge base to improve the reasoning capabilities of the QA system and capture the semantic relationships between the concepts. 

### Architecture
![alt text](https://github.com/Mounika2405/FactoidQA/blob/master/demo/QAArchitecture-Final.png)


### Demo

![alt text](https://github.com/Mounika2405/FactoidQA/blob/master/demo/img1.png)
![alt text](https://github.com/Mounika2405/FactoidQA/blob/master/demo/img2.png)


### References

[1] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ questions for machine comprehension of text”. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016.

[2] P. Rajpurkar, R. Jia, and P. Liang, “Know what you don’t know: Unanswerable questions for SQuAD”. In Proceedings Association for Computational Linguistics, 2018.[3] Tom Young, Erik Cambria, Iti Chaturvedi, Hao Zhou, Subham Biswas, Minline Huang, “Augmenting End-to-End Dialogue Systems With Commonsense Knowledge”. In AAAI, 2018. 

[3] Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi and Hannaneh Hajishirzi. “Bidirectional Attention Flow for Machine Comprehension.” In ICLR 2017.

[4] Robyn Speer, Joshua Chin, and Catherine Havasi. “ConceptNet 5.5: an open multilingual graph of general knowledge”. In AAAI 2017.
