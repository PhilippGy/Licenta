A fully local processing pipeline consisting of automated web crawling of official admissions information, 
data preprocessing and structuring, chunking and embedding generation, retrieval with subsequent reranking, 
and response generation using locally deployed Large Language Models. Subsequently, an LLM-as-a-Judge will 
evaluate the quality and consistency of the generated responses.

To run the models, use LM Studio (or similar desktop apps) and load the models on the local server.
As of now, the models present in the current arhitecture are Gemma 4 E4B (chatbot) and
Gemma 2 9B (judge).

To view the models in action, refer to Current Arhitecture/main.ipynb.
