# RAGwthGemini
Implement RAG to Gemini API and use Streamlit to deploy website :
# All files
   1. Doc : The folder contain 30 Docs about arts and poetry that feeding in retrieve.py to being the database of this project
   2. Retrieve : This file transform all document in Doc into vector database
      1. Using Wandb for log the activity 
      2. Segment the data into 2 part 1.Content and 2.Metadata , The metadata contain lenght of file and relative of the path
      3. Separate the file into Chunk break it into small pieces 300 tokens make it able to feed into LLM
      4. Clean all chunked -like remove emoji and some special character
      5. Using Weave for storing vector data into vector database
   3. script.py : This file called from Retrieve.py doing the textprocessing and called from justloaad.py, test.py for using TFIDF retrieving function
   4. justload.py : This file contain code that using for test calling wandb api to retrieve document that related to query
   5. test.py : this is the streamlit web deployment that connect the user Gemini API .When user put the query/question the program will compute the relative document from wandb-weave and feed it to be data to Gemini and show the output throught the pages
   
# TechStack
- Python
- Wandb
- Weave
- TFIDF
- Streamlit
- Gemini Chat model

---
# Key take away
- implement RAG to LLM(Gemini) is more efficient to get the answer but it come at a cost that the time ,this process take 1-2 minute to retrieve all chunked and get the top 5 feed to LLM 
- another cost is LLM enable only around 520 token feeded so if 1 chunked contain 300 tokens that mean only 2 chunked maximum to feed to LLM
- Future feature ,I need to learn how to preload background