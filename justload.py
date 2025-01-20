import wandb
import weave
wandb.login()

WANDB_PROJECT = "RAG_Streamlit"
run = wandb.init(
    project=WANDB_PROJECT,
    group="EiEi",
)
weave_client = weave.init(WANDB_PROJECT)
chunked_data = weave.ref("chunked_data:v1").get()
# print(chunked_data.rows[:2])


# print('already get chunked_data')
from script import TFIDFRetriever
retriever = TFIDFRetriever()
retriever.index_data(list(map(dict, chunked_data.rows)))
print('retrieved')

query = "how to express homer poetry ?"
search_results = retriever.search(query)
for result in search_results:
    print(result)
print('finish')