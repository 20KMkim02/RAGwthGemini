#Using wandb to log the question

import wandb
import weave
# Using display source เพื่อใช้ในการดูparameterในการรับค่าของโค๊ดต่างๆ
# from scripts.uwtils import display_source

import nltk
nltk.download("wordnet")

# Wandb API
wandb.login()
WANDB_PROJECT = "RAG_Streamlit"

run = wandb.init(
    project=WANDB_PROJECT,
    group="EiEi",
)

# Pathlib ช่วยในการจัดการpathให้ง่ายขึ้น
import pathlib

data_dir = "Doc"

# เก็บlist ของ file path ลงใน docs_files
docs_dir = pathlib.Path(data_dir)
docs_files = sorted(docs_dir.rglob("*.txt"))

# print(f"Number of files: {len(docs_files)}\n")
# print("First 5 files:\n{files}".format(files="\n".join(map(str, docs_files[:5]))))

# print(docs_files[0].read_text())

# ทำตัวretrieve โดยการจัดFormat ของ file ให้มีmeta data
data = []
for file in docs_files:
    content = file.read_text()
    data.append(
        {
            "content": content,
            "metadata": {
                "source": str(file.relative_to(docs_dir)),
                "raw_tokens": len(content.split()),
            },
        }
    )

total_tokens = sum(map(lambda x: x["metadata"]["raw_tokens"], data))
print(f"Total Tokens in dataset: {total_tokens}")

weave_client = weave.init(WANDB_PROJECT)

# build weave dataset
raw_data = weave.Dataset(name="raw_data", rows=data)

# publish the dataset
weave.publish(raw_data)

# Slicing data
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0

from typing import List
def split_into_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Function to split the text into chunks of a maximum number of tokens
    ensure that the chunks are of size CHUNK_SIZE and overlap by chunk_overlap tokens
    use the `tokenizer.encode` method to tokenize the text
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        start = end - chunk_overlap
    return chunks

# download the `raw_data` Dataset
raw_data = weave.ref("raw_data:v0").get()

# print(raw_data.rows[:2])

chunked_data = []
for doc in raw_data.rows:
    chunks = split_into_chunks(doc["content"])
    for chunk in chunks:
        chunked_data.append(
            {
                "content": chunk,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "raw_tokens": len(chunk.split()),
                },
            }
        )

from script import make_text_tokenization_safe
cleaned_data = []
for doc in chunked_data:
    cleaned_doc = doc.copy()
    cleaned_doc["cleaned_content"] = make_text_tokenization_safe(doc["content"])
    cleaned_doc["metadata"]["cleaned_tokens"] = len(
        cleaned_doc["cleaned_content"].split()
    )
    cleaned_data.append(cleaned_doc)
# print(chunked_data[:2])

dataset = weave.Dataset(name="chunked_data", rows=cleaned_data)
weave.publish(dataset)

# Doing vectorize
chunked_data = weave.ref("chunked_data:v1").get()
print(chunked_data.rows[:2])
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