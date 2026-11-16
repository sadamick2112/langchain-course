import os

from dotenv import load_dotenv

# ServerlessSpec is used to tell where our Pinecone index is hosted
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":
    # create the Pinecone client using our API key
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # get our index name
    index_name = os.environ["INDEX_NAME"]

    if pc.has_index(index_name):
        print(f"Index '{index_name}' found, deleting...")
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        # dense vector is the embedding vector which is multidimensional where close vectors are semantically related
        # dense vectors are used to search by "meaning" and not necessarily "text" matches
        # sparse vector is mostly zeroes is used to represent documents or queries that captures keyword information. Each dimension in a sparse vector typically represents a word from a dictionary, the non-zero values represent the importance of these words in the document
        # spare vectors are mostly for "text" matches
        vector_type="dense",
        # index must match the embedding model we choose
        dimension=1536,  # OpenAIEmbeddings default (text-embedding-3-small)
        # Metric is how pinecone mesasures how similar the 2 vectors are.
        # The rule is to pick the metric your embedding model was built for. Open AI, recommends "cosine"
        # "cosine" uses the angle between the vectors (same direction means similar)
        # "euclidean" uses the distance between the 2 points (smaller means more similar)
        # "dotproduct" uses the length and angle (bigger means more similar)
        metric="cosine",
        # where the index is hosted in the cloud with a cloud type and region
        # Pinecone runs on 3 clouds (aws, gcp, azure). Enterprise will pick the cloud and region that fit them.
        # Once cloud and region are picked, they cannot be changed
        # The Pinecone starter uses aws, us-east-1
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created")
