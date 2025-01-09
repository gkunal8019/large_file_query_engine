
# Advanced Query System with VectorStore and Retrieval Mechanisms

## Overview

This project implements a query system that leverages advanced retrieval and vector search techniques to process and answer user queries. The system uses a combination of a **Milvus Vector Store**, **LLMs (Large Language Models)**, and **retriever techniques** to efficiently handle large-scale documents and provide accurate, contextually relevant responses. The system also uses a **hybrid search strategy** (combining vector-based and metadata-based search) to improve the quality of results.

## Key Components

### 1. **Settings**
The configuration settings for the system include the selection of models and context window sizes. The parameters include:
- **LLM Selection**: The system is set to use `nvllm` as the LLM.
- **Embedding Model**: The model used for embedding textual data is called `embedder`.
- **Context Window**: The context window is set to 4096 tokens.
- **Output Length**: The system is configured to generate a maximum of 1100 tokens per response.

```python
Settings.llm = nvllm
Settings.embed_model = embedder
Settings.context_window = 4096
Settings.num_output = 1100
```

### 2. **Vector Store Setup**
The vector store used for storing document vectors is **Milvus**, with a URI provided to connect to a Milvus instance. The dimensionality of the embeddings is 1024. The vector store is linked to the collection `"pdf_data"`.

```python
vector_store = MilvusVectorStore(
    uri="http:localhost:19530",
    dim=1024,
    collection_name=data_collection,
    overwrite=False
)
```

### 3. **Storage Context**
The storage context is initialized using the `vector_store` and provides access to documents within the vector store.

```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

### 4. **Index Setup**
An index is created from the vector store, allowing the retrieval of documents and nodes during query execution. This index is crucial for efficient document retrieval.

```python
INDEX = VectorStoreIndex.from_vector_store(vector_store=vector_store, store_nodes_override=False)
```

### 5. **Query with Retriever Function**
The `query_with_retriever` function processes user input and retrieves relevant documents from the vector store. It performs the following steps:
- **Base Retriever**: Sets up the base retriever with a similarity search to retrieve the top 12 documents.
- **Metadata Filters**: Filters are created to narrow down the search based on the metadata associated with the documents (source: `du_website`, type: `academic`, year: `2024`).
- **Node ID Selection**: Selects the first five node IDs from the document store to restrict the search space.
- **Retriever Setup**: A `VectorIndexRetriever` is configured with specific parameters like **similarity search**, **MMR (Maximum Marginal Relevance)**, **hybrid search**, etc.
- **Chat Engine**: A `ContextChatEngine` is created that uses the retriever to provide context and generate responses based on the user's query.

```python
def query_with_retriever(message, history):
    base_retriever = INDEX.as_retriever(similarity_top_k=12)
    
    metadata_filters = MetadataFilters(
        filters=[
            {"key": "source", "value": "du_website"},
            {"key": "type", "value": "academic"},
            {"key": "year", "value": "2024"}
        ]
    )
    
    available_node_ids = list(INDEX.docstore.docs.keys())[:5]
    
    retriever = VectorIndexRetriever(
        index=INDEX,
        similarity_top_k=12,
        vector_store_query_mode=VectorStoreQueryMode.MMR,
        alpha=0.5,
        node_ids=available_node_ids,
        hybrid_top_k=12,
        callback_manager=CallbackManager([]),
        embed_model=base_retriever._embed_model,
        verbose=True
    )

    chat_engine = ContextChatEngine(
        retriever=retriever,
        llm=nvllm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3900),
        prefix_messages=[],
        context_template=(
            "\ntell doccument about or context:\n"
            "{context_str}"
            "\nInstruction the data context above to answer the user's query. "
            "Provide a conclusion answer "
        ),
        callback_manager=CallbackManager([])
    )
    
    try:
        response = chat_engine.stream_chat(message)
        partial_response = ""
        for delta in response.response_gen:
            partial_response += delta
            yield partial_response
    except Exception as e:
        yield f"Error occurred: {str(e)}"
```

1. **`index=INDEX`**:
   - Specifies the vector store index from which documents will be retrieved. In this case, `INDEX` is an instance of `VectorStoreIndex` that has been previously defined.

2. **`similarity_top_k=12`**:
   - Determines the number of top-k most similar documents to retrieve based on the query. Setting this to 12 means the retriever will return the top 12 documents that are most similar to the query. 

3. **`vector_store_query_mode=VectorStoreQueryMode.MMR`**:
   - Specifies the query mode to use when interacting with the vector store. `VectorStoreQueryMode.MMR` indicates the use of Maximum Marginal Relevance (MMR) for retrieval, which balances relevance and diversity in the retrieved documents. 

4. **`alpha=0.5`**:
   - In the context of hybrid search, `alpha` is a parameter that balances the contribution of sparse (e.g., keyword-based) and dense (e.g., vector-based) retrieval methods. An `alpha` value of 0.5 means an equal weighting between the two methods. 

5. **`node_ids=available_node_ids`**:
   - Restricts the retrieval to specific nodes identified by their IDs. `available_node_ids` is a list of node IDs that have been predefined or selected for the retrieval process. This parameter helps in narrowing down the search space to relevant nodes. 

6. **`hybrid_top_k=12`**:
   - Specifies the number of top-k documents to retrieve when using hybrid search methods that combine multiple retrieval strategies. Setting this to 12 means the retriever will return the top 12 documents after applying the hybrid search strategy. 

7. **`callback_manager=CallbackManager([])`**:
   - Manages callbacks during the retrieval process. In this case, an empty list is passed, indicating no callbacks are registered. Callback managers are useful for monitoring and handling events during the retrieval process. 

8. **`embed_model=base_retriever._embed_model`**:
   - Specifies the embedding model to use for encoding the query and documents. `base_retriever._embed_model` refers to the embedding model associated with the base retriever, ensuring consistency in how queries and documents are represented in the vector space. 

9. **`verbose=True`**:
   - Enables verbose mode, which provides detailed logging and information during the retrieval process. Setting this to `True` allows for better monitoring and debugging of the retrieval operations. 




### 6. **Hybrid Search and Retrieval**
The retriever uses a **hybrid search** strategy combining both vector-based and metadata-based filtering. This increases the relevancy of the documents retrieved based on the user's query and metadata parameters.

### 7. **Chat Engine and Response Generation**
The `ContextChatEngine` is used to generate responses to the user's query based on the context provided by the retrieved documents. The response is generated progressively using a **streaming chat mechanism**, ensuring that the response is dynamically built as the context is processed.

### 8. **Error Handling**
If any errors occur during the retrieval or response generation process, they are caught, and an appropriate error message is returned.
