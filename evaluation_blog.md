# Evaluating RAG

When working with RAG applications, it can often be difficult to understand the quality of the application you have just built or where and how to improve results if they're not up to par. Likely, you have used it a fair bit and have observed that it provides reasonable results to the questions you have provided, but you don't have time to read through every piece of context returned for every answer, and you may not yet know how users will interact with the chat application. This can make going to production with confidence difficult because you don't know how your system is performing beyond the anecdotal evidence from your dev and QA teams. Moreover, it can be really difficult to sift through the signal and noise on LinkedIn and Twitter, where it seems a new revolutionary technique is introduced every 30 minutes. Establishing a baseline of tangible metrics to be improved over time provides essential insight into which techniques suit your application and gives you confidence going to production.

In this blog, we will cover how to measure and establish such a baseline to begin to better understand app performance. Specifically, we will walk through how to implement the popular and pragmatic RAGAs framework for a basic RAG app. In addition we will get into the specifics of how the metrics are calculated and what they do and do not tell us.

# The code

Let's say you've built a simple RAG app using LangChain, Redis, and OpenAI to answer questions about financial documents. In this example, we will use Nike's 2023 10-K document, but feel free to tailor it to your use case.

## Split and load the doc:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader

source_doc = "resources/nike-10k-2023.pdf"

loader = UnstructuredFileLoader(
    source_doc, mode="single", strategy="fast"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500, chunk_overlap=0
)

chunks = loader.load_and_split(text_splitter)
```

## Create vector embeddings for the chunks and populate and store in Redis as the vector store
```python
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis as LangChainRedis

# define embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# set the index name for this example
index_name = "ragas_ex"

# with langchain we can manually modify the default vector schema configuration
vector_schema = {
    "name": "chunk_vector",        # name of the vector field in langchain
    "algorithm": "HNSW",           # could use HNSW instead
    "dims": 384,                   # set based on the HF model embedding dimension
    "distance_metric": "COSINE",   # could use EUCLIDEAN or IP
    "datatype": "FLOAT32",
}

# here we can define the entire schema spec for our index in LangChain
index_schema = {
    "vector": [vector_schema],
    "text": [{"name": "source"}, {"name": "content"}],
    "content_vector_key": "chunk_vector"    # name of the vector field in langchain
}


# construct the vector store class from texts and metadata
rds = LangChainRedis.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    redis_url=REDIS_URL,
    index_schema=index_schema,
)

```

## Define a RetrievalQA chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=rds.as_retriever(
        search_type="similarity_distance_threshold",
        search_kwargs={"distance_threshold":0.5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": get_prompt()},
)

```

## Test it out

```python
query = "What was nike's revenue last year?"
res=qa.invoke(query)
```

Output
```json
{
    'query': "What was nike's revenue last year?",
    'result': " Nike's revenue for the last fiscal year (2022) was $46,710 million.",
    'source_documents': [
        Document(page_content='As discussed in Note 15 — Operating Segments and Related Information in the accompanying Notes to the Consolidated Financial Statements, our operating segments are evidence of the structure of the Company\'s internal organization. The NIKE Brand segments are defined by geographic regions for operations participating in NIKE Brand sales activity.\n\nThe breakdown of Revenues is as follows:\n\n(Dollars in millions)\n\nFISCAL 2023 FISCAL 2022\n\n% CHANGE\n\n% CHANGE EXCLUDING CURRENCY (1) CHANGES FISCAL 2021\n\n% CHANGE\n\nNorth America Europe, Middle East & Africa Greater China\n\n$\n\n21,608 $ 13,418 7,248\n\n18,353 12,479 7,547\n\n18 % 8 % -4 %\n\n18 % $ 21 % 4 %\n\n17,179 11,456 8,290\n\n7 % 9 % -9 %\n\nAsia Pacific & Latin America Global Brand Divisions\n\n(3)\n\n(2)\n\n6,431 58\n\n5,955 102\n\n8 % -43 %\n\n17 % -43 %\n\n5,343 25\n\n11 % 308 %\n\nTOTAL NIKE BRAND Converse\n\n$\n\n48,763 $ 2,427\n\n44,436 2,346\n\n10 % 3 %\n\n16 % $ 8 %\n\n42,293 2,205\n\n5 % 6 %\n\n(4)\n\nCorporate TOTAL NIKE, INC. REVENUES\n\n$\n\n27\n\n51,217 $\n\n(72) 46,710\n\n— 10 %\n\n— 16 % $\n\n40 44,538\n\n— 5 %\n\n(1) The percent change excluding currency changes represents a non-GAAP financial measure. For further information, see "Use of Non-GAAP Financial Measures".\n\n(2) For additional information on the transition of our NIKE Brand businesses within our CASA territory to a third-party distributor, see Note 18 — Acquisitions and Divestitures of the Notes to Consolidated\n\nFinancial Statements contained in Item 8 of this Annual Report.\n\n(3) Global Brand Divisions revenues include NIKE Brand licensing and other miscellaneous revenues that are not part of a geographic operating segment.\n\n(4) Corporate revenues primarily consist of foreign currency hedge gains and losses related to revenues generated by entities within the NIKE Brand geographic operating segments and Converse, but\n\nmanaged through our central foreign exchange risk management program.\n\nThe primary financial measure used by the Company to evaluate performance is Earnings Before Interest and Taxes ("EBIT"). As discussed in Note 15 — Operating Segments and Related Information in the accompanying Notes to the Consolidated Financial Statements, certain corporate costs are not included in EBIT.\n\nThe breakdown of EBIT is as follows:\n\n(Dollars in millions)\n\nFISCAL 2023\n\nFISCAL 2022\n\n% CHANGE\n\nFISCAL 2021\n\nNorth America Europe, Middle East & Africa Greater China\n\n$\n\n5,454 3,531 2,283\n\n$\n\n5,114 3,293 2,365\n\n7 % $ 7 % -3 %\n\n5,089 2,435 3,243\n\nAsia Pacific & Latin America Global Brand Divisions (1)', metadata={'id': 'doc:ragas_ex:747825ff51c742f39aa569bce001ad5f', 'source': 'resources/nike-10k-2023.pdf'}),
        ... <other docs>
        ]
}
```

## Viola we have a RAG app - Let's start evaluating.

The RAGAs framework consists of four primary metrics: **faithfulness**, **answer relevancy**, **context precision**, and **context recall**. Context precision and recall quantify the performance of retrieval from the vector store, while faithfulness and answer relevance quantify how well the system performed in generating results based on the context. These metrics work in tandem to provide a full picture of how the app is performing.

In order to calculate these metrics, we need to collect four pieces of information from our RAG interactions: the question that was asked, the answer that was generated, the context that was provided to the LLM to generate the answer, and, depending on which metrics you are interested in, a ground truth answer determined either by a critic LLM or a human-in-the-loop process.

Let's test this out for the question: `Where is Nike headquartered and when was it founded?` with ground truth `Nike is headquartered Beaverton, Oregon and was founded in 1964.`

## Execute test in code

```python
# helper function to convert the output of our RAG app to an eval friendly version
def parse_res(res, ground_truth=""):
    return {
        "question": [res["query"]],
        "answer": [res["result"]],
        "contexts": [[doc.page_content for doc in res["source_documents"]]],
        "ground_truth": [ground_truth]
    }

# invoke the RAG app to generate a result and parse
question = "Where is Nike headquartered and when was it founded?"
res = qa.invoke(question)
parsed_res = parse_res(res, ground_truth="Nike is headquartered Beaverton, Oregon and was founded in 1964.")

# utilize the ragas python library to import the desired metrics and evaluation function to execute
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset

ds = Dataset.from_dict(parsed_res)

# generate the result and store as a pandas dataframe for easy viewing
eval_results = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
eval_df = eval_results.to_pandas()
eval_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
```

## Results of our test
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.5</td>
      <td>0.963133</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

## What do these values mean?

Let's start with the metrics that look good. Answer relevancy is calculated under the hood by asking an LLM to generate hypothetical questions for the given answer and taking the average cosine similarity between those questions. A high score here illustrates to us that there weren't too many other ways to arrive at this answer as a proxy for how directly *relevant* the answer was to the specific question asked. Intuitively, this makes sense since it's fairly obvious what the likely question was for the answer "Nike is headquartered in Beaverton, Oregon and was founded in 1967."

We also observed that the context precision for our question/answer pair was 1.0. Context precision quantifies how *good* the returned context was and is defined as:

$$
Context\ precision = \frac{True\ Positives}{True\ Positives\ + False\ Positives}
$$

A true positive is defined as a document that is relevant and was returned in the result set and a false positive is a document that was not relevant and returned in the result set. In this case, the evaluation determined that all the docs returned were relevant to the ground truth provided. This is a positive indication but does require a bit of faith in the LLM's ability to determine what is relevant, which is a topic unto itself. I recommend reading the [full paper](https://arxiv.org/pdf/2309.15217) for those interested in gaining more insight on this front.

Faithfulness is defined:

$$
Faithfullness\ = \frac{Number\ of\ claims\ in\ the\ generated\ answer\ that\ can\ be\ inferred\ from\ the\ given\ context}{Total\ number\ of\ claim\ in\ the\ generated\ answer}
$$
In this case, there are two claims that can be determined from the answer: "Nike is headquartered in Beaverton, Oregon and was founded in 1967."

1. Nike is headquartered in Beaverton, Oregon.
2. Nike was founded in 1967.

From the context provided, there is no mention of Nike being located in Beaverton, Oregon; therefore, the claim cannot be inferred from the given context. The claim that Nike was founded in 1967, however, can be inferred from the context since the doc specifically mentions Nike being incorporated in 1967. This result highlights an important point about faithfulness as a metric. **Faithfulness does not measure accuracy**. What's interesting about this example is the claim that could **not** have been inferred from context (Nike is located in Beaverton) is factually correct. However, the claim that Nike was founded in 1967 is incorrect but can be inferred from the context. **Faithfulness measures how true to the text an answer was. It does not tell us if the answer was correct**.

Accuracy can be understood from context recall and is defined as:

$$
Context\ recall = \frac{Ground\ Truth\ sentences\ that\ can\ be\ attributed\ to\ context}{Total\ number\ of\ sentences\ in\ the\ ground\ truth}
$$

The ground truth we provided for this example was `Nike is headquartered in Beaverton, Oregon and was founded in 1964` which could be broken down into two sentences (or claims):
1. Nike is headquartered in Beaverton.
2. Nike was founded in 1964.

Neither of these claim can be inferred *correctly* from the context; therefore, context recall = 0/2 or 0.

The first example question provided here is intentionally general and meant to bring up an important point about RAG: RAG is an architecture designed to answer **specific** questions about a context. It is not necessarily ideal for answering **general** questions—that is what an LLM is for. The question `Where is Nike located and when was it founded?` is a general knowledge question that isn't specific to the 10-K document we loaded into our context. When designing a test and educating users about how to best interact with a RAG app, it's important to emphasize what type of questions are meant to be answered by the chat app. This is also why an agent layer can be so essential to a good chat experience because general questions should be handled by a general language model, while specific contextual questions should be handled by RAG, and a layer to determine the difference can greatly improve performance.


## Let's ask a different question

```python
question = "What is NIKE's policy regarding securities analysts and their reports?"
res = qa.invoke(question)

parsed = parse_res(res, ground_truth="NIKE's policy is to not disclose any material non-public information or other confidential commercial information to securities analysts. NIKE also does not confirm financial forecasts or projections issued by others. Therefore, shareholders should not assume that NIKE agrees with any statement or report issued by any analyst, regardless of the content.")

ds = Dataset.from_dict(parsed)

eval_results = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
eval_df = eval_results.to_pandas()
```

## Results
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.0</td>
      <td>0.946619</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

## Analysis

For this test, we can see that our RAGAs scores were very good. This is largely because the question is well-suited for our example RAG application.

- It asks a question that is directly related to the context.
- It uses specific terms that make matching in the vector space more likely.
- The ground truth is similar to the doc content.

With RAG, the question format really matters in the same way that formatting and using the right terms in a Google search really matters. Because we are using math to process natural language, we have to be mindful of interacting with the system in a way that lends itself to that paradigm. Coincidentally, this is why query rewriting in your applications can be really powerful, because you are performing conversions that, while obvious to humans, would not be obvious to a machine and can greatly improve performance.

# Creating a test dataset

Now that we have an understanding of the metrics in play and a better idea of what they tell us about our app, the next question becomes: how do we go about creating a dataset to test our specific apps? This is one of the more powerful pieces of the [RAGAs library](https://docs.ragas.io/en/latest/getstarted/testset_generation.html). RAGAs is designed to be 'reference-free' and provides a helper class for auto-generating a test set. The second example question was generated this way. It is worth noting that generating a synthetic dataset is not a replacement for collecting appropriate user data or labeling your own set of test questions with ground truth; however, it is a very effective baseline for getting an initial sense of app performance when a polished test set is not yet available or feasible. In the initial paper proposing RAGAs, a pairwise comparison between human annotators and the RAGAs approach found that the two were in agreement 95%, 78%, and 70% of the time, respectively, for faithfulness, answer relevance, and contextual relevance ([pg 5](https://arxiv.org/pdf/2309.15217)). Note: this was research done on the WikiEval dataset, which is probably one of the easier datasets for LLMs; however, it should add some confidence in RAGAs as a very good first pass.


The code to create an initial dataset would look like:
```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.core import SimpleDirectoryReader

generator_llm = # your choice of llm
critic_llm = # your choice of llm usually a bigger model for critic
embeddings = # your choice of embedding model

generator = TestsetGenerator.from_llama_index(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

reader = SimpleDirectoryReader(input_files=[SOURCE_DOC])

documents = reader.load_data()

testset = generator.generate_with_llamaindex_docs(
    documents,
    test_size=20,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
```

# Review

In this blog we have covered:

- The importance of establishing a metrics based baseline
- How to get started with RAGAs
- What the key metrics of RAGAs do and do not tell us
- Design considerations around when a RAG architecture is appropriate
- How to generate a test set to get started

For a full RAGAs example plus more AI recipes from the team at Redis check out our [ai resources repo](https://github.com/redis-developer/redis-ai-resources).