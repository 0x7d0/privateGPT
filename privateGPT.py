from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os

load_dotenv()

# Environment Variables
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY')
MODEL_TYPE = os.getenv('MODEL_TYPE')
MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_N_CTX = os.getenv('MODEL_N_CTX')
MODEL_N_BATCH = int(os.getenv('MODEL_N_BATCH', 8))
TARGET_SOURCE_CHUNKS = int(os.getenv('TARGET_SOURCE_CHUNKS', 4))
# Ensure required environment variables are set
REQUIRED_ENV_VARS = ['EMBEDDINGS_MODEL_NAME', 'PERSIST_DIRECTORY', 'MODEL_TYPE', 'MODEL_PATH', 'MODEL_N_CTX']
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise Exception(f"Missing required environment variable {var}")

app = Flask(__name__)
CORS(app)
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

callbacks = [StreamingStdOutCallbackHandler()]

# Prepare the LLM
if MODEL_TYPE == "LlamaCpp":
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=MODEL_N_CTX, n_batch=MODEL_N_BATCH, callbacks=callbacks, verbose=False)
elif MODEL_TYPE == "GPT4All":
    llm = GPT4All(model=MODEL_PATH, n_ctx=MODEL_N_CTX, backend='gptj', n_batch=MODEL_N_BATCH, callbacks=callbacks, verbose=False)
else:
    raise Exception(f"Model {MODEL_TYPE} not supported!")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.route('/ask', methods=['POST'])
@limiter.limit("60/minute")  # Limit to 60 requests per minute
def ask():
    query = request.json.get('query')
    if not query:
        abort(400, description="Missing 'query' in request data")
    try:
        res = qa(query)
        answer = res['result']
        return jsonify({'answer': answer})
    except Exception as e:
        abort(500, description=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
