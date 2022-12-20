# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import sys
from types import FrameType

from flask import Flask, request

from collections import Counter
import re
from math import sqrt
from google.cloud import storage

from utils.logging import logger

app = Flask(__name__)


@app.route("/")
def hello() -> str:
    import time
    start = time.time()
    #request_json = request.get_json()
    #text1 = request_json['text1']
    #text2 = request_json['text2']
    text1 = ""
    text2 = ""
    try:
        text1 = str(request.args.get('text1'))
        text2 = str(request.args.get('text2'))
    except Exception as e:
        return e.message + + "text1= " + str(text1)
    
    similarity = compute_similarity(text1, text2)
    end = time.time()
    time_result = end-start

        # Get the request parameters
    string1 = request.args.get('text1')
    string2 = request.args.get('text2')
    
    # Create a client for interacting with Google Cloud Storage
    storage_client = storage.Client()
    
    # Get the bucket where the files will be stored
    bucket = storage_client.bucket("storage_time_and_result")
    
    # Write string1 to a file called "file1.txt"
    file1 = bucket.blob("function_time.txt")
    file1.upload_from_string(str(time_result))
    
    # Write string2 to a file called "file2.txt"
    file2 = bucket.blob("function_result.txt")
    file2.upload_from_string(str(similarity))

    return "function result:" + str(similarity) +  " with time: " + str(end-start)

def preprocess_text(text):
    # Remove punctuation and make all characters lowercase
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def compute_similarity(text1, text2):
    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Tokenize the texts
    tokens1 = text1.split()
    tokens2 = text2.split()

    # Compute the frequency of each word in each text
    frequency1 = Counter(tokens1)
    frequency2 = Counter(tokens2)

    # Compute the dot product of the frequency vectors
    dot_product = sum(frequency1[token] *
frequency2[token] for token in frequency1)

    # Compute the Euclidean norms of the frequency vectors
    norm1 = sqrt(sum(frequency1[token] ** 2 for token in frequency1))
    norm2 = sqrt(sum(frequency2[token] ** 2 for token in frequency2))

    # Return the cosine similarity
    return dot_product / (norm1 * norm2)


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
