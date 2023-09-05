import requests
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec
from mlserver.codecs.numpy import NumpyRequestCodec

explain_parameters = {
    "model": "llama.cpp.llama2",
    "messages": "explain quantum theory in basic terms",
    "max_tokens" : 190,
}
x = np.array(explain_parameters)

inference_request = NumpyRequestCodec.encode_request(x)

endpoint = "http://localhost:8087/v2/models/llama.cpp.llama2/infer"
response = requests.post(endpoint, json=inference_request.dict())

print(response.text)
