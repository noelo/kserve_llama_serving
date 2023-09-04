import requests
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec
from mlserver.codecs.numpy import NumpyRequestCodec

explain_parameters={"model": "gpt-3.5-turbo","messages": "explain quantum theory in basic terms"}
x = np.array(explain_parameters)

inference_request = NumpyRequestCodec.encode_request(x)

endpoint = "http://localhost:8087/v2/models/custom_serving_llm/infer"
response = requests.post(endpoint, json=inference_request.dict())

response.json()