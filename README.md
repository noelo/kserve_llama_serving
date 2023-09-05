# Kserve custom serving runtime using llama.cpp & mlserver

* Get access to the llama2 models from Meta (https://ai.meta.com/llama/)
* Use llama.cpp (https://github.com/ggerganov/llama.cpp) to convert and quantize the models

```
    python3 ../convert.py ~/dev/llama/llama-2-7b/

    quantize ./ggml-model-f16.gguf ./ggml-model-q4_0.gguf q4_0    
```


## To run in podman

```
podman run -it --rm -p 8087:8087 -v ../llama.cpp/models/:/mnt/models/:ro,z -e MLSERVER_MODEL_URI=/mnt/models/ggml-model-q4_0.gguf quay.io/noeloc/llamacppserving

```

## Example Inference request

```python
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

endpoint = "http://your_hostname_port_here/v2/models/llama.cpp.llama2/infer"
response = requests.post(endpoint, json=inference_request.dict())

print(response.text)

```