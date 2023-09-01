# Kserve custom serving runtime using llama.cpp

* Get access to the llama2 models from Meta (https://ai.meta.com/llama/)
* Use llama.cpp (https://github.com/ggerganov/llama.cpp) to convert and quantize the models

```
    python3 ../convert.py ~/dev/llama/llama-2-7b/

    quantize ./ggml-model-f16.gguf ./ggml-model-q4_0.gguf q4_0    
```


## To run in podman

```
podman run --rm -p 9090:8080 -v ./dev/llama.cpp/models/:/mnt/models/:ro,z -e STORAGE_URI=pvc://ggml-model-q4_0.gguf -e MODEL_MNT=/mnt/models/ quay.io/noeloc/llama_serving
```