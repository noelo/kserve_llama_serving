import os
import numpy
import json
import logging
import mlserver
from llama_cpp import Llama
from typing import List
from mlserver import MLModel, types
from mlserver.codecs import NumpyCodec
from mlserver.errors import InferenceError
from mlserver.utils import get_model_uri

logger = logging.getLogger("LLama2Model")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)

CTX_SIZE: int = os.getenv("CTX_SIZE")

if CTX_SIZE is None:
    CTX_SIZE = 512


class LLama2Model(MLModel):
    async def load(self) -> bool:
        # get URI to model data
        model_uri = await get_model_uri(self._settings)

        # parse/process file and instantiate the model
        self._model = self._load_model_from_file(model_uri)

        mlserver.register("max_tokens_metric", "Max Tokens per request")

        # set ready to signal that model is loaded
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)
        output = self._predict_outputs(payload)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=output,
        )

    def _load_model_from_file(self, file_uri):
        logging.info(f"Loading model {file_uri}")
        return Llama(model_path=file_uri, n_ctx=CTX_SIZE)

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        try:
            # parse and check for errors
            (json.loads(payload.inputs[0].data.json()))[0]
        except Exception as exc:
            raise InferenceError("Invalid request payload ") from exc
        return payload

    def _predict_outputs(
        self, payload: types.InferenceRequest
    ) -> List[types.ResponseOutput]:
        logging.info(f"predict request ==> {payload}")

        json_request = (json.loads(payload.inputs[0].data.json()))[0]

        prompt = json_request["messages"]
        temp = json_request.get("temperature", 0.8)
        max_tokens = json_request.get("max_tokens", 128)
        rep_penalty = json_request.get("repetition penalty", 1.1)
        top_k = json_request.get("top_k", 40)
        top_p = json_request.get("top_p", 0.95)

        mlserver.log(max_tokens_metric=max_tokens)

        model_reponse = self._model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=rep_penalty,
        )

        model_inference_response = []
        model_inference_response.append(
            NumpyCodec.encode_output(self.name, numpy.array(model_reponse))
        )

        logging.info(model_inference_response)
        return model_inference_response
