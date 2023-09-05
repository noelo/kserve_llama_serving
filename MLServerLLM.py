import os
from llama_cpp import Llama
from typing import List
import logging
from mlserver import MLModel, types
from mlserver.errors import InferenceError
from mlserver.utils import get_model_uri
from mlserver.codecs.numpy import NumpyRequestCodec
import numpy
import json

# foo = '{ "name":"John", "age":30, "city":"This is a test of how this all works?"}'
# # foo = ["bar", "bar2"]
# explain_parameters={"threshold": 0.95,"p_sample": 0.5,"tau": 0.25,"message":"this is a test message"}
# x = numpy.array(explain_parameters)
# from mlserver.codecs.string import StringRequestCodec
# from mlserver.codecs.numpy import NumpyRequestCodec
# inference_request = NumpyRequestCodec.encode_request(x)
# # print(inference_request)
# raw_request = inference_request.dict()
# # print(raw_request)
# z = inference_request.inputs[0].data.json()
# # print(z.dict())
# k = (json.loads(z))[0]
# # l = k[0]
# print(k["threshold"])
# inference_response = NumpyRequestCodec.encode_response(model_name="testmodel", payload=x)
# print(inference_response)
# from mlserver.codecs.numpy import NumpyRequestCodec
# import numpy
# thisdict = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 1964
# }
# print(NumpyRequestCodec.encode_response(model_name="testmodel", payload=numpy.array(thisdict)))


logger = logging.getLogger()
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)

STORAGE_URI: str = os.getenv("STORAGE_URI")
MODEL_MNT: str = os.getenv("MODEL_MNT")
CTX_SIZE: int = os.getenv("CTX_SIZE")

if CTX_SIZE is None:
    CTX_SIZE = 512

if STORAGE_URI is None:
    STORAGE_URI = ""


class LLama2Model(MLModel):
    async def load(self) -> bool:
        # get URI to model data
        model_uri = await get_model_uri(self._settings)

        # parse/process file and instantiate the model
        self._model = self._load_model_from_file(model_uri)

        # set ready to signal that model is loaded
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)
        output = self._predict_outputs(payload)

        logging.debug(f"Output {output}")
        return output

        # return types.InferenceResponse(
        #     model_name=self.name,
        #     model_version=self.version,
        #     outputs=fred,
        # )

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
            NumpyRequestCodec.encode_response(
                model_name=self.name, payload=numpy.array(model_reponse)
            )
        )
        logging.info(model_inference_response)
        return model_inference_response
