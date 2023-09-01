import os
from llama_cpp import Llama
from typing import Dict
from kserve import Model, ModelServer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)

STORAGE_URI:str = os.getenv("STORAGE_URI")
MODEL_MNT:str = os.getenv("MODEL_MNT")
CTX_SIZE:int = os.getenv("CTX_SIZE")

if CTX_SIZE is None:
    CTX_SIZE = 512

if STORAGE_URI is None:
    STORAGE_URI = ""


class LLamaModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        modelpath = STORAGE_URI
        if not modelpath.startswith("pvc://"):
            logger.error(
                "storageUri -> Only pvc models accepted atm eg. pvc://models/model.onnx"
            )
        else:
            modelpath=modelpath.replace("pvc://", MODEL_MNT)

        logging.info(f"Loading model {modelpath}")
        self.model = Llama(model_path=modelpath, n_ctx=CTX_SIZE)
        self.ready = True

    async def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        logging.info(f"predict request {headers} {payload}")

        prompt = payload["messages"]
        temp = payload.get("temperature", 0.8)
        max_tokens = payload.get("max_tokens", 128)
        rep_penalty = payload.get("repetition penalty", 1.1)
        top_k = payload.get("top_k", 40)
        top_p = payload.get("top_p", 0.95)

        reponse = self.model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=rep_penalty,
        )
        return reponse

if __name__ == "__main__":
    model = LLamaModel("llama-model")
    ModelServer().start([model])
