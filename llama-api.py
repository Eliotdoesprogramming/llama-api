import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

# initialize model & stuff
generation_conf = GenerationConfig.from_pretrained("meta-llama/Llama-2-13b-chat-hf", max_new_tokens=60, max_time=600.0)

model_id = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})



class Query(BaseModel):
    query: str


@app.post("/query")
def query(query:Query):
    query = query.query
    tokens = torch.tensor(tokenizer(query)['input_ids'])
    if torch.cuda.is_available():
        tokens.to('cuda')
    tokens = model.generate(input_ids=tokens.unsqueeze(dim=0), generation_config=generation_conf)

    resp = tokenizer.decode(tokens.squeeze())

    print(resp)
    return resp
