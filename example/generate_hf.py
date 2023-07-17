import torch
from peft import PeftModel
import transformers
import gradio as gr

from ringrwkv.configuration_rwkv_world import RwkvConfig
from ringrwkv.rwkv_tokenizer import TRIE_TOKENIZER
from ringrwkv.modehf_world import RwkvForCausalLM


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#放在本地工程根目录文件夹


model = RwkvForCausalLM.from_pretrained("RWKV-4-World-7B")
tokenizer = TRIE_TOKENIZER('./ringrwkv/rwkv_vocab_v20230424.txt')


#model= PeftModel.from_pretrained(model, "./lora-out")
model = model.to(device)


def evaluate(
    instruction,
    temperature=1,
    top_p=0.7,
    top_k = 0.1,
    penalty_alpha = 0.1,
    max_new_tokens=128,
):
    
    prompt = f'Question: {instruction.strip()}\n\nAnswer:'
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    #out = model.generate(input_ids=input_ids.to(device),max_new_tokens=40)
    out = model.generate(input_ids=input_ids.to(device),temperature=temperature,top_p=top_p,top_k=top_k,penalty_alpha=penalty_alpha,max_new_tokens=max_new_tokens)
    outlist = out[0].tolist()
    for i  in outlist:
        if i==0:
            outlist.remove(i)
    answer = tokenizer.decode(outlist)
    return answer.strip()
    #return answer.split("### Response:")[1].strip()


gr.Interface(
    fn=evaluate,#接口函数
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=2, value=1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.7, label="Top p"),
        gr.components.Slider(minimum=0, maximum=1, step=1, value=0.1, label="top_k"),
        gr.components.Slider(minimum=0, maximum=1, step=1, value=0.1, label="penalty_alpha"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="RWKV-World-Alpaca",
    description="RWKV,Easy In HF.",
).launch()