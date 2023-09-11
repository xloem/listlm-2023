import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/CodeLlama-13B-Instruct-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

#prompt = "Tell me about AI"
prompt_template='''[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]

'''

print("\n\n*** Generate:")

def forward(prompt):
    input_ids = tokenizer(prompt_template.format(prompt=prompt), return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0, max_new_tokens=512)
    return tokenizer.decode(output[0])

if __name__ == '__main__':
    import sys
    prompt = sys.argv[1:].join(' ')
    print(forward(prompt))
