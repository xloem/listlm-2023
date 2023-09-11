import sys
print('importing ...', file=sys.stderr)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Model:
    def __init__(
        self,
        name = "TheBloke/CodeLlama-13B-Instruct-GPTQ",
        revision = None,
        prompt_template='''[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]

''',
    ):
        if revision is None:
            #revision = "main"
            revision = "gptq-8bit-128g-actorder_True"
        self.name = name
        self.revision = revision
        self.prompt_template = prompt_template
        self.model = None

    def forward(self, prompt, **kwparams):
        for key, val in kwparams.items():
            if val is not None and val != getattr(self, key, None):
                setattr(self, key, val)
                if key in ('name', 'revision'):
                    self.model = None
        if self.model is None:
            print('loading ...', file=sys.stderr)
            self.model = AutoModelForCausalLM.from_pretrained(self.name,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision=self.revision)
            self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        input_ids = self.tokenizer(self.prompt_template.format(prompt=prompt), return_tensors='pt').input_ids.cuda()
        output = self.model.generate(inputs=input_ids, temperature=None, do_sample=False, max_new_tokens=512)
        return self.tokenizer.decode(output[0][input_ids.shape[0]:])

    def output(self, prompt, **kwparams):
        print(prompt)
        result = forward(prompt, **kwparams)
        print(result)
        return result

model = Model()

if __name__ == '__main__':
    import sys
    prompt = sys.argv[1:].join(' ')
    print(model.forward(prompt))
