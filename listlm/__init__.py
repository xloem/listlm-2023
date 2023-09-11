import sys
print('importing ...', file=sys.stderr)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

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

    class _StreamingOutput(StoppingCriteria):
        def __init__(self, prompt_length, callback):
            self.length = prompt_length
            self.callback = callback
        def __call__(self, input_ids, scores, **kwparams):
            self.callback(input_ids[...,self.length:])
            self.length = input_ids.shape[-1]
            return False

    def forward(self, prompt, callback = None, **kwparams):
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

        criteria = []
        if callback:
            criteria.append(self._StreamingOutput(
                input_ids.shape[-1],
                lambda input_ids: callback(self.tokenizer.decode(input_ids[0]))
            ))

        output = self.model.generate(inputs=input_ids, do_sample=False, max_new_tokens=512, stopping_criteria=criteria)
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
