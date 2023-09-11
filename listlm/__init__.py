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
        callback = None
    ):
        if revision is None:
            revision = "main"
            #revision = "gptq-8bit-128g-actorder_True"
        self.name = name
        self.revision = revision
        self.prompt_template = prompt_template
        self.model = None
        self.callback = callback

    class _StreamingOutput(StoppingCriteria):
        def __init__(self, prompt_length, tokenizer, callback):
            self.prompt_length = prompt_length
            self.text = ''
            self.tokenizer = tokenizer
            self.callback = callback
        def __call__(self, input_ids, scores, **kwparams):
            text_length = len(self.text)
            self.text = self.tokenizer.decode(input_ids[0,self.prompt_length:])
            self.callback(self.text[text_length:])
            return False

    def load(self, **kwparams):
        for key, val in kwparams.items():
            if val is not None and val != getattr(self, key, None):
                setattr(self, key, val)
                if key in ('name', 'revision'):
                    self.model = None
        if self.model is None:
            print(f'loading {self.name}@{self.revision} ...', file=sys.stderr)
            self.model = AutoModelForCausalLM.from_pretrained(self.name,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision=self.revision)
            self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)

    def forward(self, prompt, append = False, **kwparams):
        self.load(**kwparams)

        input_ids = self.tokenizer(self.prompt_template.format(prompt=prompt), return_tensors='pt').input_ids.to(self.model.device)

        if append:
            input_ids = torch.cat([self.last_output_ids, input_ids])

        criteria = []
        if self.callback:
            criteria.append(self._StreamingOutput(
                input_ids.shape[-1],
                self.tokenizer,
                self.callback,
            ))

        output = self.model.generate(inputs=input_ids, do_sample=False, max_new_tokens=512, stopping_criteria=criteria)
        self.last_output_ids = output[0]
        self.last_output = self.tokenizer.decode(output[0][...,input_ids.shape[-1]:])
        return self.last_output

    def forward_more(self, prompt, **kwparams):
        return self.forward(prompt, append=True, **kwparams)

    def output(self, prompt, **kwparams):
        print(prompt)
        result = forward(prompt, **kwparams)
        print(result)
        return result

if __name__ == '__main__':
    import sys
    prompt = sys.argv[1:].join(' ')
    print(Model().forward(prompt))
