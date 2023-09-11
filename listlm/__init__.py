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
        callback = None,
        num_beams = 5,
    ):
        if revision is None:
            revision = "main"
            #revision = "gptq-8bit-128g-actorder_True"
        self.name = name
        self.revision = revision
        self.prompt_template = prompt_template
        self.model = None
        self.callback = callback
        self.num_beams = num_beams

    class _SC_StreamingOutput(StoppingCriteria):
        def __init__(self, prompt_length, tokenizer, callback):
            self.prompt_length = prompt_length
            self.text = None
            self.tokenizer = tokenizer
            self.callback = callback
        def __call__(self, input_ids, scores, **kwparams):
            if self.text is None:
                self.text = [''] * input_ids.shape[0]
            for batch_idx in range(input_ids.shape[0]):
                text = self.tokenizer.decode(input_ids[batch_idx][self.prompt_length:])
                self.callback(text[len(self.text[batch_idx]):])
                self.text[batch_idx] = text
            return False

    class _SC_FirstBeam(StoppingCriteria):
        def __init__(self, model, num_beams):
            self.model = model
            self.eos_token_id = model.config.eos_token_id
            self.vocab_size = model.config.vocab_size
            self.score_product = torch.ones([num_beams], dtype=torch.float64, device=self.model.device)
        def __call__(self, input_ids, scores, **kwparams):
            ended = (input_ids[...,-1] == self.eos_token_id)
            if scores is not None:
                self.score_product *= torch.stack(scores).max(dim=-1)
            if ended.any():
                ended = ended.nonzero()
                self.result_idx = ended[self.score_product[ended].argmax()][0]
                self.result = input_ids[self.result_idx]
                return True
            else:
                self.result_idx = self.score_product.argmax()
                self.result = input_ids[self.result_idx]
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
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)

    def forward(self, prompt, append = False, **kwparams):
        self.load(**kwparams)

        input_ids = self.tokenizer(self.prompt_template.format(prompt=prompt), return_tensors='pt').input_ids.to(self.model.device)

        if append:
            input_ids = torch.cat([self.last_output_ids, input_ids], dim=-1)

        shortest_result_finder = self._SC_FirstBeam(self.model, self.num_beams)

        criteria = [shortest_result_finder]
        if self.callback:
            criteria.append(self._SC_StreamingOutput(
                input_ids.shape[-1],
                self.tokenizer,
                self.callback,
            ))

        output_dict = self.model.generate(
            inputs=input_ids,
            num_beams=self.num_beams,
            top_p=0.5,
            do_sample=True,
            max_new_tokens=512,
            stopping_criteria=criteria,
            return_dict_in_generate=True, # needed for generating scores passed to stopping criteria
            output_scores=True,
        )
        del output_dict
        selected_output = shortest_result_finder.result
        self.last_output_ids = selected_output
        self.last_output = self.tokenizer.decode(self.last_output_ids[...,input_ids.shape[-1]:])
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
