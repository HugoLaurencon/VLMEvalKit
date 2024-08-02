import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from transformers import AutoTokenizer
from m4.models.vllama3.modeling_vllama3 import VLlama3ForCausalLM
from m4.training.utils import build_image_transform, VisionEncoderTypes
from m4.training.packing import get_splitted_images_and_corresponding_text
import torch
from PIL import Image
import os
import math


class IDEFICS(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics-9b-instruct', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='auto'
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.file_root = osp.dirname(__file__)
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def generate_inner(self, message, dataset=None):
        prompts = (
            ['Users:']
            + [msg['value'] if msg['type'] == 'text' else Image.open(msg['value']) for msg in message]
            + ['<end_of_utterance>', '\nAssistant: ']
        )
        inputs = self.processor(
            prompts, add_end_of_utterance_token=False, return_tensors='pt'
        ).to('cuda')
        exit_condition = self.processor.tokenizer(
            '<end_of_utterance>', add_special_tokens=False
        ).input_ids
        bad_words_ids = self.processor.tokenizer(
            ['<image>', '<fake_token_around_image>'], add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            **self.kwargs,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        text = generated_text[0].split('\nAssistant: ')[-1]
        return text


class IDEFICS2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics2-8b', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation='flash_attention_2',
            device_map='cuda',
        )
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )
        torch.cuda.empty_cache()

    def _process(self, formatted_messages, formatted_images):
        inputs = self.processor(
            text=formatted_messages, images=formatted_images, return_tensors='pt'
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value'].strip()
        if add_brief:
            prompt += '\nGive a very brief answer.'
        if add_yes_or_no:
            prompt += '\nAnswer yes or no.'
        prompt += '<end_of_utterance>\nAssistant:'
        return prompt, images

    def build_prompt_puremcq(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mt(self, message):
        prompt, images = '', []
        for msg in message:
            if msg['role'] == 'user':
                prompt += 'User: '
            elif msg['role'] == 'assistant':
                prompt += 'Assistant: '
            for item in msg['content']:
                if item['type'] == 'image':
                    img = load_image(item['value'])
                    images.append(img)
                    prompt += '<image>'
                elif item['type'] == 'text':
                    prompt += item['value'].strip()
                prompt += '<end_of_utterance>\n'
        return prompt + 'Assistant: '

    def build_prompt_mmbench(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if 'Hint:' in instruction:
                    hint, question = instruction.split('\nQuestion:')
                    question, choices = question.split('\nChoices:')
                    instruction = (
                        'Question:' + question + '\n' + hint + '\nChoices:' + choices
                    )
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mmmu(self, message):
        replace_mapping = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }

        prompt, images, img_counter = 'User: Question: ', [], 1
        for msg in message:
            if msg['type'] == 'image':
                prompt += f'<image {img_counter}>:<image>\n'
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += f' <image {img_counter}> '
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def build_prompt_mathvista(self, message):
        replace_mapping = {
            '(A) ': 'A. ',
            '(B) ': 'B. ',
            '(C) ': 'C. ',
            '(D) ': 'D. ',
            '(E) ': 'E. ',
            '(F) ': 'F. ',
            '(G) ': 'G. ',
            '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        # print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response.replace("\n", "\\n"))
        return response

    def generate_inner(self, message, dataset=None):
        if dataset in [
            'MMBench_DEV_EN',
            'MMBench_TEST_EN',
            'MMBench_DEV_CN',
            'MMBench_TEST_CN',
            'MMBench',
            'MMBench_CN',
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in [
            'MME',
            'MMVet',
            'OCRVQA_TEST',
            'OCRVQA_TESTCORE',
            'TextVQA_VAL',
            'ChartQA_TEST',
            'DocVQA_VAL',
            'DocVQA_TEST',
            'InfoVQA_VAL',
            'InfoVQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == 'HallusionBench':
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            'MMStar',
            'SEEDBench_IMG',
            'AI2D_TEST',
            'ScienceQA_VAL',
            'ScienceQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        # print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response.replace("\n", "\\n"))
        return response


class IDEFICS2Large(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics2-70b', **kwargs):
        assert model_path is not None
        self.model_path = model_path

        #self.processor = AutoProcessor.from_pretrained(model_path)
        #self.model = AutoModelForVision2Seq.from_pretrained(
        #    model_path,
        #    torch_dtype=torch.bfloat16,
        #    _attn_implementation='flash_attention_2',
        #    device_map='cuda',
        #)

        def load_model_and_tokenizer(path_model):
            # Load tokenizer
            path_tokenizer = path_model  # os.path.join(path_model, "tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(path_tokenizer, truncation_side="left", use_fast=True)
            tokenizer.padding_side = "left"
            # Load model
            path_unwrapped_model = path_model  # os.path.join(path_model, "unwrapped_model")
            # model = VLlama3ForCausalLM.from_pretrained(path_unwrapped_model, torch_dtype=torch.bfloat16, device_map="auto")  # DEBUG
            model = VLlama3ForCausalLM.from_pretrained(path_unwrapped_model, torch_dtype=torch.bfloat16)
            model.to("cuda:0")  # DEBUG
            # Eval mode
            model.eval()
            return tokenizer, model
        
        def load_model_and_tokenizer_2(path_model):
            # Load tokenizer
            path_tokenizer = path_model  # os.path.join(path_model, "tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(path_tokenizer, truncation_side="left", use_fast=True)
            tokenizer.padding_side = "left"
            # Load model
            path_unwrapped_model = path_model  # os.path.join(path_model, "unwrapped_model")
            model = VLlama3ForCausalLM.from_pretrained(path_unwrapped_model, torch_dtype=torch.bfloat16, device_map="auto")  # DEBUG
            #model = VLlama3ForCausalLM.from_pretrained(path_unwrapped_model, torch_dtype=torch.bfloat16)
            #model.to("cuda:0")  # DEBUG
            # Eval mode
            model.eval()
            return tokenizer, model

        PATH_MODEL = "/fsx/hugo/Idefics3-Llama3-70B/"
        self.tokenizer, self.model = load_model_and_tokenizer_2(path_model=PATH_MODEL)

        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )
        torch.cuda.empty_cache()

    def upscale_image(self, image, res_image_side):
        width, height = image.size
        aspect_ratio = width / height

        if width >= height:
            width = res_image_side
            height = int(width / aspect_ratio)
            #if height % 2 != 0:
            #    height += 1
        elif height > width:
            height = res_image_side
            width = int(height * aspect_ratio)
            #if width % 2 != 0:
            #    width += 1

        image = image.resize((width, height), Image.LANCZOS)
        return image

    def _process(self, formatted_messages, formatted_images):
        inputs = self.processor(
            text=formatted_messages, images=formatted_images, return_tensors='pt'
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                img = self.upscale_image(image=img, res_image_side=1960)
                splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                    image=img,
                    vision_encoder_max_image_size=980,
                    max_image_size=1960,
                    pre_split_scale_up_max=None,
                    pre_split_scale_up_frequency=None,
                    image_seq_len=256,
                )
                images.extend(splitted_images_array)
                prompt += text_splitted_images
                #images.append(img)
                #prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value'].strip()
        if add_brief:
            prompt += '\nGive a very brief answer.'
        if add_yes_or_no:
            prompt += '\nAnswer yes or no.'
        prompt += '<end_of_utterance>\nAssistant:'
        return prompt, images

    def build_prompt_puremcq(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                img = self.upscale_image(image=img, res_image_side=1960)
                splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                    image=img,
                    vision_encoder_max_image_size=980,
                    max_image_size=1960,
                    pre_split_scale_up_max=None,
                    pre_split_scale_up_frequency=None,
                    image_seq_len=256,
                )
                images.extend(splitted_images_array)
                prompt += text_splitted_images
                #images.append(img)
                #prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mt(self, message):
        prompt, images = '', []
        for msg in message:
            if msg['role'] == 'user':
                prompt += 'User: '
            elif msg['role'] == 'assistant':
                prompt += 'Assistant: '
            for item in msg['content']:
                if item['type'] == 'image':
                    img = load_image(item['value'])
                    img = self.upscale_image(image=img, res_image_side=1960)
                    splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                        image=img,
                        vision_encoder_max_image_size=980,
                        max_image_size=1960,
                        pre_split_scale_up_max=None,
                        pre_split_scale_up_frequency=None,
                        image_seq_len=256,
                    )
                    images.extend(splitted_images_array)
                    prompt += text_splitted_images
                    #images.append(img)
                    #prompt += '<image>'
                elif item['type'] == 'text':
                    prompt += item['value'].strip()
                prompt += '<end_of_utterance>\n'
        return prompt + 'Assistant: '

    def build_prompt_mmbench(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                img = self.upscale_image(image=img, res_image_side=1960)
                splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                    image=img,
                    vision_encoder_max_image_size=980,
                    max_image_size=1960,
                    pre_split_scale_up_max=None,
                    pre_split_scale_up_frequency=None,
                    image_seq_len=256,
                )
                images.extend(splitted_images_array)
                prompt += text_splitted_images
                #images.append(img)
                #prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if 'Hint:' in instruction:
                    hint, question = instruction.split('\nQuestion:')
                    question, choices = question.split('\nChoices:')
                    instruction = (
                        'Question:' + question + '\n' + hint + '\nChoices:' + choices
                    )
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mmmu(self, message):
        replace_mapping = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }

        prompt, images, img_counter = 'User: Question: ', [], 1
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                img = self.upscale_image(image=img, res_image_side=1960)
                splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                    image=img,
                    vision_encoder_max_image_size=980,
                    max_image_size=1960,
                    pre_split_scale_up_max=None,
                    pre_split_scale_up_frequency=None,
                    image_seq_len=256,
                )
                images.extend(splitted_images_array)
                prompt += f'<image {img_counter}>:{text_splitted_images}\n'
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg['type'] == 'image':
                #img = load_image(msg['value'])
                #images.append(img)
                prompt += f' <image {img_counter}> '
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()

        # CoT
        #prompt = prompt.replace("Answer with the letter.", "").strip()
        #if 'A.' in prompt and 'B.' in prompt:
        #    prompt += '\Think step by step and explain your reasoning. Then, answer with the letter."<end_of_utterance>\nAssistant: Let\' think step by step.'
        #else:
        #    prompt += '\Think step by step and explain your reasoning. Then, answer with the requested format."<end_of_utterance>\nAssistant: Let\' think step by step.'

        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'

        return prompt, images

    def build_prompt_mathvista(self, message):
        replace_mapping = {
            '(A) ': 'A. ',
            '(B) ': 'B. ',
            '(C) ': 'C. ',
            '(D) ': 'D. ',
            '(E) ': 'E. ',
            '(F) ': 'F. ',
            '(G) ': 'G. ',
            '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                img = self.upscale_image(image=img, res_image_side=1960)
                splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                    image=img,
                    vision_encoder_max_image_size=980,
                    max_image_size=1960,
                    pre_split_scale_up_max=None,
                    pre_split_scale_up_frequency=None,
                    image_seq_len=256,
                )
                images.extend(splitted_images_array)
                prompt += text_splitted_images
                #images.append(img)
                #prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        # CoT
        #if 'A.' in prompt and 'B.' in prompt:
        #    prompt += '\Think step by step and explain your reasoning. Then, answer with the letter."<end_of_utterance>\nAssistant: Let\' think step by step.'
        #else:
        #    prompt += '\Think step by step and explain your reasoning. Then, answer with the requested format."<end_of_utterance>\nAssistant: Let\' think step by step.'

        #if 'A.' in prompt and 'B.' in prompt:
        #    prompt += '<end_of_utterance>\nAssistant: Answer:'
        #else:
        #    prompt += '\Think step by step and explain your reasoning. Then, answer with the requested format."<end_of_utterance>\nAssistant: Let\'s think step by step.'


        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        # print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response.replace("\n", "\\n"))
        return response

    def generate_inner(self, message, dataset=None):
        if dataset in [
            'MMBench_DEV_EN',
            'MMBench_TEST_EN',
            'MMBench_DEV_CN',
            'MMBench_TEST_CN',
            'MMBench',
            'MMBench_CN',
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in [
            'MME',
            'MMVet',
            'OCRVQA_TEST',
            'OCRVQA_TESTCORE',
            'TextVQA_VAL',
            'ChartQA_TEST',
            'DocVQA_VAL',
            'DocVQA_TEST',
            'InfoVQA_VAL',
            'InfoVQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == 'HallusionBench':
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            'MMStar',
            'SEEDBench_IMG',
            'AI2D_TEST',
            'ScienceQA_VAL',
            'ScienceQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        #inputs = self._process(formatted_messages, formatted_images)

        #generated_ids = self.model.generate(**inputs, **self.kwargs)
        #generated_text = self.processor.batch_decode(
        #    generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        #)[0]

        # Beginning of my modifications

        formatted_messages = "<|begin_of_text|>" + formatted_messages

        image_transform = build_image_transform(
            max_image_size=980,
            image_size=None,
            eval=True,
            vision_encoder_type=VisionEncoderTypes.siglip,
        )
        #pixel_values = [torch.stack([image_transform(img) for img in formatted_images])]
        #pixel_values = torch.stack(pixel_values).to(self.model.device)

        # Start modification for the padded images

        pixel_values = []
        pixel_attention_masks = []

        pv = [image_transform(img) for img in formatted_images]

        num_images = len(pv)
        max_height = max([im.size(1) for im in pv])
        max_width = max([im.size(2) for im in pv])
        padded_image_tensor = torch.zeros(num_images, 3, max_height, max_width)
        padded_pixel_attention_masks = torch.zeros(num_images, max_height, max_width, dtype=torch.bool)

        for idx, im in enumerate(pv):
            im_height, im_width = im.size(1), im.size(2)
            padded_image_tensor[idx, :, :im_height, :im_width] = im
            padded_pixel_attention_masks[idx, :im_height, :im_width] = True

        pixel_values.append(padded_image_tensor)
        pixel_attention_masks.append(padded_pixel_attention_masks)

        total_batch_size = len(pixel_values)
        max_num_images = max([i.size(0) for i in pixel_values])
        max_height = max([i.size(2) for i in pixel_values])
        max_width = max([i.size(3) for i in pixel_values])
        pixel_values = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
        pixel_attention_mask = torch.zeros(total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool)
        for idx, (sample_images, sample_pixel_attention_mask) in enumerate(
            zip(pixel_values, pixel_attention_mask)
        ):
            im_batch_height, im_batch_width = sample_images.size()[2:]
            pixel_values[idx, : sample_images.shape[0], :, :im_batch_height, :im_batch_width] = sample_images
            pixel_attention_mask[idx, : sample_pixel_attention_mask.shape[0], :im_batch_height, :im_batch_width] = (
                sample_pixel_attention_mask
            )
        pixel_values = pixel_values.to(self.model.device)
        pixel_attention_mask = pixel_attention_mask.to(self.model.device)

        # End modification for the padded images

        tokens = self.tokenizer(
            [formatted_messages],
            return_tensors="pt",
            truncation=True,
            padding=True,
            add_special_tokens=False,
        )
        input_ids = torch.stack([tokens.input_ids[0]]).to(self.model.device)
        attention_mask = torch.stack([tokens.attention_mask[0]]).to(self.model.device)

        generated_tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            num_beams=1,
            max_new_tokens=512,
            bad_words_ids=self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False)["input_ids"],
            use_cache=True,
        )
        generated_tokens = generated_tokens[:, input_ids.shape[1] :]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # or skip_special_tokens=False, don't know
        generated_text = generated_texts[0]

        # End of my modifications

        response = generated_text.strip()
        # print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response.replace("\n", "\\n"))
        return response
