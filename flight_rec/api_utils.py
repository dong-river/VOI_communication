import time
import re
import json
import os
import requests
from typing import Dict, List, Tuple, Optional, Any, Union
from openai import OpenAI
from openai import AzureOpenAI

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# transformers / torch are only needed for local HuggingFace inference. Import
# them lazily so this module loads instantly for the API-based experiments
# (a top-level `import torch` can take 30-60s due to CUDA initialization).
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


def _load_transformers():
    """Import transformers + torch on first use. Returns True if available."""
    global torch, AutoModelForCausalLM, AutoTokenizer
    if torch is not None:
        return True
    try:
        from transformers import AutoModelForCausalLM as _AM, AutoTokenizer as _AT
        import torch as _torch
        torch, AutoModelForCausalLM, AutoTokenizer = _torch, _AM, _AT
        return True
    except ImportError:
        return False

# Global variables to store loaded models and tokenizers
_LOADED_MODELS = {}
_LOADED_TOKENIZERS = {}

# torch.backends.cuda.enable_flash_sdp(False)        # ❌ disable Flash Attention
# torch.backends.cuda.enable_mem_efficient_sdp(False) # ❌ disable Memory-Efficient Attention (optional)
# torch.backends.cuda.enable_math_sdp(True)    

# API keys are read from the environment only -- never hardcoded.
#   export OPENAI_API_KEY=sk-...        # provider="openai"
#   export OPENROUTER_API_KEY=sk-or-... # provider="openrouter"
openai_api_key = os.getenv("OPENAI_API_KEY")
open_router_api_key = os.getenv("OPENROUTER_API_KEY")


def _require_key(value, env_name):
    if not value:
        raise RuntimeError(
            f"{env_name} is not set. Export it before running, e.g.\n"
            f"  export {env_name}=..."
        )
    return value

# Clients are created lazily so importing this module never touches the network
# (Vertex/Together credential resolution can otherwise hang at import time).
_CLIENTS = {}


def _client(name, factory):
    if name not in _CLIENTS:
        _CLIENTS[name] = factory()
    return _CLIENTS[name]


def get_openai_client():
    return _client(
        "openai",
        lambda: OpenAI(api_key=_require_key(openai_api_key, "OPENAI_API_KEY")),
    )


def get_openrouter_client():
    return _client(
        "openrouter",
        lambda: OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_require_key(open_router_api_key, "OPENROUTER_API_KEY"),
        ),
    )


def get_azure_client():
    return _client(
        "azure",
        lambda: AzureOpenAI(
            azure_endpoint=_require_key(os.getenv("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT"),
            api_key=_require_key(os.getenv("AZURE_OPENAI_API_KEY"), "AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        ),
    )


def get_together_client():
    from together import Together
    _require_key(os.getenv("TOGETHER_API_KEY"), "TOGETHER_API_KEY")
    return _client("together", lambda: Together())


def get_google_client():
    from google import genai
    return _client(
        "google",
        lambda: genai.Client(
            vertexai=True,
            project=_require_key(os.getenv("GOOGLE_CLOUD_PROJECT"), "GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        ),
    )


def get_gemini_gen(
    prompt,
    system_prompt="",
    model="gemini-2.5-flash",
    max_tokens=2048,
    temperature=0.7,
    stop_strs=None,
    max_depth=3,
    cur_depth=0,
):
    """
    Generate a response from Google's Gemini via Vertex AI.
    Parity with get_openai_gen:
      - Supports a system prompt
      - Supports multi-turn lists [user0, assistant0, user1, assistant1, ...]
      - Non-streaming single response
    """
    try:
        from google.genai import types
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."

        # Build contents from a single prompt or an alternating list
        if isinstance(prompt, list):
            contents = []
            for i, p in enumerate(prompt):
                role = "user" if i % 2 == 0 else "model"  # OpenAI "assistant" -> Gemini "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=str(p))]))
        else:
            contents = [types.Content(role="user", parts=[types.Part(text=str(prompt))])]
        
        if system_prompt:
            contents.insert(0, types.Content(role="user", parts=[types.Part(text=system_prompt)]))


        # Build config (non-streaming). Expose stop sequences for parity with OpenAI's 'stop'
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=max_tokens,
            stop_sequences=stop_strs or [],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = get_google_client().models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        # Non-streaming: response.text is the concatenated text for most SDK versions
        return (getattr(response, "text", None) or "").strip()

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
        return get_gemini_gen(
            prompt, system_prompt, model, max_tokens, temperature, stop_strs, max_depth, cur_depth + 1
        )

## Get logit distribution from OpenAI
def get_openai_logit(message: list, model="gpt-4", temperature=0, max_tokens=15):
    response = get_openai_client().chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5
        )
    tokens = [response.choices[0].logprobs.content[4].top_logprobs[i].token for i in range(4)]
    probs = [math.exp(response.choices[0].logprobs.content[4].top_logprobs[i].logprob) for i in range(4)]
    
    return tokens, probs, response.choices[0].message.content


def get_open_router_gen(prompt, system_prompt="", model='gpt-3.5-turbo', max_tokens=2048, 
                  temperature=0.7, stop_strs=None, max_depth=3, cur_depth=0):
    """Generate a response from OpenAI's API with retry logic"""
    try:
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."
        
        if isinstance(prompt, list):
            # In this case, make sure the prompt list is in the correct order: user, assistant, user, assistant, ...
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} 
                       for idx, p in enumerate(prompt)]
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        response = get_openrouter_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_strs,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5
        )
        
        res = response.choices[0].message.content.strip()
        print(res)
        return res
    
    except Exception as e:
        print(f"Error: {e}")
        print("Retrying in 30 seconds...")
        time.sleep(30)
        return get_openai_gen(prompt, system_prompt, model, max_tokens, 
                             temperature, stop_strs, max_depth, cur_depth + 1)

def _chat_create_with_token_compat(api_client, *, use_logprobs: bool = False, **kwargs):
    """
    Call client.chat.completions.create with max_tokens, falling back to
    max_completion_tokens if the model rejects max_tokens (e.g. o1/o3/gpt-5 series).
    """
    try:
        extra = {"logprobs": True, "top_logprobs": 5} if use_logprobs else {}
        return api_client.chat.completions.create(**kwargs, **extra)
    except Exception as e:
        err_str = str(e)
        if "max_tokens" in err_str and "max_completion_tokens" in err_str:
            max_tokens = kwargs.pop("max_tokens")
            extra = {"logprobs": True, "top_logprobs": 5} if use_logprobs else {}
            return api_client.chat.completions.create(
                max_completion_tokens=max_tokens, **kwargs, **extra
            )
        raise


def get_azure_gen(prompt, system_prompt="", model='gpt-3.5-turbo', max_tokens=2048, 
                  temperature=0.7, stop_strs=None, max_depth=3, cur_depth=0):
    """Generate a response from Azure's API with retry logic"""
    try:
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."
        
        if isinstance(prompt, list):
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} 
                       for idx, p in enumerate(prompt)]
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        response = _chat_create_with_token_compat(
            get_azure_client(),
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_strs,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
        return get_azure_gen(prompt, system_prompt, model, max_tokens, 
                             temperature, stop_strs, max_depth, cur_depth + 1)
    
def get_openai_gen(prompt, system_prompt="", model='gpt-3.5-turbo', max_tokens=2048, 
                  temperature=0.7, stop_strs=None, max_depth=3, cur_depth=0):
    """Generate a response from OpenAI's API with retry logic"""
    try:
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."
        
        if isinstance(prompt, list):
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} 
                       for idx, p in enumerate(prompt)]
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        response = _chat_create_with_token_compat(
            get_openai_client(),
            use_logprobs=True,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_strs,
            temperature=temperature,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
        return get_openai_gen(prompt, system_prompt, model, max_tokens, 
                             temperature, stop_strs, max_depth, cur_depth + 1)


# def load_llama_model(model_name):
#     """Load and cache a Llama model and tokenizer"""
#     if not TRANSFORMERS_AVAILABLE:
#         raise ImportError("transformers package not installed. Please install with: pip install transformers torch")
    
#     global _LOADED_MODELS, _LOADED_TOKENIZERS
    
#     # Check if model is already loaded
#     if model_name in _LOADED_MODELS and model_name in _LOADED_TOKENIZERS:
#         return _LOADED_TOKENIZERS[model_name], _LOADED_MODELS[model_name]
    
#     # Load model and tokenizer
#     print(f"Loading model and tokenizer for {model_name}...")
#     hf_token = os.getenv("HF_TOKEN")
#     tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#     model_instance = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         torch_dtype=torch.float16, 
#         device_map="auto",
#         # load_in_8bit=True,
#         token=hf_token
#     )
    
#     # Cache the loaded model and tokenizer
#     _LOADED_TOKENIZERS[model_name] = tokenizer
#     _LOADED_MODELS[model_name] = model_instance
    
#     return tokenizer, model_instance

def load_llama_model(model_name_or_path):
    """Load and cache a Llama model and tokenizer from either HuggingFace or local path"""
    if not _load_transformers():
        raise ImportError("transformers package not installed. Please install with: pip install transformers torch")
    
    global _LOADED_MODELS, _LOADED_TOKENIZERS
    
    # Check if model is already loaded
    if model_name_or_path in _LOADED_MODELS and model_name_or_path in _LOADED_TOKENIZERS:
        print(f"Using cached model and tokenizer for {model_name_or_path}")
        return _LOADED_TOKENIZERS[model_name_or_path], _LOADED_MODELS[model_name_or_path]
    
    # Check if it's a local path or a HuggingFace model
    is_local_path = os.path.exists(model_name_or_path)
    print(f"Loading model and tokenizer from {'local path' if is_local_path else 'HuggingFace'}: {model_name_or_path}")
    
    # Configure loading parameters
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        # "device_map": "auto",
        "device_map": {"": 0}
        # "load_in_8bit": True,  # Uncomment if needed
    }
    
    # Add HF token only if loading from HuggingFace (not for local paths)
    if not is_local_path:
        hf_token = os.getenv("HF_TOKEN")
        load_kwargs["token"] = hf_token
        tokenizer_kwargs = {"token": hf_token}
    else:
        tokenizer_kwargs = {}
    
    # Check if this is a LoRA adapter path
    is_lora = False
    if is_local_path:
        # Check for typical LoRA adapter files
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            is_lora = True
            print(f"Detected LoRA adapter at {model_name_or_path}, loading local saved LoRA model.")
    
    # Load tokenizer first
    try:
        # For LoRA, we need the base model's tokenizer
        if is_lora:
            # Try to load adapter config to get base model name
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path")
            if base_model_name:
                print(f"Loading tokenizer from base model: {base_model_name}")
                # Use base model tokenizer for LoRA
                if not is_local_path:  # If base model is on HF
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **tokenizer_kwargs)
                else:  # If base model is local
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                # If base model not specified in config, try using the adapter path
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        else:
            # Regular model (not LoRA)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
    
    # Load model
    try:
        if is_lora:
            from peft import PeftModel, PeftConfig
            
            # Get config to determine base model
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                **load_kwargs,
                use_flash_attention_2=False
            )
            
            # Load LoRA adapter onto the base model
            model_instance = PeftModel.from_pretrained(
                base_model,
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            # Load regular model (non-LoRA)
            model_instance = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **load_kwargs
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Cache the loaded model and tokenizer
    _LOADED_TOKENIZERS[model_name_or_path] = tokenizer
    _LOADED_MODELS[model_name_or_path] = model_instance
    
    return tokenizer, model_instance

def clean_decode(prompt_ids, generated_ids, tokenizer):
    # Decode once for reliability; strip special tokens but keep BOS for matching.
    bos = tokenizer.bos_token or ""
    prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True).lstrip()
    full_text  = tokenizer.decode(generated_ids[0], skip_special_tokens=True).lstrip()

    print('utils/clearn_decode:', full_text)
    # 1️⃣ If BOS appears, drop it.
    if bos and full_text.startswith(bos):
        full_text = full_text[len(bos):].lstrip()

    # 2️⃣ If the (possibly trimmed) prompt shows up at the very start, drop it.
    if full_text.startswith(prompt_text):
        full_text = full_text[len(prompt_text):].lstrip()

    return full_text

def get_llama_response(prompt, system_prompt="", model='meta-llama/Llama-2-7b-chat-hf', max_tokens=2048,
                      temperature=0.7, stop_strs=None, max_retries=3):
    """Generate a response from Llama model using HuggingFace Transformers"""
    if not _load_transformers():
        return "Error: transformers package not installed. Please install with: pip install transformers torch"
    
    # Try to get a response with retries
    for attempt in range(max_retries):
        try:
            tokenizer, model_instance = load_llama_model(model)
            
            model_instance.config.use_cache = False
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print("get_llama_response:", full_prompt)
            
            # Tokenize the input
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model_instance.device)
            inputs = {k: v.contiguous() for k, v in inputs.items()}
            
            # Configure generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_p": 0.9,
            }
            
            if stop_strs:
                gen_kwargs["stopping_criteria"] = tokenizer.convert_tokens_to_ids(stop_strs)
                            
            # Generate the response
            with torch.no_grad():
                outputs = model_instance.generate(**inputs, **gen_kwargs)
            
            # Decode the response and remove the prompt
            
            # response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
            response = clean_decode(inputs['input_ids'], outputs, tokenizer)
            
            if '!' in response:
                response = response.split('!')[1].strip()
            if ":" in response and "{" not in response:
                response = response.split(":")[1].strip()
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)  # Wait before retrying
    
    return "Sorry, I was unable to generate a response."


def unload_llama_models():
    """Unload all cached models to free memory"""
    global _LOADED_MODELS, _LOADED_TOKENIZERS
    
    if torch is None:
        return
    
    for model_name in list(_LOADED_MODELS.keys()):
        del _LOADED_MODELS[model_name]
        del _LOADED_TOKENIZERS[model_name]
    
    _LOADED_MODELS = {}
    _LOADED_TOKENIZERS = {}
    
    # Clean up CUDA memory
    torch.cuda.empty_cache()
    print("All models unloaded and memory cleared.")

import math
def get_openai_logit(message: list, model="gpt-4", temperature=0, max_tokens=15):
    response = get_openai_client().chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5
        )
    res_text = response.choices[0].message.content

    guess_idx = -1
    token_logits_list = response.choices[0].logprobs.content
    for idx, token_logits in enumerate(token_logits_list):
        if 'guess' in token_logits.top_logprobs[0].token:
            guess_idx = idx
            break
    tokens = [response.choices[0].logprobs.content[guess_idx+3].top_logprobs[i].token for i in range(4)]
    probs = [math.exp(response.choices[0].logprobs.content[guess_idx+3].top_logprobs[i].logprob) for i in range(4)]
    
    if guess_idx == -1:
        return res_text, None
    prob_dict = {}
    for i, token in enumerate(tokens):
        prob_dict[token] = probs[i]
    return res_text, prob_dict

def get_together_ai_response(prompt, system_prompt="", model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', max_tokens=2048, temperature=0.7, stop_strs=None, cur_depth=0, max_depth=3):
    print(system_prompt)
    if cur_depth >= max_depth:
        return "Error: Max retries exceeded. Unable to generate a response using together ai."
    try:
        if isinstance(prompt, list):
            # In this case, make sure the prompt list is in the correct order: user, assistant, user, assistant, ...
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} 
                       for idx, p in enumerate(prompt)]
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        stream = get_together_client().chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs
        )
        # Process the streamed response
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        return response.strip()
    except Exception as e:
        print(f"Error: {e}")
        print("Retrying in 30 seconds...")
        time.sleep(30)
        return get_together_ai_response(prompt, system_prompt, model, max_tokens, temperature, stop_strs, cur_depth + 1, max_depth)
    
def get_llm_response(prompt, system_prompt="", model='gpt-3.5-turbo', max_tokens=2048,
                    temperature=0.7, stop_strs=None, provider="openai", verbose=False):
    """
    Get a response from a language model based on provider.
    
    Args:
        prompt: The user prompt (str or list of strings for conversation)
        system_prompt: The system prompt (instructions for the model)
        model: The model name to use
        max_tokens: Maximum number of tokens in the response
        temperature: Temperature for generation
        stop_strs: Strings that will stop generation
        provider: Which provider to use ('openai' or 'llama')
        verbose: Whether to print verbose information
        
    Returns:
        str: The model's response
    """
    if verbose:
        print(f"\n--- LLM Request ({provider}) ---")
        if system_prompt:
            print(f"System: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"System: {system_prompt}")
        print(f"User: {prompt[:100]}..." if isinstance(prompt, str) and len(prompt) > 100 else f"User: {prompt}")
    
    try:
        if provider.lower() == 'together' and 'gemini' not in model.lower():
            return get_together_ai_response(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_strs=stop_strs
            )
        if 'gemini' in model.lower():
            print(f"Using Gemini model: {model}")
            return get_gemini_gen(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_strs=stop_strs
            )
        if provider.lower() == 'openrouter':
            print("USing open router")
            return get_open_router_gen(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_strs=stop_strs
            )
        if 'gpt' in model.lower():
            if 'azure' == provider.lower():
                print(f"Using Azure model: {model}")
                return get_azure_gen(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_strs=stop_strs
                )
            else:
                return get_openai_gen(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_strs=stop_strs
                )
        if 'llama' in model.lower() or 'gemma' in model.lower():
            print(f"Using Llama model: {model}")
            return get_llama_response(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_strs=stop_strs
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Valid options are 'openai' or 'llama'.")
    
    except Exception as e:
        print(f"Error in get_llm_response: {e}")
        return f"I encountered an error when trying to generate a response: {e}"


def parse_guess(raw_guess, clean_for_json=True):
    """
    Extract the guess and confidence level from a raw LLM output.
    
    Args:
        raw_guess: The raw response from the LLM
        clean_for_json: Whether to clean common JSON formatting issues
        
    Returns:
        tuple: (guess, confidence, cleaned_response)
    """
    cleaned_response = raw_guess
    
    try:
        # Clean the response to handle common formatting issues
        if clean_for_json:
            cleaned_response = raw_guess.replace("{{", "{").replace("}}", "}")
        print(f"Cleaned response: {cleaned_response}")
        
        # Extract JSON block using regex
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            guess_json = json_match.group(0)
            parsed_data = json.loads(guess_json)
            
            guess = parsed_data.get("guess", "unknown")
            confidence = parsed_data.get("confidence", 0)
            return guess, confidence, cleaned_response
        else:
            # Fallback: try to extract guess directly
            guess_match = re.search(r'guess["\s:]+([^",\n]+)', cleaned_response, re.IGNORECASE)
            if guess_match:
                return guess_match.group(1).strip(), 50, cleaned_response
            
            raise ValueError("No valid JSON or direct guess found in the input.")
    
    except Exception as e:
    
        print(f"Error parsing guess: {e}")
        return "unknown", 0, cleaned_response


def load_json_data(filepath):
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        The loaded data or None on error
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None


def save_json_data(data, filepath, indent=2):
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save
        filepath: Path to the output file
        indent: Indentation level for the JSON file
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        return False


def is_multiple_choice_question(question):
    """Check if a question is a multiple choice question"""
    return bool(re.search(r'(?:^|\n)[A-F][).] ', question))


def is_yes_no_question(question):
    """Check if a question is likely a yes/no question"""
    # Common yes/no question patterns
    yes_no_patterns = [
        r"^does .+\?",
        r"^is .+\?",
        r"^are .+\?",
        r"^can .+\?",
        r"^has .+\?",
        r"^have .+\?",
        r"^will .+\?",
        r"^would .+\?",
        r"^should .+\?",
        r"^do .+\?"
    ]
    
    # Clean the question
    clean_question = question.lower().strip()
    
    # Check if it matches any yes/no pattern
    for pattern in yes_no_patterns:
        if re.search(pattern, clean_question):
            return True
            
    return False


def normalize_text(text):
    """Normalize text for comparison"""
    # Remove punctuation and extra spaces
    clean_text = re.sub(r'[.,;:!?"\'\(\)]', '', text)
    # Convert to lowercase and strip
    return clean_text.lower().strip()