# chat

This streamlit app gives basic chat functionality using the [llm-chain package](https://github.com/shane-kercheval/llm-chain) and OpenAI.

Features:

- text streaming
- switch between GPT-3.5 and GPT-4
- shows token usage and costs for individual messages and aggregate across all messages
- has options for web-search (via DuckDuckGo) and searching Stack Overflow (adds the search results to a local vector database which is used to inject the most relevant content into the prompt)
- Allows users to create and select prompt-templates (`chat/source/prompt_templates`)

## Running

- Create a `.env` file in the project directory (i.e. same directory as this README file) in the format below. This file is used by `app.py` to load the key/value pairs as environment variables.
    - In order to use the `Use Stack Overflow` flag, you must have an entry for the `STACK_OVERFLOW_KEY` environment variable. Create an account and app at [Stack Apps](https://stackapps.com/) and use the `key` that is generated (not the `secret`). Otherwise you, can omit
    
```
OPENAI_API_KEY=....
STACK_OVERFLOW_KEY=...
```

- `make docker_run`
- open browser to `http://localhost:8501`


## Prompt Templates

# Feature Roadmap

- [ ] settings to adjust DuckDuckGoSearch options
    - [ ] number of web-search results to scrape and store (default is 3)
    - [ ] size of document chunks (default is 500 characters)
    - [ ] number of documents chunks to insert into prompt (default is 3)
- [ ] `total_tokens` might not match `prompt_tokens` plus `response_tokens`. If there are other types of tokens used, for example, tokens used for embeddings.
- [ ] add settings and code to control memory e.g `llm_chain.memory.MemoryBufferMessageWindow`
- [ ] bug
# TODO: document
# Known issues;
# Once you modify the chat message, the prompt template message will know fill the text-box
# Steps: fill out fields in prompt template; hit "Create Message"; Chat message is updated;
# change one or more fields; hit "Create Message"; Chat message is still updated; Modify chat
# message; now click "Create Message" from prompt template; Chat message will not update;
# however, if you change one or more of field values in the prompt template and hit Create Message
# then the chat message will update as expected.
