# chat

This streamlit app gives basic chat functionality using the [llm-chain package](https://github.com/shane-kercheval/llm-chain) and OpenAI.

Features:

- text streaming
- switch between GPT-3.5 and GPT-4
- shows token usage and costs for individual messages and aggregate across all messages
- has options for web-search (via DuckDuckGo) and searching Stack Overflow (adds the search results to a local vector database which is used to inject the most relevant content into the prompt)
- Allows users to create and select prompt-templates (`chat/source/prompt_templates`)

## Running

- .env


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
