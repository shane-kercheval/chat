"""Helper functions for streamlit app."""
# from functools import cache
import re
from typing import TypeVar
from collections.abc import Callable
import streamlit as st
from llm_chain.tools import StackQuestion, scrape_url
from llm_chain.base import PromptModel, ExchangeRecord, Document, Chain, Value
from llm_chain.models import OpenAIChat, StreamingEvent
from llm_chain.tools import DuckDuckGoSearch, split_documents, search_stack_overflow
from llm_chain.indexes import ChromaDocumentIndex
from llm_chain.prompt_templates import DocSearchTemplate


StreamlitWidget = TypeVar('StreamlitWidget')


def apply_css() -> None:
    """Applies css to the streamlit app."""
    css = ''
    # css for main container
    css += """
    <style>
    section.main .block-container {
        padding-top: 30px;
        padding-bottom: 20px;
    }
    </style>
    """
    # css for chat message
    css += """
    <style>
        .sender {
            background-color: #d8f0ff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 45px;
        }
        .receiver {
            background-color: #F0F0F0;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .emoji {
            font-size: 24px;
            margin-bottom: 5px;
        }
        .stCodeBlock {
            margin: 20px 0;
        }
    </style>
    """
    # css for sidebar width
    css += '<style>section[data-testid="stSidebar"]{width: 400px !important;}</style>'
    # custom style for text box e.g. prompt-template in sidebar
    css += """
    <style>
    div[data-testid="stText"] {
        background-color: white;
        padding: 10px;
        white-space: pre-wrap;
        width: 100%;
    }
    </style>
    """
     # Add custom CSS to hide the label of the prompt-template selection box in the sidebar
     # use the following if we only want to hide the label in the sidebar:
     # `section[data-testid="stSidebar"] .stSelectbox label {`
    css += """
    <style>
    .stSelectbox label {
        display: none;
    }
    </style>
    """
    # Add custom CSS to hide the label of the chat message
    css += '<style>section.main div.stTextArea label { display: none;}</style>'
    st.markdown(css, unsafe_allow_html=True)


def display_horizontal_line(margin: str = '10px 0px 30px 0px') -> None:
    """
    Displays a display_horizontal_line.

    Args:
        margin: margin in pixels
    """
    css = f"""
    <style>
        .line {{
            border: none;
            border-top: 1px solid #E6E6E6;
            margin: {margin};
        }}
    </style>
    """
    st.markdown(css + "<div class='line'></div>", unsafe_allow_html=True)


def display_chat_message(
        message: str,
        is_human: bool,
        placeholder: StreamlitWidget | None = None) -> None:
    """
    Displays a chat message and formats according to whether or not the message is from a human.

    Args:
        message:
            the message to display
        is_human:
            True if the message is from a human; False if message is OpenAI response
        placeholder:
            a placeholder is widget e.g. result from st.empty(); if provided, we will clear the
            widget and write the markdown to that widget rather using `st.markdown`.
    """
    message_html = f"<div class='{'sender' if is_human else 'receiver'}'>{message}</div>"
    if placeholder:
        placeholder.empty()
        placeholder.markdown(message_html, unsafe_allow_html=True)
    else:
        st.markdown(message_html, unsafe_allow_html=True)


def display_exchange_history(exchange_history: list[ExchangeRecord]) -> None:
    """Displays the message history and corresponding cost and token usage for each mesage."""
    chat_history = list(reversed(exchange_history))
    for chat in chat_history:
        col_messages, col_totals = st.columns([5, 1])
        with col_messages:
            display_chat_message(chat.response, is_human=False)
            display_chat_message(chat.prompt, is_human=True)
        with col_totals:
            display_totals(
                cost=chat.cost,
                total_tokens=chat.total_tokens,
                prompt_tokens=chat.prompt_tokens,
                response_tokens=chat.response_tokens,
                is_total=False,
            )


def create_prompt_template_options(templates: dict) -> StreamlitWidget:
    """Returns a drop-down widget with prompt templates."""
    template_names = sorted(templates.items(), key=lambda x: (x[1]['category'], x[1]['template']))
    template_names = [x[0] for x in template_names]
    return st.selectbox(
        '<label should be hidden>',
        ['<Select>'] + template_names,
    )


def get_fields_from_template(prompt_template: str) -> list[str]:
    """
    Extracts the fields (variable names wrapped in double curly brackets) from a prompt template.

    >>> template = "This is a template with these {{var_a}} and {{var_b}} as context."
    >>> get_fields_from_template(prompt_template=template)
    ['var_a', 'var_b']

    Args:
        prompt_template: the template to extract the fields from.
    """
    # Regular expression pattern to match values within double brackets
    pattern = r"\{\{(.*?)\}\}"
    # Find all matches of the pattern in the text
    return re.findall(pattern, prompt_template, re.DOTALL)


def display_totals(
        cost: float,
        total_tokens: int,
        prompt_tokens: int,
        response_tokens: int,
        is_total: bool,
        placeholder: object | None = None) -> None:
    """
    Display the totals.

    Args:
        cost: cost of all tokens
        total_tokens: number of total tokens
        prompt_tokens: number of prompt tokens
        response_tokens: number of completion tokens
        is_total: whether or not the total cost of all conversations, or a single conversation
        placeholder: if not None; use this to write results to
    """
    total_label = 'Total' if is_total else 'Message'
    round_by = 4 if is_total else 6
    cost_string = f"""
    <b>{total_label} Cost</b>: <code>${cost:.{round_by}f}</code><br>
    """
    token_string = f"""
    {total_label} Tokens: <code>{total_tokens:,}</code><br>
    Prompt Tokens: <code>{prompt_tokens:,}</code><br>
    Response Tokens: <code>{response_tokens:,}</code><br>
    """
    cost_html = f"""
    <p style="font-size: 13px; text-align: right">{cost_string}</p>
    <p style="font-size: 12px; text-align: right">{token_string}</p>
    """
    if placeholder:
        placeholder.markdown(cost_html, unsafe_allow_html=True)
    else:
        st.markdown(cost_html, unsafe_allow_html=True)


def scrape_urls(search_results: dict) -> list[Document]:
    """
    For each url (i.e. `href` in `search_results`):
    - extracts text
    - replace new-lines with spaces
    - create a Document object.
    """
    results = []
    for result in search_results:
        try:  # noqa: SIM105
            doc = Document(
                content=re.sub(r'\s+', ' ', scrape_url(result['href'])),
                metadata={'url': result['href']},
            )
            results.append(doc)
        except:  # noqa: E722
            pass
    return results


def stack_overflow_results_to_docs(results: list[StackQuestion]) -> list[Document]:
    """
    Takes a list of results from the `search_stack_overflow` function, and returns a corresponding
    list of Document objects.
    """
    answers = []
    for question in results:
        for answer in question.answers:
            doc = Document(
                content=f"{question.title[0:100]} - {answer.markdown[0:1_500]}",
                metadata={'url': question.link},
            )
            answers.append(doc)
    return answers


def get_model_name(model_display_name: str) -> str:
    """Given the model name displayed o the user, return the actual model name."""
    if model_display_name == 'GPT-3.5':
        return 'gpt-3.5-turbo'
    if model_display_name == 'GPT-4':
        return 'gpt-4'
    raise ValueError(model_display_name)


def build_chain(
        chat_model: OpenAIChat,
        model_name: str,
        max_tokens: int,
        temperature: float,
        streaming_callback: Callable[[StreamingEvent], None],
        use_web_search: bool,
        use_stack_overflow: bool,
        ) -> Chain:
    """
    Build a Chain based on the options provided.

    Returns both the chain and the DocSearchTemplate object (if web-search or
    stack-overflow-search) is used so that we can display the URLs from the search to the end-user.
    """
    chat_model.model_name = model_name
    chat_model.temperature = temperature
    chat_model.max_tokens = max_tokens
    chat_model.streaming_callback = streaming_callback
    doc_prompt_template = None

    if use_web_search or use_stack_overflow:
        document_index = ChromaDocumentIndex(n_results=3)
        doc_prompt_template = DocSearchTemplate(doc_index=document_index, n_docs=3)
        # Value is a callable; when called with a value it caches and returns the value
        # when called without a value
        prompt_cache = Value()
        links = []
        if use_web_search:
            links += [
                prompt_cache,  # input is user's question; caches and returns
                DuckDuckGoSearch(top_n=3),  # get the urls of the top search results
                scrape_urls,  # scrape the websites corresponding to the URLs
                lambda docs: split_documents(docs=docs, max_chars=1_000),
                document_index,  # add docs to doc-index
            ]
        if use_stack_overflow:
            links += [
                # if use_web_search is false, the original question will get passed in
                # to prompt_cache which will cache and return it;
                # if use_web_search is true, then prompt_cache will have already been set
                # and document_index will return None, so prompt_cache will return the
                # previously cached value (i.e. the original question) and pass that to
                # the search_query
                prompt_cache,
                lambda search_query: search_stack_overflow(query=search_query, max_questions=2, max_answers=1),  # noqa
                stack_overflow_results_to_docs,
                # we almost certainly want to keep the entire answer
                lambda docs: split_documents(docs=docs, max_chars=2_500),
                document_index,  # add docs to doc-index
            ]
        links += [
            prompt_cache,
            doc_prompt_template,
            chat_model,
        ]
    else:
        links = [chat_model]

    return Chain(links=links), doc_prompt_template


def _create_fake_message() -> str:
    """Creates a fake message (random words of random length)."""
    import random
    from faker import Faker
    fake = Faker()
    return ' '.join([fake.word() for _ in range(random.randint(10, 100))])


class MockPromptModel(PromptModel):
    """Mock Chat model that creates takes a prompt and returns fake message; used for dev."""

    def __init__(self, model_name: str, temperature: float = 0, max_records: int = 0):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_records = max_records

    def _run(self, prompt: str) -> ExchangeRecord:
        return ExchangeRecord(
            prompt=prompt,
            response=_create_fake_message(),
            cost=1.25,
            prompt_tokens=100,
            response_tokens=75,
            total_tokens=175,
        )
