"""Helper functions for streamlit app."""
from functools import cache
import re
import streamlit as st
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS

import tiktoken



@cache
def get_tiktok_encoding(model: str) -> tiktoken.Encoding:
    """Helper function that returns an encoding method for a given model."""
    return tiktoken.encoding_for_model(model)


class ChatMetaData():
    """TBD."""

    def __init__(
            self,
            model_name: str,
            human_question: HumanMessage,
            full_question: str,
            ai_response: AIMessage,
            prompt_tokens: int | None = None,
            completion_tokens: int | None = None,
            total_tokens: int | None = None,
            cost: float | None = None,
        ) -> None:
        """
        full_question is the question and any history or prompt template that langchain sent.
        If tokens/cost is not provided, they will be calculated.
        prompt_history is not necessarily the same thing as all history. It is the history used in
        the prompt, but not all history is necessarily used.
        """
        assert model_name in {'gpt-3.5-turbo', 'gpt-4'}

        encoding = get_tiktok_encoding(model=model_name)
        self.model_name = model_name
        self.human_question = human_question
        self.full_question = full_question
        self.ai_response = ai_response
        if not prompt_tokens:
            prompt_tokens = len(encoding.encode(full_question))
        self.prompt_tokens = prompt_tokens
        if not completion_tokens:
            completion_tokens = len(encoding.encode(ai_response.content))
        self.completion_tokens = completion_tokens
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        self.total_tokens = total_tokens
        if not cost:
            cost = MODEL_COST_PER_1K_TOKENS[model_name] * (total_tokens / 1_000)
        self.cost = cost



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
            margin-bottom: 30px;
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
    css += """
    <style>
    section[data-testid="stSidebar"] .stSelectbox label {
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


def display_chat_message(message: str, is_human: bool) -> None:
    """
    Displays a chat message and formats according to whether or not the message is from a human.

    Args:
        message: the message to display
        is_human: True if the message is from a human; False if message is OpenAI response
    """
    sender_emoji = "👨‍💼" if is_human else "🤖"
    sender_class = "sender" if is_human else "receiver"

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown(f"<div class='emoji'>{sender_emoji}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='{sender_class}'>{message}</div>", unsafe_allow_html=True)


def create_prompt_template_options() -> None:
    """Returns a drop-down widget with prompt templates."""
    return st.selectbox(
        '<label should be hidden>',
        (
            '<Select>',
            'Make text sound better.',
            'Summarize text.',
            'Code: Write doc strings',
        ),
    )

def get_prompt_template(template_name: str) -> str:
    """
    Given a `template_name` get the corresponding prompt-template.

    Args:
        template_name: name of the prompt-template to get
    """
    return """
            This is a prompt.

            It has fields like this `{{context}}`

            And this:

            ```
            {{more_context}}
            ```
            """


def get_fields_from_template(prompt_template: str) -> list[str]:
    """
    Extracts the fields (variable names wrapped in double curly brackets) from a prompt template.

    >>> template = "This is a template with these {{var_a}} and {{var_b}} as context."
    >>> get_fields_from_template(prompt_template=template)
    ['var_a', 'var_b']

    Args:
        prompt_template: the template to extract the fields from.
    """
    # TODO: unit test
    # Regular expression pattern to match values within double brackets
    pattern = r"\{\{(.*?)\}\}"
    # Find all matches of the pattern in the text
    return re.findall(pattern, prompt_template, re.DOTALL)


def display_totals(
        cost: float,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        is_total: bool,
        placeholder: object | None = None) -> None:
    """
    Display the totals.

    Args:
        cost: cost of all tokens
        total_tokens: number of total tokens
        prompt_tokens: number of prompt tokens
        completion_tokens: number of completion tokens
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
    Completion Tokens: <code>{completion_tokens:,}</code><br>
    """
    cost_html = f"""
    <p style="font-size: 13px; text-align: right">{cost_string}</p>
    <p style="font-size: 12px; text-align: right">{token_string}</p>
    """
    if placeholder:
        placeholder.markdown(cost_html, unsafe_allow_html=True)
    else:
        st.markdown(cost_html, unsafe_allow_html=True)



def _create_mock_message_chain(num_conversations: int = 10) -> list[BaseMessage]:
    from faker import Faker
    import random

    random.seed(123)
    fake = Faker()
    Faker(123)
    """Returns a mock conversation with ChatGPT."""
    message = """
    This is some python:

    ```
    def python():
        return True
    ```

    It is code.
    """
    messages = [SystemMessage(content="You are a helpful assistant.")]
    for _ in range(num_conversations):
        fake_human = ' '.join([fake.word() for _ in range(random.randint(5, 10))])
        fake_ai = ' '.join([fake.word() for _ in range(random.randint(5, 10))])
        messages += [
            HumanMessage(content="Question: " + fake_human),
            AIMessage(content="Answer: " + fake_ai),
        ]
    messages += [
        HumanMessage(content="Question: " + message),
        AIMessage(content="Answer: " + message),
    ]
    return messages


def _create_mock_chat_history(message_chain: list[BaseMessage]) -> list[ChatMetaData]:
    # message_chain = list(reversed(message_chain))
    chat_history = []
    for i in range(1, len(message_chain), 2):
        human_message = message_chain[i]
        assert isinstance(human_message, HumanMessage)
        ai_message = message_chain[i + 1]
        assert isinstance(ai_message, AIMessage)
        chat_history.append(ChatMetaData(
            model_name='gpt-3.5-turbo',
            human_question=human_message,
            full_question=human_message.content + "this is some history and context sent in",
            ai_response=ai_message,
        ))
    return chat_history

# import streamlit as st

# def main():
#     state = st.session_state.setdefault('state', {'text_area_b': ''})
#     # state = {'text_area_b': ''}
#     print(state)

#     text_area_a = st.text_area("Enter text for TextArea A")
#     button_a = st.button("Button A")

#     print(f"text_area_a - 1: `{text_area_a}`")
#     if button_a:
#         # Update contents of TextArea B with contents of TextArea A
#         print("updating state")
#         state['text_area_b'] = text_area_a
#         print(f"state: `{state['text_area_b']}`")

#     print(f"state - 2: `{state['text_area_b']}`")
#     text_area_b = st.text_area("TextArea B", value=state['text_area_b'], key='text_area_b')
#     print(f"text_area_b - 1: `{text_area_b}`")

#     button_b = st.button("Button B")
#     if button_b:
#         print(f"text_area_b - 2: `{text_area_b}`")
#         st.write("Contents of TextArea B:")
#         st.write(text_area_b)
#         state['text_area_b'] = text_area_b  # Update session state value

#     print("-----end------")

# if __name__ == "__main__":
#     main()
