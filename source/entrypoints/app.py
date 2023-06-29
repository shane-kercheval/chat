"""
Streamlit app that enables conversations with ChatGPT and additional features like prompt
templates.
"""
import os
from dotenv import load_dotenv
import streamlit as st
from llm_chain.base import ChatModel, Chain, Session, Value
from llm_chain.tools import DuckDuckGoSearch, split_documents, search_stack_overflow
from llm_chain.indexes import ChromaDocumentIndex
from llm_chain.prompt_templates import DocSearchTemplate

from llm_chain.models import OpenAIChat, StreamingRecord
import source.library.streamlit_helpers as sh

# TODO: document
# Known issues;
# Once you modify the chat message, the prompt template message will know fill the text-box
# Steps: fill out fields in prompt template; hit "Create Message"; Chat message is updated;
# change one or more fields; hit "Create Message"; Chat message is still updated; Modify chat
# message; now click "Create Message" from prompt template; Chat message will not update;
# however, if you change one or more of field values in the prompt template and hit Create Message
# then the chat message will update as expected.

st.set_page_config(
    page_title="ChatGPT",
    page_icon="🤖",
    layout='wide',
)

def create_chat_model() -> ChatModel:
    """TODO."""
    # return sh.MockChatModel(model_name='mock')
    return OpenAIChat(model_name='gpt-3.5-turbo')

@st.cache_data
def load_prompt_templates() -> dict:
    """TBD."""
    import os
    import yaml
    template_files = os.listdir('/code/prompt_templates/')
    templates = {}
    for file_name in template_files:
        with open(os.path.join('/code/prompt_templates/', file_name)) as handle:
            yaml_data = yaml.safe_load(handle)
            template_name = yaml_data.pop('name')
            assert template_name not in templates
            templates[template_name] = yaml_data
    return templates


def initialize() -> None:
    """Initialize environment and app."""
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY")
    sh.apply_css()
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = Session()

def main() -> None:
    """Defines the application structure and behavior."""
    initialize()
    message_state = st.session_state.setdefault('state', {'chat_message': ''})

    with st.sidebar:
        st.markdown('# Options')
        openai_model_name = st.selectbox(label="Model", options=('GPT-3.5', 'GPT-4'))

        with st.expander("Additional Options"):
            temperature = st.slider(
                label="Temperature",
                min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",  # noqa
            )
            max_tokens = st.slider(
                label="Max Tokens",
                min_value=100, max_value=3000, value=2000, step=100,
                help="The maximum number of tokens to generate in the completion."  # noqa
            )

        st.markdown("# Prompt Template")
        prompt_templates = load_prompt_templates()
        template_name = sh.create_prompt_template_options(templates=prompt_templates)
        if template_name != '<Select>':
            prompt_template = prompt_templates[template_name]['template']
            template_fields = sh.get_fields_from_template(prompt_template=prompt_template)
            field_values = []
            # use form to prevent text_areas from refreshing app after focus moves away
            with st.form(key="prompt_template_fields_form"):
                st.markdown("### Fields")
                st.markdown(
                    '<p style="font-size: 11px; "><i>Fill in the information for the following fields referred to in the prompt template, shown below.</i></p>',  # noqa
                    unsafe_allow_html=True,
                )
                for field in template_fields:
                    field_values.append((field, st.text_area(field, height=100, key=field)))
                    # field_values.append((match, st.sidebar.text_area(match, height=100)))

                # when this button is pressed, we will build up the message from the template
                # and the values from the user and fill in the chat message text_area
                create_message_submit = st.form_submit_button(label='Create message')

            if create_message_submit:
                chat_gpt_message = prompt_template
                # replace all instances of `{{field}}` with the value from the user
                for field, value in field_values:
                    chat_gpt_message = chat_gpt_message.replace("{{" + field + "}}", value)

                message_state['chat_message'] = chat_gpt_message
                print(f"message_state - updated: `{message_state}`")

            st.markdown('### Template:')
            st.sidebar.text(prompt_template)


    # display the chat text_area and total cost side-by-side
    col_craft_message, col_conversation_totals = st.columns([5, 1])
    with col_craft_message:
        user_input = st.text_area(
            "<label should be hidden>",
            key='chat_message',
            value=message_state['chat_message'],
            placeholder="Ask a question.",
            height=150,
        )
    with col_conversation_totals:
        result_placeholder = st.empty()

    col_submit,  col_search, col_stack, col_clear =  st.columns([2, 4, 4, 2])
    with col_submit:
        submit_button = st.button("Submit")
    with col_search:
        use_web_search = st.checkbox(label="Use Web Search (DuckDuckGo)", value=False)
    with col_stack:
        use_stack_overflow = st.checkbox(label="Use Stack Overflow", value=False)
    with col_clear:
        clear_button = st.button("Clear")
        if clear_button:
            st.session_state['chat_session'] = Session()

    sh.display_horizontal_line()
    chat_session = st.session_state.get('chat_session', Session())  # TODO.. is default needed???

    # placeholders for the next submissions
    # TODO: if i select a different model under Options, the totals for all messages disappears
    if submit_button and user_input:
        col_messages, col_totals = st.columns([5, 1])
        with col_messages:
            placeholder_response = st.empty()
            placeholder_prompt = st.empty()
        with col_totals:
            placeholder_message_totals = st.empty()

    # display previous history: i.e. history at point before we hit submit
    if chat_session.message_history:
        sh.display_message_history(chat_session.message_history)

    if submit_button and user_input:
        # with st.spinner("Thinking..."):
        if openai_model_name == 'GPT-3.5':
            model_name = 'gpt-3.5-turbo'
        elif openai_model_name == 'GPT-4':
            model_name = 'gpt-4'
        else:
            raise ValueError(openai_model_name)

        sh.display_chat_message(user_input, is_human=True, placeholder=placeholder_prompt)

        message = ""
        def _update_message(x: StreamingRecord) -> None:
            nonlocal message
            message += x.response
            sh.display_chat_message(message, is_human=False, placeholder=placeholder_response)

        chain = sh.build_chain(
            chat_model=create_chat_model(),
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            streaming_callback=_update_message,
            use_web_search=use_web_search,
            use_stack_overflow=use_stack_overflow,
        )
        chat_session.append(chain=chain)
        chat_session(user_input)
        last_message = chat_session.message_history[-1]
        sh.display_totals(
            cost=last_message.cost,
            total_tokens=last_message.total_tokens,
            prompt_tokens=last_message.prompt_tokens,
            response_tokens=last_message.response_tokens,
            is_total=False,
            placeholder=placeholder_message_totals,
        )

        # display totals for entire conversation; need to do this after we are done with the last
        # submission
        # TODO: need to change this to a chain and track all costs not just message costs
        # need to display an info icon indicating that non-chat message costs/tokens are not
        # displayed and so won't match totals
        sh.display_totals(
            cost=chat_session.cost,
            total_tokens=chat_session.total_tokens,
            prompt_tokens=chat_session.prompt_tokens,  # TODO: should we add this to Session/Chain? or does it make sense to display  # noqa
            response_tokens=chat_session.response_tokens,  # TODO: should we add this to Session/Chain? or does it make sense to display # noqa
            is_total=True,
            placeholder=result_placeholder,
        )


if __name__ == '__main__':
    main()
