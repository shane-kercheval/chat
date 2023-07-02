"""
Streamlit app that enables conversations with ChatGPT and has additional features like prompt
templates.
"""
import os
from dotenv import load_dotenv
import yaml
import streamlit as st
import streamlit.components.v1 as components
from llm_chain.base import Session
from llm_chain.models import OpenAIChat, StreamingEvent
import source.streamlit_helpers as sh


PROMPT_TEMPLATE_DIR = '/code/source/prompt_templates/'


st.set_page_config(
    page_title="ChatGPT",
    page_icon="🤖",
    layout='wide',
)


@st.cache_data
def load_prompt_templates() -> dict:
    """
    Loads the prompt templates from a directory and returns them as a dictionary with the name of
    the template as the key and template and the value.
    """
    template_files = os.listdir(PROMPT_TEMPLATE_DIR)
    templates = {}
    for file_name in template_files:
        with open(os.path.join(PROMPT_TEMPLATE_DIR, file_name)) as handle:
            yaml_data = yaml.safe_load(handle)
            template_name = yaml_data.pop('name')
            assert template_name not in templates  # don't duplicate template names
            templates[template_name] = yaml_data
    return templates


def initialize() -> None:
    """Initializes environment and app."""
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY")
    sh.apply_css()
    if 'chat_session' not in st.session_state:
        # chat_session tracks the history of messages/chains used during the session
        # the `Clear` button (or page refresh) clears the session
        st.session_state.chat_session = Session()

    # the user_input state caches the value from the text-box where the user enters their question
    # we need this for updating the text-box from the prompt-templates
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''


def main() -> None:
    """Defines the application structure and behavior."""
    initialize()

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
                help="The maximum number of tokens to generate in the completion.",
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
                # when this button is pressed, we will build up the message from the template
                # and the values from the user and fill in the chat message text_area
                create_message_submit = st.form_submit_button(label='Create message')

            if create_message_submit:
                prompt = prompt_template
                # replace all instances of `{{field}}` with the value from the user
                for field, value in field_values:
                    prompt = prompt.replace("{{" + field + "}}", value)

                st.session_state.user_input = prompt

            st.markdown('### Template:')
            st.sidebar.text(prompt_template)

    # display the chat text_area and total cost side-by-side
    col_craft_message, col_conversation_totals = st.columns([5, 1])
    with col_craft_message:
        user_input = st.text_area(
            "<label should be hidden>",
            key='chat_message',
            value=st.session_state.user_input,
            placeholder="Ask a question.",
            height=150,
        )
    with col_conversation_totals:
        result_placeholder = st.empty()

    col_submit,  col_web_search, col_stack_search, col_clear =  st.columns([2, 4, 4, 2])
    with col_submit:
        submit_button = st.button("Submit")
    with col_web_search:
        use_web_search = st.checkbox(
            label="Use Web Search (DuckDuckGo)",
            value=False,
            help="Use DuckDuckGo to do a web-search based on the current input above and find the most relevant content and use that in the prompt to ChatGPT.",  # noqa
        )
    with col_stack_search:
        # only display the stack overflow checkbox if the environment variable exists
        if os.getenv('STACK_OVERFLOW_KEY', None):
            use_stack_overflow = st.checkbox(
                label="Use Stack Overflow",
                value=False,
                help="Search Stack Overflow based on the current input above and find the most relevant content and use that in the prompt to ChatGPT.",  # noqa
            )
        else:
            use_stack_overflow = False
    with col_clear:
        clear_button = st.button("Clear Session")
        if clear_button:
            st.session_state.chat_session = Session()

    sh.display_horizontal_line()

    # define the placeholders for the next chat prompt/response so that we can place them at the
    # tope of the (previous) messages
    if submit_button and user_input:
        col_messages, col_totals = st.columns([5, 1])
        with col_messages:
            placeholder_response = st.empty()
            placeholder_info = st.empty()
            placeholder_documents = st.empty()
            placeholder_prompt = st.empty()
        with col_totals:
            placeholder_message_totals = st.empty()

        with st.spinner("Loading..."):
            chat_session = st.session_state.chat_session
            if chat_session.message_history:
                sh.display_message_history(chat_session.message_history)

            sh.display_chat_message(user_input, is_human=True, placeholder=placeholder_prompt)

            message = ""
            def _update_message(x: StreamingEvent) -> None:
                nonlocal message
                message += x.response
                sh.display_chat_message(
                    message,
                    is_human=False,
                    placeholder=placeholder_response,
                )

            chain, doc_prompt_template = sh.build_chain(
                # chat_model=sh.MockChatModel(model_name='mock'),
                chat_model=OpenAIChat(model_name='gpt-3.5-turbo'),
                model_name=sh.get_model_name(model_display_name=openai_model_name),
                max_tokens=max_tokens,
                temperature=temperature,
                streaming_callback=_update_message,
                use_web_search=use_web_search,
                use_stack_overflow=use_stack_overflow,
            )
            chat_session.append(chain=chain)
            _ = chat_session(user_input)
            last_message = chat_session.message_history[-1]
            if last_message.prompt != user_input:
                # if prompt was updated then make some indication that the underlying prompt
                # has changed
                placeholder_info.info("The prompt was modified within the chain.", icon="ℹ️")
                sh.display_chat_message(
                    last_message.prompt,
                    is_human=True,
                    placeholder=placeholder_prompt,
                )

            if doc_prompt_template and doc_prompt_template.similar_docs:
                values = [x.metadata['url'] for x in doc_prompt_template.similar_docs]
                placeholder_documents.info(
                    "Information from the following URLs was provided to the model:\n\n" +\
                    "\n\n".join(values),
                    icon="ℹ️",
                )

            sh.display_totals(
                cost=last_message.cost,
                total_tokens=last_message.total_tokens,
                prompt_tokens=last_message.prompt_tokens,
                response_tokens=last_message.response_tokens,
                is_total=False,
                placeholder=placeholder_message_totals,
            )
    elif st.session_state.chat_session.message_history:
        sh.display_message_history(st.session_state.chat_session.message_history)

    if st.session_state.chat_session.message_history:
        # display totals for entire conversation; need to do this after we are done with the last
        # submission
        sh.display_totals(
            cost=st.session_state.chat_session.cost,
            total_tokens=st.session_state.chat_session.total_tokens,
            prompt_tokens=st.session_state.chat_session.prompt_tokens,
            response_tokens=st.session_state.chat_session.response_tokens,
            is_total=True,
            placeholder=result_placeholder,
        )

    # add javascript to add cmd+enter keyboard-shortcut for Submit button
    script = """
    <script>
    const doc = window.parent.document  // break out of the IFrame
    doc.addEventListener('keydown', function(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        console.log('asdfasdfasdfasd')
        var submitButton = doc.querySelector('button > div > p');
        if (submitButton && submitButton.textContent === 'Submit') {
        submitButton.parentElement.parentElement.click();
        }
    }
    });
    </script>
    """
    components.html(script, width=0, height=0)


if __name__ == '__main__':
    main()
