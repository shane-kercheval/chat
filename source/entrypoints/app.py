"""
Streamlit app that enables conversations with ChatGPT and additional features like prompt
templates.
"""
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
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


def initialize() -> None:
    """Initialize environment and app."""
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")


def main() -> None:
    """Defines the application structure and behavior."""
    initialize()
    sh.apply_css()
    message_state = st.session_state.setdefault('state', {'chat_message': ''})
    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = sh._return_mock_conversation()

    with st.sidebar:
        st.markdown('# Options')
        openai_model = st.selectbox(label="Model", options=('GPT-3.5', 'GPT-4'))

        with st.expander("Additional Options"):
            temperature = st.slider(
                label="Temperature",
                min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",  # noqa
            )
            max_tokens = st.slider(
                label="Max Tokens",
                min_value=100, max_value=3000, value=1000, step=100,
                help="The maximum number of tokens to generate in the completion."  # noqa
            )

        st.markdown('# Prompt Template')
        template_name = sh.create_prompt_template_options()
        if template_name != '<Select>':
            prompt_template = sh.get_prompt_template(template_name=template_name)
            template_fields = sh.get_fields_from_template(prompt_template=prompt_template)
            field_values = []
            # use form to prevent text_areas from refreshing app after focus moves away
            with st.form(key="prompt_template_fields_form"):
                st.markdown("### Fields")
                st.markdown("Fill in the information for the following fields referred to in the prompt template, shown below.")  # noqa
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
        # this submit_button when send the message to ChatGPT
        submit_button = st.button("Submit")
    with col_conversation_totals:
        sh.display_totals(
            cost=5.2222,
            total_tokens=3000,
            prompt_tokens=2750,
            completion_tokens=25,
        )

    sh.display_horizontal_line()

    if submit_button and user_input:
        human_message = HumanMessage(content=user_input)
        st.session_state.messages.append(human_message)
        with st.spinner("Thinking..."):
            if openai_model == 'GPT-3.5':
                model_name = 'gpt-3.5-turbo'
            elif openai_model == 'GPT-4':
                model_name = 'gpt-4'
            else:
                raise ValueError(openai_model)

            print(f"Calling ChatGPT: model={model_name}; temp={temperature}; max_tokens={max_tokens}")  # noqa
            chat = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
            # TODO: pass all messages and/or figure out memory buffer strategy
            response = chat([st.session_state.messages[0]] + [st.session_state.messages[-1]])
        st.session_state.messages.append(response)
        message_state['chat_message'] = user_input  # Update session state value

    # display message history
    messages = st.session_state.get('messages', [])
    messages = list(reversed(messages))
    for i in range(0, len(messages) - 1, 2):
        human_message = messages[i + 1]
        assert isinstance(human_message, HumanMessage)
        ai_message = messages[i]
        assert isinstance(ai_message, AIMessage)
        col_messages, col_totals = st.columns([5, 1])
        with col_messages:
            sh.display_chat_message(ai_message.content, is_human=False)
            sh.display_chat_message(human_message.content, is_human=True)
        with col_totals:
            sh.display_totals(
                cost=1.25,
                total_tokens=1000,
                prompt_tokens=750,
                completion_tokens=25,
            )

    print("--------------END--------------")

if __name__ == '__main__':
    main()
