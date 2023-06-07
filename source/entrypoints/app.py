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
    # We have to differentiate between the entire message chain/history of the conversation
    # (including the SystemMessage) and the history used for any given chat message (which may be a
    # subset of that history) and the corresponding metadata of the message (# of tokens, cost)
    # In other words, we can't regenerate the cost based on the entire list of messages
    # because that doesn't show the entire prompt/message that was sent to ChatGPT (i.e. it only
    # shows our question, not the context)
    if 'message_chain' not in st.session_state:
        # list of langhain Messages
        st.session_state.message_chain = sh._create_mock_message_chain()
    if 'chat_history' not in st.session_state:
        # list of ChatMetaData
        st.session_state.chat_history = sh._create_mock_chat_history(message_chain=st.session_state.message_chain)  # noqa

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

        st.markdown("# Prompt Template")
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

        st.markdown("# History")


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
        chat_history = st.session_state.get('chat_history', [])
        result_placeholder = st.empty()


    sh.display_horizontal_line()

    if submit_button and user_input:
        human_message = HumanMessage(content=user_input)
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
            history = [st.session_state.message_chain[0]] + [st.session_state.message_chain[-1]]
            response = chat(history)

        chat_data = sh.ChatMetaData(
            model_name=model_name,
            human_question=human_message,
            full_question=' '.join([x.content for x in history]),
            ai_response=response,
        )
        st.session_state.chat_history.append(chat_data)
        st.session_state.message_chain.append(human_message)
        st.session_state.message_chain.append(response)
        message_state['chat_message'] = user_input  # Update session state value

    # display message history
    chat_history = st.session_state.get('chat_history', [])
    chat_history = list(reversed(chat_history))
    for chat in chat_history:
        col_messages, col_totals = st.columns([5, 1])
        with col_messages:
            sh.display_chat_message(chat.ai_response.content, is_human=False)
            sh.display_chat_message(chat.human_question.content, is_human=True)
        with col_totals:
            sh.display_totals(
                cost=chat.cost,
                total_tokens=chat.total_tokens,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )

    sh.display_totals(
        cost=sum(x.cost for x in chat_history),
        total_tokens=sum(x.total_tokens for x in chat_history),
        prompt_tokens=sum(x.prompt_tokens for x in chat_history),
        completion_tokens=sum(x.completion_tokens for x in chat_history),
        round_cost_digits=4,
        placeholder=result_placeholder,
    )

    print("--------------END--------------")

if __name__ == '__main__':
    main()
