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
            # st.sidebar.markdown(f'<div class="sidebar-text">{prompt_example}</div>', unsafe_allow_html=True)


    print("A")

    col_1, col_2 = st.columns([5, 1])
    with col_1:
        user_input = st.text_area(
            "Send a message.",
            key='chat_message',
            value=message_state['chat_message'],
            placeholder="Ask a question.",
            height=150,
        )
        print(f"User Input: `{str(user_input)}`")
        submit_button = st.button("Submit")
    with col_2:
        cost_string = f"""
        <b>Total Cost</b>: <code>$1.00</code><br>
        """
        token_string = f"""
        Total Tokens: <code>1,000</code><br>
        Prompt Tokens: <code>1,000</code><br>
        Completion Tokens: <code>1,000</code><br>
        """
        cost_html = f"""
        <p style="font-size: 13px; text-align: right">{cost_string}</p>
        <p style="font-size: 12px; text-align: right">{token_string}</p>
        """
        st.markdown(cost_html, unsafe_allow_html=True)

    sh.display_horizontal_line()
    # handle user input
    if submit_button and user_input:
        print("B")
        print(f"`{str(user_input)}`")
        human_message = HumanMessage(content=user_input)
        st.session_state.messages.append(human_message)
        # message(human_message.content, is_user=True, key=str(i) + '_user')
        with st.spinner("Thinking..."):
            print("CALLING CHATGPT")
            print(f"`{str(user_input)}`")
            print(st.session_state.messages[-1])
            # TODO: pass all messages and/or figure out memory buffer strategy
            if openai_model == 'GPT-3.5':
                model_name = 'gpt-3.5-turbo'
            elif openai_model == 'GPT-4':
                model_name = 'gpt-4'
            else:
                raise ValueError(openai_model)

            chat = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
            response = chat([st.session_state.messages[0]] + [st.session_state.messages[-1]])
        st.session_state.messages.append(response)
        message_state['chat_message'] = user_input  # Update session state value
        print(f"updating state - 2: {message_state}")


    print("C")
    # display message history
    messages = st.session_state.get('messages', [])
    messages = list(reversed(messages))
    print(messages)
    print(f"len: {len(messages)}")
    for i in range(0, len(messages) - 1, 2):
        print(f"i: {i}")

        human_message = messages[i + 1]
        print(f"human: {human_message}")
        assert isinstance(human_message, HumanMessage)
        ai_message = messages[i]
        assert isinstance(ai_message, AIMessage)
        col_1, col_2 = st.columns([5, 1])
        with col_1:
            sh.display_chat_message(ai_message.content, is_human=False)
            sh.display_chat_message(human_message.content, is_human=True)
        with col_2:
            cost_string = f"""
            <b>Message Cost</b>: <code>$1.00</code><br>
            """
            token_string = f"""
            Message Tokens: <code>1,000</code><br>
            Prompt Tokens: <code>1,000</code><br>
            Completion Tokens: <code>1,000</code><br>
            """
            cost_html = f"""
            <p style="font-size: 13px; text-align: right">{cost_string}</p>
            <p style="font-size: 12px; text-align: right">{token_string}</p>
            """
            st.markdown(cost_html, unsafe_allow_html=True)
            # st.markdown("<style>p { font-size: 11px; padding: 1px;}</style>", unsafe_allow_html=True)
            # st.markdown(cost_string)

            # st.markdown(cost_string)
            # st.markdown("Totat Tokens: `1,000`")
            # st.markdown("Prompt Tokens: 1,000")
            # st.markdown("Completion Tokens: 1,000")
            # st.markdown("Total Cost: $1.00")

    # for i, msg in reversed(list(enumerate(messages[1:]))):
        # col_1, col_2 = st.columns([10, 1])
        # with col_1:
        #     display_chat_message(msg.content, isinstance(msg, HumanMessage))
        # with col_2:
        #     st.markdown("Totat Tokens: 1,000")
        #     st.markdown("Prompt Tokens: 1,000")
        #     st.markdown("Completion Tokens: 1,000")
        #     st.markdown("Total Cost: $1.00")
        # st.markdown(msg.content)
        # st.markdown('---')
        # if i % 2 == 0:
        #     message(msg.content, is_user=True, key=str(i) + '_user')
        # else:
        #     message(msg.content, is_user=False, key=str(i) + '_ai')
        # message(response.content, is_user=False, key=str(i) + '_ai')
        
        # st.experimental_rerun()

    print("--------------END--------------")

if __name__ == '__main__':
    main()
