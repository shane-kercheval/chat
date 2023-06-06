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


import streamlit as st
#Known issues;
#Once you modify the chat message, the prompt template message will know fill the text-box
# Steps: fill out fields in prompt template; hit "Create Message"; Chat message is updated;
# change one or more fields; hit "Create Message"; Chat message is still updated; Modify chat
# message; now click "Create Message" from prompt template; Chat message will not update;
# however, if you change one or more of field values in the prompt template and hit Create Message
# then the chat message will update as expected.

    # setup streamlit page
st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="🤖",
        layout='wide',
    )

def display_horizontal_line():
    css = """
    <style>
        .line {
            border: none;
            border-top: 1px solid #E6E6E6;
            margin: 10px 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<div class='line'></div>", unsafe_allow_html=True)


def display_chat_message(message, is_sender):
    sender_emoji = "👨‍💼" if is_sender else "🤖"
    sender_class = "sender" if is_sender else "receiver"

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown(f"<div class='emoji'>{sender_emoji}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='{sender_class}'>{message}</div>", unsafe_allow_html=True)

    # st.markdown(f"<div class='{sender_class}'><span class='emoji'>{sender_emoji}</span> {message}</div>", unsafe_allow_html=True)
    # st.markdown(f"<div class='{sender_class}'>{message}</div>", unsafe_allow_html=True)

# # Sample messages
# message = """
# This is some python

# ```
# def python():
#     return True
# ```
# """
# messages = [
#     {"sender": "John", "message": "asdf"},
#     {"sender": "Jane", "message": message},
#     {"sender": "John", "message": "I'm good, thanks!"},
#     {"sender": "Jane", "message": "That's great to hear!"},
#     {"sender": "John", "message": "Yeah, it's been a good day."},
# ]

# # Displaying the chat messages
# for message in reversed(messages):
#     display_chat_message(message["message"], message["sender"] == "John")
#     # display_horizontal_line()



# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")



message = """
This is some python:

```
def python():
    return True
```

It is code.
"""

def main():
    init()

    css_messages = f"""
    <style>
        .sender {{
            background-color: #d8f0ff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .receiver {{
            background-color: #F0F0F0;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .emoji {{
            font-size: 24px;
            margin-bottom: 5px;
        }}
        .stCodeBlock {{
            margin: 20px 0;
        }}
    </style>
    """
    st.markdown(css_messages, unsafe_allow_html=True)

    st.markdown(
        '<style>section[data-testid="stSidebar"]{width: 500px !important;}</style>',
        unsafe_allow_html=True,
    )

    # Apply custom CSS style to the sidebar
    st.markdown(
        """
        <style>
        div[data-testid="stText"] {
            background-color: white;
            padding: 10px;
            white-space: pre-wrap;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        # Add custom CSS to hide the label of the prompt-template selection box in the sidebar
        """
        <style>
        section[data-testid="stSidebar"] .stSelectbox label {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        # Add custom CSS to hide the label of the chat message
        '<style>section.main div.stTextArea label { display: none;}</style>',
        unsafe_allow_html=True,
    )

    state = st.session_state.setdefault('state', {'chat_message': ''})
    print(f"state: {state}")

    with st.sidebar:
        st.markdown('# Options')
        openai_model = st.selectbox(label="Model", options=('GPT-3.5', 'GPT-4'))
        temperature = st.slider(
            label="Temperature",
            min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        )
        max_tokens = st.slider(
            label="Temperature",
            min_value=100, max_value=3000, value=1000, step=100,
        )

        st.markdown('# Prompt Template')

        template_option = st.selectbox(
            '<do not display>',
            (
                '<Select>',
                'Make text sound better.',
                'Summarize text.',
                'Code: Write doc strings',
            ),
        )
        if template_option != '<Select>':
            prompt_example = """
            This is a prompt.

            It has fields like this {{context}}

            And this:

            ```
            {{more_context}}
            ```
            """

            # Regular expression pattern to match values within double brackets
            pattern = r"\{\{(.*?)\}\}"

            import re
            # Find all matches of the pattern in the text
            matches = re.findall(pattern, prompt_example, re.DOTALL)

            # Create a dictionary to store the extracted values
            prompt_field_values = []

            # Create the sidebar text areas
            with st.form(key="my_form"):
                st.markdown("### Fields")
                st.markdown("Fill in the information for the following fields referred to in the prompt template, shown below.")
                for match in matches:
                    prompt_field_values.append((match, st.text_area(match, height=100, key=match)))
                    # prompt_field_values.append((match, st.sidebar.text_area(match, height=100)))
                create_message_submit = st.form_submit_button(label='Create message')

            # Create the button to extract the text
            if create_message_submit:
                # extracted_values = {key: values_dict[key] for key in values_dict if values_dict[key]}

                chat_gpt_message = prompt_example.strip()
                for key, value in prompt_field_values:
                    chat_gpt_message = chat_gpt_message.replace("{{" + key + "}}", value)
                
                    # Update contents of TextArea B with contents of TextArea A
                print("updating chat_message state")
                state['chat_message'] = chat_gpt_message
                print(f"state - 2: `{state}`")
                

                # st.sidebar.write("Extracted Values:")
                # st.sidebar.write(extracted_values)

            st.markdown('### Template:')
            st.sidebar.text(prompt_example)
            # st.sidebar.markdown(f'<div class="sidebar-text">{prompt_example}</div>', unsafe_allow_html=True)


    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content='The capital of France is Paris.', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is 1 + 1?"),
            AIMessage(content='2', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is 2 + 2?"),
            AIMessage(content='4', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content='The capital of France is Paris.', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is 1 + 1?"),
            AIMessage(content='2', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is 2 + 2?"),
            AIMessage(content='4', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content='The capital of France is Paris.', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is 1 + 1?"),
            AIMessage(content='2', additional_kwargs={}, example=False),  # noqa
            HumanMessage(content="What is 2 + 2?"),
            AIMessage(content=message, additional_kwargs={}, example=False),  # noqa
        ]

    # st.header("Your own ChatGPT 🤖")

    print("A")

    user_input = st.text_area("Send a message.", key='chat_message', value=state['chat_message'], placeholder="Ask a question.")
    print(f"User Input: `{str(user_input)}`")
    submit_button = st.button("Submit")
    display_horizontal_line()
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
            elif openai_model == 'GPT-3.5':
                model_name = 'gpt-4'
            else:
                raise ValueError(openai_model)

            chat = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
            response = chat([st.session_state.messages[0]] + [st.session_state.messages[-1]])
        st.session_state.messages.append(response)
        state['chat_message'] = user_input  # Update session state value
        print(f"updating state - 2: {state}")


    print("C")
    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in reversed(list(enumerate(messages[1:]))):
        display_chat_message(msg.content, isinstance(msg, HumanMessage))
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
