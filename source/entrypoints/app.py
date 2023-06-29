"""
Streamlit app that enables conversations with ChatGPT and additional features like prompt
templates.
"""
import os
from dotenv import load_dotenv
import streamlit as st
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage, AIMessage, SystemMessage
from llm_chain.base import ChatModel, Chain, Session, Value
from llm_chain.tools import DuckDuckGoSearch, split_documents, search_stack_overflow
from llm_chain.indexes import ChromaDocumentIndex
from llm_chain.prompt_templates import DocSearchTemplate

from llm_chain.models import OpenAIChat
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
    return OpenAIChat(
        model_name='gpt-3.5-turbo',
        # streaming_callback=lambda x: st.write(x.response),
    )

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

def main() -> None:
    """Defines the application structure and behavior."""
    initialize()
    sh.apply_css()
    message_state = st.session_state.setdefault('state', {'chat_message': ''})
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = Session()

    with st.sidebar:
        st.markdown('# Options')
        openai_model_name = st.selectbox(label="Model", options=('GPT-3.5', 'GPT-4'))
        # use_web_search = st.checkbox(label="Use Web Search (DuckDuckGo)", value=False)
        # use_stack_overflow = st.checkbox(label="Use Stack Overflow", value=False)

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

        # st.markdown("# Tools")
        # tool_names = [
        #     'Summarize PDF',
        #     'Summarize URL (Single Page)',
        #     'Agents??',
        # ]
        # template_name = st.selectbox(
        #     '<label should be hidden>',
        #     ['<Select>'] + tool_names,
        # )

        # st.markdown("# History")
        # conversation_history = sh._create_mock_history(num_history=5)
        # first_messages = [x.message_chain[1].content[0:40] for x in conversation_history]
        # st.radio("history", first_messages)

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

    # with col_search:
    #     tool_names = [
    #         '<Select an optional tool>',
    #         'Use Web-search (DuckDuckGo)',
    #         'Use Stack Overflow',
    #         'Summarize URL (Single Page) (TBD)',
    #         'Summarize PDF (TBD)',
    #         'Agents?? (TBD)',
    #     ]
    #     tool_selection = st.selectbox('<should not see this label>', tool_names)
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


    # display message history
    ##### history at point before we hit submit
    if submit_button and user_input:
        col_messages, col_totals = st.columns([5, 1])
        with col_messages:
            placeholder_response = st.empty()
            placeholder_prompt = st.empty()



    if chat_session.message_history:
        chat_history = list(reversed(chat_session.message_history))
        for chat in chat_history:
            col_messages, col_totals = st.columns([5, 1])
            with col_messages:
                sh.display_chat_message(chat.response, is_human=False)
                sh.display_chat_message(chat.prompt, is_human=True)
            with col_totals:
                sh.display_totals(
                    cost=chat.cost,
                    total_tokens=chat.total_tokens,
                    prompt_tokens=chat.prompt_tokens,
                    response_tokens=chat.response_tokens,
                    is_total=False,
                )

        # display totals for entire conversation
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

    if submit_button and user_input:
        # human_message = HumanMessage(content=user_input)
        # chat_model.message_chain.append(human_message)
        # prompt_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # sh.display_chat_message('TBD', is_human=False)
            # sh.display_chat_message(user_input, is_human=True)
            # TODO: need to set this in teh model
            if openai_model_name == 'GPT-3.5':
                model_name = 'gpt-3.5-turbo'
            elif openai_model_name == 'GPT-4':
                model_name = 'gpt-4'
            else:
                raise ValueError(openai_model_name)

            print(f"Calling ChatGPT: model={model_name}; temp={temperature}; max_tokens={max_tokens}")  # noqa

            # chat = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
            # TODO: pass all messages and/or figure out memory buffer strategy
            # history = [chat_model.message_chain[0]] + [chat_model.message_chain[-1]]
            # print("history: " + str(history))
            # response = chat(history)
            # response = AIMessage(content = sh._create_mock_message())
            # st.session_state['chat_message'] = ""
            chat_model = create_chat_model()
            chat_model.model_name = model_name
            chat_model.temperature = temperature
            chat_model.max_tokens = max_tokens
            # chat_model._streaming_callback = lambda x: st.markdown(f'<script>document.getElementById("{div_id}").innerHTML += "{x.response}";</script>', unsafe_allow_html=True)
            # chat_model._streaming_callback = lambda x: st.write(f'{div_id} - {x.response}', unsafe_allow_html=True)
            from llm_chain.models import StreamingRecord
            sh.display_chat_message(user_input, is_human=True, placeholder=placeholder_prompt)
            message = ""
            def temp(x: StreamingRecord):
                nonlocal message
                message += x.response
                sh.display_chat_message(message, is_human=False, placeholder=placeholder_response)
                # placeholder_asdf.empty()
                # placeholder_asdf.write(x.response)
                # sh.display_chat_message(div_id, is_human=False, div_id=div_id)
            chat_model._streaming_callback = temp

            # ddg_search = DuckDuckGoSearch(top_n=3)
            # use_web_search = tool_selection == 'Use Web-search (DuckDuckGo)'
            # use_stack_overflow = tool_selection == 'Use Stack Overflow'
            if use_web_search or use_stack_overflow:
                document_index = ChromaDocumentIndex(n_results=3)
                question_1 = Value()
                links = []
                if use_web_search:
                    links += [
                        question_1,
                        DuckDuckGoSearch(top_n=3),  # get the urls of the top search results
                        sh.scrape_urls,  # scrape the websites corresponding to the URLs
                        lambda docs: split_documents(docs=docs, max_chars=1_000),
                        document_index,  # add docs to doc-index
                    ]
                if use_stack_overflow:
                    links += [
                        question_1,
                        lambda search_query: search_stack_overflow(query=search_query, max_questions=2, max_answers=1),  # noqa
                        sh.stack_overflow_results_to_docs,
                        # we almost certainly want to keep the entire answer
                        lambda docs: split_documents(docs=docs, max_chars=2_500),
                        document_index,  # add docs to doc-index
                    ]

                links += [
                    question_1,
                    DocSearchTemplate(doc_index=document_index, n_docs=3),
                    chat_model,
                ]
            else:
                links = [chat_model]

            chat_session.append(chain=Chain(links=links))
            response = chat_session(user_input)
            # st.write(response)
            # st.markdown(
            #     f'<script>document.getElementById("{div_id}").innerHTML += "{response}";</script>',
            #     unsafe_allow_html=True,
            # )


            # if use_web_search:
            #     print(ddg_search.history[0])

        # chat_data = sh.MessageMetaData(
        #     model_name=model_name,
        #     human_question=human_message,
        #     full_question=' '.join([x.content for x in history]),
        #     ai_response=response,
        # )
        # chat_model.chat_history.append(chat_data)
        # chat_model.message_chain.append(response)
        # message_state['chat_message'] = user_input  # Update session state value
        # message_state['chat_message'] = ''


    print("--------------END----------ddddd----")

if __name__ == '__main__':
    main()
