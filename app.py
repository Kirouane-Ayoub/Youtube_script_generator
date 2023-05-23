from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st 
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

tab1 , tab2 = st.tabs(["Home" , "APP"])
with tab1 : 
    st.header("Youtube script generator:")
    st.image("home.png")
    st.write("""
        This system revolutionizes YouTube script creation using the "flan-t5-large" LLM model,
        LangChain, and integrated Wikipedia research. 
        It empowers creators to effortlessly generate engaging, high-quality scripts tailored for YouTube.
        By connecting the LLM model with diverse data sources through LangChain's data awareness,
        the system enriches scripts with accurate and up-to-date information. 
        The seamless integration of Wikipedia research enhances content comprehensiveness and reliability.
        This project showcases the potential of combining advanced language models and frameworks to streamline script-making, providing intelligent
        and data-driven automation. Prepare for a new era of efficient YouTube script creation with this cutting-edge system.
    """)

with st.sidebar:
    st.image("icon.png" , width=200)
    prompt = st.text_input("Entre Your Prompt here : ")
with tab2 : 
    with st.spinner('Wait for it...'):
        model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        pipe = pipeline(
            "text2text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=100
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)

        title_template = PromptTemplate(
            input_variables = ['topic'], 
            template='write me a youtube video title about {topic}'
        )

        script_template = PromptTemplate(
            input_variables = ['title', 'wikipedia_research'], 
            template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
        )

        title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
        script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

        title_chain = LLMChain(llm=local_llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
        script_chain = LLMChain(llm=local_llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

        wiki = WikipediaAPIWrapper()

        if prompt : 
            title = title_chain.run(prompt)
            wiki_research = wiki.run(prompt) 
            script = script_chain.run(title=title, wikipedia_research=wiki_research)

        with st.expander('Research , Title , Memory history : '):

            st.header("Wikipedia research : ")
            st.text(wiki_research)
            st.header("Title : ")
            st.write(title_memory.buffer)
            st.header("Script : ")
            st.write(script_memory.buffer)
    st.success('Done!')
