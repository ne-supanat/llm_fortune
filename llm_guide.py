import time
from convert_pdf_to_collection import get_db
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda


def main():
    db = get_db()
    llm = ChatOllama(model="phi3:mini", verbose=True)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    
    {context}
    
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    retriever = MultiQueryRetriever.from_llm(
        db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT,
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    def inspect(state):
        print("\n--- context ---\n")
        print(state["context"].replace("\t", " ").replace("\n\n", "\n\n---\n\n"))
        return state

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(inspect)
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        user_input = input(
            "Ask a query about your documents (or type 'quit' to exit): "
        )
        if user_input.lower() == "quit":
            break

        start_time = time.time()

        response = chain.invoke(user_input)

        print()

        print("\n--- response ---\n")
        print(response)

        print("\n--- %s seconds ---\n" % (time.time() - start_time))


main()
