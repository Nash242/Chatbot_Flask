from flask import Flask, render_template, request, jsonify
from langchain.chains.question_answering import load_qa_chain

import os, re
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain.document_loaders import PyPDFLoader
import csv
import time

app = Flask(__name__)


def load_pdf(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            return pages

def list1(pages):
    l1 = []
    for i in pages:
        data = i.page_content
        l1.append(data)
    print(l1)
    return l1

def fetch_questions(l1):
    questions = []
    pattern = r'Q\d+\..*?\?'

    # Extract questions
    for text in l1:
        matches = re.findall(pattern, text, re.DOTALL)
        questions.extend(matches)

    # # Print questions
    # for question in questions:
    #     print(question.strip())
    return questions

def get_pdf_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=20,
        length_function=len,
        separators=['\nQ','\n\n','\n']
    )
    chunks = text_splitter.create_documents(documents)
    print(chunks)
    return chunks

def fetch_answers(l1):
    pattern = r'Answer:.*?(?=Q\d+\.|$)'
    answers = []

    # Extract answers
    for text in l1:
        matches = re.findall(pattern, text, re.DOTALL)
        answers.extend(matches)

    return answers

def m_chunks(chunks, answers):
    for chunk, answer in zip(chunks, answers):
        chunk_metadata = {"answer": answer}
        chunk.metadata = chunk_metadata
    return chunks

def persist(chunks, user_question):
    embeddings = OpenAIEmbeddings()
    persist_directory = "temp"
    retriever2 = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    retriever2.persist()
    quest = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    #quest = retriever2.similarity_search_with_score(user_question,k=1)
    return quest

def rerank_top_n(query,output,n_chunks):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    input_lst=[]
    for chunk in output:
        tup=(query,chunk[0].page_content)
        input_lst.append(tup)
    if len(input_lst) == 0:
        return []
    scores = model.predict(input_lst)
    total_data=zip(output,scores)
    reranked = sorted(total_data, key=lambda x: x[1],reverse=True)
    try:
        return reranked[:n_chunks]
    except:
        print(f"Value of n_chunks is greater than the number of input chunks. Reduce the number of n_chunks.")
        return reranked


#================================= Text File =====================================================

def load_docs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(content)
    return documents


def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.create_documents(documents)
    return chunks


def create_vectors(persist_directory, directory, docs):
    embeddings = OpenAIEmbeddings()
    #embeddings = AzureOpenAIEmbeddings(azure_deployment="test2", open_api_version="2023-05-15")
    processed_files = {}
    processed_files_path = "processed.txt"
    
    if os.path.exists(persist_directory):
        if os.path.exists(processed_files_path):
            with open(processed_files_path, "r") as file:
                for line in file:
                    file_name, modification_time = line.strip().split(":")
                    processed_files[file_name] = float(modification_time)

        current_files = os.listdir(directory)
        new_files = []
        for file in current_files:
            file_path = os.path.join(directory, file)
            if file not in processed_files or os.path.getmtime(file_path) > processed_files[file]:
                new_files.append(file)

        if new_files:
            print("New or modified files detected:", new_files)
            documents = load_docs(directory)
            docs = split_docs(documents)

            # Create or fetch Chroma vector database
            vectors = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
            vectors.persist()
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

            # Update processed files list with modification times
            with open(processed_files_path, "w") as file:
                for file_name in new_files:
                    file_path = os.path.join(directory, file_name)
                    modification_time = os.path.getmtime(file_path)
                    processed_files[file_name] = modification_time
                    file.write(f"{file_name}:{modification_time}\n")

            return vectordb
        else:
            print("No new or modified files detected. Using existing vector database.")
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectordb
    else:
        os.makedirs(persist_directory)
        
        # Create Chroma vector database
        vectors = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
        vectors.persist()
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Create and initialize processed.txt
        with open(processed_files_path, "w") as file:
            file.write("")
        
        return vectordb

# Route for GET requests
@app.route('/', methods=['GET','POST'])
def get_route():
    return render_template('index.html')


@app.route('/getanswer', methods=['POST','GET'])
def post_route():
    try:
        if request.method == 'POST':
            que=request.form['question']
            print(que)
            data=runmain(que)
            print(20*'-')
            print(data)
            return jsonify({'message': data, "status":"success"})
    except Exception as e:
        return jsonify({'message': e, "status":"fail"})


@app.route('/dislikeresponse', methods=['POST'])
def write_to_first_csv():
    usermassage = request.form['usermassage']
    botresponse = request.form['botresponse']
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    csv_file_path = "thumbs_down_log.csv"
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        fieldnames = ['Timestamp', 'Question', 'Answer']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({'Timestamp': timestamp, 'Question': usermassage, 'Answer': botresponse})
    return jsonify({'msg':'success'})

@app.route('/likeresponse', methods=['POST'])
def write_to_second_csv():
    usermassage = request.form['usermassage']
    botresponse = request.form['botresponse']
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    csv_file_path = "thumbs_up_log.csv"
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        fieldnames = ['Timestamp', 'Question', 'Answer']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({'Timestamp': timestamp, 'Question': usermassage, 'Answer': botresponse})
    return jsonify({'msg':'success'})


def runmain(user_que):
    print('in runMAIN')
    os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"
    directory = "contents"
    persist_directory = "chroma_db"    
    documents = load_docs(directory)
    docs = split_docs(documents)
    vectordb = create_vectors(persist_directory, directory, docs)
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.1, model=llm_model)    
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    user_question =user_que #user input
    if user_question:
        if (user_question.lower().strip() in ['hi','hello','good morning','good afternoon','good evening','Hi']):
            question = user_question
        else:
            question = user_question + " Answer in step by step points only"
        embeddings = OpenAIEmbeddings()
        directory1_path = "temp"
        try:
            if os.path.isdir(directory1_path):
                db3 = Chroma(persist_directory=directory1_path, embedding_function=embeddings)
                quest = db3.similarity_search_with_score(user_question,k=3)
                print(quest)
                top = rerank_top_n(user_question,quest,1)
                print("======================",top)
                if top[0][1]*10 > 60:
                    return top[0][0][0].metadata["answer"]
                else:
                        if (user_question.lower().strip() in ['hi','hello','good morning','good afternoon','good evening']):
                            question = user_question
                        else:
                            question = user_question + " Answer in step by step points only"
                        #refine_question = question + " Answer in step by step points only"
                        matching_docs = vectordb.similarity_search(question)
                        input_data = {'question': question, 'input_documents': matching_docs}
                        #prompt = ChatPromptTemplate.from_template("Answer the the {question} in step by step points from the {matching_docs} only")
                        answer = qa_chain.invoke(input=input_data)
                        print(answer)
                        if " Answer in step by step points only" in question:
                            user_que=question.split(" Answer in step by step points only")[0]
                        else:
                            return answer['output_text']
                        unanswered_phrases = ["I don't know.", "I don't have much information", "I don't have any information", "I'm unable to provide an answer","I don't have enough information to answer that question.", "I don't have information about in the provided context."]  # Add more phrases if needed
                        if any(phrase in answer.get('output_text', '') for phrase in unanswered_phrases):
                            unanswered_question = f"Unanswered question: {question}\n"
                            log_file_path = "log.txt"
                            with open(log_file_path, "a") as log_file:
                                log_file.write(unanswered_question)
                            return "Sorry, I'm unable to provide an answer please rephrase your question ."
                        return answer['output_text']
                                    #print(f"Source:{document.metadata['source']}")
            else:
                pages = load_pdf(directory)
                l1 = list1(pages)
                questions = fetch_questions(l1)
                chunks = get_pdf_chunks(questions)
                answers = fetch_answers(l1)
                chunk_wm = m_chunks(chunks, answers)
                quest = persist(chunk_wm, user_question)
                db1 = quest.similarity_search_with_score(user_question,k=3)
                top = rerank_top_n(user_question,db1,1)
                if top[0][1]*10 > 60:
                    print("Inside if")
                    return top[0][0][0].metadata["answer"]
                else:
                    if (user_question.lower().strip() in ['hi','hello','good morning','good afternoon','good evening']):
                        question = user_question
                    else:
                        question = user_question + " Answer in step by step points only"
                    #refine_question = question + " Answer in step by step points only"
                    matching_docs = vectordb.similarity_search(question, k=3)
                    input_data = {'question': question, 'input_documents': matching_docs}
                    #prompt = ChatPromptTemplate.from_template("Answer the the {question} in step by step points from the {matching_docs} only")
                    answer = qa_chain.invoke(input=input_data)
                    if " Answer in step by step points only" in question:
                        user_que=question.split(" Answer in step by step points only")[0]
                    else:
                        return answer['output_text']
                    unanswered_phrases = ["I don't know.", "I don't have much information", "I don't have any information", "I'm unable to provide an answer","I don't have enough information to answer that question.", "I don't have information about in the provided context."]  # Add more phrases if needed
                    if any(phrase in answer.get('output_text', '') for phrase in unanswered_phrases):
                        unanswered_question = f"Unanswered question: {question}\n"
                        log_file_path = "log.txt"
                        with open(log_file_path, "a") as log_file:
                            log_file.write(unanswered_question)
                        return "Sorry, I'm unable to provide an answer please rephrase your question ."
                    return answer['output_text']
        except Exception as e:
            return (f"Failure {e}")




if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(e)
