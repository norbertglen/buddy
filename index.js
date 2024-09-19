const env = require('dotenv');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { OpenAIEmbeddings, ChatOpenAI } = require('@langchain/openai');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const pdf = require("pdf-parse")

env.config();
const openaiApiKey = process.env.OPENAI_API_KEY;

async function extractTextFromPDF(filePath) {
    try {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdf(dataBuffer);
        return data.text;
    } catch (error) {
        console.error(`Error reading PDF file ${filePath}:`, error);
        return '';
    }
}

async function getEmbeddingsFromFiles(directoryPath) {
    const files = fs.readdirSync(directoryPath);
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    });

    let allDocuments = [];

    for (const file of files) {
        const filePath = path.join(directoryPath, file);
        let content = '';

        try {
            if (path.extname(file) === '.pdf') {
                // Handle PDF files
                content = await extractTextFromPDF(filePath);
            } else {
                // Handle text files
                content = fs.readFileSync(filePath, 'utf-8');
            }

            const document = { pageContent: content, metadata: { filename: file } };
            const splits = await textSplitter.splitDocuments([document]);
            allDocuments.push(...splits);
        } catch (error) {
            console.error(`Error processing file ${filePath}:`, error);
        }
    }

    const vectorStore = await MemoryVectorStore.fromDocuments(allDocuments, new OpenAIEmbeddings());
    return vectorStore.asRetriever({ k: 6, searchType: 'similarity' });
}

async function generateAnswer(question, context) {
    const llm = new ChatOpenAI({
        apiKey: openaiApiKey,
        temperature: 0.7,
    });

    const prompt = `You are a smart assistant helping answer user questions based on the context you are given.
  Here is the context you are provided with:
  \n ------- \n
  ${context} 
  \n ------- \n
  Here is the user question: ${question}
  If you cannot find the answer from the context above, just say you don't know.
  `;

    try {
        const response = await llm.invoke(prompt);
        return response.text || 'No answer received.';
    } catch (error) {
        console.error('Error generating answer:', error);
        return 'Sorry, there was an error generating the answer.';
    }
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function askQuestion(prompt) {
    return new Promise(resolve => rl.question(prompt, resolve));
}

async function startRAGSystem() {
    const directoryPath = './data';
    const retriever = await getEmbeddingsFromFiles(directoryPath);

    let question = await askQuestion('Ask a question: ');

    while (question.toLowerCase() !== 'exit') {
        try {
            const result = await retriever.invoke(question);
            if (result && result.length > 0) {
                const context = result.map(doc => doc.pageContent).join('\n\n');  // Join contexts from multiple documents
                const answer = await generateAnswer(question, context);
                console.log(`Answer: ${answer}`);
            } else {
                console.log("Sorry, couldn't find relevant information.");
            }
        } catch (error) {
            console.error('Error retrieving context:', error);
            console.log("Sorry, couldn't retrieve context.");
        }

        question = await askQuestion('\nAsk another question or type "exit" to quit: ');
    }

    rl.close();
}

startRAGSystem();
