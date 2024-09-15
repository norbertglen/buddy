require('dotenv').config();
const fs = require('fs');
const readline = require('readline');
const axios = require('axios');

const openaiApiKey = process.env.OPENAI_API_KEY;

async function getEmbedding(text) {
  const response = await axios.post(
    'https://api.openai.com/v1/embeddings',
    { input: text, model: 'text-embedding-ada-002' },
    {
      headers: {
        Authorization: `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
    }
  );
  return response.data.data[0].embedding;
}

async function generateOpenAIResponse(userQuery, documentText) {
  const response = await axios.post(
    'https://api.openai.com/v1/chat/completions',
    {
      model: 'gpt-4',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        {
          role: 'user',
          content: `Based on the following document, answer the question: "${userQuery}"\n\nDocument: ${documentText}`,
        },
      ],
    },
    {
      headers: {
        Authorization: `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
    }
  );
  return response.data.choices[0].message.content;
}

function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((acc, val, idx) => acc + val * vecB[idx], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((acc, val) => acc + val * val, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((acc, val) => acc + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

function readFilesFromFolder(folderPath) {
  const files = fs.readdirSync(folderPath);
  const fileContents = files.map((file) => {
    const filePath = `${folderPath}/${file}`;
    const content = fs.readFileSync(filePath, 'utf8');
    return { filename: file, content };
  });
  return fileContents;
}

async function createDocumentsFromFiles(fileContents) {
  const documents = fileContents.map((file) => ({
    id: file.filename,
    text: file.content,
  }));

  const embeddings = await Promise.all(
    documents.map(async (doc) => {
      const embedding = await getEmbedding(doc.text);
      return { doc, embedding };
    })
  );

  return embeddings;
}

async function queryAssistant(userQuery, documentEmbeddings) {
  const queryEmbedding = await getEmbedding(userQuery);

  const similarities = documentEmbeddings.map(({ doc, embedding }) => {
    const similarity = cosineSimilarity(queryEmbedding, embedding);
    return { doc, similarity };
  });

  similarities.sort((a, b) => b.similarity - a.similarity);
  const mostRelevantDoc = similarities[0].doc;

  return await generateOpenAIResponse(userQuery, mostRelevantDoc.text);
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'Ask your question: ',
});

const folderPath = './data'; 
let documentEmbeddings = [];

(async () => {
  console.log('Loading documents and creating embeddings...');
  const fileContents = readFilesFromFolder(folderPath);
  documentEmbeddings = await createDocumentsFromFiles(fileContents);

  console.log('Embeddings are ready! You can now start asking questions.');
  rl.prompt();
})();

rl.on('line', async (line) => {
  const userQuery = line.trim();

  if (userQuery) {
    const response = await queryAssistant(userQuery, documentEmbeddings);
    console.log(`Answer: ${response}`);
  } else {
    console.log('Please ask a question.');
  }

  rl.prompt(); // Prompt the user to ask another question
}).on('close', () => {
  console.log('Goodbye!');
  process.exit(0);
});
