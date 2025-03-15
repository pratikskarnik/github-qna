// src/App.jsx
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [repoUrl, setRepoUrl] = useState('');
  const [repos, setRepos] = useState([]);
  const [selectedRepo, setSelectedRepo] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [processingRepo, setProcessingRepo] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Fetch available repositories on component mount
  useEffect(() => {
    fetchRepos();
  }, []);

  const fetchRepos = async () => {
    try {
      const response = await fetch('http://localhost:8000/repos');
      const data = await response.json();
      setRepos(data.repos);
    } catch (error) {
      console.error('Error fetching repos:', error);
      setErrorMessage('Failed to fetch repositories');
    }
  };

  const handleProcessRepo = async (e) => {
    e.preventDefault();
    if (!repoUrl) {
      setErrorMessage('Please enter a GitHub repository URL');
      return;
    }

    setProcessingRepo(true);
    setErrorMessage('');

    try {
      const response = await fetch('http://localhost:8000/process-repo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ repo_url: repoUrl }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process repository');
      }

      const data = await response.json();
      setRepoUrl('');
      fetchRepos();
      setSelectedRepo(data.repo_name);
    } catch (error) {
      console.error('Error processing repo:', error);
      setErrorMessage(error.message);
    } finally {
      setProcessingRepo(false);
    }
  };

  const handleAskQuestion = async (e) => {
    e.preventDefault();
    if (!selectedRepo) {
      setErrorMessage('Please select a repository first');
      return;
    }
    if (!question) {
      setErrorMessage('Please enter a question');
      return;
    }

    setLoading(true);
    setErrorMessage('');

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_name: selectedRepo,
          question: question,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get answer');
      }

      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      console.error('Error asking question:', error);
      setErrorMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-900 mb-8">
          GitHub Repository QnA System
        </h1>

        {errorMessage && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {errorMessage}
          </div>
        )}

        <div className="bg-white shadow-md rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Process a GitHub Repository</h2>
          <form onSubmit={handleProcessRepo} className="flex flex-col space-y-4">
            <div>
              <label htmlFor="repoUrl" className="block text-sm font-medium text-gray-700 mb-1">
                GitHub Repository URL
              </label>
              <input
                id="repoUrl"
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/username/repository"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <button
              type="submit"
              disabled={processingRepo}
              className="bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:bg-indigo-400"
            >
              {processingRepo ? 'Processing...' : 'Process Repository'}
            </button>
          </form>
        </div>

        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Ask a Question</h2>

          <div className="mb-4">
            <label htmlFor="repoSelect" className="block text-sm font-medium text-gray-700 mb-1">
              Select Repository
            </label>
            <select
              id="repoSelect"
              value={selectedRepo}
              onChange={(e) => setSelectedRepo(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="">Select a repository</option>
              {repos.map((repo) => (
                <option key={repo} value={repo}>
                  {repo}
                </option>
              ))}
            </select>
          </div>

          <form onSubmit={handleAskQuestion} className="flex flex-col space-y-4">
            <div>
              <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-1">
                Your Question
              </label>
              <textarea
                id="question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What does this repository do? How does a specific feature work?"
                rows="3"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:bg-indigo-400"
            >
              {loading ? 'Loading...' : 'Ask Question'}
            </button>
          </form>

          {answer && (
            <div className="mt-6">
              <h3 className="font-medium text-lg mb-2">Answer:</h3>
              <div className="bg-gray-50 p-4 rounded-md whitespace-pre-wrap">{answer}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;