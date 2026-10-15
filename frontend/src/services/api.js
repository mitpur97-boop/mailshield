/**
 * MailShield API Service
 * Handles all communication with the FastAPI backend.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Classify a message as spam or not spam.
 * @param {string} message - The message text to classify
 * @returns {Promise<Object>} Prediction result
 */
export async function classifyMessage(message) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(error.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch model information and metrics from the backend.
 * @returns {Promise<Object>} Model info
 */
export async function getModelInfo() {
  const response = await fetch(`${API_URL}/model-info`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(error.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Check backend health status.
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_URL}/health`);
  return response.json();
}
