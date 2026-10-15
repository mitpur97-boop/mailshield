import { useState, useRef } from 'react';
import { Search, Trash2, Loader2, AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import { classifyMessage } from '../services/api';
import './Classifier.css';

const EXAMPLE_MESSAGES = [
  {
    label: 'Obvious Spam',
    text: 'WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.',
  },
  {
    label: 'Legitimate',
    text: 'Hey, are we still meeting at the coffee shop at 3pm today? Let me know if you need a ride.',
  },
  {
    label: 'Promotional',
    text: 'Hi! We noticed you haven\'t visited in a while. Come back and enjoy 20% off your next order. Use code WELCOME20 at checkout.',
  },
  {
    label: 'Ambiguous',
    text: 'Your account security requires immediate attention. Please verify your details to continue using our service.',
  },
];

const MAX_CHARS = 5000;

export default function Classifier() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const textareaRef = useRef(null);
  const requestRef = useRef(false);

  const handleAnalyze = async () => {
    if (!message.trim() || loading || requestRef.current) return;
    requestRef.current = true;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = await classifyMessage(message);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze message. Is the backend running?');
    } finally {
      setLoading(false);
      requestRef.current = false;
    }
  };

  const handleClear = () => {
    setMessage('');
    setResult(null);
    setError('');
    textareaRef.current?.focus();
  };

  const handleExample = (text) => {
    setMessage(text);
    setResult(null);
    setError('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAnalyze();
    }
  };

  const isSpam = result?.prediction === 'SPAM';

  return (
    <section id="classifier" className="classifier-section">
      <div className="section-container">
        <h2 className="section-title">Message Classifier</h2>
        <p className="section-subtitle">Paste a message below to analyze it for spam</p>

        {/* Example Messages */}
        <div className="examples">
          <span className="examples-label">Try an example:</span>
          <div className="examples-list">
            {EXAMPLE_MESSAGES.map((ex, i) => (
              <button key={i} className="example-btn" onClick={() => handleExample(ex.text)}>
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        {/* Input Area */}
        <div className="input-card">
          <textarea
            ref={textareaRef}
            className="message-input"
            placeholder="Paste an email or message here..."
            value={message}
            onChange={(e) => setMessage(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={handleKeyDown}
            rows={6}
            maxLength={MAX_CHARS}
          />
          <div className="input-footer">
            <span className="char-count">{message.length} / {MAX_CHARS}</span>
            <div className="input-actions">
              {message && (
                <button className="btn btn-clear" onClick={handleClear}>
                  <Trash2 size={16} /> Clear
                </button>
              )}
              <button
                className="btn btn-analyze"
                onClick={handleAnalyze}
                disabled={!message.trim() || loading}
              >
                {loading ? (
                  <><Loader2 size={16} className="spin" /> Analyzing...</>
                ) : (
                  <><Search size={16} /> Analyze Message</>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="error-card">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={`result-card ${isSpam ? 'result-spam' : 'result-ham'}`}>
            <div className="result-header">
              {isSpam ? (
                <><AlertTriangle size={28} /> <span className="result-label">SPAM DETECTED</span></>
              ) : (
                <><CheckCircle2 size={28} /> <span className="result-label">NOT SPAM</span></>
              )}
            </div>
            <p className="result-confidence">
              {(result.confidence * 100).toFixed(1)}% confidence
            </p>

            {/* Probability Bars */}
            <div className="prob-bars">
              <div className="prob-row">
                <span className="prob-label">Spam</span>
                <div className="prob-track">
                  <div
                    className="prob-fill prob-spam"
                    style={{ width: `${result.spam_probability * 100}%` }}
                  />
                </div>
                <span className="prob-value">{(result.spam_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-row">
                <span className="prob-label">Legit</span>
                <div className="prob-track">
                  <div
                    className="prob-fill prob-ham"
                    style={{ width: `${result.ham_probability * 100}%` }}
                  />
                </div>
                <span className="prob-value">{(result.ham_probability * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Contributing Words */}
            {result.top_contributing_words && result.top_contributing_words.length > 0 && (
              <div className="contributing-words">
                <div className="cw-header">
                  <Info size={16} />
                  <span>Key words influencing this prediction:</span>
                </div>
                <div className="cw-list">
                  {result.top_contributing_words.map((w, i) => (
                    <span key={i} className="cw-tag">{w.word}</span>
                  ))}
                </div>
              </div>
            )}

            <p className="privacy-note">
              Messages are analyzed for classification and are not stored by this application.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
