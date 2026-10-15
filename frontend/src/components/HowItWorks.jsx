import { MessageSquareText, BarChart3, Brain, CheckCircle } from 'lucide-react';
import './HowItWorks.css';

const STEPS = [
  {
    icon: <MessageSquareText size={28} />,
    title: '1. Enter Message',
    description: 'Paste any email or text message you want to analyze into the classifier.',
  },
  {
    icon: <BarChart3 size={28} />,
    title: '2. Text to Features',
    description: 'Your message is converted into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency). This captures how important each word is relative to the training corpus.',
  },
  {
    icon: <Brain size={28} />,
    title: '3. Naive Bayes Classification',
    description: 'Multinomial Naive Bayes evaluates the features and calculates the probability that the message belongs to the spam or legitimate class, using Bayes\' theorem.',
  },
  {
    icon: <CheckCircle size={28} />,
    title: '4. Get Prediction',
    description: 'The model returns its prediction (Spam or Not Spam) along with a confidence score showing how certain it is about the classification.',
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="how-section">
      <div className="how-container">
        <h2 className="section-title">How It Works</h2>
        <p className="section-subtitle">A 4-step machine learning pipeline</p>
        <div className="steps-grid">
          {STEPS.map((step, i) => (
            <div key={i} className="step-card">
              <div className="step-icon">{step.icon}</div>
              <h3>{step.title}</h3>
              <p>{step.description}</p>
            </div>
          ))}
        </div>

        {/* About Naive Bayes */}
        <div className="about-nb">
          <h3>What is Multinomial Naive Bayes?</h3>
          <ul>
            <li>A <strong>supervised classification algorithm</strong> that learns from labelled training data.</li>
            <li>Commonly used for <strong>text classification</strong> tasks like spam detection, sentiment analysis, and topic categorization.</li>
            <li>Applies <strong>Bayes' theorem</strong> to calculate the probability that a message belongs to each class.</li>
            <li>Assumes <strong>conditional independence</strong> between features (words) — a simplifying assumption that works surprisingly well in practice.</li>
            <li>Works efficiently with <strong>high-dimensional text data</strong> because it only needs word frequency counts, not complex relationships.</li>
            <li>"Multinomial" refers to its use of <strong>word frequency distributions</strong> (weighted by TF-IDF) rather than simple presence/absence.</li>
          </ul>
        </div>
      </div>
    </section>
  );
}
