import { Shield } from 'lucide-react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <Shield size={20} />
          <span>MAILSHIELD</span>
        </div>
        <p className="footer-text">
          A machine learning project using Multinomial Naive Bayes for spam classification.
        </p>
        <p className="footer-copy">
          Built with React, FastAPI, and scikit-learn
        </p>
      </div>
    </footer>
  );
}
