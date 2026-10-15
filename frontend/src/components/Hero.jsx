import { ShieldCheck } from 'lucide-react';
import './Hero.css';

export default function Hero() {
  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section id="hero" className="hero">
      <div className="hero-content">
        <div className="hero-icon">
          <ShieldCheck size={48} />
        </div>
        <h1>Detect Spam Before It Reaches You</h1>
        <p className="hero-subtitle">
          MailShield uses machine learning to classify suspicious messages.
          Powered by Multinomial Naive Bayes and TF-IDF text analysis.
        </p>
        <button className="hero-cta" onClick={() => scrollTo('classifier')}>
          Try the Classifier
        </button>
      </div>
    </section>
  );
}
