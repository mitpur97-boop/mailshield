import { Shield, Menu, X } from 'lucide-react';
import { useState } from 'react';
import './Header.css';

export default function Header() {
  const [menuOpen, setMenuOpen] = useState(false);

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setMenuOpen(false);
  };

  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand" onClick={() => scrollTo('hero')}>
          <Shield size={28} strokeWidth={2.2} />
          <span className="header-title">MAILSHIELD</span>
        </div>
        <button className="menu-toggle" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle menu">
          {menuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        <nav className={`header-nav ${menuOpen ? 'open' : ''}`}>
          <a onClick={() => scrollTo('classifier')}>Classifier</a>
          <a onClick={() => scrollTo('how-it-works')}>How It Works</a>
          <a onClick={() => scrollTo('model-insights')}>Model Insights</a>
        </nav>
      </div>
    </header>
  );
}
