import Header from './components/Header';
import Hero from './components/Hero';
import Classifier from './components/Classifier';
import HowItWorks from './components/HowItWorks';
import ModelInsights from './components/ModelInsights';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="app">
      <Header />
      <Hero />
      <Classifier />
      <HowItWorks />
      <ModelInsights />
      <Footer />
    </div>
  );
}

export default App;
