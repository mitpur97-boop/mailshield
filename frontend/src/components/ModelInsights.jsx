import { useState, useEffect } from 'react';
import { Database, BarChart2, Target, Activity, Layers, AlertTriangle } from 'lucide-react';
import { getModelInfo } from '../services/api';
import './ModelInsights.css';

export default function ModelInsights() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((err) => setError(err.message || 'Failed to load model info'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <section id="model-insights" className="insights-section">
        <div className="insights-container">
          <h2 className="section-title">Model Insights</h2>
          <p className="section-subtitle">Loading model information...</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section id="model-insights" className="insights-section">
        <div className="insights-container">
          <h2 className="section-title">Model Insights</h2>
          <div className="error-card">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        </div>
      </section>
    );
  }

  const m = info.metrics;
  const cm = m?.confusion_matrix;

  return (
    <section id="model-insights" className="insights-section">
      <div className="insights-container">
        <h2 className="section-title">Model Insights</h2>
        <p className="section-subtitle">Performance metrics from the trained model</p>

        {/* Model Info */}
        <div className="info-grid">
          <InfoCard icon={<Layers size={20} />} label="Algorithm" value={info.algorithm} />
          <InfoCard icon={<BarChart2 size={20} />} label="Feature Extraction" value="TF-IDF" />
          <InfoCard icon={<Database size={20} />} label="Dataset" value={info.dataset} />
          <InfoCard icon={<Activity size={20} />} label="Vocabulary Size" value={info.vocabulary_size?.toLocaleString()} />
          <InfoCard icon={<Target size={20} />} label="Training Samples" value={info.training_samples?.toLocaleString()} />
          <InfoCard icon={<Target size={20} />} label="Testing Samples" value={info.testing_samples?.toLocaleString()} />
        </div>

        {/* Performance Metrics */}
        <h3 className="sub-heading">Model Performance</h3>
        <div className="metrics-grid">
          <MetricCard label="Accuracy" value={m.accuracy} />
          <MetricCard label="Precision" value={m.precision} />
          <MetricCard label="Recall" value={m.recall} />
          <MetricCard label="F1 Score" value={m.f1_score} />
        </div>

        {/* Confusion Matrix */}
        {cm && (
          <>
            <h3 className="sub-heading">Confusion Matrix</h3>
            <div className="cm-wrapper">
              <table className="cm-table">
                <thead>
                  <tr>
                    <th></th>
                    <th>Predicted Ham</th>
                    <th>Predicted Spam</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>Actual Ham</th>
                    <td className="cm-tn">{cm.true_negative}</td>
                    <td className="cm-fp">{cm.false_positive}</td>
                  </tr>
                  <tr>
                    <th>Actual Spam</th>
                    <td className="cm-fn">{cm.false_negative}</td>
                    <td className="cm-tp">{cm.true_positive}</td>
                  </tr>
                </tbody>
              </table>
              <div className="cm-legend">
                <span><strong>TN</strong> = True Negative — Correctly identified as Not Spam</span>
                <span><strong>FP</strong> = False Positive — Legitimate message incorrectly flagged as Spam</span>
                <span><strong>FN</strong> = False Negative — Spam message missed, classified as Not Spam</span>
                <span><strong>TP</strong> = True Positive — Correctly identified as Spam</span>
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function InfoCard({ icon, label, value }) {
  return (
    <div className="info-card">
      <div className="info-icon">{icon}</div>
      <div>
        <div className="info-label">{label}</div>
        <div className="info-value">{value}</div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className="metric-card">
      <div className="metric-value">{pct}%</div>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
