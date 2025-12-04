// App.jsx
import React, { useState, useEffect } from 'react';
import { Upload, RefreshCw, Microscope, MessageSquare, CheckCircle, AlertCircle, Loader2, Lightbulb } from 'lucide-react';
import './App.css';

const MalariaScope = () => {
  const [currentFact, setCurrentFact] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const facts = [
    "The female Anopheles mosquito is the only mosquito capable of transmitting Plasmodium to humans.",
    "There are five species of Plasmodium that infect humans: P. falciparum, P. vivax, P. ovale, P. malariae, and P. knowlesi.",
    "Plasmodium falciparum is the most deadly malaria parasite, responsible for the majority of malaria deaths worldwide.",
    "The malaria parasite has a complex life cycle involving both mosquito and human hosts.",
    "Plasmodium parasites can remain dormant in the liver for months or even years before causing symptoms.",
    "Approximately 249 million malaria cases were reported globally in 2022, with most occurring in Africa.",
    "The Plasmodium parasite was discovered by Charles Louis Alphonse Laveran in 1880, earning him a Nobel Prize.",
    "Some human genetic mutations, like sickle cell trait, provide natural resistance against severe malaria.",
    "Plasmodium vivax can cause relapses because it forms dormant liver stages called hypnozoites.",
    "The malaria parasite takes only 30 seconds to travel from a mosquito bite to the human liver.",
    "P. falciparum is the only human species commonly exhibiting multiple infections (double, triple, or more) in a single RBC.",
    "P. vivax and P. ovale typically cause the infected RBCs to become enlarged (up to 2 times their normal size)",
    "The ring forms (early trophozoites) of P. falciparum are often delicate, small, and appliqué or 'headphone' shaped, frequently seen adhered to the edge of the RBC",
    "P. malariae is known for its compact, thick, and band-shaped trophozoites that stretch across the width of the infected cell.",
    "P. vivax preferentially invades young RBCs (reticulocytes), while P. malariae selectively targets older, senescent RBCs.",
    "The presence of Schüffner's dots in infected RBCs is a characteristic feature of P. vivax and P. ovale infections.",
  ];

  useEffect(() => {
    changeFact();
  }, []);

  const changeFact = () => {
    const randomFact = facts[Math.floor(Math.random() * facts.length)];
    setCurrentFact(randomFact);
  };

  const handleFileSelect = (file) => {
    if (!file || !file.type.match('image/(jpeg|png)')) {
      alert('Please upload a JPG or PNG image.');
      return;
    }

    setSelectedFile(file);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Analysis failed');

      const infected = response.headers.get('X-Infected') === 'true';
      const message = response.headers.get('X-Prediction-Message') || 'Analysis complete';
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);

      setPreviewUrl(imageUrl);
      setResult({ infected, message });
    } catch (error) {
      alert('Error analyzing image: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      <div className="max-width-container">
        {/* Header Card */}
        <div className="header-card">
          <div className="header-content">
            <div className="brand-section">
              <div className="logo-icon">
                <Microscope className="icon-size-lg" />
              </div>
              <div>
                <h1 className="brand-title">PlasmoScan</h1>
                <p className="brand-subtitle">ML-Powered Malaria Parasite Detection</p>
              </div>
            </div>
            <a
              href="https://forms.gle/w8M3ZmSbZHQ2wKtj7"
              target="_blank"
              rel="noopener noreferrer"
              className="feedback-btn"
            >
              <MessageSquare className="icon-size-sm" />
              Feedback
            </a>
          </div>
        </div>

        {/* Main Card */}
        <div className="main-card">
          <div className="title-section">
            <h2 className="main-title">Detect Malaria Parasites</h2>
            <p className="main-subtitle">
              Upload a blood cell microscopy image for instant analysis
            </p>
          </div>

          {/* Fact Box */}
          <div className="fact-box">
            <div className="fact-content">
              <div className="fact-text-section">
                <div className="fact-header">
                  <div className="fact-icon">
                    <Lightbulb className="icon-size-sm" />
                  </div>
                  <h3 className="fact-title">Did You Know?</h3>
                </div>
                <p className="fact-description">{currentFact}</p>
              </div>
              <button onClick={changeFact} className="refresh-btn">
                <RefreshCw className="icon-refresh" />
              </button>
            </div>
          </div>

          {/* Upload Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput').click()}
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
          >
            <div className="upload-icon">
              <Upload className="icon-size-xl" />
            </div>
            <p className="upload-title">Drop your blood cell image here</p>
            <p className="upload-subtitle">
              or click to browse • JPG, PNG supported
            </p>
            <input
              id="fileInput"
              type="file"
              accept="image/jpeg,image/png"
              onChange={(e) => handleFileSelect(e.target.files[0])}
              className="file-input"
            />
          </div>

          {/* Preview */}
          {previewUrl && (
            <div className="preview-section">
              <img src={previewUrl} alt="Preview" className="preview-image" />
            </div>
          )}

          {/* Loading */}
          {isAnalyzing && (
            <div className="loading-section">
              <Loader2 className="loading-spinner" />
              <p className="loading-text">Analyzing image...</p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className={`result-box ${result.infected ? 'infected' : 'not-infected'}`}>
              <div className="result-content">
                {result.infected ? (
                  <AlertCircle className="result-icon" />
                ) : (
                  <CheckCircle className="result-icon" />
                )}
                <p className="result-message">{result.message}</p>
              </div>
            </div>
          )}

          {/* Analyze Button */}
          {selectedFile && !isAnalyzing && (
            <button onClick={handleAnalyze} className="analyze-btn">
              <Microscope className="icon-size-md" />
              Analyze Image
            </button>
          )}

          {/* How it works */}
          <div className="how-it-works">
            <h3 className="how-it-works-title">How it works</h3>
            <div className="steps-container">
              {[
                'Upload a microscopy image of blood cells (JPG or PNG)',
                'Our ML model analyzes the image for Plasmodium parasites',
                'Get instant results with annotated detection if parasites are found'
              ].map((step, index) => (
                <div key={index} className="step-item">
                  <div className="step-number">{index + 1}</div>
                  <p className="step-text">{step}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MalariaScope;