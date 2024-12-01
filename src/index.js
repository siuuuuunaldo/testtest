// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css'; // 스타일 파일 (필요한 경우)
import App from './App'; // App 컴포넌트 (필요한 경우)

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
