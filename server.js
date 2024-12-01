const express = require('express');
const cors = require('cors');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const KakaoStrategy = require('passport-kakao').Strategy;
const path = require('path'); // 파일 경로 처리
const authRoutes = require('./routes/auth'); // 인증 관련 라우트
const db = require('./db'); // 데이터베이스 연결
require('dotenv').config(); // 환경 변수 로드

const app = express();
const PORT = process.env.PORT || 3000;

// CORS 설정
app.use(cors({
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
}));

// JSON 요청 본문 처리
app.use(express.json());

// 정적 파일 제공 (done.html, login.html, register.html 등)
app.use(express.static(path.join(__dirname, 'public')));

// 세션 설정
app.use(session({
    secret: process.env.SESSION_SECRET || 'your_secret_key',
    resave: false,
    saveUninitialized: true,
}));

// Passport 초기화
app.use(passport.initialize());
app.use(passport.session());

// Google OAuth 설정
passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: `${process.env.BASE_URL}/auth/google/callback`,
  },
  function (token, tokenSecret, profile, done) {
    return done(null, profile); // Google 프로필 반환
  }
));

// Kakao OAuth 설정
passport.use(new KakaoStrategy({
    clientID: process.env.KAKAO_CLIENT_ID,
    clientSecret: process.env.KAKAO_CLIENT_SECRET,
    callbackURL: `${process.env.BASE_URL}/auth/kakao/callback`,
  },
  function (accessToken, refreshToken, profile, done) {
    return done(null, profile); // Kakao 프로필 반환
  }
));

// 세션에 사용자 정보 저장
passport.serializeUser((user, done) => done(null, user));
passport.deserializeUser((id, done) => done(null, id));

// API 라우트 연결
app.use('/api/auth', authRoutes);

// Google OAuth 라우트
app.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

app.get('/auth/google/callback',
  passport.authenticate('google', { failureRedirect: '/login.html' }),
  (req, res) => {
    res.redirect('/home'); // 로그인 후 리디렉션
  }
);

// Kakao OAuth 라우트
app.get('/auth/kakao', passport.authenticate('kakao'));

app.get('/auth/kakao/callback',
  passport.authenticate('kakao', { failureRedirect: '/login.html' }),
  (req, res) => {
    res.redirect('/home'); // 로그인 후 리디렉션
  }
);

// 기타 HTML 파일 경로 처리
app.get('/done.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'done.html'));
});

// 홈 페이지 예제
app.get('/home', (req, res) => {
    if (req.session.user) {
        res.send(`Welcome, ${req.session.user.fullName}`);
    } else {
        res.redirect('/login.html');
    }
});

app.listen(PORT, () => {
  console.log('Server is running on http://localhost:3000');
}); 