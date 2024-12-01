from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import FinanceDataReader as fdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import Adam
import asyncio
from news_monitor import search_naver_news, get_article_content
from text_preprocessing import preprocess_news, clean_text
from sentiment_analysis import analyze_sentiment
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 주식 예측 진행률 전역 변수
progress = 0
news_results = []  # 뉴스 데이터를 저장할 전역 변수

# 주식 관련 키워드 리스트
stock_keywords = ["삼성전자", "sk하이닉스", "네이버", "현대차", "셀트리온"]

# === 주식 예측 기능 ===
# 진행률 반환 API
@app.route('/progress')
def get_progress():
    return jsonify({'progress': progress})

# 진행률 업데이트 콜백 클래스
class ProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global progress
        progress = int((epoch + 1) / self.params['epochs'] * 100)

# 주식 예측 API
@app.route('/predict_stock')
def predict_stock():
    global progress
    progress = 0  # 진행률 초기화

    code = request.args.get('code')
    if not code:
        return "종목 코드가 필요합니다."

    stock = fdr.DataReader(code)
    scaler_close = MinMaxScaler()
    scaled_close = scaler_close.fit_transform(stock[['Close']])
    df = pd.DataFrame(scaled_close, columns=['Close'])

    x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', axis=1), df['Close'], test_size=0.2, random_state=0, shuffle=False)

    def windowed_dataset(series, window_size, batch_size, shuffle):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.map(lambda w: (w[:-1], w[-1]))
        return ds.batch(batch_size).prefetch(1)

    WINDOW_SIZE = 20
    BATCH_SIZE = 64
    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

    model = Sequential([
        Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
        LSTM(16, activation='tanh'),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.0005), metrics=['mse'])

    os.makedirs('tmp', exist_ok=True)
    filename = os.path.join('tmp', 'checkpoint.weights.h5')
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint = ModelCheckpoint(filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)

    history = model.fit(
        train_data, 
        validation_data=(test_data), 
        epochs=100, 
        callbacks=[checkpoint, earlystopping, ProgressCallback()]
    )
    model.load_weights(filename)

    def forecast_windowed_dataset(series, window_size, batch_size):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        return ds.batch(batch_size).prefetch(1)

    forecast = []
    input_series = np.asarray(y_test[-WINDOW_SIZE:])

    for day in range(30):
        input_data = forecast_windowed_dataset(input_series, WINDOW_SIZE, BATCH_SIZE)
        prediction = model.predict(input_data)
        forecast.append(prediction[0, 0])
        input_series = np.append(input_series[1:], prediction[0, 0])

    forecast = scaler_close.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    actual_prices = scaler_close.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()

    last_date = stock.index[-1]
    forecast_dates = pd.date_range(last_date, periods=30, freq='B')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Candlestick(
            x=stock.index,
            open=stock['Open'],
            high=stock['High'],
            low=stock['Low'],
            close=stock['Close'],
            increasing_line_color='blue',
            decreasing_line_color='red',
            name="가격",
            increasing_line_width=2,
            decreasing_line_width=2,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='예측 가격'),
        secondary_y=False,
    )

    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title="가격 (KRW)",
        hovermode="x unified",
        xaxis=dict(
            rangeslider_visible=True,
            type='date',
            range=['2000-01-01', stock.index[-1].strftime('%Y-%m-%d')],
        ),
        yaxis=dict(
            range=[min(actual_prices) * 0.95, max(actual_prices) * 1.05],
            ticks="outside",
            ticklen=5
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        xaxis_rangeslider=dict(visible=True, thickness=0.05),
        autosize=True,
    )

    fig_html = fig.to_html(full_html=False)

    return render_template('prediction_result.html', stock_code=code, plot=fig_html)

# === 뉴스 모니터링 기능 ===
@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(monitor_news_with_preprocessing(stock_keywords))
    return jsonify({"status": "Monitoring started!"})

@app.route('/get-news', methods=['GET'])
def get_news():
    return jsonify(news_results)

@app.route('/')
def index():
    return render_template('news.html', news_results=news_results)

def extract_stock_related_sentences(text, keywords):
    relevant_sentences = []
    sentences = text.split('. ')
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                relevant_sentences.append(sentence)
                break
    return ' '.join(relevant_sentences)

async def monitor_news_with_preprocessing(keywords):
    global news_results
    tasks = []
    for keyword in keywords:
        tasks.append(fetch_and_analyze_news(keyword))
    await asyncio.gather(*tasks)

async def fetch_and_analyze_news(keyword):
    global news_results
    news_data = search_naver_news(keyword, display=5)
    if news_data and 'items' in news_data:
        for item in news_data['items']:
            title = item.get('title', '').strip()
            link = item.get('link', '')
            description = clean_text(item.get('description', ''))
            pub_date = item.get('pubDate', '')

            article_content = get_article_content(link)
            if article_content:
                article_content = clean_text(article_content)
                news_text = f"{description} {article_content}"
                stock_related_text = extract_stock_related_sentences(news_text, stock_keywords)

                if stock_related_text:
                    preprocessed_text = preprocess_news(stock_related_text)
                    sentiment, score = analyze_sentiment(preprocessed_text)

                    if score is None:
                        sentiment_label = "중립"
                        score = 0.5
                    else:
                        sentiment_label = "긍정" if score >= 0.5 else "중립" if score >= 0.45 else "부정"

                    news_results.append({
                        "keyword": keyword,
                        "title": title,
                        "link": link,
                        "description": description,
                        "sentiment": sentiment_label,
                        "score": f"{score:.2f}",
                        "pub_date": pub_date,
                        "analyzed_text": preprocessed_text
                    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
