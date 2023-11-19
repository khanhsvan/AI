from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Thiết lập cấu hình hiển thị cho numpy
np.set_printoptions(precision=3, suppress=True)

# Load mô hình từ file .pkl
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file, encoding='latin1') # Thêm tham số encoding

# Load dữ liệu từ file merged_data.csv
merged_data_df = pd.read_csv('merged_data.csv') # Thêm tham số index_col
feature_names = merged_data_df.columns.tolist()
selected_columns = ["user_reviews","price_final","positive_ratio","discount","year_release","platform"]
X = merged_data_df[selected_columns]

# Load dữ liệu từ file games.csv
games_df = pd.read_csv('games.csv', index_col=0) # Thêm tham số index_col
games_df.head()
print(merged_data_df.head())
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_games', methods=['POST'])
def recommend_games():
    # Lấy dữ liệu từ request form
    user_reviews = int(request.form['user_reviews'])
    price_final = float(request.form['price_final'])
    positive_ratio = float(request.form['positive_ratio'])
    discount = float(request.form['discount'])
    year_release = int(request.form['year_release'])
    platform = int(request.form['platform'])

    # Tiền xử lý dữ liệu
    def preprocess_data(user_reviews, price_final, positive_ratio, discount, year_release, platform):
        # Dự đoán
        input_data = np.array([user_reviews,price_final,positive_ratio,discount,year_release,platform], dtype=float).reshape(1, -1)
        recommendation = model.predict(input_data) # Sử dụng hàm predict_proba

        # Lọc danh sách các app_id tương ứng với kết quả dự đoán
        similar_games = merged_data_df[
            (merged_data_df["user_reviews"] >= user_reviews) &
            (merged_data_df["price_final"] <= price_final) &
            (merged_data_df["positive_ratio"] >= positive_ratio) &
            (merged_data_df["discount"] >= discount) &
            (merged_data_df["year_release"] == year_release) &
            (merged_data_df["platform"] == platform)
        ]
        similar_games.to_numpy()
        # Trích xuất các tên (titles) của game từ các app_id phù hợp
        recommended_titles = games_df[games_df['user_reviews'].isin(similar_games['user_reviews'])]['title'].tolist()
        # print(recommended_titles)
        # print(input_data)
        # print(recommendation)
        # print(similar_games)
        return recommendation, recommended_titles  # Thêm kết quả dự đoán vào đây
    # Gọi hàm tiền xử lý dữ liệu và nhận danh sách các titles phù hợp
    recommendation, recommended_titles = preprocess_data(user_reviews,price_final,positive_ratio,discount,year_release,platform)

    # Trả về view HTML chứa danh sách các titles và app_id tương ứng
    return render_template('index.html', recommendation=recommendation, recommended_games=recommended_titles)  # Thêm recommendation vào đây để truyền dữ liệu dự đoán tới template

if __name__ == '__main__':
    app.run(debug=True)
