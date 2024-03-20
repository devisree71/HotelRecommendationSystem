from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
data = pd.read_excel("hotels_data.xlsx")
X = data[['Budget', 'Rating']]
y = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_data = request.get_json()
        user_budget = user_data['budget']
        user_rating = user_data['rating']
        user_data_df = pd.DataFrame({'Budget': [user_budget], 'Rating': [user_rating]})
        user_predicted_rating = model.predict(user_data_df)[0]
        filtered_data = data[(data['Budget'] <= user_budget) & (data['Rating'] >= user_rating)]
    # Top 5 hotels based on user input
        if not filtered_data.empty:
            recommended_hotels = filtered_data.head(5)[['hotel_name', 'Rating']].to_dict(orient='records')
        else:
            recommended_hotels = []
        return jsonify(predicted_rating=user_predicted_rating, recommended_hotels=recommended_hotels)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
