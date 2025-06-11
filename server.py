import os
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, jsonify
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

# Database connection setup
DATABASE_USERNAME = "wy2470"
DATABASE_PASSWRD = "342930"
DATABASE_HOST = "34.148.223.31"
DATABASEURI = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWRD}@{DATABASE_HOST}/proj1part2"

engine = create_engine(DATABASEURI)

@app.before_request
def connect_db():
	try:
		g.conn = engine.connect()
	except Exception as e:
		logger.error(f"Database connection error: {str(e)}")
		g.conn = None

@app.teardown_request
def close_db(exception):
	try:
		g.conn.close()
	except Exception as e:
		pass

@app.route('/')
def index():
	return render_template("welcome.html")

@app.route('/restaurants')
def restaurants():
	try:
		query = """
		SELECT r.Restaurant_ID AS restaurant_id, r.Restaurant_name AS restaurant_name, r.Price_range, r.Category, r.Michelin_stars, r.Popular_dishes, r.Opening_hours, AVG(rev.Rating) AS avg_rating
		FROM Restaurant r
		LEFT JOIN Review rev ON r.Restaurant_ID = rev.Restaurant_ID
		GROUP BY r.Restaurant_ID, r.Restaurant_name, r.Price_range, r.Category, r.Michelin_stars, r.Popular_dishes, r.Opening_hours
		"""
		restaurants = []
		for row in g.conn.execute(text(query)):
			restaurants.append(dict(row._mapping))
		menu_query = """SELECT *, restaurant_id AS restaurant_id FROM Menu"""
		menu = {}
		for n in g.conn.execute(text(menu_query)):
			id = n._mapping['restaurant_id']
			if id not in menu:
				menu[id] = []
			menu[id].append(dict(n._mapping))
		rating_query = "SELECT *, restaurant_id AS restaurant_id FROM Review"
		ratings = {}
		for n in g.conn.execute(text(rating_query)):
			id = n._mapping['restaurant_id']
			if id not in ratings:
				ratings[id] = []
			ratings[id].append(dict(n._mapping))

		return render_template('restaurants.html', restaurants=restaurants, ratings=ratings, menu=menu)
	except Exception as e:
		return render_template('restaurants.html', restaurants=[], ratings=[], error="Error loading restaurants")

@app.route('/reservation/<int:restaurant_id>', methods=['GET', 'POST'])
def reservation(restaurant_id):
	if request.method == 'POST':
		try:
			party_size = request.form['party_size']
			date = request.form['date']
			time = request.form['time']
			special_event = request.form.get('special_event', None)
			last_name = request.form['last_name']
			phone_number = request.form['phone_number']

			user_query = """SELECT User_ID FROM Users WHERE Last_Name = :last_name AND Phone_Number = :phone_number"""
			user_cursor = g.conn.execute(text(user_query), {'last_name': last_name,'phone_number': phone_number})
			user = user_cursor.fetchone()
			user_cursor.close()
			
			if not user:
				return render_template("reservation.html", restaurant_id=restaurant_id, error="User not found. Please check your last name and phone number.")
			current_reservation_id_query = """SELECT Max(Reservation_ID) AS current_reservation_id FROM Reservation"""
			current_reservation_id_cursor = g.conn.execute(text(current_reservation_id_query))
			current_reservation_id = current_reservation_id_cursor.fetchone()._mapping['current_reservation_id']
			current_reservation_id_cursor.close()
			now_reservation_id = current_reservation_id + 1
			query = """
			INSERT INTO Reservation (Reservation_ID, User_ID, Restaurant_ID, Party_size, Time, Date, Special_event)
			VALUES (:reservation_id, :user_id, :restaurant_id, :party_size, :time, :date, :special_event)
			"""
			g.conn.execute(text(query), {
				'reservation_id': now_reservation_id,
				'user_id': user.user_id,
				'restaurant_id': restaurant_id,
				'party_size': party_size,
				'time': time,
				'date': date,
				'special_event': special_event
			})
			g.conn.commit()
			return redirect('/search')
		except Exception as e:
			logger.error(f"Error creating reservation: {str(e)}")
			logger.exception("Detailed traceback for reservation error:")
			return render_template("reservation.html", restaurant_id=restaurant_id, error="Error creating reservation. Please try again.")
	
	return render_template("reservation.html", restaurant_id=restaurant_id)
@app.route('/create', methods=['GET', 'POST'])
def create():
	if request.method == 'POST':
		first_name = request.form['first_name']
		last_name = request.form['last_name']
		phone_number = request.form['phone_number']
		email = request.form['email']
		current_user_id_query = """SELECT Max(User_ID) AS current_user_id FROM Users"""
		current_user_id_cursor = g.conn.execute(text(current_user_id_query))
		current_user_id = current_user_id_cursor.fetchone()._mapping['current_user_id']
		current_user_id_cursor.close()
		now_user_id = current_user_id + 1
		query = """INSERT INTO Users (User_ID, First_Name, Last_Name, Phone_Number, Email)
				VALUES (:user_id, :first_name, :last_name, :phone_number, :email)
				"""
		g.conn.execute(text(query), {
			'user_id': now_user_id,
			'first_name': first_name,
			'last_name': last_name,
			'phone_number': phone_number,
			'email': email
		})
		g.conn.commit()
		return redirect('/search')
	return render_template("create.html")
@app.route('/search', methods=['GET', 'POST'])
def search():
	if request.method == 'POST':
		try:
			last_name = request.form['last_name']
			phone_number = request.form['phone_number']
			
			query = """SELECT * FROM Users WHERE Last_Name = :last_name AND Phone_Number = :phone_number"""
			cursor = g.conn.execute(text(query), {
				'last_name': last_name,
				'phone_number': phone_number
			})
			user = cursor.fetchone()._mapping
			cursor.close()
			
			if not user:
				return render_template("search.html", error="User not found")

			card_query = """SELECT * FROM Card_Information WHERE User_ID = :user_id"""
			card_cursor = g.conn.execute(text(card_query), {'user_id': user.user_id})
			card = card_cursor.fetchone()
			card_cursor.close()
			
			reservation_query = """SELECT r.*, res.Restaurant_name FROM Reservation r
			JOIN Restaurant res ON r.Restaurant_ID = res.Restaurant_ID
			WHERE r.User_ID = :user_id
			ORDER BY r.Date DESC, r.Time DESC
			"""
			reservation_cursor = g.conn.execute(text(reservation_query), {'user_id': user.user_id})
			reservations = [dict(n._mapping) for n in reservation_cursor]
			reservation_cursor.close()
			
			return render_template("profile.html", user=user, card=card, reservations=reservations)
		except Exception as e:
			logger.error(f"Error searching for user: {str(e)}")
			logger.exception("Detailed traceback for user search error:")
			return render_template("search.html", error="Error searching for user information")
	
	return render_template("search.html")

def build_transition_matrix(cuisines):
	transition_matrix = defaultdict(lambda: defaultdict(int))
	for i in range(len(cuisines) - 1):
		current = cuisines[i]
		next_cuisine = cuisines[i + 1]
		transition_matrix[current][next_cuisine] += 1

	for current in transition_matrix:
		total = sum(transition_matrix[current].values())
		for next_cuisine in transition_matrix[current]:
			transition_matrix[current][next_cuisine] /= total
	
	return transition_matrix

def predict_next_cuisine(current_cuisine, transition_matrix):
	if current_cuisine not in transition_matrix:
		return None
	next_cuisines = list(transition_matrix[current_cuisine].keys())
	probabilities = list(transition_matrix[current_cuisine].values())
	next_cuisine = random.choices(next_cuisines, probabilities)[0]
	return next_cuisine

def haversine_distance(lat1, lon1, lat2, lon2):
	if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
		return float('inf')
	coords_1 = (lat1, lon1)
	coords_2 = (lat2, lon2)
	return geodesic(coords_1, coords_2).meters

def get_user_recommendations(user_id):
	history_query = """
	SELECT r.Category, res.Restaurant_ID, res.Restaurant_name, r.Price_range
	FROM Reservation res
	JOIN Restaurant r ON res.Restaurant_ID = r.Restaurant_ID
	WHERE res.User_ID = :user_id
	ORDER BY res.Date DESC, res.Time DESC
	"""
	
	user_history = []
	for row in g.conn.execute(text(history_query), {'user_id': user_id}):
		user_history.append(dict(row._mapping))
	
	if not user_history:
		return None, "No reservation history found for this user"
	
	visited_cuisines = [visit['category'] for visit in user_history if visit['category']]
	
	if len(visited_cuisines) < 2:
		cuisine_counts = defaultdict(int)
		for cuisine in visited_cuisines:
			cuisine_counts[cuisine] += 1
		predicted_cuisine = max(cuisine_counts.keys(), key=lambda k: cuisine_counts[k])
	else:
		transition_matrix = build_transition_matrix(visited_cuisines)
		last_cuisine = visited_cuisines[-1]
		predicted_cuisine = predict_next_cuisine(last_cuisine, transition_matrix)
		
		if predicted_cuisine is None:
			cuisine_counts = defaultdict(int)
			for cuisine in visited_cuisines:
				cuisine_counts[cuisine] += 1
			predicted_cuisine = max(cuisine_counts.keys(), key=lambda k: cuisine_counts[k])
	
	all_restaurants_query = """
	SELECT r.Restaurant_ID, r.Restaurant_name, r.Price_range, r.Category, 
	       AVG(rev.Rating) as avg_rating
	FROM Restaurant r
	LEFT JOIN Review rev ON r.Restaurant_ID = rev.Restaurant_ID
	GROUP BY r.Restaurant_ID, r.Restaurant_name, r.Price_range, r.Category
	"""
	restaurants_data = []
	for row in g.conn.execute(text(all_restaurants_query)):
		restaurants_data.append(dict(row._mapping))
	df = pd.DataFrame(restaurants_data)
	user_visits_df = pd.DataFrame(user_history)
	
	if df.empty:
		return None, "No restaurants found"
	
	visited_restaurant_ids = [visit['restaurant_id'] for visit in user_history]
	df = df[~df['restaurant_id'].isin(visited_restaurant_ids)]
	
	if df.empty:
		return None, "No new restaurants to recommend"
	
	df['cuisine_match'] = (df['category'] == predicted_cuisine).astype(int)
	
	price_encoder = LabelEncoder()
	all_prices = list(df['price_range'].dropna()) + list(user_visits_df['price_range'].dropna())
	price_encoder.fit(all_prices)
	
	df['price_encoded'] = price_encoder.transform(df['price_range'].fillna('$'))
	user_visits_df['price_encoded'] = price_encoder.transform(user_visits_df['price_range'].fillna('$'))
	
	user_avg_price_encoded = user_visits_df['price_encoded'].mean()
	user_avg_price_encoded_reshaped = np.array([user_avg_price_encoded]).reshape(1, -1)
	
	df['price_similarity'] = cosine_similarity(
		user_avg_price_encoded_reshaped,
		df['price_encoded'].values.reshape(-1, 1)
	).flatten()
	
	df['location_similarity'] = np.random.uniform(0.3, 1.0, len(df))
	
	cuisine_weight = 0.4 
	price_weight = 0.3   
	location_weight = 0.2 
	rating_weight = 0.1  
	
	df['final_score'] = (
		cuisine_weight * df['cuisine_match'] +
		price_weight * df['price_similarity'] +
		location_weight * df['location_similarity'] +
		rating_weight * (df['avg_rating'].fillna(0) / 5.0)
	)
	
	top_recommendations = df.sort_values(by='final_score', ascending=False).head(5)
	
	recommendations = []
	for _, restaurant in top_recommendations.iterrows():
		recommendations.append({
			'restaurant_id': int(restaurant['restaurant_id']),
			'name': restaurant['restaurant_name'],
			'category': restaurant['category'],
			'price_range': restaurant['price_range'],
			'avg_rating': float(restaurant['avg_rating']) if restaurant['avg_rating'] else None,
			'predicted_cuisine': predicted_cuisine,
			'final_score': float(restaurant['final_score'])
		})
	
	return {
		'user_id': user_id,
		'predicted_next_cuisine': predicted_cuisine,
		'recommendations': recommendations
	}, None

@app.route('/recommendations/<int:user_id>')
def get_recommendations(user_id):
	try:
		result, error = get_user_recommendations(user_id)
		if error:
			return jsonify({"error": error}), 404
		return jsonify(result)
		
	except Exception as e:
		logger.error(f"Error generating recommendations: {str(e)}")
		logger.exception("Detailed traceback for recommendation error:")
		return jsonify({"error": "Error generating recommendations"}), 500

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
	if request.method == 'POST':
		try:
			last_name = request.form['last_name']
			phone_number = request.form['phone_number']
			
			user_query = """SELECT User_ID FROM Users WHERE Last_Name = :last_name AND Phone_Number = :phone_number"""
			user_cursor = g.conn.execute(text(user_query), {'last_name': last_name, 'phone_number': phone_number})
			user = user_cursor.fetchone()
			user_cursor.close()
			
			if not user:
				return render_template("recommendations.html", error="User not found. Please check your last name and phone number.")
			
			result, error = get_user_recommendations(user.user_id)
			
			if error:
				return render_template("recommendations.html", error=error)
			
			return render_template("recommendations.html", 
								   recommendations=result['recommendations'],
								   predicted_cuisine=result['predicted_next_cuisine'])
				
		except Exception as e:
			logger.error(f"Error in recommendations route: {str(e)}")
			return render_template("recommendations.html", error="Error processing request")
	
	return render_template("recommendations.html")

if __name__ == "__main__":
	import click

	@click.command()
	@click.option('--debug', is_flag=True)
	@click.option('--threaded', is_flag=True)
	@click.argument('HOST', default='0.0.0.0')
	@click.argument('PORT', default=8111, type=int)
	def run(debug, threaded, host, port):
		HOST, PORT = host, port
		print("running on %s:%d" % (HOST, PORT))
		app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)

	run()
