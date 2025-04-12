import os
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect
import logging

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
