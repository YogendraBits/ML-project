from faker import Faker
import random
import pandas as pd
import datetime

fake = Faker()

# Generate synthetic data for weather and competitor pricing
def generate_weather_data(num_samples):
    weather_data = []
    for _ in range(num_samples):
        temperature = random.uniform(-10, 47)  # Temperature in Celsius
        precipitation = random.choice(['none', 'light', 'moderate', 'heavy'])  # Precipitation intensity
        weather_data.append([temperature, precipitation])
    return weather_data

def generate_competitor_pricing_data(num_samples):
    competitor_pricing_data = []
    for _ in range(num_samples):
        competitor_price = random.uniform(500,20000)  # Price in local currency
        competitor_pricing_data.append(competitor_price)
    return competitor_pricing_data

# Generate synthetic weather and competitor pricing data
num_samples = 5359
weather_data = generate_weather_data(num_samples)
competitor_pricing_data = generate_competitor_pricing_data(num_samples)

# Generate synthetic data for hotel bookings
def generate_hotel_booking_data(num_samples):
    data = []
    for i in range(num_samples):
        
       
        booking_date = fake.date_between(start_date='-30d', end_date='today')
        arrival_date = booking_date + datetime.timedelta(days=random.randint(1, 30))
        
        departure_date = arrival_date + datetime.timedelta(days=random.randint(1, 10))
        length_of_stay = (departure_date - arrival_date).days
        num_guests = random.randint(1, 4)
        room_type = random.choice(['single', 'double', 'suite'])
        
        # Adjust total_cost based on room_type
        if room_type == 'single':
            total_cost = random.uniform(500, 2000)
        elif room_type == 'double':
            total_cost = random.uniform(1000, 4000)
        elif room_type == 'suite':
            total_cost = random.uniform(8000, 20000)
        
        # Adjust the number of rooms based on the number of guests and room type
        if room_type == 'suite':
            number_of_rooms = max(1, num_guests // 2)  # Suites can accommodate fewer guests per room
        else:
            number_of_rooms = max(1, num_guests // 1)
        

        guest_type = random.choice(['individual', 'group', 'corporate'])


        if guest_type == 'corporate':
                purpose_of_visit = 'business'
        else:
                purpose_of_visit = random.choice(['leisure', 'business', 'other'])
                
        loyalty_program = random.choice([True, False])
        previous_cancellations = random.uniform(0, 20)
        reservation_status = random.choices(['checkout', 'cancelled'])
        deposit_type = random.choice(['no_deposit', 'credit_card', 'cash'])
        payment_method = random.choice(['credit_card', 'debit_card', 'cash'])
        hotel_type = random.choice(['resort', 'city hotel', 'motel'])

        # Adjust facilities based on hotel type
        if hotel_type == 'resort':
            facilities = random.choices(['pool', 'gym', 'restaurant', 'spa', 'kids club'], k=random.randint(1, 5))
        elif hotel_type == 'city hotel':
            facilities = random.choices(['gym', 'restaurant', 'business center'], k=random.randint(1, 3))
        elif hotel_type == 'motel':
            facilities = random.choices(['parking', 'wifi'], k=random.randint(1, 2))
        
        lead_time = (arrival_date - booking_date).days
        seasonality = arrival_date.strftime('%B')


        
        
        # Generate cancellation data
        if random.random() < 0.2:  # 20% chance of cancellation
            cancellation_date = booking_date + datetime.timedelta(days=random.randint(1, 10))
            if guest_type == 'corporate':
                reason_for_cancellation = random.choice(['change of plans','unforeseen circumstances'])
            else:
                reason_for_cancellation = random.choice(['change of plans', 'financial reasons', 'unforeseen circumstances'])
            lead_time_at_cancellation = (cancellation_date - booking_date).days
        else:
            cancellation_date = None
            reason_for_cancellation = None
            lead_time_at_cancellation = None

        # Additional features
        # Guest-related features
        guest_age = random.randint(18, 80)
        guest_nationality = fake.country_code(representation="alpha-3")

        # Property-related features
        property_rating = random.randint(1, 5)
        distance_to_attractions = random.uniform(0.1, 10.0)

        # Temporal features
        day_of_week = arrival_date.strftime('%A')
        time_of_day_booking = fake.time(pattern="%H:%M:%S")
        time_of_day_cancellation = fake.time(pattern="%H:%M:%S")

        # Economic and market data
        gdp = random.uniform(1000, 10000)
        inflation_rate = random.uniform(0.01, 0.1)

        # Event-related data
        presence_of_event = random.choice([True, False])
        event_date = fake.date_between(start_date='today', end_date='+30d')

        # Weather data
        temperature, precipitation = weather_data[i]

        # Competitor pricing data
        competitor_price = competitor_pricing_data[i]
        total_cost = competitor_pricing_data[i]

        data.append([booking_date, arrival_date, departure_date, length_of_stay, num_guests, room_type,
                     guest_type,purpose_of_visit, loyalty_program,
                     previous_cancellations, reservation_status, deposit_type, total_cost, payment_method,
                     hotel_type,number_of_rooms, facilities, lead_time, seasonality,
                     cancellation_date, reason_for_cancellation, lead_time_at_cancellation,
                     guest_age, guest_nationality, property_rating, distance_to_attractions,
                     day_of_week, time_of_day_booking, time_of_day_cancellation,
                     gdp, inflation_rate, presence_of_event, event_date,
                     temperature, precipitation, competitor_price])

    return data

# Generate synthetic data
data = generate_hotel_booking_data(num_samples)

# Create a DataFrame from the generated data
columns = ['booking_date', 'arrival_date', 'departure_date', 'length_of_stay', 'num_guests',
           'room_type','guest_type','purpose_of_visit',
           'loyalty_program', 'previous_cancellations', 'reservation_status', 'deposit_type', 'total_cost',
           'payment_method', 'hotel_type','number_of_rooms', 'facilities', 'lead_time',
           'seasonality', 'cancellation_date', 'reason_for_cancellation', 'lead_time_at_cancellation',
           'guest_age', 'guest_nationality', 'property_rating', 'distance_to_attractions',
           'day_of_week', 'time_of_day_booking', 'time_of_day_cancellation',
           'gdp', 'inflation_rate', 'presence_of_event', 'event_date',
           'temperature', 'precipitation', 'competitor_price']
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('hotel_booking_data_extended.csv', index=False)
