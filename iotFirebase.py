import firebase_admin
from firebase_admin import credentials, db

def check_thresholds(data):
    alerts = []

    # Check battery temperature thresholds
    battery_temp = data.get('batteryTemp')
    if battery_temp is not None:
        if battery_temp < 20:
            alerts.append("Battery temperature below optimal!")
        elif battery_temp > 45:
            alerts.append("Battery temperature above optimal!")
    
    # Check energy consumption and grid battery threshold
    energy_consumption = data.get('energyConsumption')
    grid_battery = data.get('gridBattery')
    
    if energy_consumption is not None and energy_consumption > 750:
        alerts.append("Energy consumption exceeds production rate!")
    
    if grid_battery is not None and grid_battery < 30:
        alerts.append("Grid battery needs recharging!")

    # Check operation status threshold
    operation_status = data.get('operationStatus')
    if operation_status == 0:
        alerts.append("Infrastructure operational error!")
    
    return alerts


def call_firebase():
    # Initialize Firebase Admin SDK with the service account JSON file
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase.json')  # Replace with your actual Firebase JSON file path
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://cropp-cbe63-default-rtdb.firebaseio.com'  # Your Firebase Realtime Database URL
        })

    # Reference to the Firebase Realtime Database path
    ref = db.reference('/energyData')

    # Retrieve the latest data based on the key (or timestamp)
    latest_data = ref.order_by_key().limit_to_last(1).get()

    if latest_data:
        # Extract the latest data
        latest_entry = list(latest_data.values())[0]  # Get the first (and only) entry
        
        print(f"Latest Data: {latest_entry}")

        # Check for threshold conditions
        alerts = check_thresholds(latest_entry)
        
        # If there are any alerts, print them
        if alerts:
            for alert in alerts:
                print(alert)
        else:
            print("All values are within optimal thresholds.")
    else:
        print("No data found.")
    
    return None  # Return None if no data is found

call_firebase()
