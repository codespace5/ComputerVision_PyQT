import requests

response = requests.post('https://api.mynotifier.app', {
                "apiKey": '6edbf34e-58ce-4145-a31d-275a790e5f71',
                "message": "111",
                "description": "111",
                "type": "info",  # info, error, warning, or success
            })
if response.status_code == 200:
    print("Notification sent successfully!")
else:
    print(f"Failed to send the notification. Status code: {response.status_code}")
  
