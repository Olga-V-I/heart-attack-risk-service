# client_test.py
import requests
import json

URL = "http://127.0.0.1:8000/predict"

# путь к тестовому файлу
payload = "heart_test.csv"

# отправляем POST-запрос с JSON-строкой
response = requests.post(URL, json=payload)

# печатаем статус ответа
print("Status:", response.status_code)

# выводим JSON с отступами
print(json.dumps(response.json(), ensure_ascii=False, indent=2))