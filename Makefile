run-server:
	uvicorn main:app --reload

get-params:
	curl -X 'GET' \
		'http://127.0.0.1:8000/params/' \
		-H 'accept: application/json'

test-predict:
	curl -X 'POST' \
		'http://127.0.0.1:8000/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{"symbol": "EURJPY", "data": [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}]}'
