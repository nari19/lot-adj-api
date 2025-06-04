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
		-d @test_data.json
