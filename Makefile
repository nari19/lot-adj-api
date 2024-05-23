run-server:
	uvicorn main:app --reload

# http://127.0.0.1:8000/params/
get-params:
	curl -X 'GET' \
		'http://127.0.0.1:8000/params/' \
		-H 'accept: application/json'

send-predict-request1:
	curl -X 'POST' \
		'http://127.0.0.1:8000/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
		"Symbol": "EURCHF", \
		"BuySell": "Buy", \
		"Indi1": 4, \
		"Indi2": 41, \
		"Param1": 5, \
		"Param2": 17, \
		"Day": 3, \
		"Hour": 0, \
		"Minute": 5, \
		"RSI1": 18.72580654667253, \
		"RSI2": 26.134190042276217, \
		"RSI3": 31.70680291597936, \
		"RSI4": 34.51854383961684, \
		"RSI5": 40.47923677790335, \
		"MACD1": -0.0005981052118797, \
		"MACD_Sig1": -0.0002368242880685, \
		"MACD2": -0.0006581705049784, \
		"MACD_Sig2": -0.0005007348827782, \
		"MACD3": -0.0007418691195377, \
		"MACD_Sig3": -0.0004891677971046, \
		"MACD4": -0.0007987172584982, \
		"MACD_Sig4": -0.0004899872037783, \
		"MACD5": -0.0008729080754881, \
		"MACD_Sig5": -0.0005609283938151 \
	}'

send-predict-request2:
	curl -X 'POST' \
		'http://127.0.0.1:8000/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
		"Symbol": "EURCHF", \
		"BuySell": "Sell", \
		"Indi1": 4, \
		"Indi2": 41, \
		"Param1": 5, \
		"Param2": 17, \
		"Day": 4, \
		"Hour": 8, \
		"Minute": 0, \
		"RSI1": 67.51473686991753, \
		"RSI2": 64.68070652381499, \
		"RSI3": 61.70520619276791, \
		"RSI4": 59.334711388491385, \
		"RSI5": 53.07214154879348, \
		"MACD1": 0.0002837459484352, \
		"MACD_Sig1": 0.0002679037928884, \
		"MACD2": 0.0003546894604335, \
		"MACD_Sig2": 0.0002446246734312, \
		"MACD3": 0.0003809711389868, \
		"MACD_Sig3": 0.0002333747251092, \
		"MACD4": 0.0003056477572096, \
		"MACD_Sig4": 0.0001055828849192, \
		"MACD5": -0.0002212349382868, \
		"MACD_Sig5": -0.0005990844010359 \
	}'

send-predict-request3:
	curl -X 'POST' \
		'https://lot-adj.onrender.com/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
		"Symbol": "EURCHF", \
		"BuySell": "Buy", \
		"Indi1": 4, \
		"Indi2": 41, \
		"Param1": 5, \
		"Param2": 17, \
		"Day": 3, \
		"Hour": 0, \
		"Minute": 5, \
		"RSI1": 18.72580654667253, \
		"RSI2": 26.134190042276217, \
		"RSI3": 31.70680291597936, \
		"RSI4": 34.51854383961684, \
		"RSI5": 40.47923677790335, \
		"MACD1": -0.0005981052118797, \
		"MACD_Sig1": -0.0002368242880685, \
		"MACD2": -0.0006581705049784, \
		"MACD_Sig2": -0.0005007348827782, \
		"MACD3": -0.0007418691195377, \
		"MACD_Sig3": -0.0004891677971046, \
		"MACD4": -0.0007987172584982, \
		"MACD_Sig4": -0.0004899872037783, \
		"MACD5": -0.0008729080754881, \
		"MACD_Sig5": -0.0005609283938151 \
	}'

send-predict-request4:
	curl -X 'POST' \
		'https://lot-adj.onrender.com/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
		"Symbol": "EURCHF", \
		"BuySell": "Sell", \
		"Indi1": 4, \
		"Indi2": 41, \
		"Param1": 5, \
		"Param2": 17, \
		"Day": 4, \
		"Hour": 8, \
		"Minute": 0, \
		"RSI1": 67.51473686991753, \
		"RSI2": 64.68070652381499, \
		"RSI3": 61.70520619276791, \
		"RSI4": 59.334711388491385, \
		"RSI5": 53.07214154879348, \
		"MACD1": 0.0002837459484352, \
		"MACD_Sig1": 0.0002679037928884, \
		"MACD2": 0.0003546894604335, \
		"MACD_Sig2": 0.0002446246734312, \
		"MACD3": 0.0003809711389868, \
		"MACD_Sig3": 0.0002333747251092, \
		"MACD4": 0.0003056477572096, \
		"MACD_Sig4": 0.0001055828849192, \
		"MACD5": -0.0002212349382868, \
		"MACD_Sig5": -0.0005990844010359 \
	}'
