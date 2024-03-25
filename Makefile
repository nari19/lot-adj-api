run-server:
	uvicorn main:app --reload

send-predict-request:
	curl -X 'POST' \
		'https://lot-adj.onrender.com/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
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
		'https://lot-adj.onrender.com/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
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
		"Day": 6, \
		"Hour": 9, \
		"Minute": 10, \
		"RSI1": 41.41796563411295, \
		"RSI2": 49.18070946799118, \
		"RSI3": 52.17859486387651, \
		"RSI4": 52.62355619402437, \
		"RSI5": 52.51385109706518, \
		"MACD1": 4.79420619146e-05, \
		"MACD_Sig1": 3.68784963564e-05, \
		"MACD2": 0.0001270361503162, \
		"MACD_Sig2": 0.0001433002660545, \
		"MACD3": 0.0002577105364731, \
		"MACD_Sig3": 0.0002652194539544, \
		"MACD4": 0.0003287374719731, \
		"MACD_Sig4": 0.0002291721884661, \
		"MACD5": 0.0005843611751272, \
		"MACD_Sig5": 0.0005432953337931 \
	}'

send-predict-request4:
	curl -X 'POST' \
		'https://lot-adj.onrender.com/predict/' \
		-H 'Accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{ \
		"Day": 20, \
		"Hour": 5, \
		"Minute": 9, \
		"RSI1": 42.652037352141946, \
		"RSI2": 47.44176218956845, \
		"RSI3": 50.1868727036748, \
		"RSI4": 51.27505130004888, \
		"RSI5": 53.05241451135397, \
		"MACD1": -0.0002741320872732267, \
		"MACD_Sig1": -0.00017334846848114904, \
		"MACD2": 0.00008469404334965347, \
		"MACD_Sig2": 0.00043782962448892036, \
		"MACD3": 0.0005353323255183895, \
		"MACD_Sig3": 0.0006311114842023087, \
		"MACD4": 0.0013867799156970229, \
		"MACD_Sig4": 0.001507162907236508, \
		"MACD5": 0.0027625468737997316, \
		"MACD_Sig5": 0.0005432953337931 \
	}'
