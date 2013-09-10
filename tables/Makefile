all:
	rm runsvm.obv-all.json ../split-tables/*
	cat runsvm.obv-*.json > runsvm.obv-all.json
	split -a 3 -d -l 1 runsvm.obv-all.json ../split-tables/runsvm.obv-all.json-
