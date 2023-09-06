.PHONY: archive

archive:
	find . -name "*.csv" -o -name "*.pkl" | tar -czvf data.tgz -T -