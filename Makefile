all:
	python imputation.py 	# add missing data
	python multirf.py	 	# construct reflow oven model
	python main.py		 	# run the sequence