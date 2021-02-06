.PHONY: all

all:
	python imputation.py 		# add missing data
	python self_alignment.py --train	# construct reflow oven model
	# python self_alignment.py --test --test_size 10 --load_path ./models/MODEL.pkl --chip R1005 # to test
	python main.py		 		# run the sequence