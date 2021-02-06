.PHONY: all

all:
	python imputation.py 		# add missing data
	python self_alignment.py --train	# construct reflow oven model
	# python self_alignment.py --test --test_size 10 --load_path ./reflow_oven/models/MODEL.pkl --chip R1005 # to test
	python main.py --model np		 		# run the sequence