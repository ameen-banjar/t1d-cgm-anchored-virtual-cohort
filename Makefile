.PHONY: install demo test clean

install:
	python3 -m pip install -e .

demo:
	PYTHONPATH=src python3 -m t1d_virtual_cohort.cli demo --output outputs/demo

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

clean:
	rm -rf outputs/demo
