default: sand

all: sand

sand: lstanchors
	pdflatex sand.tex
	bibtex sand
	pdflatex sand.tex
	pdflatex sand.tex

strip:
	python strip_examples.py

lstanchors: strip
	./lstanchors.pl paper.tex

clean:
	rm -f sand.blg
	rm -f sand.bbl
	rm -f sand.pdf
	rm -f sand.log
	rm -f sand.lot
	rm -f sand.lof
	rm -f sand.out
	rm -f sand.toc
	#
	rm -f *.aux
	rm -f examples/*.pyc
	rm -f examples/results.yml
	rm -f examples/*_strip.py

