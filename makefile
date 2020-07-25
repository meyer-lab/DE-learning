SHELL := /bin/bash

.PHONY: clean test juliaInstall

flist = 1 2 3
flistFull = $(patsubst %, output/figure%.svg, $(flist))

all: pylint.log $(flistFull) output/manuscript.md coverage.xml

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

juliaInstall:
	julia -e 'using Pkg; Pkg.add("https://github.com/meyer-lab/DE.jl.git"); Pkg.add("PyCall")'
	julia -e 'using Pkg; Pkg.update(); Pkg.build(); Pkg.precompile(); Pkg.gc()'

output/figure%.svg: genFigures.py de/figures/figure%.py venv
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md venv/bin/activate
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --verbose -t docx $(pandocCommon) \
		--reference-doc=common/templates/manubot/default.docx \
		--resource-path=.:content \
		-o $@ output/manuscript.md

test: venv
	. venv/bin/activate && pytest -s

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=de --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc de > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf coverage.xml junit.xml output venv
