SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4
flistFull = $(patsubst %, output/figure%.svg, $(flist))

all: $(flistFull) output/manuscript.md

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: de/figures/figure%.svg
	@ mkdir -p ./output
	cp $< $@

output/figure%.svg: genFigures.py de/figures/figure%.py venv
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml output/manuscript.md

test: venv
	. venv/bin/activate && pytest -s -x

clean:
	rm -rf output venv de/data/GSE*

download: de/data/GSE106127_inst_info.txt.xz de/data/GSE92742_Broad_LINCS_Level2.csv.xz de/data/GSE70138_Broad_LINCS_Level2.csv.xz

de/data/%.xz:
	wget -N -P ./de/data https://syno.seas.ucla.edu:9001/de-learning/$*.xz
