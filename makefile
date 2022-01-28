SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard de/figures/figure*.py)

all: $(patsubst de/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: de/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v

clean:
	rm -rf output de/data/GSE*

download: de/data/GSE106127_inst_info.txt.xz de/data/GSE92742_Broad_LINCS_Level2.csv.xz de/data/GSE70138_Broad_LINCS_Level2.csv.xz

de/data/%.xz:
	wget -N -P ./de/data https://syno.seas.ucla.edu:9001/de-learning/$*.xz
