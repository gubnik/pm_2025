TEX=xelatex
TEX_FLAGS=

MAIN_BUILD_SCRIPT=main.py
TITLEPAGE_BUILD_SCRIPT=titlepage.py

GENERATED_IMAGES_DIR=img/gen

GRAPHS :=
GRAPHS += img/gen/pulse_1.png
GRAPHS += img/gen/pulse_3.png
GRAPHS += img/gen/pulse_5.png
GRAPHS += img/gen/pulse_7.png

all: gen.main.pdf

gen.main.pdf: gen.main.tex gen.titlepage.tex $(GRAPHS)
	$(TEX) $(TEX_FLAGS) $<
	$(TEX) $(TEX_FLAGS) $<

gen.main.tex: main.tex gen.titlepage.tex $(MAIN_BUILD_SCRIPT)
	./$(MAIN_BUILD_SCRIPT) $< $@

gen.titlepage.tex: titlepage.tex $(TITLEPAGE_BUILD_SCRIPT)
	./$(TITLEPAGE_BUILD_SCRIPT) $< $@

$(GENERATED_IMAGES_DIR)/%.png: graphs.py
	@mkdir -p $(GENERATED_IMAGES_DIR)
	python3 $<

clean:
	rm -f *.toc *.out *.aux *.bbl *.blg *.log main.pdf
