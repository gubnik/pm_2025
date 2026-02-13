TEX=xelatex
TEX_FLAGS=

MAIN_BUILD_SCRIPT=main.py
TITLEPAGE_BUILD_SCRIPT=titlepage.py

DRAWIO_BUILD_DIR=build/img
DRAWIO=~/.local/share/applications/drawio-x86_64-29.0.3.AppImage

DIAGRAMS :=

all: gen.main.pdf

gen.main.pdf: gen.main.tex gen.titlepage.tex ${DIAGRAMS}
	$(TEX) $(TEX_FLAGS) $<
	$(TEX) $(TEX_FLAGS) $<

gen.main.tex: main.tex gen.titlepage.tex $(MAIN_BUILD_SCRIPT)
	./$(MAIN_BUILD_SCRIPT) $< $@

gen.titlepage.tex: titlepage.tex $(TITLEPAGE_BUILD_SCRIPT)
	./$(TITLEPAGE_BUILD_SCRIPT) $< $@

$(DRAWIO_BUILD_DIR)/%.png: diagrams/%.drawio
	@mkdir -p $(DRAWIO_BUILD_DIR)
	$(DRAWIO) -x -f png -o $@ $<

clean:
	rm -f *.toc *.out *.aux *.bbl *.blg *.log main.pdf
