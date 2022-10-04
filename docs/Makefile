APER := thesis
BUILD_DIR := ./build

ARTIFACT_DIR := ../artifacts

CONTENT_TEX := $(wildcard content/*.tex)
BUILD_EXTENSIONS := {aux,bbl,blg,log,out,thm,toc}

# Create the pdf in BUILD_DIR and clean build files
build: $(PAPER).pdf move clean

artifacts: $(PAPER).pdf
	mkdir $(ARTIFACT_DIR)
	mv $< $(ARTIFACT_DIR)

# Call with e.g. "make main.pdf"
%.pdf: %.tex $(CONTENT_TEX)
	pdflatex $<
	- bibtex $*
	pdflatex $<
	pdflatex $<

move:
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)
	mv *.pdf $(BUILD_DIR)/

.PHONY: clean
clean:
	rm -f comment.cut $(PAPER).$(BUILD_EXTENSIONS) 
	rm -f content/*.$(BUILD_EXTENSIONS)
	rm -f *-converted-to.pdf
