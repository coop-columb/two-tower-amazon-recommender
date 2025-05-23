# Minimal Makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# For now, create minimal HTML documentation
html:
	@echo "Building documentation..."
	@mkdir -p $(BUILDDIR)/html
	@cp index.html $(BUILDDIR)/html/ 2>/dev/null || echo "No index.html found, creating minimal documentation"
	@echo '<!DOCTYPE html><html><head><title>Two-Tower Amazon Recommender</title></head><body><h1>Two-Tower Amazon Recommender Documentation</h1><p>Documentation is under development.</p></body></html>' > $(BUILDDIR)/html/index.html