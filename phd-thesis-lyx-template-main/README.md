
> **Important** :warning: This template was created in 
July 2023 following the University of Manchester's Presentation of Theses Policy from _June 2022_. Make sure you check the [website](http://www.regulations.manchester.ac.uk/pgr-presentation-theses/) for the updated policy.

## How to use this template

First, understand the files. Here is a brief description of them:

- **main.lyx**: this is the main structure of the document.
- **title_page.lyx**: first page of your document. You do not need to edit it. It is included in main.lyx.
- **preliminary_pages.lyx**: this file organises the preliminary pages of the document. You do not need to edit it. It is included in main.lyx.
- **macros.lyx**: this file defines some shortcuts you can use when in Math mode. For example, you can use `\dq{h}` to write a dual quaternion $\boldsymbol{\underline{h}}$ with bold and underline notation, instead of the long `\boldsymbol{\underline{h}}`. You do not need to edit it. It is included in main.lyx.
- **introduction.lyx, appendix.lyx, publications.lyx, ...**: chapter files. Add one of these for each new chapter and include them in main.lyx. 
- **list_of_abbreviations.lyx and list_of_symbols.lyx**: self-explanatory files included in preliminary_pages.lyx.
- **references.bib**: BIB file with the list of references. It is included in main.lxy.

Ideally, you should only edit _main.lyx_ with your information and create your chapter files, making sure to include them. The _main.lyx_ file has notes to help you edit it, as long as some guidelines from the theses policy.

Table of contents and lists of figures and tables are automatically generated. You should create/edit any other list yourself, such as the lists of abbreviations and symbols. Replace _references.bib_ with your own BIB file (e.g., one generated with Zotero, Mendeley, etc) and update the file name in _main.lyx_ if necessary.


## If you are new to Lyx, here are some useful tips:
- Instead of creating a blank Lyx file for a new chapter, use the _introduction.lyx_ file (or any other one from the template) as a template. It is already configured!
- When editing other files, open _main.lyx_ in another tab in the same Lyx window. It will update the section numbers considering the whole document, you will be able to include cross references from other chapters and citations more easily, and you will have a better visualisation of expressions that use commands defined in _macros.lyx_.
