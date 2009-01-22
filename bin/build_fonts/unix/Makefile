#    Makefile for the fonts
#  1/2009 Ansley Manke From Jerry Davison's files
#
#    builds and runs the executable to write the
#    binary character and symbol font files. 
#    Just use long lists of files.
#    Expects environment variables F77 and PLOTFONTS to be defined.
#    No need for any compiler flags.

all:    binary 

binary:	binary.f
	$(F77) $(FFLAGS) -o binary binary.f
	binary   in_fnt01a.chr  fnt01.chr
	binary   in_fnt02a.chr  fnt02.chr
	binary   in_fnt03a.chr  fnt03.chr
	binary   in_fnt04a.chr  fnt04.chr
	binary   in_fnt05a.chr  fnt05.chr
	binary   in_fnt06a.chr  fnt06.chr
	binary   in_fnt07a.chr  fnt07.chr
	binary   in_fnt08a.chr  fnt08.chr
	binary   in_fnt09a.chr  fnt09.chr
	binary   in_fnt10a.chr  fnt10.chr
	binary   in_fnt11a.chr  fnt11.chr
	binary   in_fnt12a.chr  fnt12.chr
	binary   in_fnt13a.chr  fnt13.chr
	binary   in_fnt14a.chr  fnt14.chr
	binary   in_fnt15a.chr  fnt15.chr
	binary   in_fnt16a.chr  fnt16.chr
	binary   in_fnt17a.chr  fnt17.chr
	binary   in_fnt18a.chr  fnt18.chr
	binary   in_fnt19a.chr  fnt19.chr
	binary   in_fnt20a.sym  fnt20.sym
	binary   in_fnt21a.sym  fnt21.sym
	binary   in_fnt22a.sym  fnt22.sym
	binary   in_fnt23a.sym  fnt23.sym
	binary   in_fnt24a.sym  fnt24.sym
	binary   in_fnt25a.sym  fnt25.sym
	binary   in_fnt26a.sym  fnt26.sym
	binary   in_fnt27a.sym  fnt27.sym
	binary   in_fnt28a.sym  fnt28.sym
	binary   in_fnt29a.sym  fnt29.sym
	binary   in_fnt30a.sym  fnt30.sym
	binary   in_fnt31a.chr  fnt31.chr
	binary   in_fnt32a.chr  fnt32.chr
	binary   in_fnt33a.sym  fnt33.sym

install:
	cp fnt01.chr $(PLOTFONTS)
	cp fnt02.chr $(PLOTFONTS)
	cp fnt03.chr $(PLOTFONTS)
	cp fnt04.chr $(PLOTFONTS)
	cp fnt05.chr $(PLOTFONTS)
	cp fnt06.chr $(PLOTFONTS)
	cp fnt07.chr $(PLOTFONTS)
	cp fnt08.chr $(PLOTFONTS)
	cp fnt09.chr $(PLOTFONTS)
	cp fnt10.chr $(PLOTFONTS)
	cp fnt11.chr $(PLOTFONTS)
	cp fnt12.chr $(PLOTFONTS)
	cp fnt13.chr $(PLOTFONTS)
	cp fnt14.chr $(PLOTFONTS)
	cp fnt15.chr $(PLOTFONTS)
	cp fnt16.chr $(PLOTFONTS)
	cp fnt17.chr $(PLOTFONTS)
	cp fnt18.chr $(PLOTFONTS)
	cp fnt19.chr $(PLOTFONTS)
	cp fnt20.sym $(PLOTFONTS)
	cp fnt21.sym $(PLOTFONTS)
	cp fnt22.sym $(PLOTFONTS)
	cp fnt23.sym $(PLOTFONTS)
	cp fnt24.sym $(PLOTFONTS)
	cp fnt25.sym $(PLOTFONTS)
	cp fnt26.sym $(PLOTFONTS)
	cp fnt27.sym $(PLOTFONTS)
	cp fnt28.sym $(PLOTFONTS)
	cp fnt29.sym $(PLOTFONTS)
	cp fnt30.sym $(PLOTFONTS)
	cp fnt31.chr $(PLOTFONTS)
	cp fnt32.chr $(PLOTFONTS)
	cp fnt33.sym $(PLOTFONTS)
	cp fondat.dat $(PLOTFONTS)

clean: 
	rm -f binary
