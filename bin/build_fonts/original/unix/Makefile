#    make file for the fonts
#

binary:	binary.f
	$(F77) $(FFLAGS) -o binary binary.f
	binary   fnt01a.chr   font01
	binary   fnt02a.chr   font02
	binary   fnt03a.chr   font03
	binary   fnt04a.chr   font04
	binary   fnt05a.chr   font05
	binary   fnt06a.chr   font06
	binary   fnt07a.chr   font07
	binary   fnt08a.chr   font08
	binary   fnt09a.chr   font09
	binary   fnt10a.chr   font10
	binary   fnt11a.chr   font11
	binary   fnt12a.chr   font12
	binary   fnt13a.chr   font13
	binary   fnt14a.chr   font14
	binary   fnt15a.chr   font15
	binary   fnt16a.chr   font16
	binary   fnt17a.chr   font17
	binary   fnt18a.chr   font18
	binary   fnt19a.chr   font19
	binary   fnt20a.sym   font20
	binary   fnt21a.sym   font21
	binary   fnt22a.sym   font22
	binary   fnt23a.sym   font23
	binary   fnt24a.sym   font24
	binary   fnt25a.sym   font25
	binary   fnt26a.sym   font26
	binary   fnt27a.sym   font27
	binary   fnt28a.sym   font28
	binary   fnt29a.sym   font29
	binary   fnt30a.sym   font30
	binary   fnt31a.chr   font31
	binary   fnt32a.chr   font32
	binary   fnt33a.sym   font33
	sed -e "s;/usr/local/lib/pplus/;$(FONTDIR);" < list.tplate > font.list

