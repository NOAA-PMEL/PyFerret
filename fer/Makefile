# Makefile for Program FERRET  
# consolidated SUN, DEC, SGI version 2/2/93 *sh*
# Modified to run on AXP *jd* 9.28.93
# added LDFLAGS for link stmt *sh* 10/19/93
# for reorganized development (debug) directories *sh* 1/94
# 3/1: to build the Solaris version
#	make gcompile     # ("sun4" default works fine for this)
#	make solarisxgks
# 3/22: Note: to make solaris version use "make HOSTTYPE=solaris"
# 11/94: had to interchange the ordering of the libraries 
#	$(PPLDIR)/plotlib/pll.a and $(PPLDIR)/symlib/sym.a because for
#	mysterious reasons the backslash after sym.a was passing through on
#	SunOS makes (though not on OSF makes) ???
# 12/94 official changeover to version 320 -- expanded list of "special"
#	routines 
# 8/95 added "GUI" and "DEBUG" and automated the insertion of this information
#	into the program
# 8/95 Makefile totally revamped *kob*
#	- each system now has only two entries
#		- hosttype_fortran
#		- hosttype
#	- the special flags which turn on debuggin, gui output, etc are all
#	  culled from the below make options
#	- to use FHECKOUT or TMCHECKOUT areas, they must be explicitly 
# 	  passed into the makefile..ie:
#		 make FCHECKOUT="/home/rogue/hankin/fer/src/checkdir/*.o"
# 9/95: removed reference to GKS3D object file in DEC entries *kob*
# 9/96: add compile options for hp *kob*
# 9/96: add hp_gui target *jc*
# 11/96: modify locations for netcdf and hdf libraries for 
#	 fisheries sgi (cyborg) *kob*
# 11/96: when "debug" is specificed link with tmap_debuglib.a instead of
#	tmap_lib.a *sh*
#	==> follow up 11/22/96 to eliminate unneeded "/"
# 11/96: pass "-static" option to linker for alpha_DODS *kob*
# 11/96: add -DX_REFRESH to all sgi entries...was only in nogcompile *kob*
# 11/96: New HDF lib (4.0r2) requires "-ljpeg". (Possibly will revert to old).
# 12/96: Make above change for gui link on osf *kob*
# 12/96: modify sed command when updating revision data ro be rid of "/" *kob*
# usage:
#	make 		- create distributable FERRET on current platform
#	make gcompile	- compile with debug all the code on current platform
#	make nogcompile	- compile without debug all the code ...
#	make libraries	- build all the libraries on the current platform
#			  (needed when .obj files are changed but .f are not)
#	make update	- create next DEC beta version including date stamp
#	make debug	- build custom debugging version on current platform
#			  (update and debug require TMCHECKOUT="...")
#	make update_date- update date stamp COMMON DATA only
#	make WHATGKS=xgks - to control GKS libraries used
#	make EXECNAME=special_ferret - to control executable name
# added 8/95 *kob*
#	make fortran 	- creates a debug version of the ferret fortran
#			  executable
#	make gui	- creates a distributable version of ferret which
#			  also encompasses the gui
#	make debug_gui	- creates a debug version of the above
#  !!! make MEMSIZE=Fsmall_mem - to control FERRET memory size
#  !!! Note: this didn't work because common is included with FORTRAN includes
#            rather than cpp includes so the #ifdef is illegal!!
# 10/95 *kob* - added GUICHECKOUT and PPLCHECKOUT logic
# 11/95 *kob* - added F77 recursive variable to allow for use of either
#               f77 or g77
# 11/95 *sh* - added passing of GUICHECKOUT and PPLCHECKOUT under "update"
#	     - also corrected bug that PPLCHECKOUT wasn't used in the link stmt
# 1/96  *kob* - folded in ibm mods to official makefile
# 3/96  *JC* - added -llist to GUI_LIB under sun4_gui: target
#            - the library came from http://www.cis.ufl.edu/~bcs/list-2.1/
# 3/15/96 *kob* - addding a alpha_DODS entry
# 5/01/96 *JC*  - added solaris_gui: target
# 6/03/96 *kob* - implement modified fer dir structure to include a 
#		  c source code directory called ccr
# 8/21/96 *jc*  - added aix_gui: target
# 01/13/97 *jc* - CNSD put a new compiler on corona
#                 Making changes to solaris compilation: use -lM77 
#		  instead of absolute path
# 01/15/97 *kob* - need to put in explicit reference to new F77 lib's on 
#		   solaris because of new compiler
# 06/27/97 *kob* - added in linux entries
#		 - added a reference to save_arg_pointers.o to FER_OBJS 
#		   for alpha entries(will need to do all) - this is
#		   necessary due to new curvilinear coordinate stuff
# 7/7/97 *kob* - updated solaris sections for ferret v5

# 7/24/97 *kob* - add a FER_LINK_DIR so that CVS makes wil now work
FER_LINK_DIR=..
GUI =no GUI
DEBUG =
GFLAG =
DEB_LIB =
WHATGKS = "xgks"
F77 =f77
TMCHECKOUT =
FCHECKOUT =
GUICHECKOUT=
PPLCHECKOUT=
GKSFFLAGS = -D$(WHATGKS)
MEMSIZE = Flarge_mem
MEMFLAGS = -D$(MEMSIZE)
CCINCLUDES = -I/usr/local/include -I../common \
        -I$(TMAP_LOCAL)/src/xgks-2.5.5/port \
        -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib \
        -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm
FCHK = /home/rogue/hankin/fer/src/checkdir
FERMAIN =
GUI_LIB=
DIRS = ctx dat doo fmt gnl ino mem ocn plt rpn stk utl xeq special ccr


all_systems:
	${MAKE} $(HOSTTYPE) F77="$(F77)"

debug:
	${MAKE} $(HOSTTYPE) GFLAG="-g"  DEB="debug" \
				F77="$(F77)" \
				FCHECKOUT="$(FCHECKOUT)"  \
				TMCHECKOUT="$(TMCHECKOUT)" \
			 	GUICHECKOUT="$(GUICHECKOUT)" \
				PPLCHECKOUT="$(PPLCHECKOUT)"
gui:
	${MAKE} $(HOSTTYPE)_gui GUI="GUI" \
				F77="$(F77)" 

debug_gui:
	${MAKE} $(HOSTTYPE)_gui DEB="debug" GFLAG="-g"  \
				F77="$(F77)" \
				FCHECKOUT="$(FCHECKOUT)"  \
				TMCHECKOUT="$(TMCHECKOUT)" \
			 	GUICHECKOUT="$(GUICHECKOUT)" \
				PPLCHECKOUT="$(PPLCHECKOUT)"
debug_DODS:
	${MAKE} $(HOSTTYPE)_DODS DEB="debug" GFLAG="-g"  \
				F77="$(F77)" \
				FCHECKOUT="$(FCHECKOUT)"  \
				TMCHECKOUT="$(TMCHECKOUT)" \
			 	GUICHECKOUT="$(GUICHECKOUT)" \
				PPLCHECKOUT="$(PPLCHECKOUT)"

make fortran:
	${MAKE} $(HOSTTYPE)_FORTRAN

update:
	${MAKE} update_date DATE=`/bin/date +%D`
	${MAKE} debug	FCHECKOUT="$(FCHECKOUT)" \
		 	TMCHECKOUT="$(TMCHECKOUT)" \
			GUICHECKOUT="$(GUICHECKOUT)" \
			PPLCHECKOUT="$(PPLCHECKOUT)"
				
update_date:
	sed "s&xxxxxxxx&$(DATE)&" <special/xmake_date_data.template >special/xmake_date_data.F

decstation_FORTRAN:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=f77 \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c $(GFLAG)" \
		FFLAGS="-c -u -cpp -G 0 -fpe1 $(GFLAG) -C -Df7732_bug -Ddebug\
			-DDEC_FORTRAN $(MEMFLAGS) $(GKSFFLAGS)"\
		LDFLAGS=" " \
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD="ppl_bdead.o" \
		MOVIELIB=" " \
		TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
		GKSOBJ= \
		GKSLIB="-lGKS3Dmotif -lGKS3Ddforbnd -lGKS3D" \
		HDFLIB="-ldf" \
		CDFLIB="-lnetcdf" \
		SYSLIB="-lMrm -lXm -lXt -ldwt -lcursesX -li -lc -lX11" \
		RANLIB="/usr/bin/ranlib"

decstation:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=cc \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c $(GFLAG)" \
		FFLAGS="-c -u -cpp -G 0 -fpe1 $(GFLAG) -C -Df7732_bug -Ddebug\
			-DDEC_FORTRAN $(MEMFLAGS) $(GKSFFLAGS)"\
		LDFLAGS=" " \
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD="ppl_bdead.o" \
		MOVIELIB=" " \
		TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
		GKSOBJ= \
		GKSLIB="-lGKS3Dmotif -lGKS3Ddforbnd -lGKS3D" \
		HDFLIB="-ldf" \
		CDFLIB="-lnetcdf" \
		SYSLIB="-lMrm -lXm -lXt -ldwt -lcursesX -li -lc -lX11" \
		RANLIB="/usr/bin/ranlib"

sun4_FORTRAN:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=f77 \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -D_NO_PROTO -c $(GFLAG)" \
                FFLAGS="-c -Nl160 $(GFLAG) -C -u -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="" \
		XTRA_OBJ="special/ppl_fake_externals.o" \
		FEROBJS="special/fermain_fortran.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-lX11 -Bstatic -lV77" \
		RANLIB="/usr/bin/ranlib"

sun4:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=cc \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -D_NO_PROTO -c $(GFLAG)" \
                FFLAGS="-c -Nl160 $(GFLAG) -C -u -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="" \
		XTRA_OBJ="special/ppl_fake_externals.o" \
		FEROBJS="ccr/fermain_c.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf" \
		CDFLIB="-lnetcdf" \
		SYSLIB="-L/usr/local/lib -L/home/e1/tmap/local/sun/lib -L/usr/lang/SC1.0/cg87 -L/usr/lang/SC1.0 -lX11 -Bstatic -lV77 -lF77 -lm" \
		RANLIB="/usr/bin/ranlib"



sun4_gui:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		GUI="GUI" \
		LD=cc \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -D_NO_PROTO -c $(GFLAG)" \
                FFLAGS="-c -Nl160 $(GFLAG) -C -u -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="" \
		FERMAIN="gui/ferret.o" \
		GUI_LIB="gui/libgui.a -llist"\
		XTRA_OBJ="special/ppl_fake_externals.o" \
		FEROBJS="$(FEROBJS_GUI)" \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB="" \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf" \
		CDFLIB="-lnetcdf" \
		SYSLIB="-L/usr/local/lib -L/home/e1/tmap/local/sun/lib -L/usr/lang/SC1.0/cg87 -L/usr/lang/SC1.0 -L/usr/lib -Bstatic -lXm -lXt -lc $(TMAP_LOCAL)/lib/libXpm.a -lX11 -lV77 -lF77 -lm" \
		RANLIB="/usr/bin/ranlib"


alpha_FORTRAN:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=f77 \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c $(GFLAG) -DUSG -DDEC_ALPHA" \
                FFLAGS="-c -u $(GFLAG) -cpp -w1 -align dcommons -align records -integer_size 32 -real_size  32 -check underflow -check overflow -Df7732_bug -DDEC_FORTRAN -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-non_shared" \
		XTRA_OBJ=" " \
		FEROBJS="special/fermain_fortran.o ccr/save_arg_pointers.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-lX11 -ldnet_stub"\
		RANLIB="/usr/bin/ranlib"

alpha:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=cc \
		CCFLAGS="-c $(GFLAG) -DUSG -DDEC_ALPHA" \
		F77="$(F77)" \
		CC=cc \
                FFLAGS="-c -u $(GFLAG) -cpp -w1 -align dcommons -align records -integer_size 32 -real_size  32 -check underflow -check overflow -Df7732_bug -DDEC_FORTRAN -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-non_shared $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="ccr/fermain_c.o ccr/save_arg_pointers.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf -ljpeg -lz" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-lm -lUfor -lfor -lFutil -lots -lX11 -ldnet_stub"\
		RANLIB="/usr/bin/ranlib"

alpha_DODS:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=g++ \
		CCFLAGS="-c $(GFLAG) -DUSG -DDEC_ALPHA" \
		F77="$(F77)" \
		CC=cc \
                FFLAGS="-c -u $(GFLAG) -cpp -w1 -align dcommons -align records -integer_size 32 -real_size  32 -check underflow -check overflow -Df7732_bug -DDEC_FORTRAN -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-static -non_shared $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="ccr/fermain_c.o ccr/save_arg_pointers.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c_DODS" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf -ljpeg -lz" \
                CDFLIB="-L/home/e2/tmap/local/osf/src/DODS/lib -lnc-dods -ldap++ -lnc-dods -ldap++ -lwww -lexpect5.21 -ltcl7.6 -lg++" \
                SYSLIB="-lm -lUfor -lfor -lFutil -lots -lX11 -ldnet_stub /usr/local/lib/gcc-lib/alpha-dec-osf3.0/2.7.2/libgcc.a"\
		RANLIB="/usr/bin/ranlib"

alpha_gui:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		GUI="GUI" \
		LD=cc \
		CCFLAGS="-c $(GFLAG) -DUSG -DDEC_ALPHA" \
		F77="$(F77)" \
		CC=cc \
		FERMAIN="gui/ferret.o" \
		GUI_LIB="gui/libgui.a -llist" \
                FFLAGS="-c -u $(GFLAG) -cpp -w1 -align dcommons -align records -integer_size 32 -real_size  32 -check underflow -check overflow -Df7732_bug -DDEC_FORTRAN -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-non_shared $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="ccr/save_arg_pointers.o $(FEROBJS_GUI)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB="$(TMAP_LOCAL)/lib/libXpm.a" \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf -ljpeg -lz" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-lXm -lXt -lc -lm -lUfor -lfor -lFutil -lots -lX11 -ldnet_stub"\
		RANLIB="/usr/bin/ranlib"

alpha_DODS_gui:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		GUI="GUI"\
		LD=g++ \
		CCFLAGS="-c $(GFLAG) -DUSG -DDEC_ALPHA" \
		F77="$(F77)" \
		CC=cc \
		FERMAIN="gui/ferret.o" \
		GUI_LIB="gui/libgui.a -llist" \
                FFLAGS="-c -u $(GFLAG) -cpp -w1 -align dcommons -align records -integer_size 32 -real_size  32 -check underflow -check overflow -Df7732_bug -DDEC_FORTRAN -Ddebug $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-static -non_shared $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="ccr/save_arg_pointers.o $(FEROBJS_GUI)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_DODS" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB="$(TMAP_LOCAL)/lib/libXpm.a" \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib -ldf -ljpeg -lz" \
                CDFLIB="-L/home/e2/tmap/local/osf/src/DODS/lib -lnc-dods -ldap++ -lnc-dods -ldap++ -lwww -lexpect5.21 -ltcl7.6 -lg++" \
                SYSLIB="-lXm -lXt -lm -lUfor -lfor -lFutil -lots -lX11 -ldnet_stub /usr/local/lib/gcc-lib/alpha-dec-osf3.0/2.7.2/libgcc.a"\
		RANLIB="/usr/bin/ranlib"

solaris_FORTRAN:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=f77 \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -c -I/usr/openwin/include" \
		F77="$(F77)" \
                FFLAGS="-c -Nl160 $(GFLAG) -C -u -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Dsolaris -Ddebug -DX_REFRESH $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-L/usr/openwin/lib -v" \
		XTRA_OBJ="" \
		FEROBJS="special/fermain_fortran.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="/opt/lib/libdf.a" \
		CDFLIB="/opt/lib/libnetcdf.a" \
                SYSLIB=" -lX11 -ldl -lc -Bstatic -L/usr/ucblib -lucb -L/opt/SUNWspro/SC4.0/lib -lV77 -lelf -Bdynamic" \
		RANLIB="/bin/touch"

solaris:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=cc \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -c -I/usr/openwin/include" \
		F77="$(F77)" \
                FFLAGS="-c -Nl160 $(GFLAG) -C -u -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Dsolaris -Ddebug -DX_REFRESH $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-L/usr/openwin/lib -v" \
		XTRA_OBJ="" \
		FEROBJS="ccr/fermain_c.o ccr/save_arg_pointers.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="/opt/lib/libdf.a" \
		CDFLIB="/opt/lib/libnetcdf.a" \
                SYSLIB=" -lX11 -ldl -lc -Bstatic -L/usr/ucblib -lucb -L/opt/SUNWspro/SC4.0/lib -lV77 -lelf -lM77 -lF77 -Bdynamic -lsunmath -lm -lc" \
		RANLIB="/bin/touch"

# 97.01.13 Adjustments for new F77 compiler on corona. This was the old line.
#                SYSLIB=" -lX11 -ldl -lc -Bstatic -L/usr/ucblib -lucb -lV77 -lelf /opt/SUNWspro/SC3.0/lib/libM77.a -lF77 -lsunmath -lm -lc" \

# Solaris link -- 8/25/95 Berkeley signals experiment:
# we tested inserting /usr/ucblib/libucb.a ahead of -lc -- no effect.  The 
# windows still fail to update when resized if MODE REFRESH is off


solaris_gui:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		GUI="GUI" \
		LD=cc \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -c -I/usr/openwin/include" \
		F77="$(F77)" \
                FFLAGS="-c -Nl160 $(GFLAG) -C -u -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Dsolaris -Ddebug -DX_REFRESH $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-L/usr/openwin/lib -v" \
		FERMAIN="gui/ferret.o" \
		GUI_LIB="gui/libgui.a -L$(TMAP_LOCAL)/lib -llist"\
		XTRA_OBJ="" \
		FEROBJS="ccr/save_arg_pointers.o $(FEROBJS_GUI)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="/opt/lib/libdf.a" \
		CDFLIB="/opt/lib/libnetcdf.a" \
                SYSLIB=" -lX11 -ldl -lc -Bstatic -L/usr/ucblib -lucb -L/opt/SUNWspro/SC4.0/lib -lV77 -lelf -lM77 -lF77 $(TMAP_LOCAL)/lib/libXpm.a -lXm -lXt -Bdynamic -lsunmath -lm -lc /usr/ccs/lib/libgen.a" \
		RANLIB="/bin/touch"

#                SYSLIB=" -lX11 -ldl -lc -Bstatic -L/usr/ucblib -lucb -L/opt/SUNWspro/SC4.0/lib -lV77 -lelf -lM77 -lF77 -Bdynamic -lsunmath -lm -lc $(TMAP_LOCAL)/lib/libXpm.a -lXm -lXt /usr/ccs/lib/libgen.a" \
#                SYSLIB=" -lX11 -ldl -lc -Bstatic -L/usr/ucblib -lucb -lV77 -lelf /opt/SUNWspro/SC3.0/lib/libM77.a -lF77 -lsunmath -lm -lc $(TMAP_LOCAL)/lib/libXpm.a -lXm -lXt /usr/ccs/lib/libgen.a" \

sgi_FORTRAN:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=f77 \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="$(GFLAG) -cckr -c -G 0 -Dsgi -Dsun -DSGI_SIGNALS" \
		FFLAGS="$(GFLAG) -c -Nl160 -u  -Dsun -Dsgi -G 0 -DX_REFRESH $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-v" \
		FEROBJS="special/fermain_fortran.o $(FEROBJS_DFLT)"\
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
		HDFLIB="-L$(TMAP_LOCAL)/lib -ldf" \
		CDFLIB="-L$(TMAP_LOCAL)/lib -lnetcdf" \
                SYSLIB="-lX11 -lsun"\
		RANLIB="ar ts"

sgi:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=cc \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="$(GFLAG) -cckr -c -G 0 -Dsgi -Dsun -DSGI_SIGNALS" \
		FFLAGS="$(GFLAG) -c -Nl160 -u  -Dsgi -Dsun -G 0 -DX_REFRESH $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-v " \
		FEROBJS="ccr/fermain_c.o $(FEROBJS_DFLT)"\
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
		HDFLIB="-L$(TMAP_LOCAL)/lib -ldf" \
		CDFLIB="-L$(TMAP_LOCAL)/lib -lnetcdf" \
                SYSLIB="-lX11 -lsun -nocount -lftn -lm -lc /usr/lib/crtn.o"\
		RANLIB="ar ts"


sgi_gui:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		GUI="GUI" \
		LD=cc \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="$(GFLAG) -cckr -c -G 0 -Dsgi -Dsun -DSGI_SIGNALS -DSGI_POPUPS" \
		FFLAGS="$(GFLAG) -c -Nl160 -u  -Dsgi -Dsun -G 0 -DX_REFRESH $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-v " \
                FERMAIN="gui/ferret.o" \
                GUI_LIB="gui/libgui.a -L$(TMAP_LOCAL)/lib -llist"\
		FEROBJS="$(FEROBJS_GUI)"\
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
		HDFLIB="-L$(TMAP_LOCAL)/lib -ldf" \
		CDFLIB="-L$(TMAP_LOCAL)/lib -lnetcdf" \
                SYSLIB="sgi/sgi_intl.o sgi/sgi_visual.o -lXm sgi/Shell.o -lXt -lX11 -lsun -nocount -lftn -lm -lc $(TMAP_LOCAL)/lib/libXpm.a /usr/lib/crtn.o"\
		RANLIB="ar ts"


hp_FORTRAN:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=f77 \
		F77="$(F77)" \
		CC=c89 \
		CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/include/X11R5/ -I../common -I$TMAP_LOCAL/src/xgks-2.5.5/port -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm -c -Dsun -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -N -Ae" \
		FFLAGS=" -c +es +e -u -w +U77 -N +B +E4 $(GFLAG) $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-v" \
		FEROBJS="special/aix_routines.o special/fermain_fortran.o $(FEROBJS_DFLT)"\
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="../fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
		HDFLIB="/users/ferret/HDF3.3r4/hdf/lib/libdf.a" \
		CDFLIB="/usr/local/lib/libnetcdf.a" \
                SYSLIB="/usr/lib/X11R5/libX11.a -lm -lBSD /usr/lib/libU77.a"\
		RANLIB="/bin/touch"

hp:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		LD=c89 \
		F77="$(F77)" \
		CC=c89 \
		CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/include/X11R5/ -I../common -I$TMAP_LOCAL/src/xgks-2.5.5/port -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm -c -Dsun -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -N -Ae " \
		FFLAGS=" -c +es +e -u -w +U77 -N +B +E4 $(GFLAG) $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-v" \
		FEROBJS="ccr/fermain_c.o special/aix_routines.o  $(FEROBJS_DFLT)"\
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="../fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
		HDFLIB="/users/ferret/HDF3.3r4/hdf/lib/libdf.a" \
		CDFLIB="/usr/local/lib/libnetcdf.a" \
                SYSLIB="/usr/lib/X11R5/libX11.a -lm -lBSD /usr/lib/libU77.a -lcl -lisamstub"\
		RANLIB="/bin/touch"


hp_gui:
	${MAKE} ferret \
		DEBUG="$(DEB)" \
		GUI="GUI" \
		LD=c89 \
		F77="$(F77)" \
		CC=c89 \
		CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/include/X11R5/ -I../common -I$TMAP_LOCAL/src/xgks-2.5.5/port -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm -c -Dsun -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -N -Ae -DSOLARIS" \
		FFLAGS=" -c +es +e -u -w +U77 -N +B +E4 $(GFLAG) $(MEMFLAGS) $(GKSFFLAGS)" \
		LDFLAGS="-v" \
                FERMAIN="gui/ferret.o" \
                GUI_LIB="gui/libgui.a -L$(TMAP_LOCAL)/lib -llist"\
		FEROBJS="special/aix_routines.o  $(FEROBJS_GUI)"\
		XTRA_OBJ=" " \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="../fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
		HDFLIB="/users/ferret/HDF3.3r4/hdf/lib/libdf.a" \
		CDFLIB="/usr/local/lib/libnetcdf.a" \
                SYSLIB="/usr/lib/Motif1.2/libXm.a /usr/lib/X11R5/libXt.a /usr/lib/X11R5/libX11.a $(TMAP_LOCAL)/lib/libXpm.a -lm -lBSD /usr/lib/libU77.a -lcl -lisamstub"\
		RANLIB="/bin/touch"


aix_FORTRAN:
	${MAKE} ferret_aix \
                F77=xlf \
		LD=xlf \
		CC=cc \
		FFLAGS="-c -u -qfixed=80 -qxlf77=oldboz" \
		CPPFLAGS="-P -Dunix  -DNO_SNGL  -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF" \
		LDFLAGS="-v" \
		XTRA_OBJ="ccr/fermain_c.o special/aix_routines.o $(FEROBJS_DFLT)" \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib/hdf -ldf" \
		CDFLIB="-L/usr/local/lib -lnetcdf" \
                SYSLIB="-lc -lX11 " \
		RANLIB="/usr/bin/ranlib"

#removed -DNO_PASSED_CONCAT and -DNO_FREE_READS with new compiler *kob* 12/95
aix:
	${MAKE} ferret_aix \
                F77=xlf \
		LD=cc \
		CC=cc \
		FFLAGS="-c -u -qfixed=80 -qxlf77=oldboz" \
		CPPFLAGS="-P -Dunix  -DNO_SNGL  -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF" \
		LDFLAGS="-v -bloadmap:load.txt" \
		XTRA_OBJ="ccr/fermain_c.o special/aix_routines.o $(FEROBJS_DFLT)" \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib/hdf -ldf" \
		CDFLIB="-L/usr/local/lib -lnetcdf" \
                SYSLIB="-lX11 -lm -lc -lIM -liconv -bnso -bI:/lib/syscalls.exp -lxlf -lxlf90 -bI:/usr/lpp/xlf/lib/lowsys.exp -bI:/usr/lpp/X11/bin/smt.exp" \
		RANLIB="/usr/bin/ranlib"

aix_gui:
	${MAKE} ferret_aix \
                F77=xlf \
		LD=cc \
		CC=cc \
                FERMAIN="gui/ferret.o" \
                GUI_LIB="gui/libgui.a -llist" \
                FFLAGS="-c -u -qfixed=80 -qxlf77=oldboz" \
		CPPFLAGS="-P -Dunix  -DNO_SNGL  -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF" \
		LDFLAGS="-v -bloadmap:load.txt" \
		XTRA_OBJ="ccr/fermain_c.o special/aix_routines.o $(FEROBJS_DFLT)" \
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
                GUICHECKOUT="$(GUICHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB="$(TMAP_LOCAL)/lib/libXpm.a" \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib/hdf -ldf" \
		CDFLIB="-L/usr/local/lib -lnetcdf" \
                SYSLIB="-lX11 -lXm -lXt -lm -lc -lIM -liconv -bnso -bI:/lib/syscalls.exp -lxlf -lxlf90 -bI:/usr/lpp/xlf/lib/lowsys.exp -bI:/usr/lpp/X11/bin/smt.exp" \
		RANLIB="/usr/bin/ranlib"

linux_fortran:
	${MAKE} ferret_linux \
		DEBUG="$(DEB)" \
		LD=f90 \
		CCFLAGS="-c $(GFLAG) " \
		F77="$(F77)" \
		CC=cc \
		FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl90 -x77 -w" \
		LDFLAGS="-v $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="special/fermain_fortran.o  special/linux_routines.o special/ppl_fake_externals.o dat/*.o $(FER_LINK_DIR)/fmt/src/x*.o ccr/save_arg_pointers.o $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_fortran" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib  -ldf -ljpeg -lz" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-L/usr/X11R6/lib -lX11 /usr/lib/libf2c.a /usr/lib/libc.a /usr/lib/libc_p.a /home/l1/kobrien/src/f2c/libF77/libF77.a /home/l1/kobrien/src/f2c/libI77/libI77.a /usr/local/lib/f90/libf90.a"\
		RANLIB="/usr/bin/ranlib"
#-lg -lm -L/usr/local/lib/f90 -lf90 -L/usr/X11R6/lib -lX11 -lf2c

linux:
	${MAKE} ferret_linux \
		DEBUG="$(DEB)" \
		LD=cc \
		CCFLAGS="-c $(GFLAG) " \
		F77="$(F77)" \
		CC=cc \
		FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl90 -x77 -w" \
		LDFLAGS="-v -static $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="ccr/fermain_c.o  special/linux_routines.o special/ppl_fake_externals.o dat/*.o $(FER_LINK_DIR)/fmt/src/x*.o ccr/save_arg_pointers.o  $(FEROBJS_DFLT)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferret_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB=" " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib  -ldf -ljpeg -lz" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-L/usr/X11R6/lib -lX11 -static /usr/lib/libf2c.a /usr/lib/libc.a /usr/lib/libc_p.a /home/l1/kobrien/src/f2c/libF77/libF77.a /home/l1/kobrien/src/f2c/libI77/libI77.a /usr/local/lib/f90/libf90.a -lm"\
		RANLIB="/usr/bin/ranlib"

linux_gui:
	${MAKE} ferret_linux \
		DEBUG="$(DEB)" \
		GUI="GUI" \
		LD=cc \
		CCFLAGS="-c $(GFLAG) " \
		F77="$(F77)" \
		CC=cc \
		FERMAIN="gui/ferret.o" \
		GUI_LIB="gui/libgui.a -L$(TMAP_LOCAL) -llist" \
		FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl90 -x77 -w" \
		LDFLAGS="-v -static $(GFLAG)" \
		XTRA_OBJ=" " \
		FEROBJS="special/linux_routines.o special/ppl_fake_externals.o dat/*.o $(FER_LINK_DIR)/fmt/src/x*.o ccr/save_arg_pointers.o $(FEROBJS_GUI)"\
		FCHECKOUT="$(FCHECKOUT)" \
		TMCHECKOUT="$(TMCHECKOUT)" \
		GUICHECKOUT="$(GUICHECKOUT)" \
		PPLCHECKOUT="$(PPLCHECKOUT)" \
		WHATGKS="xgks" \
		EXECNAME="ferretgui_c" \
		PPLDIR="$(FER_LINK_DIR)/ppl" \
		BDEAD=" " \
		MOVIELIB="$(TMAP_LOCAL)/lib/libXpm.a " \
                TMLIB="$(FER_LINK_DIR)/fmt/libs/tmap_$(DEB)lib.a" \
                GKSOBJ=" " \
		GKSLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		HDFLIB="-L/usr/local/lib  -ldf -ljpeg -lz" \
		CDFLIB="-lnetcdf" \
                SYSLIB="-L/usr/X11R6/lib -static /usr/lib/libf2c.a /usr/lib/libc.a /usr/lib/libc_p.a /home/l1/kobrien/src/f2c/libF77/libF77.a /home/l1/kobrien/src/f2c/libI77/libI77.a /usr/local/lib/f90/libf90.a -lm $(TMAP_LOCAL)/lib/libXpm.a -lXm -lXt -lXext -lX11 -lSM -lICE"\
		RANLIB="/usr/bin/ranlib"



gcompile:
	${MAKE} $(HOSTTYPE)_gcompile\
		 F77="$(F77)" \
		 GKSFFLAGS=$(GKSFFLAGS)\
		 MEMFLAGS=$(MEMFLAGS)

decstation_gcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c -g" \
		FFLAGS="-c -u -cpp -fpe1 -g -C -Df7732_bug -DDEC_FORTRAN\
		-Ddebug $(GKSFFLAGS) $(MEMFLAGS)"

alpha_gcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c -g -DUSG -DDEC_ALPHA" \
		FFLAGS="-c -u -cpp -g -C  -Df7732_bug -DDEC_FORTRAN\
		-integer_size 32 -real_size 32 -align dcommons -align records\
		-Ddebug $(GKSFFLAGS) $(MEMFLAGS)"

sun4_gcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -D_NO_PROTO -c -g" \
		FFLAGS="-c -Nl160 -u  -C -g -Ddebug $(GKSFFLAGS) $(MEMFLAGS)"

solaris_gcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -c -g -I/usr/openwin/include" \
		FFLAGS="-c -Nl160 -u  -C -g -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Dsolaris -Ddebug -DX_REFRESH\
		$(GKSFFLAGS) $(MEMFLAGS)"

sgi_gcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-cckr -c -g -G 0 -Dsgi -Dsun -DSGI_SIGNALS" \
		FFLAGS="-c -Nl160 -u  -Dsun -Dsgi -G 0 -C -g -Ddebug -DX_REFRESH	$(GKSFFLAGS) $(MEMFLAGS)"

hp_gcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-g -I/usr/ucbinclude -I/usr/include/X11R5 -Dsun -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -DHP_ADDS -DFlarge_mem -Dxgks -DSTDERR_UNIT_SEVEN -N -Ae" \
		FFLAGS="-g -c +e +es -u -w +U77 -N +B +E4 -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -DHP_ADDS -DFlarge_mem -Dxgks -DSTDERR_UNIT_SEVEN $(GKSFFLAGS) $(MEMFLAGS)" 

# removed -DNO_PASSED_CONCAT and -DNO_FREE_READS with new compiler *kob* 12/95
aix_gcompile:
	${MAKE} go_compile_aix \
                F77=xlf \
		FFLAGS="-c -u -gi -qfixed=85 -qxlf77=oldboz" \
		CPPFLAGS="-P -DMANDATORY_FORMAT_WIDTHS -Dunix -Ddebug -DNO_SNGL  -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF -DNO_ENTRY_NAME_UNDERSCORES" \
		CCFLAGS="-Ireadline -I. -I../common -I$TMAP_LOCAL/src/xgks-2.5.5/port -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm -c -DSGI_SIGNALS"

aix_gcompile_1:
	${MAKE} go_compile_aix_1 \
		SOURCE=${SOURCE} \
                DIR=${DIR} \
		F77=xlf \
		FFLAGS="-c -u -g -qfixed=80" \
		CPPFLAGS="-P -DMANDATORY_FORMAT_WIDTHS -Dunix -DNO_SNGL -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF -DNO_ENTRY_NAME_UNDERSCORES"

linux_gcompile:
	${MAKE} go_compile_linux \
		F77=f77 \
		FFLAGS="-132 -c -dusty -C -f77 -fieee -g -Nl90 -x77 -w" \
		CPPFLAGS="-P -Dunix -DNEED_IAND -DFORTRAN_90 -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE -DNO_SNGL -DF90_DATE_TIME -DF90_SYSTEM_ERROR_CALLS -DMANDATORY_FORMAT_WIDTHS -DX_REFRESH -Dreclen_in_bytes -DNO_DOUBLE_ESCAPE_SLASH -DF90_OPEN_FILE_APPEND" \
		CCFLAGS="-g -Dunix -DNEED_IAND -DFORTRAN_90 -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE -DNO_SNGL -DF90_DATE_TIME -DF90_SYSTEM_ERROR_CALLS -DMANDATORY_FORMAT_WIDTHS -DX_REFRESH -Dreclen_in_bytes -DNO_DOUBLE_ESCAPE_SLASH -DF90_OPEN_FILE_APPEND-Ireadline -I. -I../common -I$TMAP_LOCAL/src/xgks-2.5.5/port -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm -c"
		
	
nogcompile:
	${MAKE} $(HOSTTYPE)_nogcompile\
		 F77="$(F77)" \
		 GKSFFLAGS=$(GKSFFLAGS)\
		 MEMFLAGS=$(MEMFLAGS)

decstation_nogcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c" \
		FFLAGS="-c -u -cpp -G 0 -fpe1 -Df7732_bug -DDEC_FORTRAN\
		$(GKSFFLAGS) $(MEMFLAGS)"

alpha_nogcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-c -DUSG -DDEC_ALPHA" \
		FFLAGS="-c -u -cpp -C  -Df7732_bug -DDEC_FORTRAN\
		-integer_size 32 -real_size 32 -align dcommons -align records\
		$(GKSFFLAGS) $(MEMFLAGS)"

sun4_nogcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -D_NO_PROTO -c" \
                FFLAGS="-c -Nl160 -u $(GKSFFLAGS) $(MEMFLAGS)"

solaris_nogcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -c -I/usr/openwin/include" \
		FFLAGS="-c -Nl160 -u -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Dsolaris -DX_REFRESH $(GKSFFLAGS) $(MEMFLAGS)"

sgi_nogcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-cckr -c -G 0 -Dsgi -Dsun -DSGI_SIGNALS" \
		FFLAGS="-c -Nl160 -u -Dsun -Dsgi -G 0 -DX_REFRESH $(GKSFFLAGS) $(MEMFLAGS)"

hp_nogcompile:
	${MAKE} go_compile \
		F77="$(F77)" \
		CC=cc \
		CCFLAGS="-I/usr/ucbinclude -I/usr/include/X11R5 -Dsun -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -DHP_ADDS -DFlarge_mem -Dxgks -DSTDERR_UNIT_SEVEN -N -Ae" \
		FFLAGS="-c +e +es -u -w +U77 -N +B +E4 -DREADONLY_ALLOWED -Dunix -DNO_ENTRY_NAME_UNDERSCORES -DNO_SNGL -DNO_WIN_UTIL_H -DHP_SIGNALS -DNEED_IAND -DHP_ADDS -DFlarge_mem -Dxgks -DSTDERR_UNIT_SEVEN $(GKSFFLAGS) $(MEMFLAGS)" 


#removed -DNO_FREE_READS and -DNO_PASSED_CONCAT for new aix compiler 12/95
# added static link options *kob* 1/96
aix_nogcompile:
	${MAKE} go_compile_aix \
                F77=xlf \
		FFLAGS="-c -u -qfixed=85  -qxlf77=oldboz -bnso -bI:/usr/lib/syscalls.exp" \
		CPPFLAGS="-P -Dunix -DNO_SNGL -DMANDATORY_FORMAT_WIDTHS -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF -DNO_ENTRY_NAME_UNDERSCORES" \
		CCFLAGS="-Ireadline -I. -I../common -I$TMAP_LOCAL/src/xgks-2.5.5/port -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm -c -DSGI_SIGNALS"

aix_nogcompile_1:
	${MAKE} go_compile_aix_1 \
		SOURCE=${SOURCE} \
                DIR=${DIR} \
		F77=xlf \
		FFLAGS="-c -u -qfixed=85 -qxlf77=oldboz -bnso -bI:/usr/lib/syscalls.exp" \
		CPPFLAGS="-P -Dunix -DMANDATORY_FORMAT_WIDTHS -DNO_SNGL -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE $(GKSFFLAGS) $(MEMFLAGS) -Daix -DAIX_XLF -DNO_ENTRY_NAME_UNDERSCORES"

linux_nogcompile:
	${MAKE} go_compile_linux \
		F77=f77 \
		FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl90 -x77 -w" \
		CPPFLAGS="-P -Dunix -DNEED_IAND -DFORTRAN_90 -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE -DNO_SNGL -DF90_DATE_TIME -DF90_SYSTEM_ERROR_CALLS -DMANDATORY_FORMAT_WIDTHS -DX_REFRESH -Dreclen_in_bytes -DNO_DOUBLE_ESCAPE_SLASH -DF90_OPEN_FILE_APPEND " \
		CCFLAGS="-Dunix -DNEED_IAND -DFORTRAN_90 -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE -DNO_SNGL -DF90_DATE_TIME -DF90_SYSTEM_ERROR_CALLS -DMANDATORY_FORMAT_WIDTHS -DX_REFRESH -Dreclen_in_bytes -DNO_DOUBLE_ESCAPE_SLASH -DF90_OPEN_FILE_APPEND -I. -I../common -I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c"


# FERRET code (order of libraries is essential to correct loading)
FERLIBS =      xeq/xeqlib.a  gnl/gnllib.a  plt/pltlib.a  rpn/rpnlib.a\
               stk/stklib.a  doo/doolib.a  ocn/ocnlib.a  ctx/ctxlib.a\
               fmt/fmtlib.a  ino/inolib.a  mem/memlib.a  utl/utllib.a\
               dat/datlib.a  ccr/ccrlib.a

# removed c objects because they are now found in ccrlib.a *kob* 6/96
FEROBJS_DFLT=special/ferret_dispatch.o special/xmake_date_data.o special/fakes3.o special/ferret_query_f.o special/gui_fakes.o  special/xrevision_type_data.o
FEROBJS_GUI= special/ferret_dispatch.o special/xmake_date_data.o special/fakes3.o special/ferret_query_f.o special/xrevision_type_data.o

# PLOT+ code
PPLOBJS = \
 $(PPLDIR)/mods$(WHATGKS)/*.o\
 $(PPLDIR)/adds$(WHATGKS)/*.o
PPLLIBS = \
 $(PPLDIR)/plot/plt.a\
 $(PPLDIR)/plotlib/pll.a\
 $(PPLDIR)/symlib/sym.a\
 $(PPLDIR)/complot/cmp.a\
 $(PPLDIR)/ourlib/our.a\
 $(PPLDIR)/pplepic/epi.a \
 $(PPLDIR)/pplusr/usr.a


# compilation definitions
memobjs = mem/*.o
xeqobjs = xeq/*.o
datobjs = dat/*.o
pltobjs = plt/*.o
gnlobjs = gnl/*.o
stkobjs = stk/*.o
dooobjs = doo/*.o
rpnobjs = rpn/*.o
ocnobjs = ocn/*.o
utlobjs = utl/*.o
fmtobjs = fmt/*.o
inoobjs = ino/*.o
ctxobjs = ctx/*.o
ccrobjs = ccr/*.o

ferret:	$(FERLIBS) $(FEROBJS) $(TMLIB) $(MOVIELIB) $(FCHECKOUT) $(TMCHECKOUT) $(GUICHECKOUT)
#	echo $(PPLLIBS)
# auto-generate the "debug" and "GUI" information about this link (8/95)
	sed -e 's&mmmmmmmm&($(DEBUG)/$(GUI))&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	(cd special;$(F77) $(FFLAGS) xrevision_type_data.F) 
	- mv $(EXECNAME) $(EXECNAME)_pre_`date +'%d%h'|tr '[A-Z]' '[a-z]'`
	- $(LD) $(LDFLAGS) -o $(EXECNAME) \
		$(FERMAIN) \
		$(FCHECKOUT) \
		$(TMCHECKOUT) \
		$(GUICHECKOUT) \
		$(PPLCHECKOUT) \
		$(FEROBJS) \
		$(GUI_LIB) \
		$(XTRA_OBJ) \
		$(GKSOBJ) \
		$(PPLOBJS) \
		$(BDEAD) \
		$(FERLIBS) \
		$(PPLLIBS) \
		$(MOVIELIB) \
		$(TMLIB) \
		-ltermcap \
		$(CDFLIB) \
		$(HDFLIB) \
		$(GKSLIB) \
		$(SYSLIB)
	chmod +x $(EXECNAME)

ferret_aix:	$(FERLIBS) $(FEROBJS) $(TMLIB) $(MOVIELIB) $(FCHECKOUT) $(TMCHECKOUT)
#	echo $(PPLLIBS)
# auto-generate the "debug" and "GUI" information about this link (8/95)
#	sed -e 's&mmmmmmmm&($(DEBUG)$(GUI))&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	sed -e 's&mmmmmmmm&($(DEBUG)/$(GUI))&' -e 's&\(/&(' -e 's&/\)&)' < special/xrevision_type_data.template > special/xrevision_type_data.F
	(cd special;/usr/lib/cpp -P ${CPPFLAGS} xrevision_type_data.F > xrevision_type_data.f; $(F77) $(FFLAGS) xrevision_type_data.f) 
	- mv $(EXECNAME) $(EXECNAME)_pre_`date +'%d%h'|tr '[A-Z]' '[a-z]'`
	- $(LD) $(LDFLAGS) -o $(EXECNAME) \
		$(FERMAIN) \
		$(GUI_LIB) \
		$(FEROBJS) \
		$(FCHECKOUT) \
		$(TMCHECKOUT) \
		$(XTRA_OBJ) \
		$(GKSOBJ) \
		$(PPLOBJS) \
		$(BDEAD) \
		$(FERLIBS) \
		$(PPLLIBS) \
		$(MOVIELIB) \
		$(TMLIB) \
		-ltermcap \
		$(CDFLIB) \
		$(HDFLIB) \
		$(GKSLIB) \
		$(SYSLIB)
	chmod +x $(EXECNAME)
ferret_linux:	$(FERLIBS) $(FEROBJS) $(TMLIB) $(MOVIELIB) $(FCHECKOUT) $(TMCHECKOUT)
	sed -e 's&mmmmmmmm&($(DEBUG)/$(GUI))&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	(cd special;/lib/cpp -P ${CPPFLAGS} xrevision_type_data.F > xrevision_type_data.f; $(F77) $(FFLAGS) xrevision_type_data.f) 
	- mv $(EXECNAME) $(EXECNAME)_pre_`date +'%d%h'|tr '[A-Z]' '[a-z]'`
	- $(LD) $(LDFLAGS) -o $(EXECNAME) \
		$(FERMAIN) \
		$(GUI_LIB) \
		$(FEROBJS) \
		$(FCHECKOUT) \
		$(TMCHECKOUT) \
		$(XTRA_OBJ) \
		$(GKSOBJ) \
		$(PPLOBJS) \
		$(BDEAD) \
		$(FERLIBS) \
		$(PPLLIBS) \
		$(MOVIELIB) \
		$(TMLIB) \
		-ltermcap \
		$(CDFLIB) \
		$(HDFLIB) \
		$(GKSLIB) \
		$(SYSLIB)
	chmod +x $(EXECNAME)



go_compile:
	date
	(cd mem;$(F77) $(FFLAGS) *.[fF])
	(cd xeq;$(F77) $(FFLAGS) *.[fF])
	(cd plt;$(F77) $(FFLAGS) *.[fF])
	(cd gnl;$(F77) $(FFLAGS) *.[fF])
	(cd stk;$(F77) $(FFLAGS) *.[fF])
	(cd doo;$(F77) $(FFLAGS) *.[fF])
	(cd rpn;$(F77) $(FFLAGS) *.[fF])
	(cd ocn;$(F77) $(FFLAGS) *.[fF])
	(cd utl;$(F77) $(FFLAGS) *.[fF])
	(cd fmt;$(F77) $(FFLAGS) *.[fF])
	(cd ino;$(F77) $(FFLAGS) *.[fF])
	(cd ctx;$(F77) $(FFLAGS) *.[fF])
	(cd dat;$(F77) $(FFLAGS) *.[fF])
	(cd ccr;$(CC) $(CCFLAGS) $(CCINCLUDES) *.c)
	(cd special;$(F77) $(FFLAGS) *.[fF])

	date


go_compile_aix:
	for dir in ${DIRS} ; \
	do \
		echo $${dir}; cd $${dir}; \
			for file in *.F ; do \
				echo $$file;\
				touch `echo $$file | sed -e 's/.F/.f/g'`; \
                		rm `echo $$file | sed -e 's/.F/.f/g'`; \
		           	sed -e 's/???/?/g' $$file |sed -e 's/??/?/g' |	        /usr/lib/cpp ${CPPFLAGS}    | sed -e 's/	/      /'|sed -e 's/, /,/g'|sed -e 's/\.	/\./g' | sed -e 's/( /(/g' | sed -e 's/ )/)/g' | sed -e 's/\.      /\./g' | sed -e 's/ =/=/g' | sed -e 's/= /=/g'| sed -e 's/          /      /g' | sed -e 's/        /      /g' | sed -e 's/) /)/g' | sed -e 's/ (/(/g' | sed -e 's/??/?/g' | sed -e 's&/FORMAT=&/FORMAT = &' -e 's&/SKIP=&/SKIP = &' -e 's&/COLUMNS=&/COLUMNS = &' -e 's&momentum Am=&momentum Am = &' -e 's&/PRECISION=&/PRECISION = &' -e 's&//'\''('\''//&//'\'' ('\''//&' -e 's&//'\'' ,'\''//&//'\'' , '\''//&' > `echo $$file | sed -e 's/.F/.f/g'`; \
                		${F77} ${FFLAGS} `echo $$file | sed -e 's/.F/.f/g'`; \
			done;\
		cd ..; echo `pwd`; \
	done

go_compile_aix_1:
	for dir in ${DIR} ; \
	do \
		echo $${dir}; cd $${dir}; \
			for file in ${SOURCE} ; do \
				echo $$file;\
				touch `echo $$file | sed -e 's/.F/.f/g'`; \
                		rm `echo $$file | sed -e 's/.F/.f/g'`; \
		           	sed -e 's/???/?/g' $$file |sed -e 's/??/?/g' |	        /usr/lib/cpp ${CPPFLAGS}    | sed -e 's/	/      /'|sed -e 's/, /,/g'|sed -e 's/\.	/\./g' | sed -e 's/( /(/g' | sed -e 's/ )/)/g' | sed -e 's/\.      /\./g' | sed -e 's/ =/=/g' | sed -e 's/= /=/g'| sed -e 's/          /      /g' | sed -e 's/        /      /g' | sed -e 's/) /)/g' | sed -e 's/ (/(/g' | sed -e 's/??/?/g' | sed -e 's&/FORMAT=&/FORMAT = &' -e 's&/SKIP=&/SKIP = &' -e 's&/COLUMNS=&/COLUMNS = &' -e 's&momentum Am=&momentum Am = &' -e 's&/PRECISION=&/PRECISION = &' -e 's&//'\''('\''//&//'\'' ('\''//&' -e 's&//'\'' ,'\''//&//'\'' , '\''//&' > `echo $$file | sed -e 's/.F/.f/g'`; \
                		${F77} ${FFLAGS} `echo $$file | sed -e 's/.F/.f/g'`; \
			done;\
		cd ..; echo `pwd`; \
	done

go_compile_linux:
	for dir in ${DIRS} ; \
	do \
		echo $${dir}; cd $${dir}; \
			for file in *.F ; do \
                		rm -f `echo $$file | sed -e 's/.F/.f/g'`; \
				echo $$file; \
				/lib/cpp  ${CPPFLAGS} $$file      | sed -e 's/de	/de /g' | sed -e  's/de 	/de /g'> `echo $$file | sed -e 's/.F/.f/g'`; ${F77} ${FFLAGS} `echo $$file | sed -e 's/.F/.f/g'`; \
			done;\
		cd ..; echo `pwd`; \
	done

	(cd ccr;  $(CC)  $(CCFLAGS) *.c)


libraries:
	${MAKE} mem/memlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} xeq/xeqlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} dat/datlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} plt/pltlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} gnl/gnllib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} stk/stklib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} doo/doolib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} rpn/rpnlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} ocn/ocnlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} utl/utllib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} fmt/fmtlib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} ino/inolib.a  RANLIB="/usr/bin/ranlib"
	${MAKE} ctx/ctxlib.a  RANLIB="/usr/bin/ranlib"	
	${MAKE} ccr/ccrlib.a  RANLIB="/usr/bin/ranlib"

pplcheck: $(PPLOBJS) $(PPLLIBS)
	@ echo ppl code is out of date

# make the libraries
# Note: F77 and FFLAGS are not defined within this procedure

mem/memlib.a:	$(memobjs)
	(cd mem;/bin/rm -f memlib.a; ar cq memlib.a *.o; $(RANLIB) memlib.a)
xeq/xeqlib.a:	$(xeqobjs)
	(cd xeq;/bin/rm -f xeqlib.a; ar cq xeqlib.a *.o; $(RANLIB) xeqlib.a)
dat/datlib.a:	$(datobjs)
	(cd dat;/bin/rm -f datlib.a; ar cq datlib.a *.o; $(RANLIB) datlib.a)
plt/pltlib.a:	$(pltobjs)
	(cd plt;/bin/rm -f pltlib.a; ar cq pltlib.a *.o; $(RANLIB) pltlib.a)
gnl/gnllib.a:	$(gnlobjs)
	(cd gnl;/bin/rm -f gnllib.a; ar cq gnllib.a *.o; $(RANLIB) gnllib.a)
stk/stklib.a:	$(stkobjs)
	(cd stk;/bin/rm -f stklib.a; ar cq stklib.a *.o; $(RANLIB) stklib.a)
rpn/rpnlib.a:	$(rpnobjs)
	(cd rpn;/bin/rm -f rpnlib.a; ar cq rpnlib.a *.o; $(RANLIB) rpnlib.a)
ocn/ocnlib.a:	$(ocnobjs)
	(cd ocn;/bin/rm -f ocnlib.a; ar cq ocnlib.a *.o; $(RANLIB) ocnlib.a)
utl/utllib.a:	$(utlobjs)
	(cd utl;/bin/rm -f utllib.a; ar cq utllib.a *.o; $(RANLIB) utllib.a)
fmt/fmtlib.a:	$(fmtobjs)
	(cd fmt;/bin/rm -f fmtlib.a; ar cq fmtlib.a *.o; $(RANLIB) fmtlib.a)
ino/inolib.a:	$(inoobjs)
	(cd ino;/bin/rm -f inolib.a; ar cq inolib.a *.o; $(RANLIB) inolib.a)
ctx/ctxlib.a:	$(ctxobjs)
	(cd ctx;/bin/rm -f ctxlib.a; ar cq ctxlib.a *.o; $(RANLIB) ctxlib.a)
doo/doolib.a:	$(dooobjs)
	(cd doo;/bin/rm -f doolib.a; ar cq doolib.a *.o; $(RANLIB) doolib.a)
ccr/ccrlib.a:	$(ccrobjs)
	(cd ccr;/bin/rm -f ccrlib.a; ar cq ccrlib.a *.o; $(RANLIB) ccrlib.a)

.f.o: 
	(cd $(<D);$(F77) $(FFLAGS) $(<F))

.c.o: 
	(cd $(<D);$(CC) $(CCFLAGS) $(CCINCLUDES) $(<F))

.F.o: 
	(cd $(<D);$(F77) $(FFLAGS) $(<F))

