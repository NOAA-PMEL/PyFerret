	SUBROUTINE RSMNMX

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOT]MINMAX.FOR

* search for "TMAP" to find all revisions
* *sh* - 1/27/89 corrected bug: min/max search improperly rejects LIMITs points
* *sh* - 12/8/93 corrected bug: vectors blow on divied by zero attempt
* *jd* - 4.20.99 Don't delete symbols {X,Y}{MIN,MAX} in overlay plots

C
C	RESET MIN MAX SYMBOLS AND SET LINE_COUNT TO ZERO
C
#ifdef unix
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
#else
        INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
        INCLUDE 'PPLV11INC:PARAMPL5.DAT'
        INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
        INCLUDE 'PPLV11INC:LINES.INC'
        INCLUDE 'TMAP_PPLV11INC:SWITCH_INC.DECL'
        INCLUDE 'PPLV11INC:SWITCH.INC'
#endif
	CHARACTER SYM*30,TEMP*2
C
	SYM='*PPL$LINE_COUNT'
	TEMP='0'
	CALL PUTSYM(SYM,TEMP,1,IER)
C
	DO 10 I=1,LNUM
	WRITE(SYM,999)'XMIN',I
999	FORMAT('*PPL$',A,'(',I2,')')
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'XMAX',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YMIN',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YMAX',I
	CALL DELSYM(SYM,IER)
C
	WRITE(SYM,999)'XFIRST',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'XLAST',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YFIRST',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YLAST',I
	CALL DELSYM(SYM,IER)
10	CONTINUE
C
*jd*begin
	IF(.NOT.SOVER)THEN
	   SYM='*PPL$XMIN'
	   CALL DELSYM(SYM,IER)
	   SYM='*PPL$XMAX'
	   CALL DELSYM(SYM,IER)
	   SYM='*PPL$YMIN'
	   CALL DELSYM(SYM,IER)
	   SYM='*PPL$YMAX'
	   CALL DELSYM(SYM,IER)
	ENDIF
*jd*end
	SYM='*PPL$ZMIN'
	CALL DELSYM(SYM,IER)
	SYM='*PPL$ZMAX'
	CALL DELSYM(SYM,IER)
	sym='*PPL$XHIGH'
	CALL DELSYM(SYM,IER)
	sym='*PPL$YHIGH'
	CALL DELSYM(SYM,IER)
	sym='*PPL$XLOW'
	CALL DELSYM(SYM,IER)
	sym='*PPL$YLOW'
	CALL DELSYM(SYM,IER)
	RETURN
	END
