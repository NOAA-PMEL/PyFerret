	INTEGER FUNCTION NEST_REGION_CLASS( idim, cx, uvar )

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
* determine the region-limits class (uvar_given) of a user-defined variable
* used as a component in another user-defined variable

* for example,	DEFINE VARIABLE A = U[K=@AVE]
*		DEFINE VARIABLE B = A[K=1:5] * 5
* references to B have the K limits reduced to a point although it isn't
* evident in the definition of B, alone.  This routine is used to determine
* the uvar_given character of A[K=1:5] used as a component in B.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/18/89
* V230:  1/27/92 - changed uvlim_*_hi (unused) to uvlim_*_lohi to describe
*                  @WEQ - exaxt limits required but a range of data
*                - also added @RSUM (like @IIN)
*           ==>  commandeered "hi" to become "lohi" for @WEQ
*    note: those table entries using "lohi" are less thoroughly tested
	
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations
	INTEGER	idim, cx, uvar

* local variable declarations
	INTEGER	REGION_CLASS, iins, iout, nested(0:3,0:3)

* local parameter declarations
	INTEGER		gnl,   lo,   lohi,   xact
	PARAMETER     ( gnl=0, lo=1, lohi=2, xact=3 )

* * * * NESTING TABLE * * * *
*			    ---  underlying user var definition  ---
* (paradigm)	      U[K=1:5]	     U[K=@IIN]	     U[K=@WEQ]	    U[K=@AVE]
*			gnl		lo		lohi		xact
*		--------------------------------------------------------------
* N C |	gnl	|	gnl		lo		lohi		xact |
* E O |	lo	|	 lo		lo		lohi		xact |
* W M |	lohi	|	lohi		lohi		lohi		xact |
*   P |	xact	|	xact		xact		xact		xact |
*		--------------------------------------------------------------
* across the top of the table we have the class of the underlying definition
* on the left we have the manner in which the new component is referenced
* for example, if "DEFINE VARIABLE A=U[K=@AVE]" defines the underlying var
*	       and "A[K=1:5]" is the new component then the column is "xact"
*	       and the row is "gnl" indicating that the result is "xact".
* (This makes sense as the new component will function like A[K=1:5@AVE]
*	- reducing to a point and requiring an exact match to locate it in mem.)
*  or (ex#2)   if "DEFINE VARIABLE A=U" defines the underlying var
*	       and "A[K=@AVE]" is the new component then the column is "gnl"
*	       and the row is "xact" indicating that the result is "xact".

* gnl  == no match required e.g. "U"		or "U[K=1:5]"
* lo   == match lo limit    e.g. "U[K=@IIN]"	or "U[K=1:5@IIN]"
* lohi == match both limits (range of data)  e.g. "U[K=@WEQ]"
* xact == match both limits (single point)   e.g. "U[K=@AVE]"
* "xact" is special because it refers to transformations that reduce a range
*	of data to a point (like averaging).

	DATA	(nested(iins,0),iins=0,3) / gnl,   lo, lohi, xact /,
     .		(nested(iins,1),iins=0,3) /  lo,   lo, lohi, xact /,
     .		(nested(iins,2),iins=0,3) /lohi, lohi, lohi, xact /,
     .		(nested(iins,3),iins=0,3) /xact, xact, xact, xact /

**************************************************************************

* terminology: in the example reference of A[K=1:5] where DEF VAR A=U[K=@AVE]
* the "A[K=1:5]" is referred to as the "outside reference" and "U[K=@AVE]" is
* the inside reference

* get region-limits class of outside reference
	iout = REGION_CLASS( idim, cx )

* get region-limits class of inside reference
	iins = MOD( uvar_given(idim,uvar), 4 )

* pull nested reference from table
	NEST_REGION_CLASS = nested( iins, iout )
	
	RETURN
	END
