/*
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
*/



/*      tm_world_recur - recursive C routine to locate the value of a
	coordinate from its index where the underlying axis is the child
	of some other axis

  See the FORTRAN routine
	DOUBLE PRECISION FUNCTION TM_WORLD
  as a reference

  TMAP interactive data analysis program

  programmer - steve hankin
  NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

   revision history:
   10/95 - original
    5/99 *sh* - bug fix: modulo irregular axes get wrong memory subscript
    V510: 4/00 *sh* - the arrays line_parent and line_class are now indexed
                      from zero instead of from max_lines
    V541: 2/02 *sh* - added support for subspan modulo axes
                    - fixed bug in strides on modulo parent axis

    V542: 10/02 *sh* - serious bug fixes in non-modulo subspan modulo

   compile this with
   cc -c -g tm_world_recur.c
   (and use -D_NO_PROTO for non-ANSI compilers)
*/ 

/* local macro definitions */
#define PLINE_CLASS_BASIC   0
#define PLINE_CLASS_STRIDE  1
#define PLINE_CLASS_MIDPT   2
#define PLINE_CLASS_FOREIGN 3
#define PLINE_CLASS_FFT     4
#define BOX_LO_LIM          1
#define BOX_MIDDLE          2
#define BOX_HI_LIM          3

#define MIN(x, y) (( (x) < (y)) ? (x) : (y))
#define MAX(x, y) (( (x) < (y)) ? (y) : (x))

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

/* prototype for FORTRAN boolean function */
int  FORTRAN(tm_its_subspan_modulo) (int *axis);
void FORTRAN(tm_ww_axlims) (int *axis, double *lo, double *hi);
double FORTRAN(tm_modulo_axlen) (int *axis);

double FORTRAN(tm_world_recur)
     ( int *isubscript, int *iaxis, int *where_in_box,
       int *max_lines, double line_mem[], int line_parent[],
       int line_class[], int line_dim[], 
       double line_start[], double line_delta[],
       int line_subsc1[], int line_modulo[], double line_modulo_len[],
       int line_regular[] )

{
  double tempwld, tm_world;
  int isub, rmod;
  int axis = *iaxis;   /* these FORTRAN arrs start at 0 like C */
  int line_len = line_dim[axis];

/*
   if this is a child (e.g. every Nth point) of an irregularly-spaced
   axis then use a recursive C routine to find the answer
*/
  int recursive = axis > *max_lines;
  if ( recursive ) {
    recursive = line_parent[axis] != 0;    /* could use "&&" */
    if ( recursive ) {
      int new_ss, lo_ss, hi_ss, parent_len;
      int new_where;
      switch (line_class[axis]) {
      case PLINE_CLASS_STRIDE:
/* 5/99 - in irreg axis striding the box edges cannot simply be read from
          the box edge array (think about it) So we have xtra logic here.
*/
	new_ss = (int)line_start[axis]
	             +(*isubscript-1)*(int)line_delta[axis];
	if ( line_regular[axis] || *where_in_box==BOX_MIDDLE ) {

	  tm_world = FORTRAN(tm_world_recur)
	    (&new_ss,
	     &(line_parent[axis]),
	     where_in_box,
	     max_lines, line_mem, line_parent,
	     line_class, line_dim,
	     line_start, line_delta,
	     line_subsc1, line_modulo, line_modulo_len,
	     line_regular );

	} else {   /*  !!!! EXTRA LOGIC FOR IRREGULAR AXIS STRIDES */
	  /* cases to consider: interpolate to neighbor above or below
	                        use lower or upper limit of entire axis
	  */	  
	  if (*where_in_box ==  BOX_LO_LIM) {
	    lo_ss = (int)line_start[axis]
	             +(*isubscript-2)*(int)line_delta[axis];
	    hi_ss = new_ss;
	  } else {
	    lo_ss = new_ss;
	    hi_ss =  (int)line_start[axis]
	             +(*isubscript-0)*(int)line_delta[axis];
	  }

	  parent_len = line_dim[ line_parent[axis] ];
	  if ( line_modulo[ line_parent[axis] ]   /* 2/02 bug fix */
	       || (lo_ss>=1 && hi_ss<=parent_len) ) {  /* interpolate */
	    new_where = BOX_MIDDLE;
	    tm_world = 0.5 * (
			      FORTRAN(tm_world_recur)
			      (&lo_ss,
			       &(line_parent[axis]),
			       &new_where,
			       max_lines, line_mem, line_parent,
			       line_class, line_dim,
			       line_start, line_delta,
			       line_subsc1, line_modulo, line_modulo_len,
			       line_regular)
	                  +   
			      FORTRAN(tm_world_recur)
			      (&hi_ss,
			       &(line_parent[axis]),
			       &new_where,
			       max_lines, line_mem, line_parent,
			       line_class, line_dim,
			       line_start, line_delta,
			       line_subsc1, line_modulo, line_modulo_len,
			       line_regular)
			      );
	  } else if (*where_in_box ==  BOX_LO_LIM) { /* lower axis edge */
	    new_ss = 1;
	    tm_world = FORTRAN(tm_world_recur)
	      (&new_ss,
	       &(line_parent[axis]),
	       where_in_box,
	       max_lines, line_mem, line_parent,
	       line_class, line_dim,
	       line_start, line_delta,
	       line_subsc1, line_modulo, line_modulo_len, line_regular);
	  } else {  /* upper axis edge */
	    new_ss = parent_len;
	    tm_world = FORTRAN(tm_world_recur)
	      (&new_ss,
	       &(line_parent[axis]),
	       where_in_box,
	       max_lines, line_mem, line_parent,
	       line_class, line_dim,
	       line_start, line_delta,
	       line_subsc1, line_modulo, line_modulo_len, line_regular);
	  }
	}
	break;

      case PLINE_CLASS_MIDPT:
	tm_world = FORTRAN(tm_world_recur)
	  (isubscript,iaxis,where_in_box,
	   max_lines, line_mem, line_parent,
	   line_class, line_dim,
	   line_start, line_delta,
	   line_subsc1, line_modulo, line_modulo_len, line_regular);
	break;
      default:
	tm_world = -999.;
      }
      return(tm_world);
    }
  }

/* 
   not a recursive access - return the same result that TM_WORLD would have.
   Force given subsc to data range as appropriate for modulo or non-modulo axes
*/
  if ( FORTRAN(tm_its_subspan_modulo) (&axis) ) line_len++;  /* 2/02 mod */
  if ( line_modulo[axis] ) {
    isub = ((*isubscript-1)%line_len) + 1 ;  /* inserted "+1" 5/99 */
    if (isub <= 0)
      isub += line_len;
  }
  else
      isub = MIN( line_len, MAX( 1, *isubscript ) );

/*
    the given index  falls in the "void" region of a subspan modulo axis
    ... get the box_hi_lim of the Nth point in the core region
*/
  if  ( FORTRAN(tm_its_subspan_modulo) (&axis)
	&& isub == line_len ) {
    double lo, hi;
    FORTRAN(tm_ww_axlims) (&axis,&lo, &hi);
/* ... now where within the grid box ? */
    if ( *where_in_box == BOX_LO_LIM )
      tempwld = hi;
    else if ( *where_in_box == BOX_MIDDLE )
      tempwld = ( hi + (lo+line_modulo_len[axis]) )/2.;
    else
      tempwld = lo + line_modulo_len[axis];
    
    if (*isubscript <= 0)
      rmod = *isubscript/line_len - 1;
    else
      rmod = (*isubscript-1)/line_len;
    tm_world = tempwld + rmod*line_modulo_len[axis];
	    

/*
   regularly spaced points
*/
    } else if ( line_regular[axis] ) {
/* ... calculate midpoint and box_size values */
    double midpoint = line_start[axis] + (isub-1)*line_delta[axis];
    double box_size = line_delta[axis];
/* ... now where within the grid box ? */
    if ( *where_in_box == BOX_LO_LIM )
      tempwld = midpoint - ( box_size / 2. );
    else if ( *where_in_box == BOX_MIDDLE )
      tempwld = midpoint;
    else
      tempwld = midpoint + ( box_size / 2. );
    
    if ( line_modulo[axis] ) {
      if (*isubscript <= 0)
	rmod = ( *isubscript/line_len - 1 );
      else
	rmod = ( (*isubscript-1)/line_len );
      tm_world = tempwld + rmod * FORTRAN(tm_modulo_axlen) (&axis);
    }
    else
      tm_world = tempwld;
  }
  else
    {
/*
  irregularly spaced points
*/
/* ... xlate subscript to location in line_mem array */
      isub  += line_subsc1[axis] - 1;
      isub--;      /* 5/99 switch to C-style zero-referenced indexing */
/* ... now, where within the grid box ? */
      if      ( *where_in_box == BOX_LO_LIM )
	tempwld = line_mem[isub+line_dim[axis]];
      else if ( *where_in_box == BOX_MIDDLE )
	tempwld = line_mem[isub];
      else
	tempwld = line_mem[isub+line_dim[axis]+1];
      
      if ( line_modulo[axis] ) {
	if (*isubscript <= 0)
	  rmod = *isubscript/line_len - 1;
	else
	  rmod = (*isubscript-1)/line_len;
	
	tm_world = tempwld + rmod * FORTRAN(tm_modulo_axlen) (&axis);
      }
      else
	tm_world = tempwld;
    }
  return(tm_world);
  
}

