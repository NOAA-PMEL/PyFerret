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
   10/95

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
#  define ENTRY_NAME tm_world_recur
#else
#  define ENTRY_NAME tm_world_recur_
#endif

#ifdef _NO_PROTO
double ENTRY_NAME( isubscript, iaxis, where_in_box,
		  max_lines, line_mem, line_parent,
		  line_class, line_dim,
		  line_start, line_delta,
		  line_subsc1, line_modulo, line_regular )
int *isubscript, *iaxis, *where_in_box, *max_lines, line_parent[],
    line_class[], line_dim[], line_subsc1[], line_modulo[], line_regular[];
double line_mem[], line_start[], line_delta[];
#else
double ENTRY_NAME( int *isubscript, int *iaxis, int *where_in_box,
		  int *max_lines, double line_mem[], int line_parent[],
		  int line_class[], int line_dim[], 
		  double line_start[], double line_delta[],
		  int line_subsc1[], int line_modulo[], int line_regular[] )
#endif

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
    recursive = line_parent[axis-*max_lines] != 0;    /* could use "&&" */
    if ( recursive ) {
      int new_ss;
      switch (line_class[axis-*max_lines]) {
      case PLINE_CLASS_STRIDE:
	new_ss = (int)line_start[axis]
	             +(*isubscript-1)*(int)line_delta[axis];
	tm_world = ENTRY_NAME(&new_ss,
			      &(line_parent[axis-*max_lines]),
			      where_in_box,
			      max_lines, line_mem, line_parent,
			      line_class, line_dim,
			      line_start, line_delta,
			      line_subsc1, line_modulo,line_regular);
	break;

      case PLINE_CLASS_MIDPT:
	tm_world = ENTRY_NAME(isubscript,iaxis,where_in_box,
			      max_lines, line_mem, line_parent,
			      line_class, line_dim,
			      line_start, line_delta,
			      line_subsc1, line_modulo,line_regular);
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
  if ( line_modulo[axis] ) {
    isub = ((*isubscript-1)%line_len) ;  /* would add 1 in FORTRAN */
    if (isub <= 0)
      isub += line_len;
  }
  else
      isub = MIN( line_len, MAX( 1, *isubscript ) );

/*

   regularly spaced points
*/
  if ( line_regular[axis] ) {
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
	rmod = line_len * ( *isubscript/line_len - 1 );
      else
	rmod = line_len * ( (*isubscript-1)/line_len );
      tm_world = tempwld + rmod*line_delta[axis];
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
/** ... now, where within the grid box ? */
      if      ( *where_in_box == BOX_LO_LIM )
	tempwld = line_mem[isub+line_len];
      else if ( *where_in_box == BOX_MIDDLE )
	tempwld = line_mem[isub];
      else
	tempwld = line_mem[isub+line_len+1];
      
      if ( line_modulo[axis] ) {
	if (*isubscript <= 0)
	  rmod = *isubscript/line_len - 1;
	else
	  rmod = (*isubscript-1)/line_len;
	
	tm_world = tempwld + rmod*
	  ( line_mem[line_subsc1[axis]+2*line_len] -
	   line_mem[line_subsc1[axis]+  line_len] );
      }
      else
	tm_world = tempwld;
    }
  return(tm_world);
  
}

