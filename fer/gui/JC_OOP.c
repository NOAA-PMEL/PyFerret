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



/*
 * JC_OOP.c
 *
 * Jonathan Callahan
 * Nov 30th 1995
 *
 * This file contains functions which interact with the new structures I have
 * defined for the Ferret GUI.
 *
 * I have endeavored to stay very close to true Object-Oriented-Programming
 * principles and have in general followed the C++ guidelines as I learned
 * them from "Weiskamp & Flamig:  The complete C++ primer".
 *
 * NB_ This file is included in main.c and shouldn't have any definitions or declarations.
 */


/* .................... JC_Axis methods .................... */


void JC_Axis_Clear( JC_Axis *this )
{
     memset(this, 0, sizeof(JC_Axis));
}


int JC_Axis_ReturnIndex( JC_Axis *this, float value )
{
     double delta=0.0, oldDelta=32000000000.0;
     double dindex=0.0;
     int i=0, index=-1, xyzt=0;

     if ( this->ss[LO] == IRRELEVANT_AXIS )
	  return IRRELEVANT_AXIS;

     if ( value < this->ww[LO] )
	  return 0;
     else if ( value > this->ww[HI] )
	  return (this->num_points-1);
     
     if ( this->has_regular_spacing ) {

	  dindex = ((double)value - this->start) / this->delta;
	  if ( (dindex - (int)dindex) < 0.5 )
	       return (int)dindex;
	  else
	       return (int)dindex + 1;

     } else /* use the lookup list */ {

	  /*
	   * - Find the value with the smallest difference between it and the desired value.
	   * - Do some error checking and return it's index.
	   */    
	  for (i=0; i<this->num_points; i++) {
	       delta = value - this->value[i];
	       delta = (delta<0.0) ? -delta : delta;
	       if (delta > oldDelta) {
		    index = i - 1;
		    break;
	       }
	       oldDelta = delta;
	  }
     
	  if ( index != -1 )
	       return index;
	  else if ( i == this->num_points )
	       return i-1;
	  else
	       return 0;

     }
     
}


float JC_Axis_ReturnNearestValue( JC_Axis *this, float value )
{
     double delta=0.0, oldDelta=32000000000.0;
     double dindex=0.0;
     int i=0, index=-1, xyzt=0;

     if ( this->ss[LO] == IRRELEVANT_AXIS )
	  return 0.0;

     if ( value < this->ww[LO] )
	  return (this->ww[LO]);
     else if ( value > this->ww[HI] )
	  return (this->ww[HI]);
     
     if ( this->has_regular_spacing ) {

	  dindex = ((double)value - this->start) / this->delta;
	  if ( (dindex - (int)dindex) < 0.5 )
	       return ( dindex * (this->delta) + this->start );
	  else
	       return ( (dindex+1) * (this->delta) + this->start );

     } else /* use the lookup list */ {

	  /*
	   * - Find the value with the smallest difference between it and the desired value.
	   * - Do some error checking and return that value.
	   */    
	  for (i=0; i<this->num_points; i++) {
	       delta = value - this->value[i];
	       delta = (delta<0.0) ? -delta : delta;
	       if (delta > oldDelta) {
		    index = i - 1;
		    break;
	       }
	       oldDelta = delta;
	  }
     
	  if ( index != -1 )
	    return (this->value[index]);
	  else if ( i == this->num_points )
	    return (this->value[i-1]);
	  else {
	    fprintf(stderr, "Axis_ReturnNearestValue(): error in lookup list.\n");
	    return 0.0;
	  }

     }
     
}


float JC_Axis_ReturnNearestMidpoint( JC_Axis *this, float value )
{
     float delta=0.0, oldDelta=32000000000.0;
     float dindex=0.0;
     int i=0, index=-1, xyzt=0;

     if ( this->ss[LO] == IRRELEVANT_AXIS )
	  return 0.0;

     if ( this->has_regular_spacing ) {

       if ( value < this->ww[LO] )
	 return (this->ww[LO] - this->delta/2);
       else if ( value > this->ww[HI] )
	 return (this->ww[HI] + this->delta/2);
     
       dindex = ((double)value - this->start) / this->delta;

       if ( (dindex - (int)dindex) < 0.5 )
	 return ( dindex * (this->delta) + this->start - this->delta/2 );
       else
	 return ( (dindex+1) * (this->delta) + this->start - this->delta/2 );

     } else /* use the lookup list */ {

       /*
	* TODO: fix Axis_ReturnNearestMidpoint() code to use Ferret's grid cell edges.
	*
	* Right now I calculate this here in a quick and dirty fashion.
	*/
       if ( value < this->ww[LO] )
	 return (this->ww[LO] - (this->value[1]-this->value[0])/2);
       else if ( value > this->ww[HI] )
	 return (this->ww[HI] + (this->value[this->num_points-1]-this->value[this->num_points-2])/2);
     
	  /*
	   * - Find the value with the smallest difference between it and the desired value.
	   * - Do some error checking and return it's index.
	   */    
	  for (i=0; i<this->num_points; i++) {
	       delta = value - this->value[i];
	       delta = (delta<0) ? -delta : delta;
	       if (delta > oldDelta) {
		    index = i - 1;
		    break;
	       }
	       oldDelta = delta;
	  }
     
	  if ( index != -1 )
	    return (this->value[index] - (this->value[index]-this->value[index-1])/2);
	  else if ( i == this->num_points )
	    return (this->value[i-1] - (this->value[index-1]-this->value[index-2])/2);
	  else {
	    fprintf(stderr, "Axis_ReturnNearestValue(): error in lookup list.\n");
	    return 0.0;
	  }

     }
     
}


void JC_Axis_PrintNoValues( JC_Axis *this, FILE *File_ptr )
{
     
     fprintf( File_ptr, "========== JC_Axis contents: ==========\n");

     fprintf( File_ptr, "\n\
name:                %s\n\
title:               %s\n\
units:               %s\n\
start,delta,end:     %g,%g,%g\n\
ww:                 [%g, %g, %g]\n\
ss:                 [%d, %d, %d]\n\
xyzt:                %d\n\
unit_code:           %d\n\
num_points:          %d\n",
	     this->name, this->title, this->units, this->start, this->delta, this->end,
	     this->ww[LO], this->ww[HI], this->ww[PT],
	     this->ss[LO], this->ss[HI], this->ss[PT], this->xyzt, this->unit_code,
	     this->num_points);
     
     fprintf( File_ptr, "\
is_modulo:           %s\n\
has_fancy_labeling:  %s\n\
has_regular_spacing: %s\n\
is_expandable:       %s\n",
	     this->is_modulo ? "TRUE" : "FALSE",
	     this->has_fancy_labeling ? "TRUE" : "FALSE",
	     this->has_regular_spacing ? "TRUE" : "FALSE",
	     this->is_expandable ? "TRUE" : "FALSE");
     
     if ( this->xyzt == T_AXIS ) {
	  fprintf( File_ptr, "\n\
time_type:          %d\n\
secsAtT0:           %g\n\
secsPerUnit:        %g\n",
		  this->time_type, this->secsAtT0, this->secsPerUnit);
     }

     fprintf( File_ptr, "========================================\n");

}


void JC_Axis_Printssww( JC_Axis *this, FILE *File_ptr )
{
     
     fprintf( File_ptr, "========== JC_Axis contents: ==========\n");

     fprintf( File_ptr, "\n\
title:                %s\n\
ww:                 [%g, %g, %g]\n\
ss:                 [%d, %d, %d]\n",
	     this->title,
	     this->ww[LO], this->ww[HI], this->ww[PT],
	     this->ss[LO], this->ss[HI], this->ss[PT]);

     fprintf( File_ptr, "========================================\n");

}


/* .................... JC_Span methods .................... */


void JC_Span_Clear( JC_Span *this )
{
     memset(this, 0, sizeof(JC_Span));
}


void JC_Span_NewAxis( JC_Span *this, JC_Axis *A_ptr )
{

     if ( A_ptr->ss[LO] == IRRELEVANT_AXIS ) {
	  JC_Span_Clear( this );
	  this->ss[LO] = IRRELEVANT_AXIS;
	  this->xyzt = A_ptr->xyzt;
	  return;
     }

     /*
      * - if the low end of the current span is out of the new axis range
      *      adjust appropriately
      * - create a new index value for ss[LO]
      */
     if ( this->ww[LO] < A_ptr->ww[LO] || this->ww[LO] >= A_ptr->ww[HI] )
	  this->ww[LO] = A_ptr->ww[LO];
     this->ss[LO] = JC_Axis_ReturnIndex(A_ptr, this->ww[LO]);

     /*
      * - if the high end of the current span is out of the new axis range
      *      adjust appropriately
      * - create a new index value for ss[HI]
      */
     if ( this->ww[HI] <= A_ptr->ww[LO] || this->ww[HI] > A_ptr->ww[HI] )
	  this->ww[HI] = A_ptr->ww[HI];
     this->ss[HI] = JC_Axis_ReturnIndex(A_ptr, this->ww[HI]);

     /*
      * - if the point value of the current span is out of the new axis range
      *      adjust appropriately
      * - create a new index value for ss[PT]
      */
     if ( this->ww[PT] < A_ptr->ww[LO] )
	  this->ww[PT] = A_ptr->ww[LO];
     else if ( this->ww[PT] > A_ptr->ww[HI] )
	  this->ww[PT] = A_ptr->ww[HI];
     this->ss[PT] = JC_Axis_ReturnIndex(A_ptr, this->ww[PT]);

     /*
      * - if the current span LO and HI are the same value but the current axis has a range
      *      expand the current span to the same range as the current axis
      *   (This occurs when the previous dataset has a range with only one value.
      *    For instance: all values are at a depth of 12.5m.)
      */
     if ( (this->ww[LO] == this->ww[HI]) && (A_ptr->ww[LO] != A_ptr->ww[HI]) ) {
       this->ww[LO] = A_ptr->ww[LO];
       this->ww[HI] = A_ptr->ww[HI];
       this->ss[LO] = JC_Axis_ReturnIndex(A_ptr, this->ww[LO]);
       this->ss[HI] = JC_Axis_ReturnIndex(A_ptr, this->ww[HI]);
     }

     /*
      * The following are some axis constants that it is convenient to carry around
      * with the span.
      */
     strcpy(this->title, A_ptr->title);
     this->secsAtT0 = A_ptr->secsAtT0;
     this->secsPerUnit = A_ptr->secsPerUnit;
     this->xyzt = A_ptr->xyzt;
     this->num_points = A_ptr->num_points;
     this->time_type = A_ptr->time_type;
     this->is_modulo = A_ptr->is_modulo;
     this->has_fancy_labeling = A_ptr->has_fancy_labeling;
     this->is_expandable = A_ptr->is_expandable;
  
}


void JC_Span_NewTransform(JC_Span *this, JC_Transform *T_ptr)
{

     /*
      * - if the transform compresses
      *      if 'this' is expandable
      *           show the lo and hi scrollbars
      *           make it compressed
      *      else
      *           make it uncompressed
      * 
      * - else if the transform doesn't compress
      *           make it uncompressed
      */

     if ( T_ptr->compresses ) {
	  if ( this->is_expandable ) {
	       this->needs_lo_hi_displayed_in_GUI = TRUE;
	       this->is_compressed_in_GUI = TRUE;
	  } else {
	       this->needs_lo_hi_displayed_in_GUI = FALSE;
	       this->is_compressed_in_GUI = FALSE;
	  }
     }
     else
	  this->is_compressed_in_GUI = FALSE;

}


void JC_Span_Print( JC_Span *this, FILE *File_ptr )
{

     fprintf( File_ptr, "========== JC_Span contents: ==========\n");

     fprintf( File_ptr, "\n\
title:               %s\n\
ww:                 [%g, %g, %g]\n\
ss:                 [%d, %d, %d]\n\
xyzt:                %d\n\
num_points:          %d\n",
	     this->title, this->ww[LO], this->ww[HI], this->ww[PT],
	     this->ss[LO], this->ss[HI], this->ss[PT], this->xyzt,
	     this->num_points);
     
     fprintf( File_ptr, "\
is_modulo:           %s\n\
has_fancy_labeling:  %s\n\
is_expandable:       %s\n\
needs_lo_hi_displayed_in_GUI: %s\n\
is_compressed_in_GUI:         %s\n\
by_index_in_GUI:              %s\n",
	     this->is_modulo ? "TRUE" : "FALSE",
	     this->has_fancy_labeling ? "TRUE" : "FALSE",
	     this->is_expandable ? "TRUE" : "FALSE",
	     this->needs_lo_hi_displayed_in_GUI ? "TRUE" : "FALSE",
	     this->is_compressed_in_GUI ? "TRUE" : "FALSE",
	     this->by_index_in_GUI ? "TRUE" : "FALSE");
     
     if ( this->xyzt == T_AXIS ) {
	  fprintf( File_ptr, "\n\
time_type:          %d\n\
secsAtT0:           %g\n\
secsPerUnit:        %g\n",
		  this->time_type, this->secsAtT0, this->secsPerUnit);
     }

     fprintf( File_ptr, "========================================\n");


}


/* .................... JC_Transform methods .................... */


void JC_Transform_Clear( JC_Transform *this )
{
     memset(this, 0, sizeof(JC_Transform));
}


/* .................... JC_Regridding methods .................... */


void JC_Regridding_Initialize( JC_Regridding *this )
{
     int xyzt=0;

     for ( xyzt=0; xyzt<5; xyzt++ ) {
	  strcpy(this->var[xyzt], "");
	  strcpy(this->dset[xyzt], "");
	  strcpy(this->rg_transform[xyzt], "LIN");
	  this->type = UNIFORM;
     }
}


/* .................... JC_Variable methods .................... */


void JC_Variable_Clear( JC_Variable *this )
{
     memset(this, 0, sizeof(JC_Variable));
}


void JC_Variable_New( JC_Variable *this, char *name_ptr, char *dset_ptr )
{
     int xyzt=0, i=0, j=0, num_points=0, num_vals=0;
     char temp_var_name[MAX_NAME_LENGTH], tempText[MAX_NAME_LENGTH];
     char *axis_code[]={"1", "2", "3", "4"};
     char nullStr[1] = {'\0'};
     char name[MAX_NAME_LENGTH]="";
     char dset[MAX_NAME_LENGTH]="";

/*
 * I need to have copies of 'name' and 'dset' because 'name_ptr' and 'dset_ptr' might point
 * to this->name and this->dset.  The information would be wiped out by the JC_Variable_Clear()
 * a few lines down.
 */

     strcpy(name, name_ptr);
     strcpy(dset, dset_ptr);

     if ( strlen(name) == 0 || strlen(dset) == 0 ) {
	  fprintf(stderr, "ERROR in JC_OOP.c: JC_Variable_New(): name=\"%s\", dset=\"%s\".\n", name, dset);
	  return;
     }

/*
 * - Copy "name" to "tempText" because strtok(s1,s2) will alter "s1".
 * - If this variable is a vector pair
 *      take the first variable as the new one and use it throughout
 */
     strcpy(tempText, name);
     strcpy(temp_var_name, strtok(tempText, ","));
     
     JC_Variable_Clear( this );

/*
 * - QUERY_GRID(sBuffer, dataset, variable)
 *   sBuffer gets: grid name, axis names, axis orientations.
 *
 * - Put these values into 'this' variable.
 *   (Orientation = "1/2/3/4", not worth keeping)
 */
     strcpy(this->name, name);
     strcpy(this->dset, dset);
     ferret_query(QUERY_GRID, sBuffer, dset, temp_var_name, nullStr, nullStr, nullStr);
     sBuffer->textP = &sBuffer->text[0];
     sBuffer->textP = CollectToReturn(sBuffer->textP, this->grid);
     for (xyzt=0; xyzt<4; xyzt++) {
	  sBuffer->textP = CollectToReturn(sBuffer->textP, this->axis[xyzt].name);
	  this->axis[xyzt].xyzt = xyzt;
     }
     
/*
 * - QUERY_SPAN(sBuffer, dataset, variable)
 *   sBuffer gets: grid name, axis limits
 *
 * - Put these values into 'this' variable's axes.
 */

     ferret_query( QUERY_SPAN, sBuffer, dset, temp_var_name, nullStr, nullStr, nullStr);
     sBuffer->textP = &sBuffer->text[0];
     sBuffer->textP = CollectToReturn(sBuffer->textP, this->grid);
     for (xyzt=0; xyzt<4; xyzt++) {
	       this->axis[xyzt].ss[LO] = (int)((double)sBuffer->nums[xyzt*4+0]);
	       if ( this->axis[xyzt].ss[LO] == IRRELEVANT_AXIS )
		    continue;
	       this->axis[xyzt].ss[HI] = (int)((double)sBuffer->nums[xyzt*4+1]);
	       this->axis[xyzt].ww[LO] = (double)sBuffer->nums[xyzt*4+2];
	       this->axis[xyzt].ww[HI] = (double)sBuffer->nums[xyzt*4+3];

	       if ( this->axis[xyzt].ss[LO] < 0 || this->axis[xyzt].ss[HI] < 0 ) {
		    fprintf(stderr, "ERROR in JC_OOP: Variable_New(): axis[%d].ss[LO] = %d\n", xyzt, this->axis[xyzt].ss[LO]);
		    JC_Axis_PrintNoValues(&(this->axis[xyzt]), stderr);
	       }
     }
     


/*
 * - If the axis is relevant, decrement the index values now that we are in a 'C' program.
 */

     for (xyzt=0; xyzt<4; xyzt++) {
	  if ( this->axis[xyzt].ss[LO] != IRRELEVANT_AXIS ) {
	       this->axis[xyzt].ss[LO]--;
	       this->axis[xyzt].ss[HI]--;
	       if ( this->axis[xyzt].ss[LO] < 0 ) {
		    fprintf(stderr, "ERROR in JC_OOP: Variable_New(): axis[%d].ss[LO] = %d\n", xyzt, this->axis[xyzt].ss[LO]);
		    JC_Axis_PrintNoValues(&(this->axis[xyzt]), stderr);
		    fprintf(stderr, "\t\taxis[%d].ss[LO] being reset to 0\n", xyzt);
		    this->axis[xyzt].ss[LO] = 0;
	       }
	  }
     }
     

/*
 * - QUERY_GAXIS(sBuffer, grid, orientation)
 *   sBuffer gets: title, units, flags, num_points, start, delta, time_0, time_unit.
 *
 * - Put these values into this->axis[i].
 */
     for (xyzt=0; xyzt<4; xyzt++) {

	  if ( this->axis[xyzt].ss[LO] == IRRELEVANT_AXIS )
	       continue;

	  ferret_query(QUERY_GAXIS, sBuffer, this->grid, axis_code[xyzt], nullStr, nullStr, nullStr);
	  this->axis[xyzt].num_points = (int)((double)sBuffer->nums[0]);
	  if ( this->axis[xyzt].num_points > 1 )
	       this->axis[xyzt].is_expandable = TRUE;
	  sBuffer->textP = &sBuffer->text[0];
	  sBuffer->textP = CollectToReturn(sBuffer->textP, this->axis[xyzt].title);
	  sBuffer->textP = CollectToReturn(sBuffer->textP, this->axis[xyzt].units);
	  this->axis[xyzt].is_modulo           = (Boolean)sBuffer->flags[0];
	  this->axis[xyzt].has_fancy_labeling  = (Boolean)sBuffer->flags[1];
	  this->axis[xyzt].has_regular_spacing = (Boolean)sBuffer->flags[2];
	  this->axis[xyzt].unit_code           =     (int)sBuffer->flags[3];
	  this->axis[xyzt].num_points          =     (int)sBuffer->nums[0];
	  this->axis[xyzt].start               =  (double)sBuffer->nums[1];
	  this->axis[xyzt].delta               =  (double)sBuffer->nums[2];

/*
 * - If this is the longitude axis AND it is modulo AND it has regular spacing
 *      double it's extent to match the Map in the main interface.
 *
 * - NB_ We need to add one more to ss[HI] because we decremented above to convert to 'C' arrays.
 *   (ie. If we had 10 points, ss[HI] from sBuffer was 10 we would have decremented to 9.
 *        For the modulo axis, num_points is 2 * num_points = 20.
 *        But 2 * ss[HI] = 18 so we add one more so the C array will go from 0 to 19 (20 points) )
 */
	  if ( xyzt == X_AXIS && this->axis[xyzt].is_modulo && this->axis[xyzt].has_regular_spacing ) {
	       this->axis[xyzt].num_points = 2 * this->axis[xyzt].num_points;
	       this->axis[xyzt].ss[HI] = 2 * this->axis[xyzt].ss[HI] + 1;
	       this->axis[xyzt].ww[HI] = this->axis[xyzt].ww[HI] + (double)360.0;
	  }
	  
/*
 * - If this is the time axis
 *      get calendar information
 *
 *         if it has fancy labeling
 *            it's a calendar
 *         else if it's modulo ( <= 12 )
 *            it's a climatology
 *         else if it's modulo ( > 12 )
 *            some kind of mistake
 *         else
 *            it's model time
 */	  
	  if ( xyzt == T_AXIS ) {
	       this->axis[xyzt].secsAtT0    =  (double)sBuffer->nums[3];
	       this->axis[xyzt].secsPerUnit =  (double)sBuffer->nums[4];

	       if ( this->axis[xyzt].is_modulo && this->axis[xyzt].num_points <= 12 )
		    this->axis[xyzt].time_type = CLIMATOLOGY_TIME;

	       else if ( this->axis[xyzt].has_fancy_labeling )
		    this->axis[xyzt].time_type = CALENDAR_TIME;

	       else if ( this->axis[xyzt].is_modulo && this->axis[xyzt].num_points > 12 ) {
		    fprintf(stderr, "ERROR in JC_OOP.c: JC_Variable_New(): time axis modulo && num_points = %d.\n", this->axis[xyzt].num_points);
		    JC_Axis_PrintNoValues(&(this->axis[xyzt]), stderr);

	       } else
		    this->axis[xyzt].time_type = MODEL_TIME;
	  }
	  
     }

/*
 * - If this axis doesn't have regular spacing
 *
 * -    QUERY_GCOORDS(sBuffer, grid, orientation, offset, stride)
 *      sBuffer gets: the entire set of values along an axis.
 *
 * -    Put these values into this->axis[i].values
 */
     for (xyzt=0; xyzt<4; xyzt++) {

	  if ( this->axis[xyzt].ss[LO] == IRRELEVANT_AXIS )
	       continue;
	  
	  if ( !this->axis[xyzt].has_regular_spacing ) {

	       ferret_query(QUERY_GCOORDS, sBuffer, this->grid, axis_code[xyzt], "0", "1", nullStr);
	       num_vals = (int)sBuffer->numNumbers;
	       if ( num_vals != this->axis[xyzt].num_points ) {
		    fprintf(stderr, "ERROR in JC_OOP.c: JC_Variable_New(): num_vals[%d] != num_points[%d].\n", 
			   num_vals, this->axis[xyzt].num_points);
		    fprintf(stderr, "resetting num_vals to %d.\n", this->axis[xyzt].num_points);
		    num_vals = this->axis[xyzt].num_points;
	       }

               if ( num_vals > MAX_AXIS_SIZE ) {
                    fprintf(stderr, "WARNING in JC_OOP.c: JC_Variable_New(): num_vals[%d] > MAX_AXIS_SIZE[%d]\
, num_vals being reset to %d\n", num_vals, MAX_AXIS_SIZE, MAX_AXIS_SIZE);
		    JC_Axis_PrintNoValues(&(this->axis[xyzt]), stderr);
                    num_vals = MAX_AXIS_SIZE;
               }

	       for (j=0; j<num_vals; j++)
		    this->axis[xyzt].value[j] =  (double)sBuffer->nums[j];

	       if ( xyzt == X_AXIS && this->axis[xyzt].is_modulo ) {
		    this->axis[xyzt].num_points = 2 * this->axis[xyzt].num_points;
		    this->axis[xyzt].ss[HI] = 2 * this->axis[xyzt].ss[HI];
		    this->axis[xyzt].ww[HI] = this->axis[xyzt].ww[HI] + (double)360.0;
		    for (j=num_vals; j<2*num_vals; j++)
			 this->axis[xyzt].value[j] = this->axis[xyzt].value[j-num_vals];
	       }

	  }

     }
     
     JC_Variable_SetokGeoms( this );

}


void JC_Variable_SetokGeoms( JC_Variable *this )
{
     int geometry=0, xyzt=0, geometry_is_ok=0;
     Boolean full_axis[4];
   
     /*
      * - test each axis to see if it is a full axis.
      */

     for (xyzt=0; xyzt<4; xyzt++) {
	  if ( this->axis[xyzt].ww[LO] == this->axis[xyzt].ww[HI] )
	       full_axis[xyzt] = FALSE;
	  else
	       full_axis[xyzt] = TRUE;
     }
  
 
     /*
      * - for each geometry
      *     for each axis
      *       if this axis is required, see if it exists
      * - if all axes exist
      *     make this Geometry available
      */

     for (geometry=0; geometry<16; geometry++) {
	  geometry_is_ok = TRUE;
	  for (xyzt=0; xyzt<4; xyzt++) {
	       if (geom_axes[geometry][xyzt] == 1)
		    geometry_is_ok = geometry_is_ok && full_axis[xyzt];
	  }
	  if ( geometry_is_ok )
	       this->okGeoms[geometry] = TRUE;
	  else
	       this->okGeoms[geometry] = FALSE;
     }

}


/* .................... JC_Region methods .................... */


void JC_Region_Clear( JC_Region *this )
{
  memset(this, 0, sizeof(JC_Region));
}

void JC_Region_Initialize( JC_Region *this )
{
  int xyzt=0;

  JC_Region_Clear(this);
  for (xyzt=0; xyzt<4; xyzt++)
    this->span[xyzt].ss[LO] == IRRELEVANT_AXIS;
}

/*
 * This function resets the geometry when the span has been changed because
 * of a new transform selection.
 */
void JC_Region_SetGeometryFromSpan( JC_Region *this )
{
     int xyzt=0;
     Boolean full_axis[4]={FALSE, FALSE, FALSE, FALSE};

     for (xyzt=0; xyzt<4; xyzt++) {
	  if ( this->span[xyzt].ss[LO] == IRRELEVANT_AXIS )
	       continue;
	  if ( this->span[xyzt].needs_lo_hi_displayed_in_GUI && !this->span[xyzt].is_compressed_in_GUI )
	       full_axis[xyzt] = TRUE;
     }

     if ( full_axis[X_AXIS] ) {
	  if ( full_axis[Y_AXIS] ) {
	       if ( full_axis[Z_AXIS] ) {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_XYZT;
		    else
			 this->geometry = GEOM_XYZ;
	       } else {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_XYT;
		    else
			 this->geometry = GEOM_XY;
	       }
	  } else {
	       if ( full_axis[Z_AXIS] ) {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_XZT;
		    else
			 this->geometry = GEOM_XZ;
	       } else {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_XT;
		    else
			 this->geometry = GEOM_X;
	       }
	  }

     } else /* X_AXIS is not full */{

	  if ( full_axis[Y_AXIS] ) {
	       if ( full_axis[Z_AXIS] ) {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_YZT;
		    else
			 this->geometry = GEOM_YZ;
	       } else {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_YT;
		    else
			 this->geometry = GEOM_Y;
	       }
	  } else {
	       if ( full_axis[Z_AXIS] ) {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_ZT;
		    else
			 this->geometry = GEOM_Z;
	       } else {
		    if ( full_axis[T_AXIS] )
			 this->geometry = GEOM_T;
		    else
			 this->geometry = GEOM_POINT;
	       }
	  }
     }

}


void JC_Region_NewTransform(JC_Region *this, int xyzt, JC_Transform *T_ptr)
{
     this->transform[xyzt] = *T_ptr;
     JC_Span_NewTransform( &(this->span[xyzt]), T_ptr );
     JC_Region_SetGeometryFromSpan( this );
}


void JC_Region_NewFramer( JC_Region *this, float lat[3], float lon[3] )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;

     this->span[X_AXIS].ww[LO] = lon[LO];
     this->span[X_AXIS].ww[HI] = lon[HI];
     this->span[X_AXIS].ww[PT] = lon[PT];
     this->span[X_AXIS].ss[LO] = JC_Axis_ReturnIndex(&(V_ptr->axis[X_AXIS]), lon[LO]);
     this->span[X_AXIS].ss[HI] = JC_Axis_ReturnIndex(&(V_ptr->axis[X_AXIS]), lon[HI]);
     this->span[X_AXIS].ss[PT] = JC_Axis_ReturnIndex(&(V_ptr->axis[X_AXIS]), lon[PT]);
     
     this->span[Y_AXIS].ww[LO] = lat[LO];
     this->span[Y_AXIS].ww[HI] = lat[HI];
     this->span[Y_AXIS].ww[PT] = lat[PT];
     this->span[Y_AXIS].ss[LO] = JC_Axis_ReturnIndex(&(V_ptr->axis[Y_AXIS]), lat[LO]);
     this->span[Y_AXIS].ss[HI] = JC_Axis_ReturnIndex(&(V_ptr->axis[Y_AXIS]), lat[HI]);
     this->span[Y_AXIS].ss[PT] = JC_Axis_ReturnIndex(&(V_ptr->axis[Y_AXIS]), lat[PT]);
     
}


void JC_Region_NewGeometry( JC_Region *this, int geometry )
{
     int xyzt=0;
     
     /*
      * - Set 'this' region's geometry
      *
      * - If an axis is expandable 
      *      if an axis is found in the new geometry
      *         display lo and hi values of the span
      *         turn off compression
      *         clear the transform
      *      else
      *         if the axis is being compressed by a transform
      *            display lo and hi values of the span
      *         else
      *            display pt value of the span
      *
      * - Else the axis is NOT expandable (can happen with new variable selection)
      *      only display pt value
      *      turn off compression
      *      clear the transform
      */
     
     
     this->geometry = geometry;
     
     for (xyzt=0; xyzt<4; xyzt++) {

	  if ( this->span[xyzt].ss[LO] == IRRELEVANT_AXIS )
	       continue;

	  if ( this->span[xyzt].is_expandable ) {

	    if ( geom_axes[this->geometry][xyzt] ) {
	      this->span[xyzt].needs_lo_hi_displayed_in_GUI = TRUE;
	      this->span[xyzt].is_compressed_in_GUI = FALSE;
	       JC_Transform_Clear( &(this->transform[xyzt]) );
	    } else		/* axis is not found */ {
	      if ( this->transform[xyzt].compresses ) {
		this->span[xyzt].needs_lo_hi_displayed_in_GUI = TRUE;
		this->span[xyzt].is_compressed_in_GUI = TRUE;
	      } else {
		this->span[xyzt].needs_lo_hi_displayed_in_GUI = FALSE;
		this->span[xyzt].is_compressed_in_GUI = FALSE;
	      }
	    }

	  } else /* span[xyzt] is not expandable */ {

	       this->span[xyzt].needs_lo_hi_displayed_in_GUI = FALSE;
	       this->span[xyzt].is_compressed_in_GUI = FALSE;
	       JC_Transform_Clear( &(this->transform[xyzt]) );

	  }
     }
     
}


void JC_Region_NewVariable( JC_Region *this, JC_Variable *V_ptr, Boolean first_ever )
{
     float value=0;
     int i=0, geometry=0, xyzt=0, lo_hi_pt=0;
     Boolean this_is_a_new_geometry = FALSE;

     /* 
      * - If the current geometry is not available in the new variable
      *     look for most desirable geometry to switch to
      *
      * - For each axis
      *      reconcile the spans with the new axes
      *
      * - Set "needs_lo_hi_displayed_in_GUI" and "is_compressed_in_GUI" flags on spans 
      */

     if ( !(V_ptr->okGeoms[this->geometry]) || first_ever ) {
	  geometry = geom_desirability[i++];
	  while (!V_ptr->okGeoms[geometry]) {
	       geometry = geom_desirability[i++];
	  }
	  this->geometry = geometry;
	  this_is_a_new_geometry = TRUE;
     }

     for (xyzt=0; xyzt<4; xyzt++)
	  JC_Span_NewAxis( &(this->span[xyzt]), &(V_ptr->axis[xyzt]) );

     if ( this_is_a_new_geometry || first_ever ) {

/*
 * - If this is the longitude axis AND it has regular spacing
 *      halve it's extent (It was doubled in JC_Variable_New to match the Map in the main interface).
 */
       if ( V_ptr->axis[X_AXIS].is_modulo )
	 this->span[X_AXIS].ww[HI] = V_ptr->axis[X_AXIS].ww[HI] - 360.0;
       else	  
	 this->span[X_AXIS].ww[HI] = V_ptr->axis[X_AXIS].ww[HI];

       this->span[X_AXIS].ww[LO] = V_ptr->axis[X_AXIS].ww[LO];
       this->span[Y_AXIS].ww[LO] = V_ptr->axis[Y_AXIS].ww[LO];
       this->span[Y_AXIS].ww[HI] = V_ptr->axis[Y_AXIS].ww[HI];

       this->span[X_AXIS].ss[LO] = JC_Axis_ReturnIndex(&(V_ptr->axis[X_AXIS]), this->span[X_AXIS].ww[LO]);
       this->span[X_AXIS].ss[HI] = JC_Axis_ReturnIndex(&(V_ptr->axis[X_AXIS]), this->span[X_AXIS].ww[HI]);
       this->span[Y_AXIS].ss[LO] = JC_Axis_ReturnIndex(&(V_ptr->axis[Y_AXIS]), this->span[Y_AXIS].ww[LO]);
       this->span[Y_AXIS].ss[HI] = JC_Axis_ReturnIndex(&(V_ptr->axis[Y_AXIS]), this->span[Y_AXIS].ww[HI]);
     }

     JC_Region_NewGeometry( this, this->geometry );

     for ( xyzt=0; xyzt<4; xyzt++ ) {
       if ( this->span[xyzt].ss[LO] != IRRELEVANT_AXIS ) {
	 if ( this->span[xyzt].ss[LO] < 0 )
	   fprintf(stderr, "ERROR in JC_OOP: Region_NewVariable(): axis[%d].ss[LO] = %d\n",
		   xyzt, this->span[xyzt].ss[LO]);
       }
     }

}


/* .................... JC_Object methods (aka. ClonedVariable) .................... */


void JC_Clone_New( JC_Object *this, char *name, char *dset )
{
     JC_DatasetElement *DE_ptr=NULL;
     JC_Object *O_ptr=NULL;
     int status=LIST_OK;

     O_ptr = JC_Clone_ReturnPointer( name, dset );

     if ( O_ptr == NULL )
	  return;
     
     GLOBAL_Variable = O_ptr->variable;
     GLOBAL_Region = O_ptr->region;
     GLOBAL_Regridding = O_ptr->regridding;
     
/* JC_TODO ... Need to update regridding interface if it is showing when a new Clone is selected. */

}


JC_Object *JC_Clone_ReturnPointer( char *name, char *dset )
{
     JC_DatasetElement *DE_ptr=NULL;
     int status=LIST_OK;

     status = list_traverse(GLOBAL_DatasetList, dset, JC_ListTraverse_FoundDsetMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
     if ( status != LIST_OK ) {
	  fprintf(stderr, "ERROR in JC_OOP.c: JC_Clone_ReturnPointer(): dset %s not found\n", dset);
	  return NULL;
     }
     DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);
     
     status = list_traverse(DE_ptr->cvarList, name, JC_ListTraverse_FoundCvarMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
     if ( status != LIST_OK ) {
	  fprintf(stderr, "ERROR in JC_OOP.c: JC_Clone_ReturnPointer(): cvar %s not found\n", name);
	  return NULL;
     }

     return (JC_Object *)list_curr(DE_ptr->cvarList);

}


void JC_Clone_Print( JC_Object *this, FILE *File_ptr )
{
     JC_Variable *V_ptr=&(this->variable);
     JC_Region *R_ptr=&(this->region);
     JC_Regridding *RG_ptr=&(this->regridding);
     
     fprintf( File_ptr, "========== JC_Clone contents: ==========\n\n");
     fprintf( File_ptr, "address = %d\n", (int) this);

     if ( this == NULL ) {

	  fprintf(File_ptr, "Error in JC_OOP.c: Clone_Print(): NULL object pointer.\n");
	  
     } else {
	  
	  fprintf(File_ptr, "\
var = \"%s\", dset = \"%s\"\n\
X = ss: [%d,%d,%d], ww: [%f,%f,%f]\n\
Y = ss: [%d,%d,%d], ww: [%f,%f,%f]\n\
Z = ss: [%d,%d,%d], ww: [%f,%f,%f]\n\
T = ss: [%d,%d,%d], ww: [%f,%f,%f]\n",
		  V_ptr->name, V_ptr->dset,
		  R_ptr->span[X_AXIS].ss[LO], R_ptr->span[X_AXIS].ss[PT], R_ptr->span[X_AXIS].ss[HI],
		  R_ptr->span[X_AXIS].ww[LO], R_ptr->span[X_AXIS].ww[PT], R_ptr->span[X_AXIS].ww[HI],
		  R_ptr->span[Y_AXIS].ss[LO], R_ptr->span[Y_AXIS].ss[PT], R_ptr->span[Y_AXIS].ss[HI],
		  R_ptr->span[Y_AXIS].ww[LO], R_ptr->span[Y_AXIS].ww[PT], R_ptr->span[Y_AXIS].ww[HI],
		  R_ptr->span[Z_AXIS].ss[LO], R_ptr->span[Z_AXIS].ss[PT], R_ptr->span[Z_AXIS].ss[HI],
		  R_ptr->span[Z_AXIS].ww[LO], R_ptr->span[Z_AXIS].ww[PT], R_ptr->span[Z_AXIS].ww[HI],
		  R_ptr->span[T_AXIS].ss[LO], R_ptr->span[T_AXIS].ss[PT], R_ptr->span[T_AXIS].ss[HI],
		  R_ptr->span[T_AXIS].ww[LO], R_ptr->span[T_AXIS].ww[PT], R_ptr->span[T_AXIS].ww[HI]);
     
     }

     fprintf( File_ptr, "========================================\n");
}


/* .................... JC_DatasetElement methods .................... */



void JC_DatasetElement_QueryFerret( JC_DatasetElement *this, Boolean new_dataset )
{
     JC_DefinedVariable JC_DV;
     LIST *ferret_dvars;

     int i=0, j=0, num_strings=0, status=LIST_OK;
     char tempText[MAX_NAME_LENGTH]="";
     char nullStr[1] = {'\0'};

     if ( this == NULL ) {
	  fprintf(stderr, "ERROR in JC_OOP.c: JC_DatasetElement_QueryFerret: this = NULL\n");
	  return;
     }

     if ( strlen(this->name) == 0 ) {
	  fprintf(stderr, "ERROR in JC_OOP.c: JC_DatasetElement_QueryFerret: this->name = \"\".\n");
	  return;
     }
/*
 * - Initialize the 'ferret_dvars' list.
 */

     if ( (ferret_dvars = list_init()) == NULL )
	  fprintf(stderr, "ERROR in JC_OOP.c: JC_DatasetElement_QueryFerret(): Unable to initialize ferret_dvars.\n");
     
/*
 * - If this is a new dataset:
 *      initialize the lists inside 'this' element.
 */

     if ( new_dataset ) {
	  if ( (this->varList = list_init()) == NULL )
	       fprintf(stderr, "ERROR in JC_OOP.c: JC_DatasetElement_QueryFerret(): Unable to initialize this->varList.\n");
	  if ( (this->dvarList = list_init()) == NULL )
	       fprintf(stderr, "ERROR in JC_OOP.c: JC_DatasetElement_QueryFerret(): Unable to initialize this->dvarList.\n");
	  if ( (this->cvarList = list_init()) == NULL )
	       fprintf(stderr, "ERROR in JC_OOP.c: JC_DatasetElement_QueryFerret(): Unable to initialize this->cvarList.\n");
     }

/*
 * Fill in 'this'->varList.
 *
 * - If this is a new dataset:
 *      get each variable name, convert it to uppercase and add it to the list.
 *      then check for vector pairs and add those.
 *
 *   (If this is an old dataset:  file variables cannot change)
 */

     if ( new_dataset ) {

	  ferret_query(QUERY_VARIABLE, sBuffer, this->name, nullStr, nullStr, nullStr, nullStr);
	  sBuffer->textP = &sBuffer->text[0];
	  num_strings = sBuffer->numStrings;

	  for (i=0; i<num_strings; i++) {
	       sBuffer->textP = CollectToReturn(sBuffer->textP, tempText);
	       for (j=0; j<strlen(tempText); j++)
		    tempText[j] = toupper(tempText[j]);
	       list_insert_after(this->varList, tempText, sizeof(tempText));
	  }

	  JC_List_AddVectorPairs(this->varList, this->name);

     }
     

/*
 * - Query ferret for the defined variables.
 * - Make them all upper case.
 * - Add each one to the 'ferret_dvars' list.
 */

     ferret_query(QUERY_DVARS, sBuffer, this->name, nullStr, nullStr, nullStr, nullStr);
     sBuffer->textP = &sBuffer->text[0];
     num_strings = sBuffer->numStrings;
     
     for (i=0; i<num_strings; i++) {
	  sBuffer->textP = CollectToReturn(sBuffer->textP, tempText);
	  for (j=0; j<strlen(tempText); j++)
	       tempText[j] = toupper(tempText[j]);
	  list_insert_after(ferret_dvars, tempText, sizeof(tempText));
     }

/*
 * - Go through this->dvarList.
 * - If a name isn't found in the ferret list:
 *      delete it from the this->dvarList.
 */

     if ( list_mvfront(this->dvarList) != NULL ) {
	  for (i=0; i<list_size(this->dvarList); i++) {
	       strcpy(tempText, ((JC_DefinedVariable *)list_curr(this->dvarList))->name);
	       status = list_traverse(ferret_dvars, tempText, JC_ListTraverse_FoundMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));
	       if ( status == LIST_OK )
		    list_mvnext(this->dvarList);
	       else {
		    list_remove_curr(this->dvarList);
	       }
	  }
     }
     

/*
 * - Go through the ferret list.
 * - If a name isn't found in this->dvarList:
 *      add it to this->dvarList.
 */

     if ( list_mvfront(ferret_dvars) != NULL ) {
	  for (i=0; i<list_size(ferret_dvars); i++) {
	       strcpy(JC_DV.name, list_curr(ferret_dvars));
	       status = list_traverse(this->dvarList, JC_DV.name, JC_ListTraverse_FoundDvarMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));
	       if ( status == LIST_OK ) {
		    list_mvnext(ferret_dvars);
	       } else {
		    list_mvrear(this->dvarList);
		    list_insert_after(this->dvarList, &JC_DV, sizeof(JC_DV));
		    list_mvnext(ferret_dvars);
	       }
	  }
     }

/*
 * - Deallocate the 'ferret_dvars' list.
 */
   
     list_free(ferret_dvars, LIST_DEALLOC);

/*
 * - Don't touch  'this'->cvarList.
 *
 * - (Cloned variables are added one at a time by the "clone" button and are deleted when
 *    a dataset is removed.  Otherwise the cvarList shouldn't be touched.)
 */

}


int JC_DatasetElement_VarnameExists( JC_DatasetElement *this, char *name )
{
     int status=LIST_OK;

/*
 * For each list in 'this' element.

 * If we don't get all the way through the list:
 *  - found a match.
 */
     status = list_traverse(this->varList, name, JC_ListTraverse_FoundMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));

     if ( status == LIST_OK )
	  return TRUE;

     status = list_traverse(this->dvarList, name, JC_ListTraverse_FoundDvarMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));

     if ( status == LIST_OK )
	  return TRUE;

     status = list_traverse(this->cvarList, name, JC_ListTraverse_FoundCvarMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));

     if ( status == LIST_OK )
	  return TRUE;

     return FALSE;
}


/* .................... JC_List methods .................... */



void JC_DatasetList_Print( LIST *this, FILE *File_ptr )
{
     list_traverse(this, File_ptr, JC_ListTraverse_Dsetfprintf, (LIST_FRNT | LIST_FORW | LIST_SAVE));
}


void JC_DatasetList_Free( LIST *this )
{
     list_traverse(this, NULL, JC_ListTraverse_FreeDataset, (LIST_FRNT | LIST_FORW | LIST_SAVE));
     list_free(this, LIST_DEALLOC);
}


void JC_List_AddVectorPairs( LIST *this, char *dset )
{
     int i=0, status=LIST_OK;
     char var1[MAX_NAME_LENGTH], *var1_vartype, var1_grid[MAX_NAME_LENGTH];
     char var2[MAX_NAME_LENGTH], *var2_vartype, var2_grid[MAX_NAME_LENGTH];
     char new_var[MAX_NAME_LENGTH];
     char nullStr[1] = {'\0'};
     
     struct {
	  char c1[2],c2[2];
     } vPairs[6];

     strcpy(vPairs[0].c1, "X");
     strcpy(vPairs[0].c2, "Y");
     strcpy(vPairs[1].c1, "X");
     strcpy(vPairs[1].c2, "Z");
     strcpy(vPairs[2].c1, "Y");
     strcpy(vPairs[2].c2, "Z");
     strcpy(vPairs[3].c1, "U");
     strcpy(vPairs[3].c2, "V");
     strcpy(vPairs[4].c1, "U");
     strcpy(vPairs[4].c2, "W");
     strcpy(vPairs[5].c1, "V");
     strcpy(vPairs[5].c2, "W");

     if ( list_empty(this) )
	  return;
     
     for (i=0; i<6; i++) {
	  
/*
 * Traverse the list, looking for an element with vParis[i].c1.
 *
 * If you look through the whole list:
 * - continue.
 *
 * Put the found variable in var1.
 */

	  status = list_traverse(this, vPairs[i].c1, JC_ListTraverse_strstr, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  
	  if ( status == LIST_EXTENT )
	       continue;
	  
	  strcpy(var1, list_curr(this));
	  var1_vartype = strstr(list_curr(this), vPairs[i].c1);
	  var1_vartype++;
/*
 * Traverse the list again, looking for an element with vParis[i].c2.
 *
 * If you look through the whole list:
 * - continue.
 *
 * Put the found variable in var2.
 */

	  status = list_traverse(this, vPairs[i].c2, JC_ListTraverse_strstr, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  
	  if ( status == LIST_EXTENT )
	       continue;
	  
	  strcpy(var2, list_curr(this));
	  var2_vartype = strstr(list_curr(this), vPairs[i].c2);
	  var2_vartype++;
	  
/*
 * If the variable types don't match:
 * - keep traversing, looking for the second component of the same variable type
 */

	  while ( strcmp(var1_vartype, var2_vartype) ) {
	       status = list_traverse(this, vPairs[i].c2, JC_ListTraverse_strstr, (LIST_CURR | LIST_FORW | LIST_ALTR));
	       if ( status == LIST_EXTENT )
		    break;
	       strcpy(var2, list_curr(this));
	       var2_vartype = strstr(list_curr(this), vPairs[i].c2);
	       var2_vartype++;
	       list_mvnext(this);
	       if ( list_curr(this) == list_rear(this) )
		    break;
	  }
	  
/*
 * If you find a match:
 *
 * - create a new variable
 * - make sure it doesn't already exist
 * - if the two components are on the same grid:
 *   - add this variable to the list.
 */

	  if ( !strcmp(var1_vartype, var2_vartype) ) {
	       
	       sprintf(new_var, "%s,%s", var1, var2);

	       status = list_traverse(this, new_var, JC_ListTraverse_FoundMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));
	       if ( status != LIST_EXTENT )
		    continue;
	       
	       ferret_query(QUERY_GRID, sBuffer, dset, var1, nullStr, nullStr, nullStr);
	       sBuffer->textP = &sBuffer->text[0];
	       sBuffer->textP = CollectToReturn(sBuffer->textP, var1_grid);
	       ferret_query(QUERY_GRID, sBuffer, dset, var2, nullStr, nullStr, nullStr);
	       sBuffer->textP = &sBuffer->text[0];
	       sBuffer->textP = CollectToReturn(sBuffer->textP, var2_grid);

	       if ( !strcmp(var1_grid, var2_grid) ) {
		    list_mvrear(this);
		    list_insert_after(this, new_var, sizeof(new_var));
	       }
	       
	  }
	  
     }
     
}


int JC_ListTraverse_Sort( char *data, char *curr )
{
     if ( strcmp(data, curr) > 0 )
	  return TRUE;
     else
	  return FALSE;
}


int JC_ListTraverse_fprintf( char *data, char *curr )
{
     FILE *File_ptr=(FILE *)data;

     fprintf(File_ptr, "%s\n", curr);
     return TRUE;
}


int JC_ListTraverse_Dsetfprintf( char *data, char *curr )
{
     FILE *File_ptr=(FILE *)data;
     JC_DatasetElement *DE_ptr=(JC_DatasetElement *)curr;
     
     (File_ptr, "========== JC_DatasetElement contents: ==========\n");

     fprintf(File_ptr, "\n%s\n", DE_ptr->name );

     fprintf(File_ptr, "\nVariables:\n");
     list_traverse(DE_ptr->varList, File_ptr, JC_ListTraverse_fprintf, (LIST_FRNT | LIST_FORW | LIST_SAVE));
     
     fprintf(File_ptr, "\nDefined Variables:\n");
     list_traverse(DE_ptr->dvarList, File_ptr, JC_ListTraverse_Dvarfprintf, (LIST_FRNT | LIST_FORW | LIST_SAVE));
     
     fprintf(File_ptr, "\nCloned Variables:\n");
     list_traverse(DE_ptr->cvarList, File_ptr, JC_ListTraverse_Cvarfprintf, (LIST_FRNT | LIST_FORW | LIST_SAVE));
     
     fprintf(File_ptr, "========================================\n");

     return TRUE;
}


int JC_ListTraverse_Dvarfprintf( char *data, char *curr )
{
     FILE *File_ptr=(FILE *)data;
     JC_DefinedVariable *DV_ptr=(JC_DefinedVariable *)curr;

     fprintf(File_ptr, "%s\n", DV_ptr->name);
     return TRUE;
}


int JC_ListTraverse_Cvarfprintf( char *data, char *curr )
{
     FILE *File_ptr=(FILE *)data;
     JC_Object *Obj_ptr=(JC_Object *)curr;

     fprintf(File_ptr, "%s\n", Obj_ptr->name);
     return TRUE;
}


int JC_ListTraverse_strstr( char *data, char *curr )
{

     if ( strstr(curr, data) ) {
	  return FALSE;
     } else
	  return TRUE;
}


int JC_ListTraverse_FoundMatch( char *data, char *curr )
{
     if ( !strcmp(data, curr) )
	  return FALSE;
     else
	  return TRUE;
}


int JC_ListTraverse_FoundDvarMatch( char *data, char *curr )
{
     JC_DefinedVariable *DV_ptr=(JC_DefinedVariable *)curr;

     if ( !strcmp(data, DV_ptr->name) )
	  return FALSE;
     else
	  return TRUE;
}


int JC_ListTraverse_FoundCvarMatch( char *data, char *curr )
{
     JC_Object *Obj_ptr=(JC_Object *)curr;

     if ( !strcmp(data, Obj_ptr->name) )
	  return FALSE;
     else
	  return TRUE;
}


int JC_ListTraverse_FoundDsetMatch( char *data, char *curr )
{
     JC_DatasetElement *DE_ptr=(JC_DatasetElement *)curr;

     if ( !strcmp(data, DE_ptr->name) )
	  return FALSE;
     else
	  return TRUE;
}


int JC_ListTraverse_FreeDataset( char *data, char *curr )
{
     JC_DatasetElement *DE_ptr=(JC_DatasetElement *)curr;

     list_free(DE_ptr->varList, LIST_DEALLOC);
     list_free(DE_ptr->dvarList, LIST_DEALLOC);
     list_free(DE_ptr->cvarList, LIST_DEALLOC);

     return TRUE;
}

