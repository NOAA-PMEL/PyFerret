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
 * JC_CommandGen.c
 *
 * Jonathan Callahan
 * Dec 11th 1995
 *
 * This file contains functions which create commands to be sent to Ferret.
 *
 * Each type of command is treated as an object.  Only one function needs to
 * be invoked externally for each separate type of command.
 *
 * eg. To create a plot command the PlotButton_CB() routine would create
 *     a string, grab the global Object and PlotOptions and pass pointers to
 *     all three to JC_PlotCommand_Create().  This command modifies the string 
 *     it was passed according to the O_ptr and PO_ptr contents.  The PlotButton_CB()
 *     function is the responsible for shipping this command off to Ferret.
 *
 */

/* .................... Includes .................... */

#include <stdio.h>

#include "ferret_structures.h"
#include "JC_Utility.h"

#define ALL_AXES 4

/* .................... External Declarations .................... */

extern JC_StateFlags GLOBAL_StateFlags;

JC_Object *JC_Clone_ReturnPointer( char *name, char *dset );

typedef struct { 
        int autoOrCustomName; 
        char customName[256]; 
        enum {text, binary, netcdf} fileFormat; 
        int heading, fixedTimeSpan, customOrder, append, fortranFormat; 
        char custOrder[5], ftnFormat[128]; 
} JC_CSO; 


/* .................... Internal Declarations .................... */

static void JC_Command_AddRegionQualifier( char *command, JC_Region *R_ptr, JC_Variable *V_ptr );
static void JC_Command_AddVariableExpression( char *command, JC_Object *O_ptr );

static void JC_LetCommand_AddIndividualSpan( char *var_expression, JC_Span *S_ptr );
void JC_LetCommand_CreateClonedVarExpression(char *var_expression, int var_ID , JC_DefinedVariable *DV_ptr);
static void JC_LetCommand_CreateVarExpression(char *var_expression, int var_ID , JC_DefinedVariable *DV_ptr);

static void JC_ListCommand_AddFileQualifier( char *command, JC_CSO *CSO_ptr );

static void JC_PlotCommand_AddCommand( char *command, JC_PlotOptions *PO_ptr );
static void JC_PlotCommand_AddAxisRange( char *command, JC_Span *S_ptr, JC_Axis *A_ptr );
static void JC_PlotCommand_AddOptionQualifier( char *command, JC_PlotOptions *PO_ptr );
static void JC_PlotCommand_AddOneDOptions( char *command, JC_OneDOptions *ODO_ptr, int plot_type );
static void JC_PlotCommand_AddTwoDOptions( char *command, JC_TwoDOptions *TDO_ptr );
static void JC_PlotCommand_AddVectorOptions( char *command, JC_VectorOptions *VO_ptr );
static void JC_PlotCommand_AddTransformationQualifier( char *command, JC_Region *R_ptr );
static void JC_PlotCommand_AddIndividualTransformation( char *command, JC_Region *R_ptr, int xyzt );
static void JC_PlotCommand_AddRegriddingQualifier( char *command, JC_Object *O_ptr );
static void JC_PlotCommand_AddIndividualRegridding( char *command, JC_Object *O_ptr, int xyzt );


/* .................... Command methods .................... */

static void JC_Command_AddRegionQualifier( char *command, JC_Region *R_ptr, JC_Variable *V_ptr )
{
     int xyzt=0;
	
     for (xyzt=0; xyzt<4; xyzt++) {
	  if ( R_ptr->span[xyzt].ss[LO] != IRRELEVANT_AXIS ) {
	       strcat(command, "/");
	       JC_PlotCommand_AddAxisRange( command, &(R_ptr->span[xyzt]), &(V_ptr->axis[xyzt]) );
	  }
     }
}


static void JC_Command_AddVariableExpression(char *command, JC_Object *O_ptr)
{
     JC_Object local_obj = *O_ptr;

     char tempText[MAX_NAME_LENGTH];

     if ( strchr(O_ptr->variable.name, ',' )) {

/* 
 * For some reason this recursive function call doesn't work.
 * 
 * I'll leave it in the other form for now.
 *
	  strcpy(tempText, O_ptr->variable.name);
	  strcpy(local_obj.variable.name, strtok(tempText, ","));
	  JC_Command_AddVariableExpression(command, &local_obj);
	  strcat(command, ",");
	  strcpy(local_obj.variable.name, strtok(NULL, ","));
	  JC_Command_AddVariableExpression(command, &local_obj);
*/

/*
 * - First copy "O_ptr->variable.name" to "tempText" because strtok(s1,s2) alters "s1".
 */

	  strcpy(tempText, O_ptr->variable.name);
	  strcat(command, strtok(tempText, ","));
	  strcat(command, "[d=");
	  strcat(command, O_ptr->variable.dset);
	  JC_PlotCommand_AddTransformationQualifier(command, &(O_ptr->region));
	  JC_PlotCommand_AddRegriddingQualifier(command, O_ptr); 
	  strcat(command, "],");
	  strcat(command, strtok(NULL, ","));
	  strcat(command, "[d=");
	  strcat(command, O_ptr->variable.dset);
	  JC_PlotCommand_AddTransformationQualifier(command, &(O_ptr->region));
	  JC_PlotCommand_AddRegriddingQualifier(command, O_ptr); 
	  strcat(command, "]");

     } else /* non-vector variable */ {

	  strcat(command, O_ptr->variable.name);
	  strcat(command, "[d=");
	  strcat(command, O_ptr->variable.dset);
	  JC_PlotCommand_AddTransformationQualifier(command, &(O_ptr->region));
	  JC_PlotCommand_AddRegriddingQualifier(command, O_ptr); 
	  strcat(command, "]");
     }
}


void JC_SolidLandCommand_Create(char *command, JC_Object *O_ptr, int resolution)
{
  JC_Region *R_ptr=&(O_ptr->region);

  sprintf(command, "GO fland %d gray overlay solid x=%.2f:%.2f y=%.2f:%.2f", resolution,
	  R_ptr->span[X_AXIS].ww[LO], R_ptr->span[X_AXIS].ww[HI],
	  R_ptr->span[Y_AXIS].ww[LO], R_ptr->span[Y_AXIS].ww[HI]);

  strcat(command, " ");
}


static void JC_LetCommand_AddIndividualSpan( char *var_expression, JC_Span *S_ptr )
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     char tempText[MAX_NAME_LENGTH]="";
     int xyzt=S_ptr->xyzt;

     if ( S_ptr->by_index_in_GUI ) {

	  if ( S_ptr->needs_lo_hi_displayed_in_GUI ) {
	       sprintf(tempText, "%d:", S_ptr->ss[LO]+1); /* Ferret needs indices one greater than C code */
	       strcat(var_expression, tempText);
	       sprintf(tempText, "%d", S_ptr->ss[HI]+1); /* Ferret needs indices one greater than C code */
	       strcat(var_expression, tempText);
	  } else /* needs point value */ {
	       sprintf(tempText, "%d", S_ptr->ss[PT]+1); /* Ferret needs indices one greater than C code */
	       strcat(var_expression, tempText);
	  }

     } else /* by "ww" coordinates */ {

	  if ( S_ptr->needs_lo_hi_displayed_in_GUI ) {
	       JC_String_CreateFancyFerretLabel(tempText, S_ptr->ww[LO], S_ptr, SF_ptr);
	       strcat(var_expression, tempText);
	       strcat(var_expression, ":");
	       JC_String_CreateFancyFerretLabel(tempText, S_ptr->ww[HI], S_ptr, SF_ptr);
	       strcat(var_expression, tempText);
	  } else /* needs point value */ {
	       JC_String_CreateFancyFerretLabel(tempText, S_ptr->ww[PT], S_ptr, SF_ptr);
	       strcat(var_expression, tempText);
	  }

     }

}


void JC_LetCommand_Create(char *command, JC_DefinedVariable *DV_ptr)
{
     char var_expression[4][MAX_NAME_LENGTH];
     char local_multiplier[4][MAX_NAME_LENGTH];
     char local_title[MAX_NAME_LENGTH]="", local_units[MAX_NAME_LENGTH]="";
     char asterisk[4][2]={ "*", "*", "*", "*" };
     int i=0;
     
     if ( strlen(DV_ptr->title) > 0 )
	  sprintf(local_title, "/TITLE=\"%s\"", DV_ptr->title);

     if ( strlen(DV_ptr->units) > 0 )
	  sprintf(local_units, "/UNIT=\"%s\"", DV_ptr->units);


     for ( i=0; i<DV_ptr->number_of_vars; i++ ) {
	  if ( DV_ptr->clone_ptr[i] == NULL )
	       JC_LetCommand_CreateVarExpression(var_expression[i], i, DV_ptr);
	  else
	       JC_LetCommand_CreateClonedVarExpression(var_expression[i], i, DV_ptr);
     }
     
     for ( i=0; i<4; i++ ) {
	  if ( !strcmp(DV_ptr->multiplier[i], "1.0") ) {
	       strcpy(local_multiplier[i], "");
	       strcpy(asterisk[i], "");
	  } else {
	       strcpy(local_multiplier[i], DV_ptr->multiplier[i]);
	       strcpy(asterisk[i], "*");
	  }
     }
	  
     switch (DV_ptr->type) {

/*
 * I've decided to leave parentheses off the individual components in the rest.
 */
	case FUNC_FUNCTION1:
	  sprintf(command, "LET/D=%s%s%s %s = %s(%s%s%s)",
DV_ptr->assigned_dset, local_title, local_units, DV_ptr->name,
DV_ptr->function, local_multiplier[0], asterisk[0], var_expression[0]);
	break;

	case FUNC_FUNCTION2:
	  sprintf(command, "LET/D=%s%s%s %s = %s(%s%s%s, %s%s%s)",
DV_ptr->assigned_dset, local_title, local_units, DV_ptr->name,
DV_ptr->function, local_multiplier[0], asterisk[0], var_expression[0],
local_multiplier[1], asterisk[1], var_expression[1]);
	  break;

/*
 * I've decided to leave parentheses off the individual components in the rest.
 */
	case FUNC_LINEAR_COMBINATION:
	  sprintf(command, "LET/D=%s%s%s %s = %s%s%s %s %s%s%s",
DV_ptr->assigned_dset, local_title, local_units, DV_ptr->name,
local_multiplier[0], asterisk[0], var_expression[0],
DV_ptr->operator, local_multiplier[1], asterisk[1], var_expression[1]);
	  break;

	case FUNC_PLUS_CONSTANT:
	case FUNC_EXPONENT:
	  if ( !strcmp(local_multiplier[1],"") )
	       strcpy(local_multiplier[1], "1.0");

	  sprintf(command, "LET/D=%s%s%s %s = %s%s%s %s %s",
		  DV_ptr->assigned_dset, local_title, local_units, DV_ptr->name,
		  local_multiplier[0], asterisk[0], var_expression[0],
		  DV_ptr->operator, local_multiplier[1]);
	  break;

	default:
	  fprintf(stderr, "ERROR in JC_CommandGen.c: JC_LetCommand_Create(): DV_ptr->type = %d\n", DV_ptr->type);
	  strcpy(command, "");
	  return;
	  break;

     }

}


void JC_LetCommand_CreateClonedVarExpression(char *var_expression, int ID, JC_DefinedVariable *DV_ptr)
{
     int xyzt=0;
     JC_Object *O_ptr=NULL;
     char *ss_name[4] = {"I=","J=","K=","L="};
     char *ww_name[4] = {"X=","Y=","Z=","T="};
     Boolean we_need_a_comma=FALSE;

/*
 * - Add the component variable name.
 *
 * - Get the cloned variable information.
 *   
 */
     strcpy(var_expression, DV_ptr->var[ID]);
     O_ptr = DV_ptr->clone_ptr[ID];

     if ( O_ptr == NULL ) {
	  fprintf(stderr, "ERROR in CommandGen.c: CreateClonedVarExpression(): O_ptr == NULL\n");
	  return;
     }

/*
 * - If any information is fixed, or any transform exists: begin the qualifying brackets.
 *
 * - For each axis:
 *      If the span is fixed: add it, then add the transform (if it exists).
 *      Otherwise just add the transform (if it exists).
 *
 * - If the regridding is fixed: add it.
 *
 * - Close the qualifying brackets.
 */

     if ( O_ptr->fixed_axis[X_AXIS] || O_ptr->fixed_axis[Y_AXIS] || 
	 O_ptr->fixed_axis[Z_AXIS] || O_ptr->fixed_axis[T_AXIS] ||
	 O_ptr->region.transform[X_AXIS].exists || O_ptr->region.transform[Y_AXIS].exists ||
	 O_ptr->region.transform[Z_AXIS].exists || O_ptr->region.transform[T_AXIS].exists ||
	 O_ptr->fixed_regridding ) {
	  
	  strcat(var_expression, "[");
	  
	  if ( strcmp(DV_ptr->assigned_dset, DV_ptr->dset[ID]) ) {
	       strcat(var_expression, DV_ptr->dset[ID]);
	       we_need_a_comma = TRUE;
	  }

	  for ( xyzt=0; xyzt<4; xyzt++ ) {

	       if ( O_ptr->region.span[xyzt].ss[LO] == IRRELEVANT_AXIS )
		    continue;
	
	       if ( O_ptr->fixed_axis[xyzt] ) {

		    if ( we_need_a_comma )
			 strcat(var_expression, ",");

		    if ( O_ptr->region.span[xyzt].by_index_in_GUI )
			 strcat(var_expression, ss_name[xyzt]);
		    else	/* use "ww" coordinates */
			 strcat(var_expression, ww_name[xyzt]);

		    JC_LetCommand_AddIndividualSpan(var_expression, &(O_ptr->region.span[xyzt]));

		    if ( O_ptr->region.transform[xyzt].exists )
			 JC_PlotCommand_AddIndividualTransformation(var_expression, &(O_ptr->region), xyzt);

		    we_need_a_comma = TRUE;

	       } else /* the span is not fixed */ {

		    if ( O_ptr->region.transform[xyzt].exists ) {

			 if ( we_need_a_comma )
			      strcat(var_expression, ",");
			 
			 if ( O_ptr->region.span[xyzt].by_index_in_GUI )
			      strcat(var_expression, ss_name[xyzt]);
			 else	/* use "ww" coordinates */
			      strcat(var_expression, ww_name[xyzt]);
			 
			 JC_PlotCommand_AddIndividualTransformation(var_expression, &(O_ptr->region), xyzt);

			 we_need_a_comma = TRUE;

		    }
		    
	       }
	  
	       if ( O_ptr->fixed_regridding && O_ptr->regridding.type == NON_UNIFORM ) {
		    if ( strcmp(O_ptr->regridding.var[xyzt], "") ) {
			 if ( we_need_a_comma )
			      strcat(var_expression, ",");
			 JC_PlotCommand_AddIndividualRegridding(var_expression, O_ptr, xyzt); 
		    }
	       }
	  }

	  if ( O_ptr->fixed_regridding && O_ptr->regridding.type == UNIFORM ) {
	       if ( strcmp(O_ptr->regridding.var[ALL_AXES], "") ) {
		    if ( we_need_a_comma )
			 strcat(var_expression, ",");
		    JC_PlotCommand_AddIndividualRegridding(var_expression, O_ptr, ALL_AXES); 
	       }
	  }

	  strcat(var_expression, "]");

     }
     
}


static void JC_LetCommand_CreateVarExpression(char *var_expression, int ID, JC_DefinedVariable *DV_ptr)
{
/*
 * - Add the component variable name.
 *
 * - If the component variable is not in the same dataset as the assigned dataset of the defined variable:
 *      add "[D=dset_name]"
 */

     strcpy(var_expression, DV_ptr->var[ID]);
     
     if ( strcmp(DV_ptr->assigned_dset, DV_ptr->dset[ID]) ) {
	  strcat(var_expression, "[D=");
	  strcat(var_expression, DV_ptr->dset[ID]);
	  strcat(var_expression, "]");
     }
     
}


static void JC_ListCommand_AddFileQualifier( char *command, JC_CSO *CSO_ptr )
{
     if ( !CSO_ptr->autoOrCustomName ) {
	  strcat(command, "/FILE=");
	  strcat(command, CSO_ptr->customName);
     } else
	  strcat(command, "/FILE");

     switch (CSO_ptr->fileFormat) {
	case text:
	  if (CSO_ptr->fortranFormat) {
	       strcat(command, "/FORMAT=");
	       strcat(command, CSO_ptr->ftnFormat);
	  }
	  break;
	case binary:
	  strcat(command, "/FORMAT=UNFORMATTED");
	  break;
	case netcdf:
	  strcat(command, "/FORMAT=CDF");
	  break;
     }

     if ( CSO_ptr->append )
	  strcat(command, "/APPEND");

     if ( CSO_ptr->heading )
	  strcat(command, "/HEAD");
     else if (CSO_ptr->fileFormat != binary && CSO_ptr->fileFormat != netcdf)
	  strcat(command, "/NOHEAD");

     if ( CSO_ptr->fixedTimeSpan )
	  strcat(command, "/RIGID");

     if ( CSO_ptr->customOrder ) {
	  strcat(command, "/ORDER=");
	  strcat(command, CSO_ptr->custOrder);
     }
     
}


void JC_ListCommand_Create( char *command, JC_Object *O_ptr )
{
     strcpy(command, "LIST");
     JC_Command_AddRegionQualifier(command, &(O_ptr->region), &(O_ptr->variable));
     strcat(command, " ");
     JC_Command_AddVariableExpression(command, O_ptr);
}


void JC_ListFileCommand_Create( char *command, JC_Object *O_ptr, JC_CSO *CSO_ptr )
{
     strcpy(command, "LIST");
     JC_Command_AddRegionQualifier(command, &(O_ptr->region), &(O_ptr->variable));
     JC_ListCommand_AddFileQualifier(command, CSO_ptr);
     strcat(command, " ");
     JC_Command_AddVariableExpression(command, O_ptr);
}


void JC_PlotCommand_Create( char *command, JC_Object *O_ptr, JC_PlotOptions *PO_ptr )
{

/*
 * - Clear the plot command string.
 * - Add command.
 * - Add region qualifier.
 * - Add plot option qualifiers.
 * - Add variable expression.
 */

     strcpy(command, "");
     JC_PlotCommand_AddCommand( command, PO_ptr );
     JC_Command_AddRegionQualifier( command, &(O_ptr->region), &(O_ptr->variable) );
     JC_PlotCommand_AddOptionQualifier( command, PO_ptr );
     strcat(command, " ");
     JC_Command_AddVariableExpression( command, O_ptr );

}


static void JC_PlotCommand_AddCommand( char *command, JC_PlotOptions *PO_ptr )
{

     switch ( PO_ptr->plot_type ) {
	case PLOT_LINE:
	  strcpy(command, "PLOT");
	  break;
	case PLOT_SCATTER:
	  strcpy(command, "PLOT/VS");
	  break;
	case PLOT_SHADE:
	  strcpy(command, "SHADE");
	  break;
	case PLOT_CONTOUR:
	  strcpy(command, "CONTOUR");
	  break;
	case PLOT_FILL:
	  strcpy(command, "FILL");
	  break;
	case PLOT_VECTOR:
	  strcpy(command, "VECTOR");
	  break;
	default:
	  fprintf(stderr, "ERROR in JC_CommandGen.c: JC_PlotCommand_AddCommand: \
plot_type = %d.\n", PO_ptr->plot_type);
	  break;
     }
	
     if ( PO_ptr->overlay )
	  strcat(command, "/OV");

}


static void JC_PlotCommand_AddAxisRange( char *command, JC_Span *S_ptr, JC_Axis *A_ptr )
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     char tempText[MAX_NAME_LENGTH]="";
     char *ss_name[4] = {"I=","J=","K=","L="};
     char *ww_name[4] = {"X=","Y=","Z=","T="};
     char *limit_name[4] = {"XLIMITS=", "YLIMITS=", "ZLIMITS=", "TLIMITS="};
     float span[2]={0,0}, limit[2]={0,0};
     int xyzt=S_ptr->xyzt;
     Boolean include_limits=FALSE;

     if ( S_ptr->by_index_in_GUI ) {

	  strcat(command, ss_name[xyzt]);

	  if ( S_ptr->needs_lo_hi_displayed_in_GUI ) {
	       sprintf(tempText, "%d:", S_ptr->ss[LO]+1); /* Ferret needs indices one greater than C code */
	       strcat(command, tempText);
	       sprintf(tempText, "%d", S_ptr->ss[HI]+1); /* Ferret needs indices one greater than C code */
	       strcat(command, tempText);
	  } else		/* needs point value */ {
	       sprintf(tempText, "%d", S_ptr->ss[PT]+1); /* Ferret needs indices one greater than C code */

	       strcat(command, tempText);
	  }

     } else /* by "ww" coordinates */ {

	  strcat(command, ww_name[xyzt]);

	  if ( S_ptr->needs_lo_hi_displayed_in_GUI ) {

/*
 * - If the desired region (S_ptr->ww[LO/HI]) is outside the valid region (A_ptr->ww[LO/HI])
 *      X=lo:hi will have the valid part
 *      XLIMITS=lo:hi will have the user specified part
 */

	       if ( S_ptr->ww[LO] < A_ptr->ww[LO] ) {
		    include_limits = TRUE;
		    span[LO] = A_ptr->ww[LO];
		    limit[LO] = S_ptr->ww[LO];
	       } else {
		    span[LO] = S_ptr->ww[LO];
		    limit[LO] = S_ptr->ww[LO];
	       }
	       
	       if ( S_ptr->ww[HI] > A_ptr->ww[HI] ) {
		    include_limits = TRUE;
		    span[HI] = A_ptr->ww[HI];
		    limit[HI] = S_ptr->ww[HI];
	       } else {
		    span[HI] = S_ptr->ww[HI];
		    limit[HI] = S_ptr->ww[HI];
	       }

	       JC_String_CreateFancyFerretLabel(tempText, span[LO], S_ptr, SF_ptr);
	       strcat(command, tempText);
	       strcat(command, ":");
	       JC_String_CreateFancyFerretLabel(tempText, span[HI], S_ptr, SF_ptr);
	       strcat(command, tempText);

	       if ( include_limits && (xyzt == X_AXIS || xyzt == Y_AXIS) ) {
		    strcat(command, "/");
		    strcat(command, limit_name[xyzt]);
		    JC_String_CreateFancyFerretLabel(tempText, limit[LO], S_ptr, SF_ptr);
		    strcat(command, tempText);
		    strcat(command, ":");
		    JC_String_CreateFancyFerretLabel(tempText, limit[HI], S_ptr, SF_ptr);
		    strcat(command, tempText);
	       }

	  } else /* needs point value */ {
	       JC_String_CreateFancyFerretLabel(tempText, S_ptr->ww[PT], S_ptr, SF_ptr);
	       strcat(command, tempText);
	  }

     }

}


static void JC_PlotCommand_AddOptionQualifier(char *command, JC_PlotOptions *PO_ptr)
{

     switch ( PO_ptr->plot_type ) {

	case PLOT_LINE:
	case PLOT_SCATTER:
	  JC_PlotCommand_AddOneDOptions(command, &(PO_ptr->oneD_options), PO_ptr->plot_type);
	  break;
	case PLOT_SHADE:
	case PLOT_CONTOUR:
	case PLOT_FILL:
	  JC_PlotCommand_AddTwoDOptions(command, &(PO_ptr->twoD_options));
	  break;
	case PLOT_VECTOR:
	  JC_PlotCommand_AddVectorOptions(command, &(PO_ptr->vector_options));
	  break;
     }

     if ( PO_ptr->transpose )
	  strcat(command, "/TRANSPOSE");

     if ( PO_ptr->nolabels )
	  strcat(command, "/NOLABELS");
}


static void JC_PlotCommand_AddOneDOptions( char *command, JC_OneDOptions *ODO_ptr, int plot_type )
{
     char tempText[MAX_NAME_LENGTH]="";

     if ( ODO_ptr->automatic ) {

	  if ( plot_type == PLOT_SCATTER)
	       strcat(command, "/SYMBOL");

     } else /* the user has control */ {

	  if ( ODO_ptr->style ) {
	       sprintf(tempText, "/LINE=%d", ODO_ptr->style);
	       strcat(command, tempText);
	  }
 
	  if ( ODO_ptr->symbol ) {
	       sprintf(tempText, "/SYMBOL=%d", ODO_ptr->symbol);
	       strcat(command, tempText);
	  }
     }
}


static void JC_PlotCommand_AddTwoDOptions( char *command, JC_TwoDOptions *TDO_ptr )
{
     char tempText[MAX_NAME_LENGTH]="";

     if ( TDO_ptr->level_type == LAST_LEVELS )
	  strcat(command, "/LEVELS");

     else if ( TDO_ptr->level_type == NEW_LEVELS) {
	  if ( TDO_ptr->levels[LO] == UNSET_VALUE || TDO_ptr->levels[HI]== UNSET_VALUE || TDO_ptr->levels[DELTA] == UNSET_VALUE)
	       strcat(command, "/LEVELS");
	  else {
	       sprintf(tempText, "/LEVELS=(%.2f,%.2f,%.2f", TDO_ptr->levels[LO], TDO_ptr->levels[HI], TDO_ptr->levels[DELTA]);
	       strcat(command, tempText);
	       if ( TDO_ptr->levels[NDIG] != UNSET_VALUE ) {
		    sprintf(tempText, ",%d", TDO_ptr->levels[NDIG]);
		    strcat(command, tempText);
	       }
	       strcat(command, ")");
	  }
     }

     if ( TDO_ptr->no_key )
	  strcat(command, "/NOKEY");

     if ( TDO_ptr->line ) {
	  strcat(command, "/LINE");
	  if ( TDO_ptr->key )
	       strcat(command, "/KEY");
     }
}


static void JC_PlotCommand_AddVectorOptions( char *command, JC_VectorOptions *VO_ptr )
{
     char tempText[MAX_NAME_LENGTH]="";

     if ( VO_ptr->aspect )
	  strcat(command, "/ASPECT");

     if ( VO_ptr->length_type == LAST_LENGTH )
	  strcat(command, "/LENGTH");
     else if ( VO_ptr->length_type == NEW_LENGTH && VO_ptr->length != UNSET_VALUE ) {
	  sprintf(tempText, "/LENGTH=%.2f", VO_ptr->length);
	  strcat(command, tempText);
     }

     if ( VO_ptr->xskip > 0 ) {
	  sprintf(tempText, "/XSKIP=%d", VO_ptr->xskip);
	  strcat(command, tempText);
     }

     if ( VO_ptr->yskip > 0 ) {
	  sprintf(tempText, "/YSKIP=%d", VO_ptr->yskip);
	  strcat(command, tempText);
     }
}


static void JC_PlotCommand_AddTransformationQualifier(char *command, JC_Region *R_ptr)
{
     int xyzt=0;
     char *ss_name[] = {"I=","J=","K=","L="};
     char *ww_name[] = {"X=","Y=","Z=","T="};
     
     for (xyzt=0; xyzt<4; xyzt++) {

	  if ( R_ptr->span[xyzt].ss[LO] == IRRELEVANT_AXIS )
	       continue;

	  if ( R_ptr->transform[xyzt].exists ) {

	       strcat(command, ",");

	       if ( R_ptr->span[xyzt].by_index_in_GUI )
		    strcat(command, ss_name[xyzt]);
	       else /* use "ww" coordinates */
		    strcat(command, ww_name[xyzt]);

	       JC_PlotCommand_AddIndividualTransformation(command, R_ptr, xyzt);

	  }

     }
     
}


static void JC_PlotCommand_AddIndividualTransformation(char *command, JC_Region *R_ptr, int xyzt )
{
     char tempText[MAX_NAME_LENGTH]="";

/*
 * - If this axis is relevant and a transform exists
 *      add the index/ww name
 *      add "@" and the transform name
 *      If the transformation accepts an argument
 *         fill in the argument
 */

     if ( R_ptr->span[xyzt].ss[LO] == IRRELEVANT_AXIS )
	  return;
     
     if ( R_ptr->transform[xyzt].exists ) {

	  strcat(command, "@");
	  strcat(command, R_ptr->transform[xyzt].name);
	  
	  if ( R_ptr->transform[xyzt].accepts_an_argument ) {

	       switch (R_ptr->transform[xyzt].code) {
		  case TRANS_SHN:
		  case TRANS_SBN:
		  case TRANS_SPZ:
		  case TRANS_SWL:
		  case TRANS_SBX:
		  case TRANS_FAV:
		    sprintf(tempText, ":%d", (int)R_ptr->transform[xyzt].arg);
		    break;
		  default:
		    sprintf(tempText, ":%.2f", R_ptr->transform[xyzt].arg);
		    break;
	       }

	       strcat(command, tempText);
	  }
	  
     }
     
}


static void JC_PlotCommand_AddRegriddingQualifier(char *command, JC_Object *O_ptr)
{
     JC_Regridding *RG_ptr=&(O_ptr->regridding);
     int xyzt=0;
     
     if ( !O_ptr->fixed_regridding )
	  return;
     
     if ( RG_ptr->type == UNIFORM ) {

	  if ( strcmp(RG_ptr->var[ALL_AXES], "") ) {
	       strcat(command, ",");
	       JC_PlotCommand_AddIndividualRegridding(command, O_ptr, ALL_AXES);
	  }

     } else if ( RG_ptr->type == NON_UNIFORM ) {

	  for ( xyzt=0; xyzt<4; xyzt++ ) {
	       if ( strcmp(RG_ptr->var[xyzt], "") ) {
		    strcat(command, ",");
		    JC_PlotCommand_AddIndividualRegridding(command, O_ptr, xyzt);
	       }
	  }
	  
     } else
	  fprintf(stderr, "ERROR in JC_CommandGen.c: JC_PlotCommand_AddRegriddingQualifier: RG_ptr->type = %d\n",
RG_ptr->type);

}


static void JC_PlotCommand_AddIndividualRegridding( char *command, JC_Object *O_ptr, int xyzt )
{
     JC_Variable *V_ptr=&(O_ptr->variable);
     JC_Regridding *RG_ptr=&(O_ptr->regridding);
     char *regrid_expression[5] = { "GX=", "GY=", "GZ=", "GT=", "G=" };

     strcat(command, regrid_expression[xyzt]);
     strcat(command, RG_ptr->var[xyzt]);
     if ( strcmp(RG_ptr->dset[xyzt], V_ptr->dset) ) {
	  strcat(command, "[D=");
	  strcat(command, RG_ptr->dset[xyzt]);
	  strcat(command, "]");
     }
     if ( strcmp(RG_ptr->rg_transform[xyzt], "LIN") ) {
	  strcat(command, "@");
	  strcat(command, RG_ptr->rg_transform[xyzt]);
     }
}

