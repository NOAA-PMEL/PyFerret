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



/* Main.c
 *
 * John Osborne
 * Jonathan Callahan (after Oct 95)
 *
 */
#define YES 1
#define NO 2
#define charset XmSTRING_DEFAULT_CHARSET


/* .................... Includes .................... */

#include "JC_OOP.c"
#include "JC_Map.c"

/* .................... Defines .................... */

enum { SAME_VARIABLE, NEW_VARIABLE, FIRST_EVER_VARIABLE }
VARIABLE_type;


/* .................... Internal Declarations .................... */

static void JC_DataFrame_Initialize( void );


void MaintainMainWdBtns( void )
{

}


void JC_GeometryInterfaceLine_NewSpan( JC_Span *S_ptr )
{

     /*
      *   if we need LO and HI
      *      update and display the LO textField and scrollBar
      *      update and display the HI textField and scrollBar
      *      update and hide the PT textField and scrollBar
      *   else
      *      update and hide the LO textField and scrollBar
      *      update and hide the HI textField and scrollBar
      *      update and display the PT textField and scrollBar
      */

	  if ( S_ptr->needs_lo_hi_displayed_in_GUI ) {

	       JC_textField_NewSpan( LO, TRUE, S_ptr );
	       JC_scrollBar_NewSpan( LO, TRUE, S_ptr );
	       JC_textField_NewSpan( HI, TRUE, S_ptr );
	       JC_scrollBar_NewSpan( HI, TRUE, S_ptr );
	       JC_textField_NewSpan( PT, FALSE, S_ptr );
	       JC_scrollBar_NewSpan( PT, FALSE, S_ptr );
	       
	  } else /* span  needs pt displayed */ {

	       JC_textField_NewSpan( LO, FALSE, S_ptr );
	       JC_scrollBar_NewSpan( LO, FALSE, S_ptr );
	       JC_textField_NewSpan( HI, FALSE, S_ptr );
	       JC_scrollBar_NewSpan( HI, FALSE, S_ptr );
	       JC_textField_NewSpan( PT, TRUE, S_ptr );
	       JC_scrollBar_NewSpan( PT, TRUE, S_ptr );
	       
	  }

}


void JC_GeometryMenu_CB( XtPointer UxClientData )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_Variable *V_ptr=&GLOBAL_Variable;

     int xyzt=0, geometry=0;
     Boolean it_is_a_vector=FALSE;

/*
 * - Get the geometry from the widget.
 * - Create a new region which reflects the new geometry.
 * - Update the geometry interface to reflect the new region.
 * - Update the transformations appropriately.
 * - Update the available plot types appropriately.
 */

     if ( strchr(V_ptr->name, ',' ))
	  it_is_a_vector = TRUE;
     
     sscanf( (char *)UxClientData, "%d", &geometry);
     JC_Region_NewGeometry( R_ptr, geometry );
     JC_Transforms_NewRegion( R_ptr );
     for (xyzt=0; xyzt<4; xyzt++)
	  JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[xyzt]) );

     PO_ptr->plot_type = JC_PlotFrame_MaintainRadios(R_ptr->geometry, it_is_a_vector, SF_ptr);
     JC_PlotFrame_MaintainButtons(R_ptr->geometry, PO_ptr->plot_type, SF_ptr);
     JC_II_MainMenu_Maintain(SF_ptr);
     JC_Map_NewRegion(R_ptr);
     
     if ( SF_ptr->a_clone_is_selected ) {

       for (xyzt=0; xyzt<4; xyzt++) {
	 if ( R_ptr->span[xyzt].ss[LO] != IRRELEVANT_AXIS) {
	   if ( !R_ptr->span[xyzt].needs_lo_hi_displayed_in_GUI || R_ptr->span[xyzt].is_compressed_in_GUI )
	     XmToggleButtonSetState(UxGetWidget(toggleButton_fixed[xyzt]), TRUE, TRUE);
	   else
	     XmToggleButtonSetState(UxGetWidget(toggleButton_fixed[xyzt]), FALSE, TRUE);
	 }
       }

       (SF_ptr->current_clone_ptr)->region = *R_ptr;
     }
}


void JC_GeometryMenu_NewGeometry( int geometry )
{
     static current_geometry;

     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_Variable *V_ptr=&GLOBAL_Variable;

     Boolean it_is_a_vector=FALSE;

     if ( strchr(V_ptr->name, ',' ))
	  it_is_a_vector = TRUE;
     
     if ( geometry != current_geometry ) {
	  XtVaSetValues(UxGetWidget(optionMenu_Geometry),
			XmNmenuHistory, geomOptPBs[geometry],
			NULL);
	  current_geometry = geometry;
     }

     PO_ptr->plot_type = JC_PlotFrame_MaintainRadios(geometry, it_is_a_vector, SF_ptr);
     JC_PlotFrame_MaintainButtons(geometry, PO_ptr->plot_type, SF_ptr);
     JC_II_MainMenu_Maintain(SF_ptr);
}


void JC_GeometryMenu_NewVariable( JC_Variable *V_ptr, int geometry )
{
     int geo=0;

     for (geo=0; geo<16; geo++) {
	  if (V_ptr->okGeoms[geo])
	       XtSetSensitive(geomOptPBs[geo], TRUE);
	  else
	       XtSetSensitive(geomOptPBs[geo], FALSE);
     }

     JC_GeometryMenu_NewGeometry( geometry );
}


void JC_MainInterface_NewVariable( char *var_name, char *dset_name, int variable_info )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_Object *O_ptr=SF_ptr->current_clone_ptr;

     static Boolean a_clone_was_selected=FALSE;
     static JC_Region remembered_Region;

     int xyzt=0;
     char tempText[MAX_NAME_LENGTH]="";
     Boolean regridding_is_on=FALSE;


     /*
      * - If a clone IS selected:
      *    - if previous was a clone: do nothing (remembered_Region stays remembered)
      *    - else if previous was NOT a clone: remember the GLOBAL_Region
      *
      * - Else if a clone is NOT selected:
      *    - if previous was a clone: GLOBAL_Region = remembered_Region
      *    - else if previous was NOT a clone: do nothing:
      */

     if ( SF_ptr->a_clone_is_selected && !a_clone_was_selected )
       remembered_Region = GLOBAL_Region;
     else if ( !SF_ptr->a_clone_is_selected && a_clone_was_selected )
       GLOBAL_Region = remembered_Region;

     /*
      * - If a clone is selected:
      *    - restore the variable, the fixed("locked") axes and the regridding
      *
      * - Else (a clone is NOT selected):
      *    - get the new variable info and create and modify the region if necessary.
      */

     if ( SF_ptr->a_clone_is_selected ) {
	  GLOBAL_Variable = O_ptr->variable;
	  for (xyzt=0; xyzt<4; xyzt++) {
	    if ( O_ptr->fixed_axis[xyzt] ) {
	      R_ptr->span[xyzt] = O_ptr->region.span[xyzt];
	      R_ptr->transform[xyzt] = O_ptr->region.transform[xyzt];
	    }
	  }
	  JC_Region_SetGeometryFromSpan(R_ptr);
	  GLOBAL_Regridding = O_ptr->regridding;
	  a_clone_was_selected = TRUE;
	  JC_Map_SetToolColor("Yellow");
     } else {
       if ( variable_info != SAME_VARIABLE ) {
	 JC_Variable_New( V_ptr, var_name, dset_name );
	 if ( variable_info == FIRST_EVER_VARIABLE )
	   JC_Region_NewVariable( R_ptr, V_ptr, TRUE );
	 else
	   JC_Region_NewVariable( R_ptr, V_ptr, FALSE );
       }
       a_clone_was_selected = FALSE;
       JC_Map_SetToolColor("White");
     }
     

     /*
      * If we are dealing with any kind of new variable:
      *  - tell the map.
      *  - tell the transforms about the new regon.
      */
     if ( variable_info != SAME_VARIABLE ) {
       JC_Map_NewVariable( V_ptr );
       JC_Transforms_NewRegion( R_ptr );
     }
     

     /*
      * Print the appropriate string next to the "Regridding" toggle.
      */
     regridding_is_on = XmToggleButtonGetState(UxGetWidget(toggleButton_Regridding));
     
     if ( regridding_is_on ) {
       if ( RG_ptr->type == UNIFORM )
	 sprintf(tempText, "%s[d=%s]", RG_ptr->var[ALL_AXES], RG_ptr->dset[ALL_AXES]);
       else
	 sprintf(tempText, "non-Uniform regridding");
     }

     XtVaSetValues(label_RegriddingStatus,
		   RES_CONVERT(XmNlabelString,  tempText),
		   NULL);
     
     if ( SF_ptr->a_clone_is_selected ) {

	  XtSetSensitive(UxGetWidget(pushButton_Clone), FALSE);
	  XtVaSetValues(label_DataFrameStatus,
			RES_CONVERT(XmNlabelString,  "Cloned Variable."),
			NULL);
	  XtVaSetValues(textField_Variable,
			XmNeditable,  TRUE,
			XmNcursorPositionVisible,  TRUE,
			NULL);

	  for (xyzt=0; xyzt<4; xyzt++) {
	    if ( R_ptr->span[xyzt].ss[LO] != IRRELEVANT_AXIS) {
	      XtMapWidget(UxGetWidget(toggleButton_fixed[xyzt]));
	      XmToggleButtonSetState(UxGetWidget(toggleButton_fixed[xyzt]), O_ptr->fixed_axis[xyzt], FALSE);
	    } else {
	      XtUnmapWidget(UxGetWidget(toggleButton_fixed[xyzt]));
	    }
	  }
	  XmToggleButtonSetState(UxGetWidget(toggleButton_Regridding), O_ptr->fixed_regridding, FALSE);

     } else /* a clone is NOT selected */ {

       if ( strchr(var_name, ',' )) /* it is a vector */
	 XtSetSensitive(UxGetWidget(pushButton_Clone), FALSE);
       else
	 XtSetSensitive(UxGetWidget(pushButton_Clone), TRUE);

       XtVaSetValues(label_DataFrameStatus,
		     RES_CONVERT(XmNlabelString,  ""),
		     NULL);
       XtVaSetValues(textField_Variable,
		     XmNeditable,  FALSE,
		     XmNcursorPositionVisible,  False,
		     NULL);

       for (xyzt=0; xyzt<4; xyzt++) {
	 XtUnmapWidget(UxGetWidget(toggleButton_fixed[xyzt]));
	 XmToggleButtonSetState(UxGetWidget(toggleButton_fixed[xyzt]), FALSE, FALSE);
       }
       XmToggleButtonSetState(UxGetWidget(toggleButton_Regridding), FALSE, FALSE);

     }


     if ( variable_info != SAME_VARIABLE ) {
       /*
	* - Update the Data text fields.
	* - Update the Geometry option menu.
	* - Update the Geometry interface.
	* - Maintain the PlotRadio buttons
	* - Update the map.
	* - Maintain the MainMenu.
	* - Update the Map.
	*/

       XmTextSetString(textField_Variable, var_name);
       XmTextSetString(textField_Dataset, dset_name);

       JC_GeometryMenu_NewVariable(V_ptr, R_ptr->geometry);

       for (xyzt=0; xyzt<4; xyzt++) {
	 JC_sswwMenu_NewAxis( &(V_ptr->axis[xyzt]), R_ptr->span[xyzt].by_index_in_GUI );
	 JC_scrollBar_NewAxis( LO, &(V_ptr->axis[xyzt]), R_ptr->span[xyzt].ss[LO] ); 
	 JC_scrollBar_NewAxis( HI, &(V_ptr->axis[xyzt]), R_ptr->span[xyzt].ss[HI] ); 
	 JC_scrollBar_NewAxis( PT, &(V_ptr->axis[xyzt]), R_ptr->span[xyzt].ss[PT] );
       }     
       for (xyzt=0; xyzt<4; xyzt++)
	 JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[xyzt]));

       JC_Map_NewRegion(R_ptr);

     }

}


void JC_VariableTextField_Verify_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_DatasetElement *DE_ptr=NULL;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_Object *O_ptr=GLOBAL_StateFlags.current_clone_ptr;
     int i=0;
     Boolean put_up_underscore_message=FALSE;
     char *text=NULL, tempText[MAX_COMMAND_LENGTH]="";

/*
 * - If a cloned variable is not selected, return;
 *
 * - Replace any blanks in the name with underscores.
 *
 * - Find the current DatasetElement_ptr;
 * - If the new name exists in this dataset:
 *    - If the name does not belong to the current clone:
 *       - put up a message
 *       - generate a default name
 *
 * - Put up an informational message about underscores if necessary.
 *
 * - Copy the new name to the cloned variable.
 */

     if ( !SF_ptr->a_clone_is_selected ) 
	  return;

     if ( text = (char *) XmTextGetString(UxGetWidget(textField_Variable)) ) {

	  for (i=0; i<strlen(text); i++) {
	       text[i] = toupper(text[i]);
	       if ( text[i] == ' ' ) {
		    text[i] = '_';
		    put_up_underscore_message = TRUE;
	       }
	  }
	  text[strlen(text)] = '\0';

	  list_traverse(GLOBAL_DatasetList, V_ptr->dset, JC_ListTraverse_FoundDsetMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);

	  if ( JC_DatasetElement_VarnameExists(DE_ptr, text) ) {

	       if ( O_ptr != JC_Clone_ReturnPointer(text, V_ptr->dset) ) {
		    sprintf(tempText, "\
Variable name \"%s\" already exists in dataset \"%s\".  Please choose another name.", text, V_ptr->dset);
		    JC_Message_CB(wid, tempText, NULL);

		    i=0;
		    sprintf(text, "%s_CLONE_%d", V_ptr->name, i++);
		    while ( JC_DatasetElement_VarnameExists(DE_ptr, text) ) {
			 sprintf(text, "%s_CLONE_%d", V_ptr->name, i++);
		    }
		    XmTextSetString(textField_Variable, text);
	       }
	       
	  } else if ( put_up_underscore_message ) {
	       
	       sprintf(tempText, "\
Blanks are not acceptable in a variable name and have been changed to underscores.");
	       JC_Message_CB(wid, tempText, NULL);
	       XmTextSetString(textField_Variable, text);
	       
	  }	       
	  
	  if ( O_ptr == NULL )
	       fprintf(stderr, "ERROR in main.c: VariableTextField_Verify_CB: clone_ptr = NULL\n");
	  else
	       strcpy(O_ptr->name, text);

	  JC_II_SelectMenus_Recreate(UxGetWidget(FerretMainWd));

	  XtFree(text); /* allocated with XmTextGetString() */
     }

/*
 * Move the input.
 */

     XmProcessTraversal(wid, XmTRAVERSE_NEXT_TAB_GROUP);

}

void JC_CloseDataset_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
  JC_Variable *V_ptr=&GLOBAL_Variable;
  JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
  char cmd[MAX_COMMAND_LENGTH]="";
  char tempText[MAX_COMMAND_LENGTH]="";

  /*
    The following check is now unnecessary because the close dataset button is 
    disabled in MainMenu_Maintain when you get down to one datset.
    It won't hurt to leave it here, though.
    */

  if ( list_size(GLOBAL_DatasetList) == 1 ) {

    sprintf(tempText, "You may not close the last open dataset.");
    JC_Message_CB(wid, tempText, NULL);

  } else {

    sprintf(cmd, "CANCEL DATA %s", V_ptr->dset);
    ferret_command(cmd, IGNORE_COMMAND_WIDGET);

  }
     
}

void JC_SaveButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     extern swidget SaveDataObject;

     SaveDataObject = create_SaveDataObject(NO_PARENT);
     XtVaSetValues(SaveDataObject,
		   XmNiconic, False,
		   NULL);
}

void JC_EditDefinedVar_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
  extern swidget JC_EditDefinedVar;

  JC_EditDefinedVar = create_JC_EditDefinedVar(NO_PARENT);
     XtVaSetValues(JC_EditDefinedVar,
		   XmNiconic, False,
		   NULL);
}

void JC_DefineGrid_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     char tempText[MAX_NAME_LENGTH]="";

     sprintf(tempText, "Not yet available in this release.");
     JC_Message_CB(wid, tempText, NULL);
}

void JC_IncludeHours_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     if ( XmToggleButtonGetState(UxGetWidget(IncludeHours_Button)) )
	  SF_ptr->time_resolution_includes_hours = TRUE;
     else
	  SF_ptr->time_resolution_includes_hours = FALSE;

     JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[T_AXIS]) );
}

void JC_MapShowHide_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     if ( XmToggleButtonGetState(UxGetWidget(ShowMap_Button)) )
	  JC_Map_Show();
     else
	  JC_Map_Hide();
}

void JC_PlotOptions_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     PlotOptions = create_PlotOptions(NO_PARENT);
     SetInitialPOState();
}


void JC_Viewports_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     extern swidget Viewports;

     Viewports = create_Viewports(NO_PARENT);
     XtVaSetValues(Viewports,
		   XmNiconic, False,
		   NULL);
}


void JC_WindowButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     char *tempText;
     char command[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;

     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(command, tempText);
     ferret_command(command, IGNORE_COMMAND_WIDGET);
     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void JC_X_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Span_Print( &(R_ptr->span[X_AXIS]), stderr );
}


void JC_Y_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Span_Print( &(R_ptr->span[Y_AXIS]), stderr );
}


void JC_Z_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Span_Print( &(R_ptr->span[Z_AXIS]), stderr );
}


void JC_T_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Span_Print( &(R_ptr->span[T_AXIS]), stderr );
}


void JC_X_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Axis_PrintNoValues( &(V_ptr->axis[X_AXIS]), stderr );
}


void JC_Y_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Axis_PrintNoValues( &(V_ptr->axis[Y_AXIS]), stderr );
}


void JC_Z_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Axis_PrintNoValues( &(V_ptr->axis[Z_AXIS]), stderr );
}


void JC_T_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Axis_PrintNoValues( &(V_ptr->axis[T_AXIS]), stderr );
}


void JC_scrollBar_CB( XtPointer client_data, XtPointer cbArg )
{
	char tempText[MAX_NAME_LENGTH]="";
	XmScrollBarCallbackStruct *sbArgs=(XmScrollBarCallbackStruct *)cbArg;
	JC_Region *R_ptr=&GLOBAL_Region;
	JC_Variable *V_ptr=&GLOBAL_Variable;
	JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
	int xyzt=-1, lo_hi_pt=-1, index=0;

	strcpy(tempText, (char *)client_data);
	
	/* 
	 * - Get the xyzt and lo_hi_pt indices.
	 */

	if ( tempText[0] == 'x' ) xyzt = X_AXIS;
	else if ( tempText[0] == 'y' ) xyzt = Y_AXIS;
	else if ( tempText[0] == 'z' ) xyzt = Z_AXIS;
	else if ( tempText[0] == 't' ) xyzt = T_AXIS;
	else {
	     fprintf(stderr, "main.c: scrollBar_CB(): client_data = \"%s\"\n", (char *)client_data);
	     return;
	}
	
	if ( tempText[2] == 'l' ) lo_hi_pt = LO;
	else if ( tempText[2] == 'h' ) lo_hi_pt = HI;
	else if ( tempText[2] == 'p' ) lo_hi_pt = PT;
	else {
	     fprintf(stderr, "main.c: scrollBar_CB(): client_data = \"%s\"\n", (char *)client_data);
	     return;
	}
	
	
	/*
	 * - Get the value of the scroll bar.
	 *
	 * - Check it against the min and max for that axis.
	 *
	 * - Set the ss value.
	 *
	 * - If the axis has regular spacing
	 *      create the ww value
	 * - else
	 *      look up the ww value
	 *
	 * - If we have overtaken the value of the other scrollbar
	 *      update the other scrollBar to the current value.
	 *      decrement/increment this scrollBar and value by one 'ss' unit. 
	 *
	 * - Update that element in the geometry interface.
	 * - Update the Map.
	 * - If this is a clone: update the cloned variable's region
	 */

	index = sbArgs->value;

	if ( index < 0 )
	     index = 0;
	else if ( index > V_ptr->axis[xyzt].num_points - 1 )
	     index = V_ptr->axis[xyzt].num_points - 1;
	
	R_ptr->span[xyzt].ss[lo_hi_pt] = index;
	
	if ( V_ptr->axis[xyzt].has_regular_spacing )
	     R_ptr->span[xyzt].ww[lo_hi_pt] = V_ptr->axis[xyzt].start +
		  (R_ptr->span[xyzt].ss[lo_hi_pt] * V_ptr->axis[xyzt].delta);
	else /* use the lookup array */
	     R_ptr->span[xyzt].ww[lo_hi_pt] = V_ptr->axis[xyzt].value[index];
	
	if ( lo_hi_pt == LO && R_ptr->span[xyzt].ss[LO] >= R_ptr->span[xyzt].ss[HI] ) {

	     R_ptr->span[xyzt].ss[HI] = R_ptr->span[xyzt].ss[LO];
	     R_ptr->span[xyzt].ww[HI] = R_ptr->span[xyzt].ww[LO];

	     R_ptr->span[xyzt].ss[lo_hi_pt] = R_ptr->span[xyzt].ss[lo_hi_pt] - 1;
	     index--;
	     if ( V_ptr->axis[xyzt].has_regular_spacing )
		  R_ptr->span[xyzt].ww[lo_hi_pt] = V_ptr->axis[xyzt].start +
		       (R_ptr->span[xyzt].ss[lo_hi_pt] * V_ptr->axis[xyzt].delta);
	     else		/* use the lookup array */
		  R_ptr->span[xyzt].ww[lo_hi_pt] = V_ptr->axis[xyzt].value[index];

	     JC_scrollBar_SetValue(scrollBar_widget[xyzt][HI], R_ptr->span[xyzt].ss[HI]);
	     JC_textField_SetValue(textField_widget[xyzt][HI], HI, &(R_ptr->span[xyzt]));
	     JC_scrollBar_SetValue(scrollBar_widget[xyzt][LO], R_ptr->span[xyzt].ss[LO]);
	     JC_textField_SetValue(textField_widget[xyzt][LO], LO, &(R_ptr->span[xyzt]));

	} else if ( lo_hi_pt == HI && R_ptr->span[xyzt].ss[HI] <= R_ptr->span[xyzt].ss[LO] ) {
	     
	     R_ptr->span[xyzt].ss[LO] = R_ptr->span[xyzt].ss[HI];
	     R_ptr->span[xyzt].ww[LO] = R_ptr->span[xyzt].ww[HI];

	     R_ptr->span[xyzt].ss[lo_hi_pt] = R_ptr->span[xyzt].ss[lo_hi_pt] + 1;
	     index++;
	     if ( V_ptr->axis[xyzt].has_regular_spacing )
		  R_ptr->span[xyzt].ww[lo_hi_pt] = V_ptr->axis[xyzt].start +
		       (R_ptr->span[xyzt].ss[lo_hi_pt] * V_ptr->axis[xyzt].delta);
	     else 	/* use the lookup array */
		  R_ptr->span[xyzt].ww[lo_hi_pt] = V_ptr->axis[xyzt].value[index];

	     JC_scrollBar_SetValue(scrollBar_widget[xyzt][LO], R_ptr->span[xyzt].ss[LO]);
	     JC_textField_SetValue(textField_widget[xyzt][LO], LO, &(R_ptr->span[xyzt]));
	     JC_scrollBar_SetValue(scrollBar_widget[xyzt][HI], R_ptr->span[xyzt].ss[HI]);
	     JC_textField_SetValue(textField_widget[xyzt][HI], HI, &(R_ptr->span[xyzt]));
	     
	} else {   

	  JC_textField_SetValue(textField_widget[xyzt][lo_hi_pt], lo_hi_pt, &(R_ptr->span[xyzt]));

	}

	JC_Map_NewRegion(R_ptr);
	
	if ( SF_ptr->a_clone_is_selected )
	     (SF_ptr->current_clone_ptr)->region = *R_ptr;
   }


void JC_scrollBar_NewAxis( int lo_hi_pt, JC_Axis *A_ptr, int value ) 
{
     int val=0, slider_size=0, inc=0, page_inc=0;
     int max=0, min=0;

     if ( A_ptr->ss[LO] == IRRELEVANT_AXIS )
	  return;

/*
 * -  Set up scoll bars for the an axis:
 *       XmNmaximum is set to  numValsOnAxis[xyzt] + slider_size - 1
 *         (because a 'C' index is one less than a Fortran index)
 */
     
     XmScrollBarGetValues(scrollBar_widget[A_ptr->xyzt][lo_hi_pt], &val, &slider_size, &inc, &page_inc);

     if ( A_ptr->num_points < 20 ) {
	  page_inc = 2;
	  slider_size = 2;
     } else if ( A_ptr->num_points >= 20 && A_ptr->num_points <= 100 ) {
	  page_inc = 5;
	  slider_size = 5;
     } else if ( A_ptr->num_points > 100 && A_ptr->num_points <= 1000 ) {
	  page_inc = 10;
	  slider_size = 10;
     } else /* num_points > 1000 */ {
	  page_inc = 50;
	  slider_size = 50;
     }

     max = A_ptr->num_points + slider_size - 1;

     XtVaSetValues(scrollBar_widget[A_ptr->xyzt][lo_hi_pt],
		   XmNincrement, inc,
		   XmNmaximum, max,
		   XmNminimum, min,
		   XmNpageIncrement, page_inc,
		   XmNsliderSize, slider_size,
		   XmNvalue, value,
		   NULL);

}


void JC_scrollBar_NewSpan( int lo_hi_pt, int is_displayed, JC_Span *S_ptr ) 
{
     int xyzt=S_ptr->xyzt;

     static Boolean scrollBar_is_displayed[5][3];

     /*
      * - If the axis is irrelevant OR this scrollBar should not be displayed
      *      desensitize the widget
      *      unmanage the widget
      *      remember until next time that the widget is not displayed
      */

     if ( S_ptr->ss[LO] == IRRELEVANT_AXIS || !is_displayed ) {
	  XtSetSensitive(scrollBar_widget[xyzt][lo_hi_pt], FALSE);
	  XtUnmanageChild(scrollBar_widget[xyzt][lo_hi_pt]);
	  scrollBar_is_displayed[xyzt][lo_hi_pt] = FALSE;
	  return;
     }

/*
 * - Set the scrollBar value
 */     

     JC_scrollBar_SetValue(scrollBar_widget[xyzt][lo_hi_pt], S_ptr->ss[lo_hi_pt]);
     
     /*
      * - If the widget is not currently displayed
      *      sensitize the widget
      *      manage the widget
      *      remember until next time that the widget is displayed
      */
     
     if ( !scrollBar_is_displayed[xyzt][lo_hi_pt] ) {
	  XtSetSensitive(scrollBar_widget[xyzt][lo_hi_pt], TRUE);
	  XtManageChild(scrollBar_widget[xyzt][lo_hi_pt]);
	  scrollBar_is_displayed[xyzt][lo_hi_pt] = TRUE;
     }
}


void JC_scrollBar_SetValue( Widget scrollBar_widget, int value)
{
     int val=0, ss=0, inc=0, pinc=0, max=0;
 
     XtVaGetValues(scrollBar_widget,
		   XmNmaximum, &max,
		   NULL);

     XmScrollBarGetValues(scrollBar_widget, &val, &ss, &inc, &pinc);
     XmScrollBarSetValues(scrollBar_widget, value, ss, inc, pinc, FALSE);

     if (value < 0) 
	  printf("ERROR in main.c: scrollBar_SetValue(): negative scrollbar value %d\n", value);
     if (value > max-ss) 
	  printf("ERROR in main.c: scrollBar_SetValue(): value[%d], maximum[%d], slider_size[%d]\n", value, max, ss);


}


void JC_sswwMenu_CB( int xyzt, int selection )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     
     /*
      * - Set span[xyzt].by_index_in_GUI flag.
      * - Update the textField and scrollBar.
      */ 

     switch (xyzt) {

	case X_AXIS:
	case Y_AXIS:
	case Z_AXIS:
	  R_ptr->span[xyzt].by_index_in_GUI = selection;
	  break;
	case T_AXIS:
	  R_ptr->span[xyzt].time_type = selection;
	  if ( selection == INDEX_TIME )
	       R_ptr->span[xyzt].by_index_in_GUI = TRUE;
	  else
	       R_ptr->span[xyzt].by_index_in_GUI = FALSE;
	  break;
     }

     JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[xyzt]) );

     if ( SF_ptr->a_clone_is_selected )
	  (SF_ptr->current_clone_ptr)->region = *R_ptr;
}


void JC_sswwMenu_NewAxis( JC_Axis *A_ptr, Boolean by_index_in_GUI )
{
     
     char tempText[MAX_NAME_LENGTH]="";
     Widget button_widget_id;
     int xyzt=A_ptr->xyzt;

     static Boolean sswwMenu_is_sensitive[4];
     
     if ( A_ptr->ss[LO] == IRRELEVANT_AXIS ) {
	  XtSetSensitive(axssww[xyzt], FALSE);
	  sswwMenu_is_sensitive[xyzt] = FALSE;
	  return;
     }
     
     switch (xyzt) {
	  /*
	   * - For the X, Y, and Z-axes
	   *      set the button_widget_id
	   *      create the name if it is not by index
	   */
	case X_AXIS:
	case Y_AXIS:
	case Z_AXIS:
	  if ( by_index_in_GUI )
	       button_widget_id = cxBySS[xyzt];
	  else			/* ww coordinates */ {
	       button_widget_id = cxByWW[xyzt];
	       strcpy(tempText, A_ptr->title);
	       PadOrTrunc(tempText, 11);
	       XtVaSetValues(button_widget_id,
			     RES_CONVERT(XmNlabelString,  tempText),
			     NULL);
	  }
	  break;
	  
	case T_AXIS:
	  /*
	   * - For the T axis
	   *      sensitize the Climatology/Calendar buttons as appropriate
	   *      set the button_widget_id
	   *      create the name if time_type == MODEL_TIME
	   */
	  
	  if ( A_ptr->has_fancy_labeling && !A_ptr->is_modulo )
	       XtSetSensitive(UxGetWidget(optionMenu_Tp_Calendar), TRUE);
	  else 
	       XtSetSensitive(UxGetWidget(optionMenu_Tp_Calendar), FALSE);
	  
	  if ( A_ptr->is_modulo )
	       XtSetSensitive(UxGetWidget(optionMenu_Tp_Climatology), TRUE);
	  else 
	       XtSetSensitive(UxGetWidget(optionMenu_Tp_Climatology), FALSE);
	  
	  if ( A_ptr->time_type == INDEX_TIME )
	       button_widget_id = (Widget)optionMenu_Tp_Index;
	  else			/* ww coordinates */ {
	       if ( A_ptr->time_type == MODEL_TIME ) {
		    strcpy(tempText, A_ptr->title);
		    button_widget_id = (Widget)optionMenu_Tp_Model;
		    PadOrTrunc(tempText, 7);
		    strcat(tempText, "(raw)");
		    XtVaSetValues(button_widget_id,
				  RES_CONVERT(XmNlabelString,  tempText), 
				  NULL);
	       }
	       else if ( A_ptr->time_type == CALENDAR_TIME ) {
		    strcpy(tempText, A_ptr->title);
		    button_widget_id = (Widget)optionMenu_Tp_Calendar;
		    PadOrTrunc(tempText, 12);
		    XtVaSetValues(button_widget_id,
				  RES_CONVERT(XmNlabelString,  tempText), 
				  NULL);
	       } else if ( A_ptr->time_type == CLIMATOLOGY_TIME )
		    button_widget_id = (Widget)optionMenu_Tp_Climatology;
	  }
	  break;
	  
     }
     
     
     /*
      * - Set the appropriate button in the option menu
      * - Sensitize the option menu.
      */
     
     XtVaSetValues(axssww[xyzt],
		   XmNmenuHistory, button_widget_id,
		   NULL);

     if ( !sswwMenu_is_sensitive[xyzt] ) {
	  XtSetSensitive(axssww[xyzt], TRUE);
	  sswwMenu_is_sensitive[xyzt] = TRUE;
     }
}


void JC_TransArg_CB( Widget widget_id, XtPointer clientData)
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     double value=0.0;
     char *tText;
     char xyztText[2]="";

     tText = XmTextFieldGetString(widget_id);
     sscanf(tText, "%lf", &value);
     strcpy(xyztText, (char *)clientData);

     if ( xyztText[0] == 'x' )
	  R_ptr->transform[X_AXIS].arg = (float)value;
     else if ( xyztText[0] == 'y' )
	  R_ptr->transform[Y_AXIS].arg = (float)value;
     else if ( xyztText[0] == 'z' )
	  R_ptr->transform[Z_AXIS].arg = (float)value;
     else if ( xyztText[0] == 't' )
	  R_ptr->transform[T_AXIS].arg = (float)value;
     else
	  fprintf(stderr, "ERROR in main.c: JC_TransArg_CB(): clientData = \"%s\"\n", xyztText);

     XtFree(tText); /* allocated with XmTextFieldGetString() */

     if ( SF_ptr->a_clone_is_selected )
	  (SF_ptr->current_clone_ptr)->region = *R_ptr;
}

void JC_TransMenu_CB( char *axis_transform )
{
     char label[MAX_NAME_LENGTH]="";
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     JC_Transform T={ NULL, };
     int i=0, xyzt=0, arg=0;
     Boolean arg_is_float=FALSE, sensitize=FALSE, it_is_a_vector=FALSE;

/*
 * - The JC_Transform structures consist of the following:
 *      {name, arg, code, exists, compresses, accepts_an_argument}
 *
 * - NB_ The 'axis_transform' argument passed in contains the axis name, an underscore, and a transform name
 *   as in "x_ave".  Thus we cannot just copy it into the transform name.  'axis_transform+2' is a pointer to
 *   char which points to the transform part of this character string.
 */

     /*
      * - Clear the memory in T.
      * - Copy the transform part into T.name.
      * - Convert to upper case.
      * - If one of the transformCodes is found as part of T.name
      * -    Shorten T.name (take "
      * - Determine which transform it is.
      * - Set the transform code.
      * - Depending on the transform
      *      set the default argument.
      * - Manage the TransformArguments interface.
      */
     
     JC_Transform_Clear( &T );

     strcpy(T.name, axis_transform+2);
     for (i=0; i<strlen(T.name); i++)
	  T.name[i] = toupper(T.name[i]);
     
     T.code = -1;
     for (i=0; i<NUM_TRANSFORMS; i++) {
	  if ( !strcmp(T.name, transformCodes[i]) ) {
	       T.code = i;
	       break;
	  }
     }

     switch ( T.code ) {

	case TRANS_NON:
	  sensitize = FALSE;
	  arg_is_float = FALSE;
	  T.exists = FALSE;
	  T.compresses = FALSE;
	  T.accepts_an_argument = FALSE;
	  break;

	case TRANS_SHF:
	case TRANS_FNR:
	case TRANS_FLN:
	  sensitize = TRUE;
	  arg_is_float = FALSE;
	  T.exists = TRUE;
	  T.compresses = FALSE;
	  T.accepts_an_argument = TRUE;
	  T.arg = 1;
	  break;
	  
	case TRANS_SHN:
	case TRANS_SBN:
	case TRANS_SPZ:
	case TRANS_SWL:
	case TRANS_SBX:
	case TRANS_FAV:
	  sensitize = TRUE;
	  arg_is_float = FALSE;
	  T.exists = TRUE;
	  T.compresses = FALSE;
	  T.accepts_an_argument = TRUE;
	  T.arg = 3;
	  break;

	case TRANS_LOC:
	  sensitize = TRUE;
	  arg_is_float = TRUE;
	  T.exists = TRUE;
	  T.compresses = TRUE;
	  T.accepts_an_argument = TRUE;
	  T.arg = 0.0;
	  break;
	  
	case TRANS_WEQ:
	  sensitize = TRUE;
	  arg_is_float = TRUE;
	  T.exists = TRUE;
	  T.compresses = FALSE;
	  T.accepts_an_argument = TRUE;
	  break;
	  
	case TRANS_AVE:
	case TRANS_VAR:
	case TRANS_SUM:
	case TRANS_MIN:
	case TRANS_MAX:
	case TRANS_DIN:
	case TRANS_NGD:
	case TRANS_NBD:
	  sensitize = FALSE;
	  arg_is_float = FALSE;
	  T.exists = TRUE;
	  T.compresses = TRUE;
	  T.accepts_an_argument = FALSE;
	  break;
	  
	case TRANS_RSU:
	case TRANS_DDC:
	case TRANS_DDF:
	case TRANS_DDB:
	case TRANS_IIN:
	  sensitize = FALSE;
	  arg_is_float = FALSE;
	  T.exists = TRUE;
	  T.compresses = FALSE;
	  T.accepts_an_argument = FALSE;
	  break;
	  
	default:
	  fprintf(stderr, "ERROR in main.c: JC_TransMenu_CB(): T.code = %d (unknown transform)\n", T.code);
	  break;

     }

     if ( arg_is_float )
	  sprintf(label, "%.2f", T.arg);
     else
	  sprintf(label, "%d", (int)T.arg);
     
     switch (axis_transform[0]) {
	case 'x':
	  xyzt = X_AXIS;
	  break;
	case 'y':
	  xyzt = Y_AXIS;
	  break;
	case 'z':
	  xyzt = Z_AXIS;
	  break;
	case 't':
	  xyzt = T_AXIS;
	  break;
     }

     if ( sensitize ) {
	  XtVaSetValues(UxGetWidget(axarg[xyzt]),
			XmNvalue, label,
			NULL);
	  XtSetSensitive(UxGetWidget(axarg[xyzt]), TRUE);
     } else {
	  XtVaSetValues(UxGetWidget(axarg[xyzt]),
			XmNvalue, "",
			NULL);
	  XtSetSensitive(UxGetWidget(axarg[xyzt]), FALSE);
     }
	  
     
     if ( SF_ptr->a_clone_is_selected && T.compresses )
	  XmToggleButtonSetState(toggleButton_fixed[xyzt], TRUE, TRUE);
     else
	  XmToggleButtonSetState(toggleButton_fixed[xyzt], FALSE, TRUE);


     /* - Adjust the region's span and geometry if necessary.
      * - Update the Geometry Menu.
      * - Update Geometry Interface.
      * - Update the Map.
      */
     
     JC_Region_NewTransform( R_ptr, xyzt, &T );
     JC_GeometryMenu_NewGeometry( R_ptr->geometry );
     for (xyzt=0; xyzt<4; xyzt++)
	  JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[xyzt]) );
     
     if ( strchr(V_ptr->name, ',' ))
	  it_is_a_vector = TRUE;

     JC_Map_NewRegion(R_ptr);
     if ( SF_ptr->a_clone_is_selected )
	  (SF_ptr->current_clone_ptr)->region = *R_ptr;
}


void JC_Transforms_NewRegion( JC_Region *R_ptr )
{
     int xyzt=0;

     for (xyzt=0; xyzt<4; xyzt++) {

	  if ( R_ptr->span[xyzt].ss[LO] != IRRELEVANT_AXIS ) {
	       XtManageChild(axtrans[xyzt]);
	       XtManageChild(axarg[xyzt]);
	  } else { 
	       XtUnmanageChild(axtrans[xyzt]);
	       XtUnmanageChild(axarg[xyzt]);
	  }
	  
	  if ( !R_ptr->transform[xyzt].exists ) {
	       XtVaSetValues(UxGetWidget(axtrans[xyzt]),
			     XmNmenuHistory, axtrnButton[xyzt][0],
			     NULL);
	  } else {
	       XtVaSetValues(UxGetWidget(axtrans[xyzt]),
			     XmNmenuHistory, axtrnButton[xyzt][R_ptr->transform[xyzt].code],
			     NULL);

	  }

     }

}


void JC_FixedToggle_CB( Widget wid, XtPointer client_data, XtPointer call_data  )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     char tempText[MAX_NAME_LENGTH]="";
     Boolean regridding_is_on=FALSE;

     if ( wid != UxGetWidget(toggleButton_Regridding) && SF_ptr->current_clone_ptr == NULL) {
	  fprintf(stderr, "ERROR in main.c: JC_FixedToggle_CB(): current_clone_ptr = NULL\n");
	  return;
     }

     if ( wid == UxGetWidget(toggleButton_Regridding) ) {

	  regridding_is_on = XmToggleButtonGetState(UxGetWidget(toggleButton_Regridding));

	  if ( regridding_is_on ) {
	       if ( RG_ptr->type == UNIFORM )
		    sprintf(tempText, "%s[d=%s]", RG_ptr->var[ALL_AXES], RG_ptr->dset[ALL_AXES]);
	       else
		    sprintf(tempText, "non-Uniform regridding");
	  }
	  XtVaSetValues(label_RegriddingStatus,
			RES_CONVERT(XmNlabelString,  tempText),
			NULL);
	  
	  if ( SF_ptr->current_clone_ptr != NULL )
	       (SF_ptr->current_clone_ptr)->fixed_regridding = regridding_is_on;

     } else if ( wid == UxGetWidget(toggleButton_X) )
	  (SF_ptr->current_clone_ptr)->fixed_axis[X_AXIS] = XmToggleButtonGetState(UxGetWidget(toggleButton_X));
     else if ( wid == UxGetWidget(toggleButton_Y) )
	  (SF_ptr->current_clone_ptr)->fixed_axis[Y_AXIS] = XmToggleButtonGetState(UxGetWidget(toggleButton_Y));
     else if ( wid == UxGetWidget(toggleButton_Z) )
	  (SF_ptr->current_clone_ptr)->fixed_axis[Z_AXIS] = XmToggleButtonGetState(UxGetWidget(toggleButton_Z));
     else if ( wid == UxGetWidget(toggleButton_T) )
	  (SF_ptr->current_clone_ptr)->fixed_axis[T_AXIS] = XmToggleButtonGetState(UxGetWidget(toggleButton_T));
     else
	  fprintf(stderr, "ERROR in main.c: JC_FixedToggle_CB(): unknownd wid: %d\n", (int)wid);

}


void JC_PlotTypeToggle_CB(int plot_type)
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     
     /*
      * I shouldn't change "plot_type_last_plotted" as we aren't plotting now 
      * but I have the following problem:
      *
      * Each JC_Synchronize() ends up calling JC_PlotFrame_MaintainRadios()
      * which resets the plot_type to plot_type_last_plotted.
      * If I plot something, and choose another variable this is great.
      * But if I toggle to another plot type and change to another variable
      * before plotting I get reset to the plot_type_last_plotted.
      *
      * So to solve this I'll reinterpret plot_type_last_plotted as
      * "plot_type_last_chosen".  This should solve the problem.
      */

     PO_ptr->plot_type = plot_type;
     SF_ptr->plot_type_last_plotted = PO_ptr->plot_type;
     JC_PlotFrame_MaintainButtons(R_ptr->geometry, PO_ptr->plot_type, SF_ptr);
     JC_II_MainMenu_Maintain(SF_ptr);

}


void JC_PlotButton_CB()
{
     char command[MAX_COMMAND_LENGTH]="";
     JC_Object Obj;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     int numViewportCycles[3]={3,1,1};
     char *viewPortNames[3][4] = {{"LL", "LR", "UR", "UL"},
				  {"LEFT", "RIGHT", "", "" },
				  {"UPPER", "LOWER", "", "" }};
     
     
     Obj.variable = GLOBAL_Variable;
     Obj.regridding = GLOBAL_Regridding;
     Obj.region = GLOBAL_Region;
     Obj.fixed_regridding = XmToggleButtonGetState(UxGetWidget(toggleButton_Regridding));

     PO_ptr->overlay = FALSE;

/*
 * - Create the "SET VIEWPORT" command if necessary.
 * - Create and send the "PLOT" command
 * - Update various StateFlags.
 * - Sensitize the "Overlay" and "Clear" buttons.
 */

     if ( gViewportActive ) {
	  if ( gViewportIsCycling ) {
	       if ( gCurrViewportCycle > gNumViewportCycles[gCurrViewportType] ) {
		    gCurrViewportCycle = 0;
		    ferret_command("SET WINDOW/CLEAR", IGNORE_COMMAND_WIDGET);
	       }
	  }
	  strcpy(command, "SET VIEWPORT ");
	  strcat(command, viewPortNames[gCurrViewportType][gCurrViewportCycle]);
	  ferret_command(command, IGNORE_COMMAND_WIDGET);
	  if ( gViewportIsCycling )
	       gCurrViewportCycle++;
     }

     SF_ptr->geometry_last_plotted = Obj.region.geometry;
     SF_ptr->a_plot_exists = TRUE;

     JC_PlotCommand_Create(command, &Obj, PO_ptr);
     ferret_command(command, IGNORE_COMMAND_WIDGET);

     XtSetSensitive(UxGetWidget(pushButton_Overlay), TRUE);
}


void JC_OverlayButton_CB()
{
     char command[MAX_COMMAND_LENGTH]="";
     JC_Object Obj;
     JC_PlotOptions *PO_ptr=&GLOBAL_PlotOptions;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     int numViewportCycles[3]={3,1,1};
     char *viewPortNames[3][4] = {{"LL", "LR", "UR", "UL"},
				  {"LEFT", "RIGHT", "", "" },
				  {"UPPER", "LOWER", "", "" }};
     

     Obj.variable   = GLOBAL_Variable;
     Obj.regridding = GLOBAL_Regridding;
     Obj.region     = GLOBAL_Region;

     PO_ptr->overlay = TRUE;

/*
 * - Create the "SET VIEWPORT" command if necessary.
 * - Create and send the "OVERLAY" command
 */

     if ( gViewportActive ) {
	  if ( gViewportIsCycling ) {
	       if ( gCurrViewportCycle > gNumViewportCycles[gCurrViewportType] ) {
		    gCurrViewportCycle = 0;
	       }
	  }
	  strcpy(command, "SET VIEWPORT ");
	  strcat(command, viewPortNames[gCurrViewportType][gCurrViewportCycle]);
	  ferret_command(command, IGNORE_COMMAND_WIDGET);
	  if ( gViewportIsCycling )
	       gCurrViewportCycle++;
     }

     SF_ptr->geometry_last_plotted = Obj.region.geometry;
     SF_ptr->a_plot_exists = TRUE;

     JC_PlotCommand_Create(command, &Obj, &GLOBAL_PlotOptions);
     ferret_command(command, IGNORE_COMMAND_WIDGET);

}


static void JC_ClearButton_CB()
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     ferret_command("SET WINDOW/CLEAR", IGNORE_COMMAND_WIDGET);
     gSomethingIsPlotted = 0;
     XtSetSensitive(UxGetWidget(pushButton_Plot), TRUE);
     XtSetSensitive(UxGetWidget(pushButton_Overlay), FALSE);
}


void JC_textField_SetValue( Widget textField_widget, int lo_hi_pt, JC_Span *S_ptr )
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     char tempText[MAX_NAME_LENGTH]="";
     double value=(double)S_ptr->ww[lo_hi_pt];

/*
 * - The index value is easy to set.
 * - For "FancyLabel" more work is required,
 */

     if ( S_ptr->by_index_in_GUI )
	  sprintf(tempText, "%d", S_ptr->ss[lo_hi_pt] + 1); /* Ferret needs/GUI displays index value 1 greater than C code */
     else
	  JC_String_CreateFancyLabel(tempText, value, S_ptr, SF_ptr );

     XmTextSetString(textField_widget, tempText);
}


void JC_textField_NewSpan( int lo_hi_pt, int is_displayed, JC_Span *S_ptr )
{
     int xyzt=S_ptr->xyzt;

     static Boolean textField_is_displayed[5][3];

     /*
      * - If the axis is irrelevant OR this textField should not be displayed
      *      desensitize the widget
      *      unmanage the widget
      *      remember until next time that the widget is not displayed
      */

     if ( S_ptr->ss[LO] == IRRELEVANT_AXIS || !is_displayed ) {
	  XtSetSensitive(textField_widget[xyzt][lo_hi_pt], FALSE);
	  XtUnmanageChild(textField_widget[xyzt][lo_hi_pt]);
	  textField_is_displayed[xyzt][lo_hi_pt] = FALSE;
	  return;
     }
     
     /*
      * - Set the value in the textField.
      * - If the widget is not currently displayed
      *      manage the widget
      *      remember until next time that the widget is displayed
      */
     
     JC_textField_SetValue( textField_widget[xyzt][lo_hi_pt], lo_hi_pt, S_ptr );	  
     
     if ( !textField_is_displayed[xyzt][lo_hi_pt] ) {
	  XtSetSensitive(textField_widget[xyzt][lo_hi_pt], TRUE);
	  XtManageChild(textField_widget[xyzt][lo_hi_pt]);
	  textField_is_displayed[xyzt][lo_hi_pt] = TRUE;
     }
     
}


void CloseWinCB(wid, clientData, callData)
Widget wid;
XtPointer clientData, callData;
{
	XmAnyCallbackStruct *cbs = (XmAnyCallbackStruct *)callData;

	if (cbs->reason == XmCR_WM_PROTOCOLS) {
	  DoQuit();
	}
}

Display *GetCurrDisplay()
{
	/* return the display pointer for the gui */
	return XtDisplay(UxGetWidget(FerretMainWd));
}


void JC_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Object Obj={ NULL, };
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_DatasetElement *DE_ptr=NULL;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     int i=0, xyzt=0;
     Boolean I_want_to_make_the_clone_current=TRUE;
     char var_to_restore[MAX_NAME_LENGTH], dset_to_restore[MAX_NAME_LENGTH];

     strcpy(var_to_restore, V_ptr->name);
     strcpy(dset_to_restore, V_ptr->dset);

     /*
      * - Get the information which goes into the Object.
      * 
      * - Test the various axes to see if we should fix any of them.
      */

     Obj.variable   = GLOBAL_Variable;
     Obj.regridding = GLOBAL_Regridding;
     Obj.region     = GLOBAL_Region;
     Obj.fixed_regridding = XmToggleButtonGetState(UxGetWidget(toggleButton_Regridding));

     for (xyzt=0; xyzt<4; xyzt++) {
       if ( R_ptr->span[xyzt].ss[LO] != IRRELEVANT_AXIS) {
	 if ( !R_ptr->span[xyzt].needs_lo_hi_displayed_in_GUI || R_ptr->span[xyzt].is_compressed_in_GUI )
	   Obj.fixed_axis[xyzt] = TRUE;
	 else
	   Obj.fixed_axis[xyzt] = FALSE;
       }
     }

/*
 * - Find the correct dataset.
 *
 * - Generate a new name.
 * - Put this name in the "variable" textField
 *
 * - Add this cloned variable at the end of the cvarList of the current dataset.
 *
 * - Update the "Select" menus.
 */

     list_traverse(GLOBAL_DatasetList, V_ptr->dset, JC_ListTraverse_FoundDsetMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
     DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);

     sprintf(Obj.name, "%s_CLONE_%d", V_ptr->name, i++);
     while ( JC_DatasetElement_VarnameExists(DE_ptr, Obj.name) ) {
	  sprintf(Obj.name, "%s_CLONE_%d", V_ptr->name, i++);
     }

     list_mvrear(DE_ptr->cvarList);
     SF_ptr->current_clone_ptr = (JC_Object *)list_insert_after(DE_ptr->cvarList, &Obj, sizeof(Obj));

     JC_II_SelectMenus_Recreate(FerretMainWd);

     if ( I_want_to_make_the_clone_current ) {
	  SF_ptr->a_clone_is_selected = TRUE;
	  JC_MainInterface_NewVariable( Obj.name, V_ptr->dset, FALSE );
     } else {
	  SF_ptr->current_clone_ptr = NULL;
	  SF_ptr->a_clone_is_selected = FALSE;
	  JC_MainInterface_NewVariable( var_to_restore, dset_to_restore, FALSE );
     }
	  
/*
 * Move the input.
 */

     XmProcessTraversal(wid, XmTRAVERSE_NEXT_TAB_GROUP);

}


void JC_ListButton_CB( void )
{
     char command[MAX_COMMAND_LENGTH]="";
     JC_Object Obj;
     
     Obj.variable   = GLOBAL_Variable;
     Obj.regridding = GLOBAL_Regridding;
     Obj.region     = GLOBAL_Region;

     JC_ListCommand_Create(command, &Obj);
     ferret_command(command, IGNORE_COMMAND_WIDGET);
}


static void JC_InfoButton_CB( void )
{
     JC_Variable *V_ptr=&GLOBAL_Variable;
     char tempText[MAX_NAME_LENGTH]="";

     strcpy(tempText, "SHOW DATA/VARIABLES ");
     strcat(tempText, V_ptr->dset);
     ferret_command(tempText, IGNORE_COMMAND_WIDGET);
}


void JC_PlotFrame_MaintainButtons( int geometry, int plot_type, JC_StateFlags *SF_ptr )
{

/*
 * - if a plot type is selected AND the geometry is valid (1- or 2-dimensional)
 *     sensitize the Plot button
 * - if a plot exists AND a plot type is selected AND the new geometry matches the plotted one
 *     sensitize the Overlay button
 */
	if ( plot_type && geometry > GEOM_POINT && geometry < GEOM_XYZ )
		XtSetSensitive(UxGetWidget(pushButton_Plot), TRUE);
	else
		XtSetSensitive(UxGetWidget(pushButton_Plot), FALSE);

        if ( SF_ptr->a_plot_exists && plot_type && (geometry == SF_ptr->geometry_last_plotted) )
		XtSetSensitive(UxGetWidget(pushButton_Overlay), TRUE);
	else
		XtSetSensitive(UxGetWidget(pushButton_Overlay), FALSE);

}


static void DisableIndPlotRadios( void )
{
	XtSetSensitive(UxGetWidget(toggleButton_Line), FALSE);
	XtSetSensitive(UxGetWidget(toggleButton_Scatter), FALSE);
	XtSetSensitive(UxGetWidget(toggleButton_Shade), FALSE);
	XtSetSensitive(UxGetWidget(toggleButton_Contour), FALSE);
	XtSetSensitive(UxGetWidget(toggleButton_Fill), FALSE);
	XtSetSensitive(UxGetWidget(toggleButton_Vector), FALSE);

	XtVaSetValues(UxGetWidget(toggleButton_Line),
		XmNset, FALSE,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton_Scatter),
		XmNset, FALSE,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton_Shade),
		XmNset, FALSE,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton_Contour),
		XmNset, FALSE,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton_Fill),
		XmNset, FALSE,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton_Vector),
		XmNset, FALSE,
		NULL);
}


int JC_PlotFrame_MaintainRadios( int geometry, int it_is_a_vector, JC_StateFlags *SF_ptr )
{
     int plot_type;

     /*
      * - First take care of the plot radios.
      */

     DisableIndPlotRadios();
     
     switch ( geometry ) {

	  /*
	   * - No plot options for points.
	   */
	case GEOM_POINT:
	  break;

	  /*
	   * - Plot options for 1-dimensional geometries.
	   */ 
	case GEOM_X:
	case GEOM_Y:
	case GEOM_Z:
	case GEOM_T:
	  XtSetSensitive(UxGetWidget(toggleButton_Line), TRUE);
	  XtVaSetValues(UxGetWidget(toggleButton_Line),
			XmNset, TRUE,
			NULL);
	  plot_type = PLOT_LINE;
	  break; 

	  /*
	   * - Plot options for 2-dimensional geometries.
	   * - if the variable is a vector
	   *     select VECTOR type plot
	   * - else
	   *     use the same plot type as last time
	   *     or select SHADE type if there is no last plot
	   *      
	   */
	case GEOM_XY:
	case GEOM_XZ:
	case GEOM_XT:
	case GEOM_YZ:
	case GEOM_YT:
	case GEOM_ZT:
	  if ( it_is_a_vector ) {
	       XtSetSensitive(UxGetWidget(toggleButton_Vector), TRUE);
	       XtVaSetValues(UxGetWidget(toggleButton_Vector),
			     XmNset, TRUE,
			     NULL);
	       plot_type = PLOT_VECTOR;
	  }
	  else {
	       XtSetSensitive(UxGetWidget(toggleButton_Shade), TRUE);
	       XtSetSensitive(UxGetWidget(toggleButton_Contour), TRUE);
	       XtSetSensitive(UxGetWidget(toggleButton_Fill), TRUE);

	       switch ( SF_ptr->plot_type_last_plotted ) {
		  case PLOT_NONE:
		  case PLOT_LINE:
		  case PLOT_SCATTER:
		  case PLOT_SHADE:
		  case PLOT_VECTOR:
		    XtVaSetValues(UxGetWidget(toggleButton_Shade),
				  XmNset, TRUE,
				  NULL);
		    plot_type = PLOT_SHADE;
		    break;
		  case PLOT_CONTOUR:
		    XtVaSetValues(UxGetWidget(toggleButton_Contour),
				  XmNset, TRUE,
				  NULL);
		    plot_type = PLOT_CONTOUR;
		    break;
		  case PLOT_FILL:
		    XtVaSetValues(UxGetWidget(toggleButton_Fill),
				  XmNset, TRUE,
				  NULL);
		    plot_type = PLOT_FILL;
		    break;
		  default:
		    fprintf(stderr, "ERROR in main.c: JC_PlotFrame_MaintainRadios(): plot_type = %d\n", plot_type);
		    break;
	       }
	  }
	  break; 

	  /*
	   * - Plot options for 3- and 4-dimensional geometries.
	   */
	case GEOM_XYZ:
	case GEOM_XYT:
	case GEOM_XZT:
	case GEOM_YZT:
	case GEOM_XYZT:
	  break; 

	default:
	  fprintf(stderr, "ERROR in main.c: JC_PlotFrame_MaintainRadios(): geometry = %d\n", geometry);
	  break;
     }

     return plot_type;

}


void InitFerretStructs()
{
     /*
      * - Allocate the shared buffer.
      * - Allocate the macro manager buffer.
      * - Initialize the macro buffer.
      */

     sBuffer = (sharedMem *)malloc(sizeof(sharedMem));
     macroBuffer = (char *)malloc(32000);
     strcpy(macroBuffer, "");

     GLOBAL_PlotOptions.plot_type = PLOT_SHADE;
     GLOBAL_PlotOptions.twoD_options.key = TRUE;
}

void SetInitialState(void)
{
  XmString menuLabel;
     
  /* initial state of menubar */
     
  /* macro menu */
  gMacroIsRecording = 1;
  menuLabel = XmStringCreate("Stop Recording", XmFONTLIST_DEFAULT_TAG);
  XtVaSetValues(UxGetWidget(Start_recording2),
		XmNlabelString, menuLabel,
		NULL);
     
  XmStringFree(menuLabel);
     
  memset(&GLOBAL_StateFlags, 0, sizeof(JC_StateFlags));
     
  if ( (GLOBAL_DatasetList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_DatasetList.\n");
  if ( (GLOBAL_DatasetNameList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_DatasetList.\n");
  if ( (GLOBAL_GlobalVariableList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_GlobalVariableList.\n");
  if ( (GLOBAL_GridList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_GridList.\n");
  if ( (GLOBAL_ViewportList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_ViewportList.\n");
  if ( (GLOBAL_WindowList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_WindowList.\n");
  if ( (GLOBAL_PlottedDataList = list_init()) == NULL )
    fprintf(stderr, "ERROR in main.c: SetInitialState(): Unable to initialize GLOBAL_PlottedDataList.\n");
     
  /*
   * - Initialize the regridding structure.
   * - Initialize the region structure.
   */
  JC_Regridding_Initialize(&GLOBAL_Regridding);
  JC_Region_Initialize(&GLOBAL_Region);
 
  /*
   * - Set up the menu bar.
   */
  XmToggleButtonSetState(UxGetWidget(IncludeHours_Button), TRUE, FALSE);
  GLOBAL_StateFlags.time_resolution_includes_hours = TRUE;
  XmToggleButtonSetState(UxGetWidget(ShowMap_Button), TRUE, FALSE);
  MaintainMainMenu();
  JC_II_MainMenu_Maintain(&GLOBAL_StateFlags);
     
  dataSetMenus = XmCreateMenuBar(UxGetWidget(rowColumn_Select), "menubar0", NULL, 0);

  /*
   * - Set up the "Select" menus.
   */
  JC_main_SelectMenu_Build(&dataSetMenus, JC_Main_SelectMenuButton_CB, JC_Main_SelectMenuButton_CB, JC_Main_SelectMenu_CloneButton_CB);
  XtUnmanageChild(UxGetWidget(rowColumn_Select));
  XtManageChild(UxGetWidget(rowColumn_Select));
  XtManageChild(UxGetWidget(dataSetMenus));
     
  /*
   * - Initialize the Data and Context Frames.
   * - Show the map.
   * - Set the initial modes for Ferret.
   */
  JC_DatasetNameList_Initialize();
  JC_DataFrame_Initialize();
  JC_ContextFrame_Initialize();
  JC_Map_Show();
  ferret_command("SET MODE INTERPOLATE", DONT_UPDATE_MM);

  /*
   * Create dialogs which use Popup and Popdown to control their visibility.
   */
  ErrorLog = create_ErrorLog(NO_PARENT);

}


static void JC_DataFrame_Initialize(void)
{
     XtSetSensitive(UxGetWidget(rowColumn_Select), FALSE);
     XtSetSensitive(UxGetWidget(rowColumn_Data), FALSE);
     XtVaSetValues(label_DataFrameStatus,
		   RES_CONVERT(XmNlabelString,  ""),
		   NULL);
}

void JC_ContextFrame_Initialize(void)
{
  int xyzt=0, lo_hi_pt=0;
  char tempText[MAX_NAME_LENGTH]="";
     
  for (xyzt=0; xyzt<4; xyzt++) {
    XtSetSensitive(axssww[xyzt], FALSE);
  }
     
  XtVaSetValues(label_RegriddingStatus,
		RES_CONVERT(XmNlabelString,  ""),
		NULL);
     
  XtUnmapWidget(UxGetWidget(toggleButton_X));
  XtUnmapWidget(UxGetWidget(toggleButton_Y));
  XtUnmapWidget(UxGetWidget(toggleButton_Z));
  XtUnmapWidget(UxGetWidget(toggleButton_T));

  XtSetSensitive(UxGetWidget(toggleButton_Regridding), TRUE);

  XtSetSensitive(UxGetWidget(frame_context), FALSE);
  XtSetSensitive(UxGetWidget(frame_plot), FALSE);
  XtSetSensitive(UxGetWidget(frame_map), FALSE);
  XtSetSensitive(UxGetWidget(optionMenu_Geometry), FALSE);

  XtUnmapWidget(UxGetWidget(frame_context));
}

void JC_DatasetNameList_Initialize(void)
{
     FILE *file_ptr=NULL;
     XmString motif_string;

     char *char_ptr=NULL, tempText[MAX_NAME_LENGTH]="", dset[MAX_NAME_LENGTH]="", cmd[MAX_NAME_LENGTH]="";
     char paths[8192]="";
     int i=0, status=LIST_OK;
     LIST *dsetList=NULL;

     strcpy(tempText, "\0");
     list_insert_after(GLOBAL_DatasetNameList, tempText, MAX_NAME_LENGTH);

/*
 * - Get all the paths from the "FER_DESCR" environment variable.
 *
 * - While there is another path:
 *    - get the path;
 *    - create a pipe for the "ls -1" command;
 *    - read stdout and put each file name in the dsText array;
 *
 *    - make sure the dataset ends in ".des";
 *    - add the dataset (minus '.des') to the list;
 *      
 */

     sprintf(paths, "%s", getenv("FER_DESCR"));
     
     char_ptr = strtok(paths, " \t");
     
     if ( char_ptr == NULL ) {

	  fprintf(stderr, "\
WARNING in main.c: DatasetNameListInitialize(): No paths were found in the environment variable FER_DESCR\n");

     } else {
	  
	  do {
	       sprintf(cmd, "ls -1 %s", char_ptr);
	       
	       if ( (file_ptr = popen(cmd, "r")) == (FILE *) NULL ) {
		    fprintf(stderr, "ERROR in main.c: DatasetMameList_Initialize(): Cannot open pipe.\n");
		    return;
	       }
	       
	       while ( fgets(tempText, MAX_NAME_LENGTH, file_ptr) != NULL ) {

		    JC_String_RemoveWhiteSpace(tempText);
		    if ( (char_ptr = strrchr(tempText, '/')) == NULL )
			 strcpy(dset, tempText);
		    else
			 strcpy(dset, char_ptr+1);
		    /*
		    if (JC_String_EndsWithTag(dset, ".des")) 
			 dset[strlen(dset)-4] = '\0';
		    else
			 continue;
			 */
		    if (JC_String_EndsWithTag(dset, ".des")) {
		      list_traverse(GLOBAL_DatasetNameList, dset, JC_ListTraverse_Sort, (LIST_FRNT | LIST_FORW | LIST_ALTR));
		      list_insert_before(GLOBAL_DatasetNameList, dset, sizeof(dset));
		    }
	       }
	  
	       pclose(file_ptr);

	       char_ptr = strtok(NULL, " \t");

	  } while ( char_ptr != NULL );
	  
     }

     sprintf(paths, "%s", getenv("FER_DATA"));
     
     char_ptr = strtok(paths, " \t");

     if ( char_ptr == NULL ) {

	  fprintf(stderr, "\
WARNING in main.c: DatasetNameList_Initialize(): No paths were found in the environment variable FER_DATA\n");

     } else {
	  
	  do {

	       sprintf(cmd, "ls -1 %s", char_ptr);

	       if ( (file_ptr = popen(cmd, "r")) == (FILE *) NULL ) {
		    fprintf(stderr, "ERROR in main.c: DatasetNameList_Initialize(): Cannot open pipe.\n");
		    return;
	       }

	       while ( fgets(tempText, MAX_NAME_LENGTH, file_ptr) != NULL ) {

		    JC_String_RemoveWhiteSpace(tempText);
		    if ( (char_ptr = strrchr(tempText, '/')) == NULL )
			 strcpy(dset, tempText);
		    else
			 strcpy(dset, char_ptr+1);
		    /*		    
		    if ( JC_String_EndsWithTag(dset, ".cdf") ) 
			 dset[strlen(dset)-4] = '\0';
		    else if ( JC_String_EndsWithTag(dset, ".nc") )
			 dset[strlen(dset)-3] = '\0';
		    else
			 continue;
			 */		    
		    if ( JC_String_EndsWithTag(dset, ".des") || JC_String_EndsWithTag(dset, ".cdf") 
			 || JC_String_EndsWithTag(dset, ".nc")) {
		      list_traverse(GLOBAL_DatasetNameList, dset, JC_ListTraverse_Sort, (LIST_FRNT | LIST_FORW | LIST_ALTR));
		      list_insert_before(GLOBAL_DatasetNameList, dset, sizeof(dset));
		    }
	       }
	  
	       pclose(file_ptr);

	       char_ptr = strtok(NULL, " \t");

	  } while ( char_ptr != NULL );
	  
     }
     
     list_remove_rear(GLOBAL_DatasetNameList);
}


void CancelInitialState()
{
  XtUnmapWidget(UxGetWidget(StartupMessage));
  XtMapWidget(UxGetWidget(frame_context));

  /* undim the region and transformation Frames */
  XtSetSensitive(UxGetWidget(frame_context), TRUE); /* region */
  XtSetSensitive(UxGetWidget(frame_plot), TRUE); /* plot types */
  XtSetSensitive(UxGetWidget(frame_map), TRUE); /* map */
  XtSetSensitive(UxGetWidget(rowColumn_Select), TRUE);
  XtSetSensitive(UxGetWidget(rowColumn_Data), TRUE);	/* variable */
     
  /* view stuff */
  XtSetSensitive(UxGetWidget(optionMenu_Geometry), TRUE);
}

void JC_MainMenu_SolidLand_CB( void )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Object Obj;
     float x_range=0, y_range=0, area=0;
     int resolution;
     char command[MAX_COMMAND_LENGTH]="";

     Obj.variable = GLOBAL_Variable;
     Obj.regridding = GLOBAL_Regridding;
     Obj.region = GLOBAL_Region;

     x_range = R_ptr->span[X_AXIS].ww[HI] - R_ptr->span[X_AXIS].ww[LO];
     y_range = R_ptr->span[Y_AXIS].ww[HI] - R_ptr->span[Y_AXIS].ww[LO];
     area = x_range * y_range;
     area = (area>=0) ? area : -area;
     
     if ( area > 7500 )
	  resolution = 60;
     else if ( area > 7500/9 )
	  resolution = 40;
     else if ( area > 7500/36 )
	  resolution = 20;
     else
	  resolution = 5;
     
     JC_SolidLandCommand_Create(command, &Obj, resolution);
     ferret_command(command, IGNORE_COMMAND_WIDGET);

}

void MaintainMainMenu()
{
	XmString startLabel, endLabel;

	startLabel = XmStringCreate("Start Recording", XmFONTLIST_DEFAULT_TAG);
	endLabel = XmStringCreate("Stop Recording", XmFONTLIST_DEFAULT_TAG);

	if (gMacroIsRecording) {
		/* recording--stop */
		XtVaSetValues(UxGetWidget(Start_recording2),
			XmNlabelString, endLabel,
			NULL);
	}
	else {
		/* not recording--start*/
		XtVaSetValues(UxGetWidget(Start_recording2),
			XmNlabelString, startLabel,
			NULL);
	}
	
	XmStringFree(startLabel);
	XmStringFree(endLabel);
}

void quit_cb( Widget wid, XtPointer client_data, XtPointer call_data)
{
     DoQuit();
}

void DoQuit( void )
{
     ferret_command("QUIT", UPDATE_COMMAND_WIDGET );
}


void JC_XYZTTextField_Verify_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     
     char *text=XmTextFieldGetString(wid); /* You need to free the memory allocated here. */
     char tempText[MAX_COMMAND_LENGTH], extra[MAX_NAME_LENGTH];
     float value=0.0;
     int xyzt=-1, lo_hi_pt=-1, index=0;
     int val=0, ss=0, inc=0, pinc=0;
     
     if ( ((char *)client_data)[0] == 'x' ) xyzt = X_AXIS;
     else if ( ((char *)client_data)[0] == 'y' ) xyzt = Y_AXIS;
     else if ( ((char *)client_data)[0] == 'z' ) xyzt = Z_AXIS;
     else if ( ((char *)client_data)[0] == 't' ) xyzt = T_AXIS;
     else {
	  fprintf(stderr, "main.c: XYZTTextField_Verify_CB(): client_data[0] = \"%s\"\n", (char *)client_data);
	  return;
     }
     
     if ( ((char *)client_data)[2] == 'l' ) lo_hi_pt = LO;
     else if ( ((char *)client_data)[2] == 'h' ) lo_hi_pt = HI;
     else if ( ((char *)client_data)[2] == 'p' ) lo_hi_pt = PT;
     else {
	  fprintf(stderr, "main.c: XYZTTextField_Verify_CB(): client_data[2] = \"%s\"\n", (char *)client_data);
	  return;
     }
     
/*
 * - If the text is being entered by index:
 *
 *    - read the text.
 *    - if it is outside the acceptable range:
 *         send a message and restore the value
 *    - else:
 *         let the scrollbar callbacks do all the work. (XmScrollBarSetValues(..., TRUE))
 *
 *   (NB_ the values for the slider are (0, N-1) whereas the textFields display (1, N))
 *
 */
  
     if ( R_ptr->span[xyzt].by_index_in_GUI ) {

	  sscanf(text, "%d%s", &index, extra);
	  if ( index < 1 || index > V_ptr->axis[xyzt].num_points ) {
	       sprintf(tempText, "%d is outside the range of this axis for %s", index, V_ptr->name);
	       JC_Message_CB(wid, tempText, NULL);
	       JC_textField_SetValue(textField_widget[xyzt][lo_hi_pt], lo_hi_pt, &(R_ptr->span[xyzt]));
	  } else {
	       XmScrollBarGetValues(scrollBar_widget[xyzt][lo_hi_pt], &val, &ss, &inc, &pinc);
	       XmScrollBarSetValues(scrollBar_widget[xyzt][lo_hi_pt], index-1, ss, inc, pinc, TRUE);
	  }

/*
 * - Else the text is being entered in world coordinates:
 *
 *    - convert the text to a float.
 *    - if the string is uninterpretable OR the value is outside the acceptable values:
 *         send a message and restore the value
 *    - else:
 *         set the 'ww' value
 *         get the associated index value
 *
 *       - if we have overtaken the value of the other scrollbar
 *            update the other scrollBar
 *
 *       - update that element in the geometry interface
 *       - update the Map
 *
 *       - if this is a clone: update the cloned variable's region
 *         
 */

     } else /* we are using world coordinates */ {
	  
	  if ( (value = JC_String_ConvertToFloat(text, &(R_ptr->span[xyzt]))) == INTERNAL_ERROR ) {

	       JC_textField_SetValue(textField_widget[xyzt][lo_hi_pt], lo_hi_pt, &(R_ptr->span[xyzt]));
	       sprintf(tempText, "Cannot interpret \"%s\" as a world coordinate.\nValue being reset.", text);
	       JC_Message_CB(wid, tempText, NULL);

	  } else if ( value < V_ptr->axis[xyzt].ww[LO] || value > V_ptr->axis[xyzt].ww[HI]) {

	       JC_textField_SetValue(textField_widget[xyzt][lo_hi_pt], lo_hi_pt, &(R_ptr->span[xyzt]));
	       sprintf(tempText, "%f is outside the range of this axis for %s.\nValue being reset.", value, V_ptr->name);
	       JC_Message_CB(wid, tempText, NULL);

	  } else {

	       R_ptr->span[xyzt].ww[lo_hi_pt] = value;
	       R_ptr->span[xyzt].ss[lo_hi_pt] = JC_Axis_ReturnIndex(&(V_ptr->axis[xyzt]), value);
	       
	       if ( lo_hi_pt == LO && R_ptr->span[xyzt].ww[LO] > R_ptr->span[xyzt].ww[HI] ) {
		    R_ptr->span[xyzt].ss[HI] = R_ptr->span[xyzt].ss[LO];
		    R_ptr->span[xyzt].ww[HI] = R_ptr->span[xyzt].ww[LO];
		    JC_scrollBar_SetValue(scrollBar_widget[xyzt][HI], R_ptr->span[xyzt].ss[HI]);
		    JC_textField_SetValue(textField_widget[xyzt][HI], lo_hi_pt, &(R_ptr->span[xyzt]));
	       } else if ( lo_hi_pt == HI && R_ptr->span[xyzt].ww[HI] < R_ptr->span[xyzt].ww[LO] ) {
		    R_ptr->span[xyzt].ss[LO] = R_ptr->span[xyzt].ss[HI];
		    R_ptr->span[xyzt].ww[LO] = R_ptr->span[xyzt].ww[HI];
		    JC_scrollBar_SetValue(scrollBar_widget[xyzt][LO], R_ptr->span[xyzt].ss[LO]);
		    JC_textField_SetValue(textField_widget[xyzt][LO], lo_hi_pt, &(R_ptr->span[xyzt]));
	       }	     
	       
	       JC_scrollBar_SetValue(scrollBar_widget[xyzt][lo_hi_pt], R_ptr->span[xyzt].ss[lo_hi_pt]);
	       JC_textField_SetValue(textField_widget[xyzt][lo_hi_pt], lo_hi_pt, &(R_ptr->span[xyzt]));
	       
	       JC_Map_NewRegion(R_ptr);
	       
	       if ( SF_ptr->a_clone_is_selected )
		    (SF_ptr->current_clone_ptr)->region = *R_ptr;
	  }
	  
     }
     
/*
 * - Free the memory allocated with XmTextFieldGetString().
 */
     XtFree(text);
}


void JC_main_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() )
{
     JC_MenuItem items[MAX_MENU_ITEMS]={ NULL, };

     JC_Menu_AddDsetVars(items, var_fn_ptr, dvar_fn_ptr, cvar_fn_ptr);

     JC_Menu_Build(*menubar, XmMENU_PULLDOWN, "Select", NULL, FALSE, items);

}


void JC_Main_SelectMenuButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     /*
      * - Get the dataset name.
      * - Get the variable name.
      */

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     SF_ptr->current_clone_ptr = NULL;
     SF_ptr->a_clone_is_selected = FALSE;

     JC_MainInterface_NewVariable( var_name, dset_name, NEW_VARIABLE );

     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void JC_Main_SelectMenu_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     /*
      * - Get the dataset name.
      * - Get the variable name.
      */

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     SF_ptr->current_clone_ptr = JC_Clone_ReturnPointer(var_name, dset_name);
     SF_ptr->a_clone_is_selected = TRUE;

     JC_MainInterface_NewVariable(var_name, dset_name, NEW_VARIABLE);

     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void InitGlobalWidgets()
{
     toggleButton_fixed[X_AXIS] = UxGetWidget(toggleButton_X);
     toggleButton_fixed[Y_AXIS] = UxGetWidget(toggleButton_Y);
     toggleButton_fixed[Z_AXIS] = UxGetWidget(toggleButton_Z);
     toggleButton_fixed[T_AXIS] = UxGetWidget(toggleButton_T);

	axssww[0] = UxGetWidget(optionMenu_X);
	axssww[1] = UxGetWidget(optionMenu_Y);
	axssww[2] = UxGetWidget(optionMenu_Z);
	axssww[3] = UxGetWidget(optionMenu_T);
	axssww[4] = UxGetWidget(optionMenu_T);

	cxByWW[0] = UxGetWidget(optionMenu_Xp_Longitude);
	cxByWW[1] = UxGetWidget(optionMenu_Yp_Latitude);
	cxByWW[2] = UxGetWidget(optionMenu_Zp_Depth);
	cxByWW[3] = UxGetWidget(optionMenu_Tp_Calendar);
	cxByWW[4] = UxGetWidget(optionMenu_Tp_Calendar);

	cxBySS[0] = UxGetWidget(optionMenu_Xp_Index);
	cxBySS[1] = UxGetWidget(optionMenu_Yp_Index);
	cxBySS[2] = UxGetWidget(optionMenu_Zp_Index);
	cxBySS[3] = UxGetWidget(optionMenu_Tp_Index);
	cxBySS[4] = UxGetWidget(optionMenu_Tp_Index);

	textField_widget[X_AXIS][LO] = UxGetWidget(textField_X_LO);
	textField_widget[Y_AXIS][LO] = UxGetWidget(textField_Y_LO);
	textField_widget[Z_AXIS][LO] = UxGetWidget(textField_Z_LO);
	textField_widget[T_AXIS][LO] = UxGetWidget(textField_T_LO);
 
	textField_widget[X_AXIS][HI] = UxGetWidget(textField_X_HI);
	textField_widget[Y_AXIS][HI] = UxGetWidget(textField_Y_HI);
	textField_widget[Z_AXIS][HI] = UxGetWidget(textField_Z_HI);
	textField_widget[T_AXIS][HI] = UxGetWidget(textField_T_HI);
 
	textField_widget[X_AXIS][PT] = UxGetWidget(textField_X_PT);
	textField_widget[Y_AXIS][PT] = UxGetWidget(textField_Y_PT);
	textField_widget[Z_AXIS][PT] = UxGetWidget(textField_Z_PT);
	textField_widget[T_AXIS][PT] = UxGetWidget(textField_T_PT);
 
	axtrans[0] = UxGetWidget(optionMenu_X_TRANS);
	axtrans[1] = UxGetWidget(optionMenu_Y_TRANS);
	axtrans[2] = UxGetWidget(optionMenu_Z_TRANS);
	axtrans[3] = UxGetWidget(optionMenu_T_TRANS);
	axtrans[4] = UxGetWidget(optionMenu_T_TRANS);

	axarg[0] = UxGetWidget(textField_X_ARG);
	axarg[1] = UxGetWidget(textField_Y_ARG);
	axarg[2] = UxGetWidget(textField_Z_ARG);
	axarg[3] = UxGetWidget(textField_T_ARG);
	axarg[4] = UxGetWidget(textField_T_ARG);

	axtrnButton[0][0] = UxGetWidget(optionMenu_p_b8);
	axtrnButton[0][1] = UxGetWidget(optionMenu_2_b91);
	axtrnButton[0][2] = UxGetWidget(optionMenu_2_b92);
	axtrnButton[0][3] = UxGetWidget(optionMenu_2_b93);
	axtrnButton[0][4] = UxGetWidget(optionMenu_2_b94);
	axtrnButton[0][5] = UxGetWidget(optionMenu_2_b95);
	axtrnButton[0][6] = UxGetWidget(optionMenu_2_b96);
	axtrnButton[0][7] = UxGetWidget(optionMenu_2_b97);
	axtrnButton[0][8] = UxGetWidget(optionMenu_2_b98);
	axtrnButton[0][9] = UxGetWidget(optionMenu_2_b99);
	axtrnButton[0][10] = UxGetWidget(optionMenu_2_b100);
	axtrnButton[0][11] = UxGetWidget(optionMenu_2_b101);
	axtrnButton[0][12] = UxGetWidget(optionMenu_2_b102);
	axtrnButton[0][13] = UxGetWidget(optionMenu_2_b103);
	axtrnButton[0][14] = UxGetWidget(optionMenu_2_b104);
	axtrnButton[0][15] = UxGetWidget(optionMenu_2_b105);
	axtrnButton[0][16] = UxGetWidget(optionMenu_2_b106);
	axtrnButton[0][17] = UxGetWidget(optionMenu_2_b107);
	axtrnButton[0][18] = UxGetWidget(optionMenu_2_b108);
	axtrnButton[0][19] = UxGetWidget(optionMenu_2_b109);
	axtrnButton[0][20] = UxGetWidget(optionMenu_2_b110);
	axtrnButton[0][21] = UxGetWidget(optionMenu_2_b111);
	axtrnButton[0][22] = UxGetWidget(optionMenu_2_b112);
	axtrnButton[0][23] = UxGetWidget(optionMenu_2_b113);
	axtrnButton[0][24] = UxGetWidget(optionMenu_2_b114);

	axtrnButton[1][0] = UxGetWidget(optionMenu_p_b9);
	axtrnButton[1][1] = UxGetWidget(optionMenu_2_b1);
	axtrnButton[1][2] = UxGetWidget(optionMenu_2_b6);
	axtrnButton[1][3] = UxGetWidget(optionMenu_2_b25);
	axtrnButton[1][4] = UxGetWidget(optionMenu_2_b26);
	axtrnButton[1][5] = UxGetWidget(optionMenu_2_b27);
	axtrnButton[1][6] = UxGetWidget(optionMenu_2_b28);
	axtrnButton[1][7] = UxGetWidget(optionMenu_2_b29);
	axtrnButton[1][8] = UxGetWidget(optionMenu_2_b30);
	axtrnButton[1][9] = UxGetWidget(optionMenu_2_b31);
	axtrnButton[1][10] = UxGetWidget(optionMenu_2_b32);
	axtrnButton[1][11] = UxGetWidget(optionMenu_2_b33);
	axtrnButton[1][12] = UxGetWidget(optionMenu_2_b34);
	axtrnButton[1][13] = UxGetWidget(optionMenu_2_b35);
	axtrnButton[1][14] = UxGetWidget(optionMenu_2_b36);
	axtrnButton[1][15] = UxGetWidget(optionMenu_2_b37);
	axtrnButton[1][16] = UxGetWidget(optionMenu_2_b38);
	axtrnButton[1][17] = UxGetWidget(optionMenu_2_b39);
	axtrnButton[1][18] = UxGetWidget(optionMenu_2_b40);
	axtrnButton[1][19] = UxGetWidget(optionMenu_2_b41);
	axtrnButton[1][20] = UxGetWidget(optionMenu_2_b1);
	axtrnButton[1][21] = UxGetWidget(optionMenu_2_b42);
	axtrnButton[1][22] = UxGetWidget(optionMenu_2_b43);
	axtrnButton[1][23] = UxGetWidget(optionMenu_2_b44);
	axtrnButton[1][24] = UxGetWidget(optionMenu_2_b2);

	axtrnButton[2][0] = UxGetWidget(optionMenu_p_b2);
	axtrnButton[2][1] = UxGetWidget(optionMenu_2_b2);
	axtrnButton[2][2] = UxGetWidget(optionMenu_2_b3);
	axtrnButton[2][3] = UxGetWidget(optionMenu_2_b4);
	axtrnButton[2][4] = UxGetWidget(optionMenu_2_b5);
	axtrnButton[2][5] = UxGetWidget(optionMenu_2_b7);
	axtrnButton[2][6] = UxGetWidget(optionMenu_2_b8);
	axtrnButton[2][7] = UxGetWidget(optionMenu_2_b9);
	axtrnButton[2][8] = UxGetWidget(optionMenu_2_b10);
	axtrnButton[2][9] = UxGetWidget(optionMenu_2_b11);
	axtrnButton[2][10] = UxGetWidget(optionMenu_2_b12);
	axtrnButton[2][11] = UxGetWidget(optionMenu_2_b13);
	axtrnButton[2][12] = UxGetWidget(optionMenu_2_b14);
	axtrnButton[2][13] = UxGetWidget(optionMenu_2_b15);
	axtrnButton[2][14] = UxGetWidget(optionMenu_2_b16);
	axtrnButton[2][15] = UxGetWidget(optionMenu_2_b17);
	axtrnButton[2][16] = UxGetWidget(optionMenu_2_b18);
	axtrnButton[2][17] = UxGetWidget(optionMenu_2_b19);
	axtrnButton[2][18] = UxGetWidget(optionMenu_2_b20);
	axtrnButton[2][19] = UxGetWidget(optionMenu_2_b21);
	axtrnButton[2][20] = UxGetWidget(optionMenu_3_b1);
	axtrnButton[2][21] = UxGetWidget(optionMenu_2_b22);
	axtrnButton[2][22] = UxGetWidget(optionMenu_2_b23);
	axtrnButton[2][23] = UxGetWidget(optionMenu_2_b24);
	axtrnButton[2][24] = UxGetWidget(optionMenu_3_b2);

	axtrnButton[3][0] = UxGetWidget(optionMenu_p_b12);
	axtrnButton[3][1] = UxGetWidget(optionMenu_2_b45);
	axtrnButton[3][2] = UxGetWidget(optionMenu_2_b46);
	axtrnButton[3][3] = UxGetWidget(optionMenu_2_b47);
	axtrnButton[3][4] = UxGetWidget(optionMenu_2_b48);
	axtrnButton[3][5] = UxGetWidget(optionMenu_2_b49);
	axtrnButton[3][6] = UxGetWidget(optionMenu_2_b50);
	axtrnButton[3][7] = UxGetWidget(optionMenu_2_b51);
	axtrnButton[3][8] = UxGetWidget(optionMenu_2_b52);
	axtrnButton[3][9] = UxGetWidget(optionMenu_2_b53);
	axtrnButton[3][10] = UxGetWidget(optionMenu_2_b54);
	axtrnButton[3][11] = UxGetWidget(optionMenu_2_b55);
	axtrnButton[3][12] = UxGetWidget(optionMenu_2_b56);
	axtrnButton[3][13] = UxGetWidget(optionMenu_2_b57);
	axtrnButton[3][14] = UxGetWidget(optionMenu_2_b58);
	axtrnButton[3][15] = UxGetWidget(optionMenu_2_b59);
	axtrnButton[3][16] = UxGetWidget(optionMenu_2_b60);
	axtrnButton[3][17] = UxGetWidget(optionMenu_2_b61);
	axtrnButton[3][18] = UxGetWidget(optionMenu_2_b62);
	axtrnButton[3][19] = UxGetWidget(optionMenu_2_b63);
	axtrnButton[3][20] = UxGetWidget(optionMenu_4_b1);
	axtrnButton[3][21] = UxGetWidget(optionMenu_2_b64);
	axtrnButton[3][22] = UxGetWidget(optionMenu_2_b65);
	axtrnButton[3][23] = UxGetWidget(optionMenu_2_b66);
	axtrnButton[3][24] = UxGetWidget(optionMenu_2_b2);

	axtrnButton[4][0] = UxGetWidget(optionMenu_p_b12);
	axtrnButton[4][1] = UxGetWidget(optionMenu_2_b45);
	axtrnButton[4][2] = UxGetWidget(optionMenu_2_b46);
	axtrnButton[4][3] = UxGetWidget(optionMenu_2_b47);
	axtrnButton[4][4] = UxGetWidget(optionMenu_2_b48);
	axtrnButton[4][5] = UxGetWidget(optionMenu_2_b49);
	axtrnButton[4][6] = UxGetWidget(optionMenu_2_b50);
	axtrnButton[4][7] = UxGetWidget(optionMenu_2_b51);
	axtrnButton[4][8] = UxGetWidget(optionMenu_2_b52);
	axtrnButton[4][9] = UxGetWidget(optionMenu_2_b53);
	axtrnButton[4][10] = UxGetWidget(optionMenu_2_b54);
	axtrnButton[4][11] = UxGetWidget(optionMenu_2_b55);
	axtrnButton[4][12] = UxGetWidget(optionMenu_2_b56);
	axtrnButton[4][13] = UxGetWidget(optionMenu_2_b57);
	axtrnButton[4][14] = UxGetWidget(optionMenu_2_b58);
	axtrnButton[4][15] = UxGetWidget(optionMenu_2_b59);
	axtrnButton[4][16] = UxGetWidget(optionMenu_2_b60);
	axtrnButton[4][17] = UxGetWidget(optionMenu_2_b61);
	axtrnButton[4][18] = UxGetWidget(optionMenu_2_b62);
	axtrnButton[4][19] = UxGetWidget(optionMenu_2_b63);
	axtrnButton[4][20] = UxGetWidget(optionMenu_4_b1);
	axtrnButton[4][21] = UxGetWidget(optionMenu_2_b64);
	axtrnButton[4][22] = UxGetWidget(optionMenu_2_b65);
	axtrnButton[4][23] = UxGetWidget(optionMenu_2_b66);
	axtrnButton[4][24] = UxGetWidget(optionMenu_2_b2);

	scrollBar_widget[X_AXIS][LO] = UxGetWidget(scrollBar_X_LO);
	scrollBar_widget[Y_AXIS][LO] = UxGetWidget(scrollBar_Y_LO);
	scrollBar_widget[Z_AXIS][LO] = UxGetWidget(scrollBar_Z_LO);
	scrollBar_widget[T_AXIS][LO] = UxGetWidget(scrollBar_T_LO);

	scrollBar_widget[X_AXIS][HI] = UxGetWidget(scrollBar_X_HI);
	scrollBar_widget[Y_AXIS][HI] = UxGetWidget(scrollBar_Y_HI);
	scrollBar_widget[Z_AXIS][HI] = UxGetWidget(scrollBar_Z_HI);
	scrollBar_widget[T_AXIS][HI] = UxGetWidget(scrollBar_T_HI);

	scrollBar_widget[X_AXIS][PT] = UxGetWidget(scrollBar_X_PT);
	scrollBar_widget[Y_AXIS][PT] = UxGetWidget(scrollBar_Y_PT);
	scrollBar_widget[Z_AXIS][PT] = UxGetWidget(scrollBar_Z_PT);
	scrollBar_widget[T_AXIS][PT] = UxGetWidget(scrollBar_T_PT);

	lowScroll[0] = UxGetWidget(scrollBar_X_LO);
	lowScroll[1] = UxGetWidget(scrollBar_Y_LO);
	lowScroll[2] = UxGetWidget(scrollBar_Z_LO);
	lowScroll[3] = UxGetWidget(scrollBar_T_LO);
	lowScroll[4] = UxGetWidget(scrollBar_T_LO);

	hiScroll[0] = UxGetWidget(scrollBar_X_HI);
	hiScroll[1] = UxGetWidget(scrollBar_Y_HI);
	hiScroll[2] = UxGetWidget(scrollBar_Z_HI);
	hiScroll[3] = UxGetWidget(scrollBar_T_HI);
	hiScroll[4] = UxGetWidget(scrollBar_T_HI);

	ptScroll[0] = UxGetWidget(scrollBar_X_LO);
	ptScroll[1] = UxGetWidget(scrollBar_Y_LO);
	ptScroll[2] = UxGetWidget(scrollBar_Z_LO);
	ptScroll[3] = UxGetWidget(scrollBar_T_LO);
	ptScroll[4] = UxGetWidget(scrollBar_T_LO);

	geomOptPBs[0] = UxGetWidget(optionMenu_p_b4);
	geomOptPBs[1] = UxGetWidget(optionMenu_p4_b2);
	geomOptPBs[2] = UxGetWidget(optionMenu_p4_b3);
	geomOptPBs[3] = UxGetWidget(optionMenu_p4_b4);
	geomOptPBs[4] = UxGetWidget(optionMenu_p4_b5);
	geomOptPBs[5] = UxGetWidget(optionMenu_p4_b6);
	geomOptPBs[6] = UxGetWidget(optionMenu_p4_b7);
	geomOptPBs[7] = UxGetWidget(optionMenu_p4_b8);
	geomOptPBs[8] = UxGetWidget(optionMenu_p4_b9);
	geomOptPBs[9] = UxGetWidget(optionMenu_p4_b10);
	geomOptPBs[10] = UxGetWidget(optionMenu_p4_b11);
	geomOptPBs[11] = UxGetWidget(optionMenu_p4_b12);
	geomOptPBs[12] = UxGetWidget(optionMenu_p4_b13);
	geomOptPBs[13] = UxGetWidget(optionMenu_p4_b14);
	geomOptPBs[14] = UxGetWidget(optionMenu_p4_b15);
	geomOptPBs[15] = UxGetWidget(optionMenu_p4_b16);
}


static void InitPixmaps()
{   
  struct stat buf;
  char *path, pixPath[320];
 
  path = (char *)getenv("FER_DIR");
  strcpy(pixPath, path);
  strcat(pixPath, "/gui/map_pn_final.xpm");

  /*
   * Error checking for availability of world map.
   */
  if ( stat(pixPath, &buf) == 0 ) {
    XtVaSetValues(UxGetWidget(drawingArea1),
		  XmNlabelType, XmPIXMAP,
		  XmNbackgroundPixmap, GetPixmapFromFile(pixPath),
		  NULL);
  } else {
    perror("\n\nFER_DIR/gui/map_pn_final.xpm");
    fprintf(stderr, "\
\n\
If the pixmap for the world map was not found, you need to install it.\n\
This pixmap and other appropriate support files are available in\n\
the latest Ferret patch available throuth the Ferret home page at\n\n\
\thttp://www.pmel.noaa.gov/ferret/\n\n\
Please report any other error messages to\n\n\
\tferret@pmel.noaa.gov\n\n");
    exit(1);
  }


    /*
	XtVaSetValues(UxGetWidget(drawingArea1),
		XmNbackgroundPixmap, GetPixmapFromData(map_pn_final_xpm),
		NULL);
		*/
	XtVaSetValues(UxGetWidget(DV_aVplusb),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(v1_op_num_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(DV_aUplusbV),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(v1_op_v2_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(DV_funcV),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(func_v1_xpm),
		NULL);


	XtVaSetValues(UxGetWidget(optionMenu_2_b91),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_x_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b1),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_y_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b2),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_z_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b45),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_t_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b92),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(var_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b6),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(var_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b3),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(var_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b46),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(var_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b93),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sum_x_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b25),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sum_y_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b4),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sum_z_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b47),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sum_t_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b94),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(rsum_x_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b26),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(rsum_y_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b5),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(rsum_z_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b48),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(rsum_t_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b95),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(shift_x_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b27),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(shift_y_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b7),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(shift_z_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b49),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(shift_t_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b99),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dx_for_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b100),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dx_bac_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b98),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dx_ctr_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b31),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dy_for_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b32),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dy_bac_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b30),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dy_ctr_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b11),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dz_for_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b12),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dz_bac_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b10),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dz_ctr_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b53),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dt_for_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b54),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dt_bac_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b52),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(dt_ctr_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b101),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_def_x_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b33),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_def_y_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b13),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_def_z_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b55),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_def_t_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b102),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_x_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b34),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_y_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b14),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_z_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b56),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(int_t_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b104),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(binomial_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b36),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(binomial_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b16),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(binomial_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b58),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(binomial_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b103),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(boxcar_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b35),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(boxcar_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b15),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(boxcar_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b57),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(boxcar_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b105),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(hanning_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b37),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(hanning_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b17),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(hanning_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b59),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(hanning_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b106),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(welch_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b38),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(welch_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b18),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(welch_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b60),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(welch_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b107),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(parzen_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b39),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(parzen_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b19),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(parzen_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b61),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(parzen_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b109),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_linear_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b41),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_linear_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b21),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_linear_xpm),
		NULL);
	XtVaSetValues(UxGetWidget(optionMenu_2_b63),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_linear_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b108),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_filled_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b40),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_filled_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b20),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_filled_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b62),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(ave_filled_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b110),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(nearest_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_1_b1),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(nearest_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_3_b1),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(nearest_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_4_b1),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(nearest_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_2_b114),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(weq_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_1_b2),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(weq_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_3_b2),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(weq_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(optionMenu_4_b2),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(weq_xpm),
		NULL);

}

