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
 * JC_InterInterface.c
 *
 * Jonathan Callahan
 * Jan 18'th 1996
 *
 * This file contains functions which allow interfaces in the Ferret GUI
 * to have effects on other interfaces.  It can be thought of as a central
 * hub where routing takes place.  It should be easier to maintain the
 * inter-interface connectivities if they are all contained in one file.
 *
 */


/* .................... Defines .................... */

enum { SAME_VARIABLE, NEW_VARIABLE, FIRST_EVER_VARIABLE } /* for JC_MainInterface_NewVariable() */
VARIABLE_type;


/* .................... Includes .................... */

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdio.h>
#include <Xm/Xm.h>

#include "UxXt.h"

#include "ferret.h"
#include "ferret_shared_buffer.h"
#include "ferret_structures.h"

#include "JC_Utility.h"
#include "FerretMainWd.h"

/* .................... Externals .................... */

extern JC_Region GLOBAL_Region;
extern JC_Variable GLOBAL_Variable;
extern JC_StateFlags GLOBAL_StateFlags;
extern JC_Regridding GLOBAL_Regridding;

extern Boolean gMetaCreationActive;

extern LIST *GLOBAL_DatasetList;

extern void JC_DatasetList_Free( LIST *this );
extern void JC_DatasetList_Print( LIST *this, FILE *File_ptr );
extern void JC_DatasetElement_QueryFerret( JC_DatasetElement *this, Boolean new_dataset );

extern int JC_ListTraverse_FoundMatch( char *data, char *curr );
extern int JC_ListTraverse_FoundDvarMatch( char *data, char *curr );
extern int JC_ListTraverse_FoundDsetMatch( char *data, char *curr );

extern char *CollectToReturn(char *targetStr, char *subStr);
extern int ferret_query(int query, smPtr sBuffer, char *tag,
                 char *arg1, char *arg2, char *arg3, char *arg4 );

/* For FerretMainWd */
extern void JC_MainInterface_NewVariable( char *var_name, char *dset_name, int variable_info );

/* For "Select" menus in FerretMainWd */
/*extern Widget rowColumn_Select;*/
extern Widget dataSetMenus;
extern void JC_main_SelectMenu_Build( Widget *menubar, void (*function_ptr)() );
extern void JC_Main_SelectMenuButton_CB( Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg );
extern void JC_Main_SelectMenu_CloneButton_CB( Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg );

/* For "Select" menus in JC_DefineVariable */
extern Widget JC_DefineVariable;
extern Widget JC_DV_SelectMenu1_F;
extern Widget JC_DV_SelectMenu2_F;
extern Widget JC_DV_SelectMenu3_F;
extern Widget JC_DV_SelectMenu1_LC;
extern Widget JC_DV_SelectMenu2_LC;
extern Widget JC_DV_SelectMenu1_EXP;
extern Widget rowColumn_Select2_LC;
extern Widget rowColumn_Select1_LC;
extern Widget rowColumn_Select1_EXP;
extern Widget rowColumn_Select1_F;
extern Widget rowColumn_Select_var2;
extern Widget rowColumn_Select_var3;
extern int GLOBAL_func_type;
extern void JC_DV_SelectMenu_Build( Widget *menubar, void (*function_ptr)() );
extern void JC_DV_SelectMenuButton_F1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuButton_F2_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuButton_F3_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuButton_LC1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuButton_LC2_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuButton_EXP1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuF1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuF2_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuF3_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuLC1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuLC2_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_DV_SelectMenuEXP1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );

/* For "Select" menus in JC_SelectRegridding */
extern Widget JC_SelectRegridding;
extern Widget rowColumn_Select_GX;
extern Widget rowColumn_Select_GY;
extern Widget rowColumn_Select_GZ;
extern Widget rowColumn_Select_GT;
extern Widget rowColumn_Select_G;
extern Widget SelectMenu_widget[5];
extern void JC_SR_SelectMenuButton_GX_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_SR_SelectMenuButton_GY_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_SR_SelectMenuButton_GZ_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_SR_SelectMenuButton_GT_CB( Widget wid, XtPointer client_data, XtPointer call_data );
extern void JC_SR_SelectMenuButton_G_CB( Widget wid, XtPointer client_data, XtPointer call_data );

/* For "Select" menus in EditDefinedVar */
extern Widget JC_EditDefinedVar;
extern Widget EDV_rowColumn_Select;
extern Widget EDV_SelectMenu_widget;
extern void EDV_SelectMenu_Build( Widget *menubar, void (*function_ptr)() );
extern void EDV_SelectMenuDefineButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );

extern Widget toggleButton_Regridding;
extern Widget label_RegriddingStatus;

static void JC_SelectMenu_Recreate( Widget parent, Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() );
void JC_II_SelectMenus_Recreate( swidget caller_id );

void JC_II_MainMenu_Maintain( JC_StateFlags *SF_ptr );

Boolean JC_DefineVariable_is_displayed=FALSE;
Boolean JC_EditDefinedVar_is_displayed=FALSE;
Boolean JC_SelectRegridding_is_displayed=FALSE;
Boolean JC_SelectRegridding_is_uniform=TRUE;

static JC_MenuItem items[MAX_MENU_ITEMS]={ NULL, };


/* .................... Function Declarations .................... */


void JC_II_SynchronizeWindows( void );


/* .................... Function Definitions .................... */


int JC_II_Synchronize( swidget caller_id )
{
  JC_Region *R_ptr=&GLOBAL_Region;
  JC_Variable *V_ptr=&GLOBAL_Variable;
  JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
  JC_Regridding *RG_ptr=&GLOBAL_Regridding;

  enum { FATAL_ERR=-2, NOT_FOUND_ERR, OK }
  ERR_type;

  JC_DatasetElement JC_Dset, *DE_ptr=NULL, *new_Dset_ptr=NULL;
  LIST *ferret_dsets;
  int status = LIST_OK;
     
  int numDataSets=0, numVars=0;
  int i=0, j=0, xyzt=0, return_code=OK;
  char nullStr[1] = {'\0'};
  char tempText[MAX_NAME_LENGTH]="";

  int variable_info = SAME_VARIABLE;
  Boolean deleted_current_dataset=FALSE, added_new_dataset=FALSE;

  /*
   * - Initialize the 'ferret_dsets' list.
   * - Fill 'ferret_dsets' with the names of the current ferret datasets.
   */

  if ( (ferret_dsets = list_init()) == NULL ) {
    fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_Synchronize(): Unable to initialize ferret_dsets.\n");
    return_code = FATAL_ERR;
    goto return_section;
  }

  ferret_query(QUERY_DSET, sBuffer, nullStr, nullStr, nullStr, nullStr, nullStr);
  sBuffer->textP = &sBuffer->text[0];     
  numDataSets = sBuffer->numStrings;

  if ( numDataSets == 0 ) {
    return_code = NOT_FOUND_ERR;
    goto syncronize_windows_section;
  }

  if ( SF_ptr->open_datasets < 1 && numDataSets > 0 )
    variable_info = FIRST_EVER_VARIABLE;

  SF_ptr->open_datasets = numDataSets;

  for (i=0; i<numDataSets; i++) {
    sBuffer->textP = CollectToReturn(sBuffer->textP, tempText);
    if ( list_insert_after(ferret_dsets, tempText, sizeof(tempText)) == NULL ) {
      fprintf(stderr, "ERROR in InterInterface.c: II_Synchronize(): list_insert_after returned NULL\n");
      return_code = FATAL_ERR;
      goto return_section;
    }
  }

  /*
   * DELETIONS
   * - Go through the GLOBAL list.
   * - If a name isn't found in the ferret list:
   *      delete it from the global list.
   */

  if ( list_mvfront(GLOBAL_DatasetList) != NULL ) {
    for (i=0; i<list_size(GLOBAL_DatasetList); i++) {
      strcpy(tempText, ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name);
      status = list_traverse(ferret_dsets, tempText, JC_ListTraverse_FoundMatch, (LIST_FRNT | LIST_FORW | LIST_SAVE));
      if ( status == LIST_OK )
	list_mvnext(GLOBAL_DatasetList);
      else {
	list_remove_curr(GLOBAL_DatasetList);
	if ( !strcmp(V_ptr->dset, tempText) )
	  deleted_current_dataset = TRUE;
      }
    }
  }


  /*
   * ADDITIONS
   * - Go through the ferret list.
   * - If a name isn't found in the GLOBAL list:
   *      add it to the global list.
   */

  list_mvrear(GLOBAL_DatasetList);
     
  if ( list_mvfront(ferret_dsets) != NULL ) {
    for (i=0; i<list_size(ferret_dsets); i++) {
      strcpy(JC_Dset.name, list_curr(ferret_dsets));
      status = list_traverse(GLOBAL_DatasetList, JC_Dset.name, JC_ListTraverse_FoundDsetMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
      if ( status == LIST_OK ) {
	DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);
	JC_DatasetElement_QueryFerret(DE_ptr, FALSE);
	list_mvnext(ferret_dsets);
      } else {
	DE_ptr = (JC_DatasetElement *)list_insert_after(GLOBAL_DatasetList, &JC_Dset, sizeof(JC_Dset));
	if ( DE_ptr == NULL ) {
	  fprintf(stderr, "ERROR in InterInterface.c: II_Synchronize(): DE_ptr = NULL\n");
	  return_code = FATAL_ERR;
	  goto return_section;
	}
	JC_DatasetElement_QueryFerret(DE_ptr, TRUE);
	added_new_dataset = TRUE;
	new_Dset_ptr = DE_ptr;
	list_mvnext(ferret_dsets);
      }
    }
  }

  /*
   * - Deallocate the 'ferret_dsets' list.
   * - Before we call MainInterface_NewVariable(), make sure the current_clone_ptr is turned off.
   */

  list_free(ferret_dsets, LIST_DEALLOC);
  /*JC_DatasetList_Print(GLOBAL_DatasetList, stderr);*/ /* debugging line */
  SF_ptr->a_clone_is_selected = FALSE;
  SF_ptr->current_clone_ptr = NULL;

  if ( list_mvfront(GLOBAL_DatasetList) == NULL )
    goto syncronize_windows_section;


  if ( variable_info == FIRST_EVER_VARIABLE ) {
    /*
     * - If this is the first dataset to be opened
     *      set the first variable and create an appropriate region
     */

    DE_ptr = (JC_DatasetElement *) list_front(GLOBAL_DatasetList);

    if ( list_empty(DE_ptr->varList) ) {
      fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_Synchronize(): varList empty in \"%s\"\n", DE_ptr->name);
      return_code = NOT_FOUND_ERR;
      goto syncronize_windows_section;
    }

    CancelInitialState();
    JC_MainInterface_NewVariable( list_front(DE_ptr->varList), DE_ptr->name, FIRST_EVER_VARIABLE );

  } else /* not the FIRST_EVER_VARIABLE */ {
    /*
     * If a new dataset has been opened
     *   choose the first variable in that dataset
     *
     * Else If the current dataset has been closed
     *   choose the first variable in the first dataset
     *
     * Else (the original dataset is still open)
     *   set DE_ptr to the current dataset
     *   look for the current variable in this dataset
     *   if the current variable is still available in this dataset
     *      keep the original dataset and variable
     *   else 
     *      if the current DEFINED variable is still available in this dataset
     *         keep the original dataset and variable
     *      else 
     *         keep the original dataset and choose the first variable
     *
     *      (cloned variables are not known to Ferret and they will always be reset)
     */

    if ( added_new_dataset ) {

      if ( list_empty(new_Dset_ptr->varList) ) {
	fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_Synchronize(): varList empty in \"%s\"\n", new_Dset_ptr->name);
	return_code = NOT_FOUND_ERR;
	goto syncronize_windows_section;
      }
      JC_MainInterface_NewVariable( list_front(new_Dset_ptr->varList), new_Dset_ptr->name, NEW_VARIABLE );

    } else if ( deleted_current_dataset ) {

      DE_ptr = (JC_DatasetElement *) list_front(GLOBAL_DatasetList);
      if ( list_empty(DE_ptr->varList) ) {
	fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_Synchronize(): varList empty in \"%s\"\n", DE_ptr->name);
	return_code = NOT_FOUND_ERR;
	goto syncronize_windows_section;
      }
      JC_MainInterface_NewVariable( list_front(DE_ptr->varList), DE_ptr->name, NEW_VARIABLE );

    } else /* we retain the same dataset */ {

      status = list_traverse(GLOBAL_DatasetList, V_ptr->dset, JC_ListTraverse_FoundDsetMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
      DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);
      if ( list_empty(DE_ptr->varList) ) {
	fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_Synchronize(): varList empty in \"%s\"\n", DE_ptr->name);
	return_code = NOT_FOUND_ERR;
	goto syncronize_windows_section;
      }
      status = list_traverse(DE_ptr->varList, V_ptr->name, JC_ListTraverse_FoundMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
      if ( status == LIST_OK ) {
	JC_MainInterface_NewVariable( V_ptr->name, DE_ptr->name, SAME_VARIABLE );
      } else {
	if ( list_empty(DE_ptr->dvarList) ) {
	  fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_Synchronize(): dvarList empty in \"%s\"\n", DE_ptr->name);
	  return_code = NOT_FOUND_ERR;
	  goto syncronize_windows_section;
	}
	status = list_traverse(DE_ptr->dvarList, V_ptr->name, JC_ListTraverse_FoundDvarMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	if ( status == LIST_OK )
	  JC_MainInterface_NewVariable( V_ptr->name, DE_ptr->name, SAME_VARIABLE );
	else
	  JC_MainInterface_NewVariable( list_front(DE_ptr->varList), DE_ptr->name, NEW_VARIABLE );
      }

    } /* added_new_dataset */

  } /* FIRST_EVER_VARIABLE */

  /*
   * - Update the "Select" menus.
   */

  JC_II_SelectMenus_Recreate(NULL);

  /*
   * Now we need to check the number of windows Ferret has open and update
   * the GUI menu.
   */

syncronize_windows_section:

  JC_II_SynchronizeWindows();


return_section:

  return return_code;
}


void JC_II_SynchronizeWindows( void )
{

  Widget local_WindowButton[5];
  int i=0, numWindows=0;
  char nullStr[1] = {'\0'};
  char tempText[MAX_NAME_LENGTH]="";
  
  local_WindowButton[0] = UxGetWidget(CancelWindow_1_Button);
  local_WindowButton[1] = UxGetWidget(CancelWindow_2_Button);
  local_WindowButton[2] = UxGetWidget(CancelWindow_3_Button);
  local_WindowButton[3] = UxGetWidget(CancelWindow_4_Button);
  local_WindowButton[4] = UxGetWidget(CancelWindow_5_Button);

  ferret_query(QUERY_WINDOWS, sBuffer, nullStr, nullStr, nullStr, nullStr, nullStr);
  sBuffer->textP = &sBuffer->text[0];     
  numWindows = sBuffer->numStrings;

  numWindows = (numWindows > 5) ? 5 : numWindows;

  for (i=0; i<numWindows; i++) {
    sBuffer->textP = CollectToReturn(sBuffer->textP, tempText);
    if ( !strcmp(tempText, "T" ) )
      XtSetSensitive(local_WindowButton[i], TRUE);
    else
      XtSetSensitive(local_WindowButton[i], FALSE);
  }

}


void JC_II_SelectMenus_Recreate( swidget caller_id )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;

     int xyzt=0;

#ifdef SGI_POPUPS
     /* NB__ UxTopLevel is declared external in UxXt.h */
     window_count = 0;
     windows[window_count++] = XtWindow(UxTopLevel);
#endif /* SGI_POPUPS */
     
/*
 * Recreate FerretMainWd "Select" menu.
 */
     JC_SelectMenu_Recreate(rowColumn_Select, &dataSetMenus, JC_Main_SelectMenuButton_CB, JC_Main_SelectMenuButton_CB, JC_Main_SelectMenu_CloneButton_CB);

     JC_II_MainMenu_Maintain(SF_ptr);


/*
 * Recreate DefineVariable "Select" menus.
 */
     if ( JC_DefineVariable_is_displayed ) {
	  
	  switch (GLOBAL_func_type) {

	     case FUNC_FUNCTION1:
	     case FUNC_FUNCTION2:
	     case FUNC_FUNCTION3:
	     case FUNC_FUNCTION4:
	       JC_SelectMenu_Recreate(rowColumn_Select1_F, &JC_DV_SelectMenu1_F, JC_DV_SelectMenuButton_F1_CB, JC_DV_SelectMenuButton_F1_CB, JC_DV_SelectMenuF1_CloneButton_CB);

	       if ( GLOBAL_func_type > FUNC_FUNCTION1 )
		    JC_SelectMenu_Recreate(rowColumn_Select_var2, &JC_DV_SelectMenu2_F, JC_DV_SelectMenuButton_F2_CB, JC_DV_SelectMenuButton_F2_CB, JC_DV_SelectMenuF2_CloneButton_CB);
	       break;

	     case FUNC_LINEAR_COMBINATION:
	       JC_SelectMenu_Recreate(rowColumn_Select1_LC, &JC_DV_SelectMenu1_LC, JC_DV_SelectMenuButton_LC1_CB, JC_DV_SelectMenuButton_LC1_CB, JC_DV_SelectMenuLC1_CloneButton_CB);
	       JC_SelectMenu_Recreate(rowColumn_Select2_LC, &JC_DV_SelectMenu2_LC, JC_DV_SelectMenuButton_LC2_CB, JC_DV_SelectMenuButton_LC2_CB, JC_DV_SelectMenuLC2_CloneButton_CB);
	       break;

	     case FUNC_PLUS_CONSTANT:
	     case FUNC_EXPONENT:
	       JC_SelectMenu_Recreate(rowColumn_Select1_EXP, &JC_DV_SelectMenu1_EXP, JC_DV_SelectMenuButton_EXP1_CB, JC_DV_SelectMenuButton_EXP1_CB, JC_DV_SelectMenuEXP1_CloneButton_CB);
	       break;

	     default:
	       fprintf(stderr, "ERROR: JC_InterInterface: JC_InterInterface_SelectMenus_Recreate(): GLOBAL_func_type = %d\n", GLOBAL_func_type);
	  }

     }

/*
 * Recreate SelectRegriddidng "Select" menus.
 */
     if ( JC_SelectRegridding_is_displayed ) {
	  
	  if ( JC_SelectRegridding_is_uniform ) {

	       JC_SelectMenu_Recreate(rowColumn_Select_G, &(SelectMenu_widget[ALL_AXES]), JC_SR_SelectMenuButton_G_CB, JC_SR_SelectMenuButton_G_CB, JC_SR_SelectMenuButton_G_CB);
	  
	  } else {

	       for ( xyzt=0; xyzt<4; xyzt++ ) {
		    if ( R_ptr->span[xyzt].ss[LO] != IRRELEVANT_AXIS ) {
			 if ( xyzt == X_AXIS )
			      JC_SelectMenu_Recreate(rowColumn_Select_GX, &(SelectMenu_widget[X_AXIS]), JC_SR_SelectMenuButton_GX_CB, JC_SR_SelectMenuButton_GX_CB, JC_SR_SelectMenuButton_GX_CB);
			 else if (xyzt == Y_AXIS )
			      JC_SelectMenu_Recreate(rowColumn_Select_GY, &(SelectMenu_widget[Y_AXIS]), JC_SR_SelectMenuButton_GY_CB, JC_SR_SelectMenuButton_GY_CB, JC_SR_SelectMenuButton_GY_CB);
			 else if (xyzt == Z_AXIS )
			      JC_SelectMenu_Recreate(rowColumn_Select_GZ, &(SelectMenu_widget[Z_AXIS]), JC_SR_SelectMenuButton_GZ_CB, JC_SR_SelectMenuButton_GZ_CB, JC_SR_SelectMenuButton_GZ_CB);
			 else if (xyzt == T_AXIS )
			      JC_SelectMenu_Recreate(rowColumn_Select_GT, &(SelectMenu_widget[T_AXIS]), JC_SR_SelectMenuButton_GT_CB, JC_SR_SelectMenuButton_GT_CB, JC_SR_SelectMenuButton_GT_CB);
			 else
			      fprintf(stderr, "ERROR in JC_InterInterface.c: JC_II_SelectMenus_Recreate(): xyzt = %d\n", xyzt);
		    }
	       }

	  }
     }

/*
 * Recreate EditDefinedVar "Select" menu.
 */
     if ( JC_EditDefinedVar_is_displayed ) {
	  
       JC_SelectMenu_Recreate(EDV_rowColumn_Select, &EDV_SelectMenu_widget, NULL, EDV_SelectMenuDefineButton_CB, NULL);

     }

#ifdef SGI_POPUPS
    /* NB__ UxTopLevel is declared external in UxXt.h */
     XSetWMColormapWindows(XtDisplay(UxTopLevel), XtWindow(UxTopLevel),
			   windows, window_count);
#endif /* SGI_POPUPS */

}


static void JC_SelectMenu_Recreate( Widget parent, Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() )
{

     /*
      * - Destroy the old menubar. (contained in the rowColumn widget)
      * - Create a new menubar.
      * - If this is the EditDefinedVar interface
      *    - only include user defined variables.
      * - Else
      *    - add all dsets and all variables.
      * - Build the menu.
      * - Manage the menubar.
      */

     XtDestroyWidget(*menubar);
     *menubar = XmCreateMenuBar(UxGetWidget(parent), "menbar0", NULL, 0);
     if ( UxGetWidget(parent) == EDV_rowColumn_Select )
       JC_Menu_AddDsetUvars(items, var_fn_ptr, dvar_fn_ptr, cvar_fn_ptr);
     else
       JC_Menu_AddDsetVars(items, var_fn_ptr, dvar_fn_ptr, cvar_fn_ptr);
     JC_Menu_Build(*menubar, XmMENU_PULLDOWN, "Select", NULL, FALSE, items);
     XtManageChild(UxGetWidget(*menubar));

}

/*--------------------  Non "Select" menus stuff  ------------------*/

void JC_II_MainMenu_Maintain( JC_StateFlags *SF_ptr )
{

  /*
   * - If datasets are open
   *      sensitize various buttons
   */
  if ( SF_ptr->open_datasets > 0 ) {
    /* from the "Edit" pane */
    if ( !JC_DefineVariable_is_displayed )
      XtSetSensitive(UxGetWidget(defineVariableButton), TRUE);
    else
      XtSetSensitive(UxGetWidget(defineVariableButton), FALSE);

    XtSetSensitive(UxGetWidget(editDefinedVarButton), TRUE);

    /* from the "View" pane */
    XtSetSensitive(UxGetWidget(InfoButton), TRUE);
    XtSetSensitive(UxGetWidget(ListButton), TRUE);
  } else {
    /* from the "File" pane */
    XtSetSensitive(UxGetWidget(saveButton), FALSE);
    /* from the "Edit" pane */
    XtSetSensitive(UxGetWidget(defineVariableButton), FALSE);
    XtSetSensitive(UxGetWidget(editDefinedVarButton), FALSE);
    /* from the "View" pane */
    XtSetSensitive(UxGetWidget(InfoButton), FALSE);
    XtSetSensitive(UxGetWidget(ListButton), FALSE);
  }

     
  /* from the "File" pane */
  if ( SF_ptr->open_datasets > 1 )
    XtSetSensitive(UxGetWidget(closeDsetButton), TRUE);
  else
    XtSetSensitive(UxGetWidget(closeDsetButton), FALSE);


  /*
   * - If a plot exists
   *      sensitize the "Windows" menu option.
   */
 if ( SF_ptr->a_plot_exists ) {
   /*    XtSetSensitive(UxGetWidget(windows_SubPane), TRUE);*/ /* JC: removed after SET/CANCEl WINDOW fix */
    if ( gMetaCreationActive )
      XtSetSensitive(UxGetWidget(printButton), TRUE);
    else
      XtSetSensitive(UxGetWidget(printButton), FALSE);
  } else {
    /* XtSetSensitive(UxGetWidget(windows_SubPane), FALSE);*/ /* JC: removed after SET/CANCEl WINDOW fix */
    XtSetSensitive(UxGetWidget(printButton), FALSE);
  }
     
 /*
  * - If the last geometry was XY, enable the "GO LAND" and "GO FLAND" menu options.
  */
  if ( SF_ptr->a_plot_exists && SF_ptr->geometry_last_plotted == GEOM_XY ) {
    XtSetSensitive(UxGetWidget(macro_pane_LandOutline), TRUE);
    XtSetSensitive(UxGetWidget(macro_pane_SolidLand), TRUE);
  } else {
    XtSetSensitive(UxGetWidget(macro_pane_LandOutline), FALSE);
    XtSetSensitive(UxGetWidget(macro_pane_SolidLand), FALSE);
  }
     
}


void JC_II_FixRegridding( swidget caller_id )
{
     XmToggleButtonSetState(UxGetWidget(toggleButton_Regridding), TRUE, TRUE);
}


void JC_II_ChangeRegriddingLabel( swidget caller_id )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char tempText[MAX_NAME_LENGTH]="";
     Boolean regridding_is_on=FALSE;

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
     
}
