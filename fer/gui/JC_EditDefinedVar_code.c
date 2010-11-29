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
 * JC_EditDefinedVar_code.c
 *
 * Jonathan Callahan
 * Jul 12'th 1996
 *
 * This file contains the auxiliary functions which are included by
 * JC_EditDefinedVar.c.
 *
 */

/* .................... Function Definitions .................... */


static void JC_EDV_Cancel_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     XtPopdown(UxGetWidget(JC_EditDefinedVar));
     JC_EditDefinedVar_is_displayed = FALSE;
     JC_II_MainMenu_Maintain( SF_ptr );
}


static void JC_EDV_Initialize( void )
{
  EDV_SelectMenu_widget = XmCreateMenuBar(EDV_rowColumn_Select, "menubar0", NULL, 0);
  EDV_SelectMenu_Build(&(EDV_SelectMenu_widget), NULL, EDV_SelectMenuDefineButton_CB, NULL);
  XtManageChild(EDV_SelectMenu_widget);
  XtManageChild(EDV_rowColumn_Select);
}


static void EDV_Define_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     char command_string[MAX_COMMAND_LENGTH]="";
     char *text=NULL, tempText[MAX_COMMAND_LENGTH]="";

     JC_DatasetElement *DE_ptr;
/*
 * - Get the LET command from the textField.
 * - Bail out if the string doesn't contain at least "LET a=b" : 7 letters.
 * - Issue the command to Ferret.
 * - Get the current DatasetElement.
 * - 
 * - Query Ferret about the this dataset so that this DatasetElement reflects the newly defined variable.
 * - Recreate all the Select menus.
 */

     if ( text = (char *) XmTextGetString(UxGetWidget(EDV_textField_definition)) ) {
       strcpy(command_string, text); /* TODO ... see if this can be removed */

       if ( strlen(command_string) < 7 ) {

	 sprintf(tempText, "\"%s\" is not a valid LET command.", command_string);
	 JC_Message_CB(wid, tempText, NULL);

       } else {
 
	 ferret_command(command_string, IGNORE_COMMAND_WIDGET);

	 list_traverse(GLOBAL_DatasetList, assigned_dset_name, JC_ListTraverse_FoundDsetMatch,
		       (LIST_FRNT | LIST_FORW | LIST_ALTR));
	 DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);
	 JC_DatasetElement_QueryFerret(DE_ptr, FALSE);
	 JC_II_SelectMenus_Recreate(UxGetWidget(JC_EditDefinedVar));

       }
     }

     XtFree(text); /* allocated with XmTextGetString() */
}


void EDV_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)()  )
{
     JC_MenuItem items[MAX_MENU_ITEMS]={ NULL, };

     JC_Menu_AddDsetUvars(items, var_fn_ptr, dvar_fn_ptr, cvar_fn_ptr);
     JC_Menu_Build(*menubar, XmMENU_PULLDOWN, "Select", NULL, FALSE, items);
}


void EDV_SelectMenuDefineButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
  char *tempText;
  char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]=""; 
  char command_string[MAX_COMMAND_LENGTH]="", definition_string[MAX_COMMAND_LENGTH]="";
  char title[MAX_NAME_LENGTH]="", units[MAX_NAME_LENGTH]=""; 
  char title_string[MAX_NAME_LENGTH]="", units_string[MAX_NAME_LENGTH]=""; 
  char nullStr[1] = {'\0'};
  XmString buttonLabel;
  JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
  int i=0, j=0, num_strings=0;

  /*
   * - Get the dataset name.
   * - Get the variable name.
   * - Store the dset_name so it can be used in Define_CB().
   */

  strcpy(dset_name, (char *)client_data);
  XtVaGetValues(wid,
		XmNlabelString, &buttonLabel,
		NULL);
  XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
  strcpy(var_name, tempText);
  strcpy(assigned_dset_name, dset_name);

  /*
   * - Get the definition of this variable.
   * - Get the title.
   * - Get the units.
   *
   * - Put it in the textField.
   */

  ferret_query(QUERY_UVAR_DEFINITION, sBuffer, dset_name, var_name, nullStr, nullStr, nullStr);
  sBuffer->textP = &sBuffer->text[0];
  num_strings = sBuffer->numStrings;

  sBuffer->textP = CollectToReturn(sBuffer->textP, definition_string);
  sBuffer->textP = CollectToReturn(sBuffer->textP, title);
  sBuffer->textP = CollectToReturn(sBuffer->textP, units);

  if ( strlen(title) > 0 )
    sprintf(title_string, "/TITLE=\"%s\"", title);

  if ( strlen(units) > 0 )
    sprintf(units_string, "/UNIT=\"%s\"", units);

  sprintf(command_string, "LET/D=%s%s%s %s = %s", dset_name, title_string, units_string, var_name, definition_string);

  XmTextSetString(EDV_textField_definition, command_string);


  /*
   * - Free the allocated memory.
   */

  XtFree(tempText); /* allocated with XmStringGetLtoR() */

}

