/*
 * JC_Map.h
 *
 * Jonathan Callahan
 * Dec 19th 1995
 *
 * This file contains functions which interact with the map for the Ferret GUI.
 *
 */

#ifndef _JC_MAP_H
#define _JC_MAP_H

/* .................... Defines .................... */


/* .................... Action Routines .................... */

void  JC_Map_Button1Press_Action( Widget wid, XEvent *ev, String *param, Cardinal num );
void  JC_Map_Motion1Notify_Action( Widget wid, XEvent *ev, String *param, Cardinal num );
void  JC_Map_Button1Release_Action( Widget wid, XEvent *ev, String *param, Cardinal num );


/* .................... Map Methods .................... */

void  JC_Map_Hide( void );
void  JC_Map_SetToolColor( XtPointer color_name );
void  JC_Map_NewVariable( JC_Variable *V_ptr );
void  JC_Map_NewRegion( JC_Region *R_ptr );
void  JC_Map_Show( void );

#endif
