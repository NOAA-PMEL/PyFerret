{
			static int startx, starty, x, y, type, hdl, oldx, oldy, width, height, newHeight, newWidth;
			static Boolean flag = False, xIsCompressed, yIsCompressed;
			int newx, newy, geo;
			Display *display;
			Screen *screen;
			GC lgc;
		
			XtVaGetValues(UxGetWidget(drawingArea1),
				XmNuserData, &lgc,
				NULL);
		
			/* display = XtDisplay(UxGetWidget(drawingArea1)); */
			screen = DefaultScreenOfDisplay(UxEvent->xany.display);
		
			XGrabButton(UxEvent->xany.display, Button1, AnyModifier, XtWindow(UxGetWidget(drawingArea1)), False,
				Button1MotionMask | ButtonReleaseMask,
				GrabModeAsync, GrabModeAsync, XtWindow(UxGetWidget(drawingArea1)),
				XCreateFontCursor(UxEvent->xany.display, XC_crosshair));
		
			switch (UxEvent->type) {
				case ButtonPress:
					if (UxEvent->xbutton.button == Button1) {
						XSetForeground(UxEvent->xany.display, lgc, WhitePixelOfScreen(screen));
						x = startx = UxEvent->xbutton.x;
						y = starty = UxEvent->xbutton.y;
						x = x > NUMLONPIXELS-2 ? NUMLONPIXELS-2 : x;
						x = x < 2 ? 2 : x;
						y = y > NUMLATPIXELS-2 ? NUMLATPIXELS-2 : y;;
						y = y < 2 ? 2 : y;
						flag = False;
		
						/* test whether button is down in a handle */
						if (hdl = GetHandle(x, y)) {
							gDragInHandle = True;
							type = gHandleList[hdl].type;
						}
						else {
							oldx = x;
							oldy = y;
							gDragInHandle = False;
						}
						
						/* get the state of the axes transformations */
						if (transformCompresses[gAllContexts[gCurrContext].transform[1]] == 'C')
						    yIsCompressed = True;
						else
						    yIsCompressed = False;
		
						if (transformCompresses[gAllContexts[gCurrContext].transform[0]] == 'C')
						    xIsCompressed = True;
						else
						    xIsCompressed = False;
		
						geo = gAllContexts[gCurrContext].geometry;
		
						if (!gDragInHandle) {
							/* start a tool manipulation */
							switch (toolMode) {
								case xl:
									xzLine.start.x = startx;
									xzLine.start.y = starty;
									xzLine.end.y = starty;
									break;
								case xy:
									oldx = x;
									oldy = y;
									break;
								case yl:
									yzLine.start.x = startx;
									yzLine.start.y = starty;
									yzLine.end.x = startx;
								case pt:
									mapLoc.x = startx;
									mapLoc.y = starty;
									break;
							}
						}
						else {
							/* manipulating an existing handle */
							switch (type) {
							case PT:oldx = mapLoc.x;
								oldy = mapLoc.y;
								break;
							case UR:
								oldx = xyRect.x;
								oldy = xyRect.y + xyRect.height;
								width = xyRect.width;
								height = xyRect.height;
								break;
							case UL:
								oldx = xyRect.x + xyRect.width;
								oldy = xyRect.y + xyRect.height;
								width = xyRect.width;
								height = xyRect.height;
								break;
							case LL:
								oldx = xyRect.x + xyRect.width;
								oldy = xyRect.y;
								width = xyRect.width;
								height = xyRect.height;
								break;
							case Cl:
							case LR:
							case CR:
							case CU:
							case CL:
							case CC:
								oldx = xyRect.x;
								oldy = xyRect.y;
								width = xyRect.width;
								height = xyRect.height;
								break;
							case xL:
								oldx = xzLine.end.x;
								break;
							case xR:
								oldx = xzLine.start.x;
								break;
							case xC:
								width = xzLine.end.x - xzLine.start.x;
								break;
							case yU:
								oldy = yzLine.end.y;
								break;
							case yC:
								height = yzLine.end.y - yzLine.start.y;
								break;
							case yL:
								oldy = yzLine.start.y;
								break;
								break;
							case cxL:
								oldx = xcLine.end.x;
								break;
							case cxR:
								oldx = xcLine.start.x;
								break;
							case cyU:
								oldy = ycLine.end.y;
								break;
							case cyL:
								oldy = ycLine.start.y;
								break;
						}
					}
					break;
				case MotionNotify:
					if (UxEvent->xmotion.state & Button1Mask) {
						gStoreHandle = False;
						 ClearMap();
						x = UxEvent->xmotion.x;
						y = UxEvent->xmotion.y;
						x = x > NUMLONPIXELS-2 ? NUMLONPIXELS-2 : x;
						x = x < 2 ? 2 : x;
						y = y > NUMLATPIXELS-2 ? NUMLATPIXELS-2 : y;
						y = y < 2 ? 2 : y;
						
						if (!gDragInHandle) {
						      /* dragging a tool */
						      switch (toolMode) {
							    case xy:
								   if (x >= oldx && y >= oldy) {
									 xyRect.x = oldx;
									 xyRect.y = oldy;
									 xyRect.width = x - oldx;
									 xyRect.height = y - oldy;
								   }
								   else if (x < oldx && y < oldy) {
									  xyRect.x = x;
									  xyRect.y = y;
									  xyRect.width = oldx - x;
									  xyRect.height = oldy - y;
								   }
								   else if (x < oldx && y >= oldy) {
									  xyRect.x = x;
									  xyRect.y = oldy;
									  xyRect.width = oldx - x;
									  xyRect.height = y - oldy;
								   }
								   else if (x >= oldx && y < oldy) {
									  xyRect.x = oldx;
									  xyRect.y = y;
									  xyRect.width = x - oldx;
									  xyRect.height = oldy - y;
								   }
								   DrawXyRect();
								   break;
							    case xl:
								   xzLine.end.x = x;
								   DrawXzLine();

								   if (yIsCompressed) {
									ycLine.start.x = xzLine.start.x + (xzLine.end.x - xzLine.start.x)/2;
									ycLine.end.x = ycLine.start.x;

								        /* does the compressed axis line fit on top of map? */
							    		if ((newy = y - (ycLine.end.y - ycLine.start.y)/2) > 2) {
										/* does */
									        newHeight = (ycLine.end.y - ycLine.start.y)/2;
										ycLine.start.y = xzLine.start.y - newHeight;
										ycLine.end.y = xzLine.start.y + newHeight;
									}
							    		else {
										/* doesn't fit--reduce height to fit  */
										ycLine.start.y = 2;
										newHeight = 2 * (xzLine.start.y - 2);
										ycLine.end.y = ycLine.start.y + newHeight;
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* does the compressed axis fit on bottom of map?  */
										if ((newy = y + (ycLine.end.y - ycLine.start.y)/2) <= NUMLATPIXELS - 2) {
											/* does */
										        ycLine.start.y = xzLine.start.y - newHeight;
											ycLine.end.y = xzLine.start.y + newHeight;
										}
										else {
											/* doesn't */
											ycLine.end.y = NUMLATPIXELS - 2;
											newHeight = 2 * (ycLine.end.y - xzLine.start.y);
											ycLine.start.y = ycLine.end.y - newHeight;
										}
							   		 }	
							   		flag = False;
								  	DrawYcLine();
								   }
								   break;
							    case yl:
								   yzLine.end.y = y;
								   DrawYzLine();
								   if (xIsCompressed) {
								        xcLine.start.y = yzLine.start.y + (yzLine.end.y - yzLine.start.y)/2;
								        xcLine.end.y = xcLine.start.y;
									
									/* test whether the left side of compressed axis fits on map */
							  		if ((newx = x - (xcLine.end.x - xcLine.start.x)/2) > 2) {
									        /* does */
									        newWidth = (xcLine.end.x - xcLine.start.x)/2;
										xcLine.start.x = yzLine.start.x - newWidth;
										xcLine.end.x = yzLine.start.x + newWidth;
									}
							  		else {
									       /* doesn't */
										xcLine.start.x = 2;
										newWidth = 2 * (yzLine.start.x - 2);
										xcLine.end.x =xcLine.start.x + newWidth;
										flag = True;
							    		}
									
									/* test whether right side of compressed axis fits on map */
							  		if (!flag) {
										if ((newx = x + (xcLine.end.x - xcLine.start.x)/2) <= NUMLONPIXELS - 2) {
										        /* does */
										        xcLine.start.x = yzLine.start.x - newWidth;
										        xcLine.end.x = yzLine.start.x + newWidth;
										}
										else {
											xcLine.end.x = NUMLONPIXELS - 2;
											newWidth = 2 * (xcLine.end.x - yzLine.start.x);
											xcLine.start.x = xcLine.end.x - newWidth;
										}
							  	  	}
							  	 	flag = False;
									DrawXcLine();
								   }
								   break;
							    case pt:
								   mapLoc.x = x;
								   mapLoc.y = y;
								   if (!yIsCompressed && !xIsCompressed)
								   	DrawPoint();
		
								   if (yIsCompressed) {
							  		/* translate compressed y axis to new mapLoc */
							  		height = ycLine.end.y - ycLine.start.y;
		
									/*translate in x */
							  		ycLine.start.x = mapLoc.x;
							  		ycLine.end.x = mapLoc.x;
		
									/* check whether top of y line bumps into border */
							    		if ((newy = y - height/2) > 2)
										/* doesn't */
										ycLine.start.y = mapLoc.y - height/2;
							    		else {
										/* does */
										ycLine.start.y = 2;
										ycLine.end.y = 2 + height;
										mapLoc.y = height/2 + 2;
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newy = y + height/2) <= NUMLATPIXELS - 2)
											/* doesn't */
											ycLine.end.y = mapLoc.y + height/2;
										else {
											/* does */
											ycLine.end.y = NUMLATPIXELS - 2;
											ycLine.start.y = NUMLATPIXELS - 2 - height;
											mapLoc.y = NUMLATPIXELS - 2 - height/2;
										}
							   		 }	
							   		 flag = False;
								}

								if (xIsCompressed) {
							  		/* translate compressed x axis to new mapLoc */
								  	width = xcLine.end.x - xcLine.start.x;

									/*translate in y */
							  		xcLine.start.y = mapLoc.y;
							  		xcLine.end.y = mapLoc.y;
		
									/* check whether top of y line bumps into border */
							    		if ((newx = x - width/2) > 2)
										/* doesn't */
										xcLine.start.x = mapLoc.x - width/2;
							    		else {
										/* does */
										xcLine.start.x = 2;
										xcLine.end.x = 2 + width;
										mapLoc.x = width/2 + 2;
										if (yIsCompressed) {
										      ycLine.start.x = mapLoc.x;
										      ycLine.end.x = mapLoc.x;
										}
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newx = x + width/2) <= NUMLONPIXELS - 2)
											/* doesn't */
											xcLine.end.x = mapLoc.x + width/2;
										else {
											/* does */
											xcLine.end.x = NUMLONPIXELS - 2;
											xcLine.start.x = NUMLONPIXELS - 2 - width;
											mapLoc.x = NUMLONPIXELS - 2 - width/2;
											if (yIsCompressed) {
											  ycLine.start.x = mapLoc.x;
											  ycLine.end.x = mapLoc.x;
											}
										}
							   		 }	
							   		 flag = False;
								}
							        if (yIsCompressed)
								  DrawYcLine();
							        if (xIsCompressed)
								  DrawXcLine();
								DrawPoint();
								break;
						      	} /* switch */
						}
						else {
						      /* dragging a handle */
						      if (type == PT) {
							mapLoc.x = x;
							mapLoc.y = y;
							if (!yIsCompressed && !xIsCompressed)
								DrawPoint();
		
							if (yIsCompressed) {
							  	/* translate compressed y axis to new mapLoc */
							  	height = ycLine.end.y - ycLine.start.y;
		
								/*translate in x */
							  	ycLine.start.x = mapLoc.x;
							  	ycLine.end.x = mapLoc.x;
	
								/* check whether top of y line bumps into border */
							    	if ((newy = y - height/2) > 2)
									/* doesn't */
									ycLine.start.y = mapLoc.y - height/2;
							    	else {
									/* does */
									ycLine.start.y = 2;
									ycLine.end.y = 2 + height;
									mapLoc.y = height/2 + 2;
									flag = True;
							    	 }
		
							   	 if (!flag) {
									/* check whether bottom bumps into border */
									if ((newy = y + height/2) <= NUMLATPIXELS - 2)
										/* doesn't */
										ycLine.end.y = mapLoc.y + height/2;
									else {
										/* does */
										ycLine.end.y = NUMLATPIXELS - 2;
										ycLine.start.y = NUMLATPIXELS - 2 - height;
										mapLoc.y = NUMLATPIXELS - 2 - height/2;
									}
							   	 }	
							   	 flag = False;
							      }

							      if (xIsCompressed) {
							  		/* translate compressed x axis to new mapLoc */
								  	width = xcLine.end.x - xcLine.start.x;

									/*translate in y */
							  		xcLine.start.y = mapLoc.y;
							  		xcLine.end.y = mapLoc.y;
		
									/* check whether top of y line bumps into border */
							    		if ((newx = x - width/2) > 2)
										/* doesn't */
										xcLine.start.x = mapLoc.x - width/2;
							    		else {
										/* does */
										xcLine.start.x = 2;
										xcLine.end.x = 2 + width;
										mapLoc.x = width/2 + 2;
										if (yIsCompressed) {
										  ycLine.start.x = mapLoc.x;
										  ycLine.end.x = mapLoc.x;
										}
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newx = x + width/2) <= NUMLONPIXELS - 2)
											/* doesn't */
											xcLine.end.x = mapLoc.x + width/2;
										else {
											/* does */
											xcLine.end.x = NUMLONPIXELS - 2;
											xcLine.start.x = NUMLONPIXELS - 2 - width;
											mapLoc.x = NUMLONPIXELS - 2 - width/2;
											if (yIsCompressed) {
											  ycLine.start.x = mapLoc.x;
											  ycLine.end.x = mapLoc.x;
											}
										}
							   		 }	
							   		 flag = False;
								}
							        if (yIsCompressed)
								  DrawYcLine();
							        if (xIsCompressed)
								  DrawXcLine();
							 DrawPoint();
						      }
						      else if (type == CC) {
							    if ((newx = x - width/2) > 2) 
								  xyRect.x = newx;
							      else
								  xyRect.x = 2;
		
							      if ((newx = x + width/2) > NUMLONPIXELS - 2)
								    xyRect.x = NUMLONPIXELS - 2 - width;
		
							    if ((newy = y - height/2) > 2)
								  xyRect.y = newy;
							     else
								  xyRect.y = 2;
		
							      if ((newy = y + height/2) > NUMLATPIXELS - 2)
								    xyRect.y = NUMLATPIXELS - 2 - height;
							    DrawXyRect();
							}
						      else if (type == LR) {
							    if (x >= oldx && y >= oldy) {
								  xyRect.width = x - oldx;
								  xyRect.height = y - oldy;
							    }
							    else if (x < oldx && y < oldy) {
								  xyRect.x = x;
								  xyRect.y = y;
								  xyRect.width = oldx - x;
								  xyRect.height = oldy - y;
							    }
							    else if (x < oldx && y >= oldy) {
								  xyRect.x = x;
								   xyRect.width = oldx - x;
								  xyRect.height = y - oldy;
							    }
							    else if (x >= oldx && y < oldy) {
								   xyRect.y = y;
								  xyRect.width = x - oldx;
								  xyRect.height = oldy - y;
							    }
							    DrawXyRect();
						      }
						      else if (type == UR) {
							    if (x >= oldx && y <= oldy) {
								  xyRect.x = oldx;
								  xyRect.y = y;
								  xyRect.width = x - oldx;
								  xyRect.height = oldy - y;
							    }
							    else if (x >= oldx && y > oldy) {
								xyRect.x = oldx;
								xyRect.y = oldy;
								xyRect.width = x - oldx;
								xyRect.height = y - oldy;
							    }
							    else if (x < oldx && y > oldy) {
								xyRect.x = x;
								xyRect.y = oldy;
								 xyRect.width = oldx - x;
								xyRect.height = y - oldy;
							    }
							    else if (x < oldx && y < oldy) {
								xyRect.x = x;
								 xyRect.y = y;
								xyRect.width = oldx - x;
								xyRect.height = oldy - y;
							    }
							    DrawXyRect();
						      }
						      else if (type == UL) {
							    if (x <= oldx && y <= oldy) {
								xyRect.x = x;
								xyRect.y = y;
								xyRect.width = oldx - x;
								xyRect.height = oldy - y;
							    }
							    else if (x <= oldx && y > oldy) {
								xyRect.x = x;
								xyRect.y = oldy;
								xyRect.width = oldx - x;
								xyRect.height = y - oldy;
							    }
							    else if (x > oldx && y <= oldy) {
								xyRect.x = oldx;
								xyRect.y = y;
								 xyRect.width = x - oldx;
								xyRect.height = oldy - y;
							    }
							    else if (x > oldx && y > oldy) {
								xyRect.x = oldx;
								 xyRect.y = oldy;
								xyRect.width = x - oldx;
								xyRect.height = y - oldy;
							    }
							    DrawXyRect();
						      }
						      else if (type == LL) {
							    if (x <= oldx && y >= oldy) {
								xyRect.x = x;
								xyRect.y = oldy;
								xyRect.width = oldx - x;
								xyRect.height = y - oldy;
							    }
							    else if (x <= oldx && y < oldy) {
								xyRect.x = x;
								xyRect.y = y;
								xyRect.width = oldx - x;
								xyRect.height = oldy - y;
							    }
							    else if (x > oldx && y >= oldy) {
								xyRect.x = oldx;
								xyRect.y = oldy;
								 xyRect.width = x - oldx;
								xyRect.height = y - oldy;
							    }
							    else if (x > oldx && y < oldy) {
								xyRect.x = oldx;
								 xyRect.y = y;
								xyRect.width = x - oldx;
								xyRect.height = oldy - y;
							    }
							    DrawXyRect();
						      }
						      else if (type == Cl) {
							    if (y >= oldy)
								 xyRect.height = y - oldy;
							     else if (y >= 2) {
								 xyRect.y = y;
								 xyRect.height = oldy - y;
							    }
							     else {
								 xyRect.y = 2;
								 xyRect.height = oldy - 2;
							    }
							    DrawXyRect();
						      }
						      else if (type == CR) {
							    if (x >= oldx)
								 xyRect.width = x - oldx;
							     else if (x >= 2) {
								 xyRect.x = x;
								 xyRect.width = oldx - x;
							    }
							     else {
								 xyRect.x = 2;
								 xyRect.width = oldx - 2;
							    }
							    DrawXyRect();
						      }
						      else if (type == CU) {
							    if (y <= oldy) {
								 xyRect.y = y;
								 xyRect.height = oldy - y + height;
							    }
							     else if (y >= oldy + height) {
								 xyRect.y = oldy + height;
								 xyRect.height = y - oldy - height;
							    }
							     else {
								 xyRect.y = y;
								 xyRect.height = oldy + height - y;
							    }
							    DrawXyRect();
						      }
						      else if (type == CL) {
							    if (x <= oldx) {
								 xyRect.x = x;
								 xyRect.width = oldx - x + width;
							    }
							     else if (x >= oldx + width) {
								 xyRect.x = oldx + width;
								 xyRect.width = x - oldx - width;
							    }
							     else {
								 xyRect.x = x;
								 xyRect.width = oldx + width - x;
							    }
							    DrawXyRect();
						      }
								
							   /* x line handles */
						      else if (type == xR) {
							    if (x >= oldx) {
								xzLine.end.x = x;
								xzLine.start.x = oldx;
							    }
							    else {
								xzLine.start.x = x;
								xzLine.end.x = oldx;
							    }
							    DrawXzLine();
							    if (yIsCompressed) {
								  ycLine.start.x = xzLine.start.x + (xzLine.end.x - xzLine.start.x)/2;
								  ycLine.end.x = ycLine.start.x;
								  DrawYcLine();
							    }
						       }
						      else if (type == xC) {
							    xzLine.start.y = y;
							    xzLine.end.y = y;
							    if ((newx = x - width/2) > 2)
								xzLine.start.x = newx;
							    else {
								xzLine.start.x = 2;
								xzLine.end.x = 2 + width;
								flag = True;
							     }
							    if (!flag) {
								if ((newx = x + width/2) <= NUMLONPIXELS - 2)
									xzLine.end.x = newx;
								else {
									xzLine.end.x = NUMLONPIXELS - 2;
									xzLine.start.x = NUMLONPIXELS - 2 - width;
								}
							    }
							    flag = False;
							    if (yIsCompressed) {
							  	height = ycLine.end.y - ycLine.start.y;
		
								/*translate in x */
							    	ycLine.start.x = xzLine.start.x + width/2;
							    	ycLine.end.x = ycLine.start.x;
		
								/* check whether top of y line bumps into border */
							    	if ((newy = y - height/2) > 2)
									/* doesn't */
									ycLine.start.y = newy;
							    	else {
									/* does */
									ycLine.start.y = 2;
									ycLine.end.y = 2 + height;
									xzLine.start.y = height/2 + 2;
									xzLine.end.y = xzLine.start.y;
									flag = True;
							    	 }
		
							   	 if (!flag) {
									/* check whether bottom bumps into border */
									if ((newy = y + height/2) <= NUMLATPIXELS - 2)
										/* doesn't */
										ycLine.end.y = newy;
									else {
										/* does */
										ycLine.end.y = NUMLATPIXELS - 2;
										ycLine.start.y = NUMLATPIXELS - 2 - height;
										xzLine.start.y = NUMLATPIXELS - 2 - height/2;
										xzLine.end.y = xzLine.start.y;
									}
							   	 }	
							   	 flag = False;
							  	 DrawYcLine();
							    }
							    DrawXzLine();
						       }
						      else if (type == xL) {
							    if (x <= oldx) {
								xzLine.start.x = x;
								xzLine.end.x = oldx;
							    }
							    else {
								xzLine.start.x = oldx;
								xzLine.end.x = x;
							    }
							    DrawXzLine();
							    if (yIsCompressed) {
								  ycLine.start.x = xzLine.start.x + (xzLine.end.x - xzLine.start.x)/2;
								  ycLine.end.x = ycLine.start.x;
								  DrawYcLine();
							    }
						      }
		
						      /* compressed x line handles */
						      else if (type == cxR) {
							    if (x >= oldx) {
								xcLine.end.x = x;
								xcLine.start.x = oldx;
							    }
							    else {
								xcLine.start.x = x;
								xcLine.end.x = oldx;
							    }
							    DrawXcLine();
							    if (yIsCompressed) {
								  ycLine.start.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								  ycLine.end.x = ycLine.start.x;
								  DrawYcLine();
								  if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in x */
								    mapLoc.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
							    }
							    else {
								  if (geo == 2 || geo == 8 || geo == 9 || geo == 14) {
								    yzLine.start.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    yzLine.end.x = yzLine.start.x;
								    DrawYzLine();
								  }
								  else if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in x */
								    mapLoc.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
							    }
						       }
						      else if (type == cxL) {
							    if (x <= oldx) {
								xcLine.start.x = x;
								xcLine.end.x = oldx;
							    }
							    else {
								xcLine.start.x = oldx;
								xcLine.end.x = x;
							    }
							    DrawXcLine();
							    if (yIsCompressed) {
								  ycLine.start.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								  ycLine.end.x = ycLine.start.x;
								  DrawYcLine();
								  if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in x */
								    mapLoc.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }	
							    }
							    else {
								  if (geo == 2 || geo == 8 || geo == 9 || geo == 14) {
								    yzLine.start.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    yzLine.end.x = yzLine.start.x;
								    DrawYzLine();
								  }
								  else if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in x */
								    mapLoc.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
							    }
						      }
		
						      /* y axis handles */
						      else if (type == yU) {
							    if (y >= oldy) {
								yzLine.end.y = y;
								yzLine.start.y = oldy;
							    }
							    else {
								yzLine.start.y = y;
								yzLine.end.y = oldy;
							    }
							    DrawYzLine();
							    if (xIsCompressed) {
								xcLine.start.y = yzLine.start.y + (yzLine.end.y - yzLine.start.y)/2;
								xcLine.end.y = xcLine.start.y;
								DrawXcLine();
							     }
						      }
						      else if (type == yC) {
							    yzLine.start.x = x;
							    yzLine.end.x = x;
							    if ((newy = y - height/2) > 2)
								yzLine.start.y = newy;
							    else {
								yzLine.start.y = 2;
								yzLine.end.y = 2 + height;
								flag = True;
							     }
							    if (!flag) {
								if ((newy = y + height/2) <= NUMLATPIXELS - 2)
									yzLine.end.y = newy;
								else {
									yzLine.end.y = NUMLATPIXELS - 2;
									yzLine.start.y = NUMLATPIXELS - 2 - height;
								}
							    }	
							    flag = False;
		 						if (xIsCompressed) {
								  	width = xcLine.end.x - xcLine.start.x;
		
									/* translate compressed x axis to new yC loc */
								  	xcLine.start.y = yzLine.start.y + height/2;
								  	xcLine.end.y = xcLine.start.y;
							  		if ((newx = x - width/2) > 2)
										xcLine.start.x = newx;
							  		else {
										xcLine.start.x = 2;
										xcLine.end.x = 2 + width;
										yzLine.start.x = width/2 + 2;
										yzLine.end.x = yzLine.start.x;
										flag = True;
							    		}
							  		if (!flag) {
										if ((newx = x + width/2) <= NUMLONPIXELS - 2)
											xcLine.end.x = newx;
										else {
											xcLine.end.x = NUMLONPIXELS - 2;
											xcLine.start.x = NUMLONPIXELS - 2 - width;
											yzLine.start.x = NUMLONPIXELS - 2 - width/2;
											yzLine.end.x = yzLine.start.x;
										}
							  	  	}
							  	 	flag = False;
								 	DrawXcLine();
								}
							    DrawYzLine();
						      }
						      else if (type == yL) {
							    if (y <= oldy) {
								yzLine.start.y = y;
								yzLine.end.y = oldy;
							    }
							    else {
								yzLine.end.y = y;
								yzLine.start.y = oldy;
							    }
							    DrawYzLine();
							    if (xIsCompressed) {
								xcLine.start.y = yzLine.start.y + (yzLine.end.y - yzLine.start.y)/2;
								xcLine.end.y = xcLine.start.y;
								DrawXcLine();
							     }
						      }
		
						      /* compressed y axis handles */
						      else if (type == cyU) {
							    if (y >= oldy) {
								ycLine.end.y = y;
								ycLine.start.y = oldy;
							    }
							    else {
								ycLine.start.y = y;
								ycLine.end.y = oldy;
							    }
							    DrawYcLine();
							    if (xIsCompressed) {
								  xcLine.start.y = ycLine.start.y + (ycLine.end.y - ycLine.start.y)/2;
								  xcLine.end.y = xcLine.start.y;
								  DrawXcLine();	
								  if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in y */
								    mapLoc.y = xcLine.start.y + (xcLine.end.y - xcLine.start.y)/2;
								    DrawPoint();
								  }				
							    }
							    else {
								  if (geo == 1 || geo == 6 || geo == 7 || geo == 13) {
								    xzLine.start.y = ycLine.start.y + (ycLine.end.y - ycLine.start.y)/2;
								    xzLine.end.y = xzLine.start.y;
								    DrawXzLine();
								  }
								  else if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in y */
								    mapLoc.y = xcLine.start.y + (xcLine.end.y - xcLine.start.y)/2;
								    DrawPoint();
								  }
							    }
						      }
						      else if (type == cyL) {
							    if (y <= oldy) {
								ycLine.start.y = y;
								ycLine.end.y = oldy;
							    }
							    else {
								ycLine.end.y = y;
								ycLine.start.y = oldy;
							    }
							    DrawYcLine();
							    if (xIsCompressed) {
								  xcLine.start.y = ycLine.start.y + (ycLine.end.y - ycLine.start.y)/2;
								  xcLine.end.y = xcLine.start.y;
								  DrawXcLine();	
								  if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in y */
								    mapLoc.y = xcLine.start.y + (xcLine.end.y - xcLine.start.y)/2;
								    DrawPoint();
								  }				
							    }
							    else {
								  if (geo == 1 || geo == 6 || geo == 7 || geo == 13) {
								    xzLine.start.y = ycLine.start.y + (ycLine.end.y - ycLine.start.y)/2;
								    xzLine.end.y = xzLine.start.y;
								    DrawXzLine();
								  }
								  else if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in y */
								    mapLoc.y = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
							    }
						      }
						}
					}
					break;
				case ButtonRelease:
					if (UxEvent->xbutton.button == Button1) {
						gNumHandles = 0;
						gStoreHandle = True;
						ClearMap();
						x = UxEvent->xbutton.x;
						y = UxEvent->xbutton.y;
						x = x > NUMLONPIXELS-2 ? NUMLONPIXELS-2 : x;
						x = x < 2 ? 2 : x;
						y = y > NUMLATPIXELS-2 ? NUMLATPIXELS-2 : y;;
						y = y < 2 ? 2 : y;
		
						if (!gDragInHandle) {
						      dragMode = 0;
						      switch (toolMode) {
							    case xy:
								xyRect.width = abs(x - startx);
								xyRect.height = abs(y - starty);
		
								/* find top left corner */
								xyRect.x = (xyRect.x < x) ? xyRect.x : x;
								xyRect.y = (xyRect.y < y) ? xyRect.y : y;
								DrawXyRect();
								break;
							    case xl:
								xzLine.end.x = x;
								DrawXzLine();

								   if (yIsCompressed) {
									ycLine.start.x = xzLine.start.x + (xzLine.end.x - xzLine.start.x)/2;
									ycLine.end.x = ycLine.start.x;

								        /* does the compressed axis line fit on top of map? */
							    		if ((newy = y - (ycLine.end.y - ycLine.start.y)/2) > 2) {
										/* does */
									        newHeight = (ycLine.end.y - ycLine.start.y)/2;
										ycLine.start.y = xzLine.start.y - newHeight;
										ycLine.end.y = xzLine.start.y + newHeight;
									}
							    		else {
										/* doesn't fit--reduce height to fit  */
										ycLine.start.y = 2;
										newHeight = 2 * (xzLine.start.y - 2);
										ycLine.end.y = ycLine.start.y + newHeight;
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* does the compressed axis fit on bottom of map?  */
										if ((newy = y + (ycLine.end.y - ycLine.start.y)/2) <= NUMLATPIXELS - 2) {
											/* does */
										        ycLine.start.y = xzLine.start.y - newHeight;
											ycLine.end.y = xzLine.start.y + newHeight;
										}
										else {
											/* doesn't */
											ycLine.end.y = NUMLATPIXELS - 2;
											newHeight = 2 * (ycLine.end.y - xzLine.start.y);
											ycLine.start.y = ycLine.end.y - newHeight;
										}
							   		 }	
							   		flag = False;
								  	DrawYcLine();
								   }
								break;
							    case yl:
								yzLine.end.y = y;
								DrawYzLine();
								   if (xIsCompressed) {
								        xcLine.start.y = yzLine.start.y + (yzLine.end.y - yzLine.start.y)/2;
								        xcLine.end.y = xcLine.start.y;
									
									/* test whether the left side of compressed axis fits on map */
							  		if ((newx = x - (xcLine.end.x - xcLine.start.x)/2) > 2) {
									        /* does */
									        newWidth = (xcLine.end.x - xcLine.start.x)/2;
										xcLine.start.x = yzLine.start.x - newWidth;
										xcLine.end.x = yzLine.start.x + newWidth;
									}
							  		else {
									       /* doesn't */
										xcLine.start.x = 2;
										newWidth = 2 * (yzLine.start.x - 2);
										xcLine.end.x = xcLine.start.x + newWidth;
										flag = True;
							    		}
									
									/* test whether right side of compressed axis fits on map */
							  		if (!flag) {
										if ((newx = x + (xcLine.end.x - xcLine.start.x)/2) <= NUMLONPIXELS - 2) {
										        /* does */
										        xcLine.start.x = yzLine.start.x - newWidth;
										        xcLine.end.x = yzLine.start.x + newWidth;
										}
										else {
											xcLine.end.x = NUMLONPIXELS - 2;
											newWidth = 2 * (xcLine.end.x - yzLine.start.x);
											xcLine.start.x = xcLine.end.x - newWidth;
										}
							  	  	}
							  	 	flag = False;
									DrawXcLine();
								   }
								break;
							    case pt:
								mapLoc.x = x;
								mapLoc.y = y;
								   if (!yIsCompressed && !xIsCompressed)
								   	DrawPoint();
		
								   if (yIsCompressed) {
							  		/* translate compressed y axis to new mapLoc */
							  		height = ycLine.end.y - ycLine.start.y;
		
									/*translate in x */
							  		ycLine.start.x = mapLoc.x;
							  		ycLine.end.x = mapLoc.x;
	
									/* check whether top of y line bumps into border */
							    		if ((newy = y - height/2) > 2)
										/* doesn't */
										ycLine.start.y = mapLoc.y - height/2;
							    		else {
										/* does */
										ycLine.start.y = 2;
										ycLine.end.y = 2 + height;
										mapLoc.y = height/2 + 2;
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newy = y + height/2) <= NUMLATPIXELS - 2)
											/* doesn't */
											ycLine.end.y = mapLoc.y + height/2;
										else {
											/* does */
											ycLine.end.y = NUMLATPIXELS - 2;
											ycLine.start.y = NUMLATPIXELS - 2 - height;
											mapLoc.y = NUMLATPIXELS - 2 - height/2;
										}
							   		 }	
							   		 flag = False;
								 }

								   if (xIsCompressed) {
							  		/* translate compressed x axis to new mapLoc */
								  	width = xcLine.end.x - xcLine.start.x;

									/*translate in y */
							  		xcLine.start.y = mapLoc.y;
							  		xcLine.end.y = mapLoc.y;
		
									/* check whether top of y line bumps into border */
							    		if ((newx = x - width/2) > 2)
										/* doesn't */
										xcLine.start.x = mapLoc.x - width/2;
							    		else {
										/* does */
										xcLine.start.x = 2;
										xcLine.end.x = 2 + width;
										mapLoc.x = width/2 + 2;
										if (yIsCompressed) {
										      ycLine.start.x = mapLoc.x;
										      ycLine.end.x = mapLoc.x;
										}
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newx = x + width/2) <= NUMLONPIXELS - 2)
											/* doesn't */
											xcLine.end.x = mapLoc.x + width/2;
										else {
											/* does */
											xcLine.end.x = NUMLONPIXELS - 2;
											xcLine.start.x = NUMLONPIXELS - 2 - width;
											mapLoc.x = NUMLONPIXELS - 2 - width/2;
											if (yIsCompressed) {
											  ycLine.start.x = mapLoc.x;
											  ycLine.end.x = mapLoc.x;
											}
										}
							   		 }	
							   		 flag = False;
								}
							        if (yIsCompressed)
								  DrawYcLine();
							        if (xIsCompressed)
								  DrawXcLine();
       								DrawPoint();
								break;
						      	}
						      }
						      else {
							    /* in drag handle */
							    if (type == PT) {
								mapLoc.x = x;
								mapLoc.y = y;
								if (!yIsCompressed && !xIsCompressed)
								  DrawPoint();
		
								if (yIsCompressed) {
								  /* translate compressed y axis to new mapLoc */
								  height = ycLine.end.y - ycLine.start.y;
		
								  /*translate in x */
								  ycLine.start.x = mapLoc.x;
								  ycLine.end.x = mapLoc.x;
	
								  /* check whether top of y line bumps into border */
								  if ((newy = y - height/2) > 2)
								    /* doesn't */
								    ycLine.start.y = mapLoc.y - height/2;
								  else {
								    /* does */
								    ycLine.start.y = 2;
								    ycLine.end.y = 2 + height;
								    mapLoc.y = height/2 + 2;
								    flag = True;
								  }
		
								  if (!flag) {
								    /* check whether bottom bumps into border */
								    if ((newy = y + height/2) <= NUMLATPIXELS - 2)
								      /* doesn't */
								      ycLine.end.y = mapLoc.y + height/2;
								    else {
								      /* does */
								      ycLine.end.y = NUMLATPIXELS - 2;
								      ycLine.start.y = NUMLATPIXELS - 2 - height;
								      mapLoc.y = NUMLATPIXELS - 2 - height/2;
								    }
								  }	
								  flag = False;
								}

							        if (xIsCompressed) {
							  		/* translate compressed x axis to new mapLoc */
								  	width = xcLine.end.x - xcLine.start.x;

									/*translate in y */
							  		xcLine.start.y = mapLoc.y;
							  		xcLine.end.y = mapLoc.y;
		
									/* check whether top of y line bumps into border */
							    		if ((newx = x - width/2) > 2)
										/* doesn't */
										xcLine.start.x = mapLoc.x - width/2;
							    		else {
										/* does */
										xcLine.start.x = 2;
										xcLine.end.x = 2 + width;
										mapLoc.x = width/2 + 2;
										if (yIsCompressed) {
										      ycLine.start.x = mapLoc.x;
										      ycLine.end.x = mapLoc.x;
										}
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newx = x + width/2) <= NUMLONPIXELS - 2)
											/* doesn't */
											xcLine.end.x = mapLoc.x + width/2;
										else {
											/* does */
											xcLine.end.x = NUMLONPIXELS - 2;
											xcLine.start.x = NUMLONPIXELS - 2 - width;
											mapLoc.x = NUMLONPIXELS - 2 - width/2;
											if (yIsCompressed) {
											  ycLine.start.x = mapLoc.x;
											  ycLine.end.x = mapLoc.x;
											}
										}
							   		 }	
							   		 flag = False;
								}
							        if (yIsCompressed)
								  DrawYcLine();
							        if (xIsCompressed)
								  DrawXcLine();
								DrawPoint();
								dragMode = DRAG_PT;
							    }
							    if (type == CC) {
								if (x >= width/2 + 2)
									xyRect.x = x - width/2;
								else 
									xyRect.x = 2;
		
								  if (x + width/2 > NUMLONPIXELS - 2)
									  xyRect.x = NUMLONPIXELS - 2 - width;
		
								if (y >= height/2 + 2)
									xyRect.y = y - height/2;
								else 
									xyRect.y = 2;
										
								if (y + height/2 > NUMLATPIXELS - 2)
									  xyRect.y = NUMLATPIXELS - 2 - height;
								DrawXyRect();
								dragMode = DRAG_RECT;
							    }
							    else if (type == LL || type == Cl || type == LR || type == CR ||
								     type == UR || type == CU || type == UL || type == CL) {
								DrawXyRect();
								dragMode = DRAG_RECT;
							    }
							    else if (type == xC) {
								if (x - width/2 < 2) {
									xzLine.start.x = 2;
									xzLine.end.x = xzLine.start.x + width;
								}
								else if (x + width/2 > NUMLONPIXELS - 2) {
									xzLine.end.x = NUMLONPIXELS - 2;
									xzLine.start.x = xzLine.end.x - width;
								}
								else {
									xzLine.start.x = x - width/2;
									xzLine.end.x = x + width/2;
								}
								xzLine.start.y = y;
								xzLine.end.y = y;
							  	if (yIsCompressed) {
							  		height = ycLine.end.y - ycLine.start.y;
		
									/*translate in x */
							    		ycLine.start.x = xzLine.start.x + width/2;
							    		ycLine.end.x = ycLine.start.x;
		
									/* check whether top of y line bumps into border */
							    		if ((newy = y - height/2) > 2)
										/* doesn't */
										ycLine.start.y = newy;
							    		else {
										/* does */
										ycLine.start.y = 2;
										ycLine.end.y = 2 + height;
										xzLine.start.y = height/2 + 2;
										xzLine.end.y = xzLine.start.y;
										flag = True;
							    		 }
		
							   		 if (!flag) {
										/* check whether bottom bumps into border */
										if ((newy = y + height/2) <= NUMLATPIXELS - 2)
											/* doesn't */
											ycLine.end.y = newy;
										else {
											/* does */
											ycLine.end.y = NUMLATPIXELS - 2;
											ycLine.start.y = NUMLATPIXELS - 2 - height;
											xzLine.start.y = NUMLATPIXELS - 2 - height/2;
											xzLine.end.y = xzLine.start.y;
										}
							   		 }	
							   		 flag = False;
							  		 DrawYcLine();
							   	 }
								DrawXzLine();
								dragMode = DRAG_XL;
							    }
							    else if (type == xL || type == xR) {
								DrawXzLine();
								if (yIsCompressed) {
								  ycLine.start.x = xzLine.start.x + (xzLine.end.x - xzLine.start.x)/2;
								  ycLine.end.x = ycLine.start.x;
								  DrawYcLine();
								}
								dragMode = DRAG_XL;
							    }
							    else if (type == cxL || type == cxR) {
								DrawXcLine();
								xzLine.start.x = xcLine.start.x;
								xzLine.end.x = xcLine.end.x;
								if (yIsCompressed) {
								  ycLine.start.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								  ycLine.end.x = ycLine.start.x;
								  DrawYcLine();		
								  if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in x */
								    mapLoc.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
								}
								else {
								  if (geo == 2 || geo == 8 || geo == 9 || geo == 14) {
								    yzLine.start.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    yzLine.end.x = yzLine.start.x;
								    DrawYzLine();
								  }
								  else if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in x */
								    mapLoc.x = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
								}
								dragMode = DRAG_CXL;
							    }
							    else if (type == yC) {
								if (y - height/2 < 2) {
									  yzLine.start.y = 2;
									yzLine.end.x = xzLine.start.y + height;
								}
								else if (y + height/2 > NUMLATPIXELS - 2) {
									  yzLine.end.y = NUMLATPIXELS - 2;
									yzLine.start.y = yzLine.end.y - height;
								}
								else {
									 yzLine.start.y = y - height/2;
									yzLine.end.y = y + height/2;
								}
								yzLine.start.x = x;
								yzLine.end.x = x;
		 						if (xIsCompressed) {
								  	width = xcLine.end.x - xcLine.start.x;
		
									/* translate compressed x axis to new yC loc */
								  	xcLine.start.y = yzLine.start.y + height/2;
								  	xcLine.end.y = xcLine.start.y;
							  		if ((newx = x - width/2) > 2)
										xcLine.start.x = newx;
							  		else {
										xcLine.start.x = 2;
										xcLine.end.x = 2 + width;
										yzLine.start.x = width/2 + 2;
										yzLine.end.x = yzLine.start.x;
										flag = True;
							    		}
							  		if (!flag) {
										if ((newx = x + width/2) <= NUMLONPIXELS - 2)
											xcLine.end.x = newx;
										else {
											xcLine.end.x = NUMLONPIXELS - 2;
											xcLine.start.x = NUMLONPIXELS - 2 - width;
											yzLine.start.x = NUMLONPIXELS - 2 - width/2;
											yzLine.end.x = yzLine.start.x;
										}
							  	  	}
							  	 	flag = False;
								 	DrawXcLine();
								}
								DrawYzLine();
								dragMode = DRAG_YL;
							    }
							    else if (type == yL || type == yU) {
								DrawYzLine();
								if (xIsCompressed) {
								  xcLine.start.y = yzLine.start.y + (yzLine.end.y - yzLine.start.y)/2;
								  xcLine.end.y = xcLine.start.y;
								  DrawXcLine();
								}
								dragMode = DRAG_YL;
							    }
							    else if (type == cyL || type == cyU) {
								DrawYcLine();
								yzLine.start.y = ycLine.start.y;
								yzLine.end.y = ycLine.end.y;
								if (xIsCompressed) {
								  xcLine.start.y = ycLine.start.y + (ycLine.end.y - ycLine.start.y)/2;
								  xcLine.end.y = xcLine.start.y;
								  DrawXcLine();		
								  if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in y */
								    mapLoc.y = xcLine.start.y + (xcLine.end.y - xcLine.start.y)/2;
								    DrawPoint();
								  }				
								}
								else {
								  if (geo == 1 || geo == 6 || geo == 7 || geo == 13) {
								    xzLine.start.y = ycLine.start.y + (ycLine.end.y - ycLine.start.y)/2;
								    xzLine.end.y = xzLine.start.y;
								    DrawXzLine();
								  }
								  else if (geo == 0 || geo == 3 || geo == 4 || geo == 10) {
								    /* translate point in y */
								    mapLoc.y = xcLine.start.x + (xcLine.end.x - xcLine.start.x)/2;
								    DrawPoint();
								  }
								}
								dragMode = DRAG_CYL;
							    }
						      }	  
					      }
					}
					FramerToRegion();
					gDragInHandle = False;
					break;
			}
			XUngrabButton(UxEvent->xany.display, Button1, AnyModifier, XtWindow(UxGetWidget(drawingArea1)));
}
