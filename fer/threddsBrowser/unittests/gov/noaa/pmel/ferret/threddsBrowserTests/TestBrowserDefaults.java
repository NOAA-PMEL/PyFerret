/**
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
 */
package gov.noaa.pmel.ferret.threddsBrowserTests;

import static org.junit.Assert.*;

import gov.noaa.pmel.ferret.threddsBrowser.BrowserDefaults;

import java.awt.Dimension;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;

import org.junit.Test;

import ucar.util.prefs.PreferencesExt;

/**
 * Tests for {@link BrowserDefaults}
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class TestBrowserDefaults {

	/**
	 * Test method for {@link BrowserDefaults#BrowserDefaults(PreferencesExt, String)}.
	 */
	@Test
	public void testBrowserDefaults() {
		// Test the (null,null) constructor
		BrowserDefaults defs = new BrowserDefaults(null, null);
		assertNotNull("BrowserDefaults(null,null)", defs);
		Collection<String> locs = defs.getLocationStrings();
		assertEquals("number of initial locations from (null,null)", 0, locs.size());
		int defDividerLoc = defs.getDividerLocation();
		assertTrue("divider location from (null,null) > 50", defDividerLoc > 50);
		File localDir = defs.getLocalBrowseDir();
		assertEquals("local browse directory from (null,null)", null, localDir);
		String defExtsString = defs.getExtensionsString();
		assertNotNull("extensions string from (null,null) not null", defExtsString);
		Dimension defSize = defs.getBrowserSize();
		assertNotNull("browser size from (null,null) not null", defSize);

		// Test the constructor with an undefined environment variable
		String envName = "JUNK498523851498";
		String envVal = System.getenv(envName);
		assertNull("made-up bizzare environment variable exists!", envVal);
		defs = new BrowserDefaults(null, envName);
		assertNotNull("BrowserDefaults(null," + envName + ")", defs);
		locs = defs.getLocationStrings();
		assertEquals("number of initial locations from (null," + envName + ")", 0, locs.size());
		int dividerLoc = defs.getDividerLocation();
		assertEquals("divider location from (null," + envName + ")", defDividerLoc, dividerLoc);
		localDir = defs.getLocalBrowseDir();
		assertEquals("local browse directory from (null," + envName + ")", null, localDir);
		String extsString = defs.getExtensionsString();
		assertEquals("extensions string from (null," + envName + ")", defExtsString, extsString);
		Dimension size = defs.getBrowserSize();
		assertEquals("browser size from (null," + envName + ")", defSize, size);
	}

	/**
	 * Test method for {@link BrowserDefaults#saveLocationsList(PreferencesExt, List<String>)}
	 * and {@link BrowserDefaults#getLocationStrings()}.
	 */
	@Test
	public void testSaveGetLocationStrings() {
		String[] locations = {"first", "second", "third", "fourth", "fifth"};
		PreferencesExt prefs = new PreferencesExt(null, "");
		BrowserDefaults.saveLocationsList(prefs, Arrays.asList(locations));
		BrowserDefaults defs = new BrowserDefaults(prefs, null);
		Collection<String> locsColl = defs.getLocationStrings();
		assertEquals("number of locations", locations.length, locsColl.size());
		int k = 0;
		for (String loc : locsColl) {
			assertEquals("location[" + k + "]", locations[k], loc);
			k++;
		}

		// Use "path" as the environment variable to test (no System.setenv to create one)
		String envName = "path";
		String envVal = System.getenv(envName);

		// Following are not errors; just can't perform this test as expected
		assertNotNull("\"path\" environment variable is not defined", envVal);
		assertFalse("\"path\" environment variable is blank", envVal.trim().isEmpty());
		assertFalse("\"path\" environment variable contains double quotes", envVal.contains("\""));
		assertFalse("\"path\" environment variable contains single quotes", envVal.contains("'"));

		// Get the unique space-separated strings in the given order
		LinkedHashSet<String> nameSet = new LinkedHashSet<String>(Arrays.asList(envVal.split("\\s+")));
		ArrayList<String> nameList = new ArrayList<String>(nameSet);

		// Get the locations from just using envName
		defs = new BrowserDefaults(null, envName);
		locsColl = defs.getLocationStrings();
		assertEquals("number of locations from " + envName, nameList.size(), locsColl.size());
		int numLocs = 0;
		for (String location : locsColl) {
			assertEquals("location[" + numLocs + "] from " + envName, nameList.get(numLocs), location);
			numLocs++;
		}

		// Create a list of locations, one duplicating the last one from envName 
		String myFirstLocation = "My silly first location";
		String mySecondLocation = nameList.get(nameList.size() - 1);
		ArrayList<String> prefLocs = new ArrayList<String>(2);
		prefLocs.add(myFirstLocation);
		prefLocs.add(mySecondLocation);

		// Create a BrowserDefaults with this PreferenceExt and envName
		BrowserDefaults.saveLocationsList(prefs, prefLocs);  // verify the old values are overwritten
		defs = new BrowserDefaults(prefs, envName);
		locsColl = defs.getLocationStrings();
		assertEquals("number of locations from prefs and " + envName, nameList.size() + 1, locsColl.size());
		numLocs = 0;
		for (String location : locsColl) {
			if ( numLocs == 0 )
				assertEquals("location[0] from prefs and " + envName, myFirstLocation, location);
			else if ( numLocs == 1 )
				assertEquals("location[1] from prefs and " + envName, mySecondLocation, location);
			else
				assertEquals("location[" + numLocs + "] from prefs and " + envName, nameList.get(numLocs-2), location);
			numLocs++;
		}
}

	/**
	 * Test method for {@link BrowserDefaults#saveDividerLocation(PreferencesExt, int)}
	 * and {@link BrowserDefaults#getDividerLocation()}.
	 */
	@Test
	public void testSaveGetDividerLocation() {
		int loc = 789;
		PreferencesExt prefs = new PreferencesExt(null, "");
		BrowserDefaults.saveDividerLocation(prefs, loc);
		BrowserDefaults defs = new BrowserDefaults(prefs, null);
		assertEquals("saved divider location", loc, defs.getDividerLocation());
	}

	/**
	 * Test method for {@link BrowserDefaults#saveLocalBrowseDir(PreferencesExt, File)}
	 * and {@link BrowserDefaults#getLocalBrowseDir()}.
	 */
	@Test
	public void testSaveGetLocalBrowseDir() throws IOException {
		File tmpdir = File.createTempFile("temp_", "_dir");
		PreferencesExt prefs = new PreferencesExt(null, "");
		BrowserDefaults.saveLocalBrowseDir(prefs, tmpdir);
		BrowserDefaults defs = new BrowserDefaults(prefs, null);
		assertEquals("saved local browse dir that is a file", null, defs.getLocalBrowseDir());
		tmpdir.delete();
		defs = new BrowserDefaults(prefs, null);
		assertEquals("saved local browse dir that does not exist", null, defs.getLocalBrowseDir());
		tmpdir.mkdir();
		defs = new BrowserDefaults(prefs, null);
		assertEquals("saved local browse dir that is a valid directory", tmpdir, defs.getLocalBrowseDir());
		tmpdir.delete();
	}

	/**
	 * Test method for {@link BrowserDefaults#saveExtensionsString(PreferencesExt, String)}
	 * and {@link BrowserDefaults#getExtensionsString()}.
	 */
	@Test
	public void testSaveGetExtensionsString() {
		String extsString = "this, that, and, the, other, thing";
		PreferencesExt prefs = new PreferencesExt(null, "");
		BrowserDefaults.saveExtensionsString(prefs, extsString);
		BrowserDefaults defs = new BrowserDefaults(prefs, null);
		assertEquals("saved extensions string", extsString, defs.getExtensionsString());
	}

	/**
	 * Test method for {@link BrowserDefaults#saveBrowserSize(PreferencesExt, Dimension)}
	 * and {@link BrowserDefaults#getBrowserSize()}.
	 */
	@Test
	public void testSaveGetBrowserSize() {
		Dimension size = new Dimension(234, 567);
		PreferencesExt prefs = new PreferencesExt(null, "");
		BrowserDefaults.saveBrowserSize(prefs, size);
		BrowserDefaults defs = new BrowserDefaults(prefs, null);
		assertEquals("saved browser size", size, defs.getBrowserSize());
	}

	/**
	 * Test method for {@link BrowserDefaults#parseExtensionsString(String)}
	 * and {@link BrowserDefaults#createExtensionsString(Collection)}.
	 */
	@Test
	public void testParseCreateExtensionsString() {
		String extsString = "this, that, and, the, other";
		String complicatedString = "  this  that;and , the  ,other  ; ";
		String[] extsArray = extsString.split(", ");

		// Try a single string of extensions
		Collection<String> extsColl = BrowserDefaults.parseExtensionsString(extsString);
		assertEquals("number of extensions from a simple string", extsArray.length, extsColl.size());
		for (int k = 0; k < extsArray.length; k++)
		   assertTrue("collection contains extension " + extsArray[k], extsColl.contains(extsArray[k]));
		String newStr = BrowserDefaults.createExtensionsString(extsColl);
		assertEquals("extensions string from collection", extsString, newStr);

		// Try a complicated string
		extsColl = BrowserDefaults.parseExtensionsString(complicatedString);
		assertEquals("number of extensions from a complicated string", extsArray.length, extsColl.size());
		for (int k = 0; k < extsArray.length; k++)
		   assertTrue("collection contains extension " + extsArray[k], extsColl.contains(extsArray[k]));
		newStr = BrowserDefaults.createExtensionsString(extsColl);
		assertEquals("extensions string from collection", extsString, newStr);

		// Should work fine with an empty string or empty collection
		extsColl = BrowserDefaults.parseExtensionsString("");
		assertEquals("number of extensions from an empty string", 0, extsColl.size());
		newStr = BrowserDefaults.createExtensionsString(extsColl);
		assertEquals("extensions string from empty collection", "", newStr);
	}

}
