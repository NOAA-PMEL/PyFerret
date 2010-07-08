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

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.List;

import gov.noaa.pmel.ferret.threddsBrowser.LocalDirTreeScanner;

import org.junit.Test;

import thredds.catalog.InvAccess;
import thredds.catalog.InvCatalogImpl;
import thredds.catalog.InvDataset;

/**
 * Test of {@link LocalDirTreeScanner}
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class TestLocalDirTreeScanner {

	/**
	 * Test method for {@link LocalDirTreeScanner#LocalDirTreeScanner(File)}.
	 */
	@Test
	public void testLocalDirTreeScanner() throws IOException {
		// Try constructor with null
		try {
			new LocalDirTreeScanner(null);
			fail("constructor with null did not throw an exception");
		} catch (NullPointerException e) {
			;
		}

		// Try constructor with a file
		File tmpdir = File.createTempFile("test_", "_tmpdir");
		try {
			new LocalDirTreeScanner(tmpdir);
			fail("constructor with a file did not throw an exception");
		} catch (IOException e) {
			;
		}

		// Try constructor with directory name that does not exist
		tmpdir.delete();
		try {
			new LocalDirTreeScanner(tmpdir);
			fail("constructor with a directory that does not exist did not throw an exception");
		} catch (IOException e) {
			;
		}

		// Try constructor with a valid directory
		tmpdir.mkdir();
		LocalDirTreeScanner scanner = new LocalDirTreeScanner(tmpdir);
		assertNotNull("constructor with a valid directory returned null", scanner);

		tmpdir.delete();
	}

	/**
	 * Test method for {@link LocalDirTreeScanner#generateCatalog(FileFilter)}.
	 */
	@Test
	public void testGenerateCatalog() throws IOException {
		// Create a directory tree
		File tmpdir = File.createTempFile("test_", "_tmpdir");
		tmpdir.delete();
		tmpdir.mkdir();

		File afile = new File(tmpdir, "a_file");
		afile.createNewFile();

		File zfile = new File(tmpdir, "z_file");
		zfile.createNewFile();

		File subdir = new File(tmpdir, "subdir");
		subdir.mkdir();

		File subfile = new File(subdir, "subfile");
		subfile.createNewFile();

		File asubfile = new File(subdir, "a_subfile");
		asubfile.createNewFile();

		// Create a scanner for this directory tree
		LocalDirTreeScanner scanner = new LocalDirTreeScanner(tmpdir);
		assertNotNull("constructor with a valid directory returned null", scanner);

		// Try scanning with no filter
		InvCatalogImpl catalog = scanner.generateCatalog(null);
		assertNotNull("generated catalog with null filter was null", catalog);

		// Top-level dataset is the directory itself
		List<InvDataset> datasets = catalog.getDatasets();
		assertEquals("Number of first-level datasets", 1, datasets.size());

		assertEquals("ID of first-level dataset", tmpdir.getPath(), datasets.get(0).getID());
		List<InvAccess> access = datasets.get(0).getAccess();
		assertEquals("Number of accesses for first-level dataset", 0, access.size());
		assertEquals("Number of datasets under the first-level dataset", 3, datasets.get(0).getDatasets().size());

		// Next level is the sorted contents of the directory
		datasets = datasets.get(0).getDatasets();

		assertEquals("ID of first second-level dataset", afile.getPath(), datasets.get(0).getID());
		access = datasets.get(0).getAccess();
		assertEquals("Number of accesses for first second-level dataset", 1, access.size());
		assertEquals("Number of datasets under the first second-level datset", 0, datasets.get(0).getDatasets().size());

		assertEquals("ID of second second-level dataset", subdir.getPath(), datasets.get(1).getID());
		access = datasets.get(1).getAccess();
		assertEquals("Number of accesses for second second-level dataset", 0, access.size());
		assertEquals("Number of datasets under the second second-level datset", 2, datasets.get(1).getDatasets().size());

		assertEquals("ID of third second-level dataset", zfile.getPath(), datasets.get(2).getID());
		access = datasets.get(2).getAccess();
		assertEquals("Number of accesses for third second-level dataset", 1, access.size());
		assertEquals("Number of datasets under the third second-level datset", 0, datasets.get(2).getDatasets().size());

		// Now the contents of subdir
		datasets = datasets.get(1).getDatasets();
		
		assertEquals("ID of first third-level dataset", asubfile.getPath(), datasets.get(0).getID());
		access = datasets.get(0).getAccess();
		assertEquals("Number of accesses for first third-level dataset", 1, access.size());
		assertEquals("Number of datasets under the first third-level datset", 0, datasets.get(0).getDatasets().size());

		assertEquals("ID of second third-level dataset", subfile.getPath(), datasets.get(1).getID());
		access = datasets.get(1).getAccess();
		assertEquals("Number of accesses for second third-level dataset", 1, access.size());
		assertEquals("Number of datasets under the second third-level datset", 0, datasets.get(1).getDatasets().size());

		// Try scanning with a filter
		catalog = scanner.generateCatalog(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				// Accept all directories
				if ( pathname.isDirectory() )
					return true;
				// Accept files with an underscore in their name
				if ( pathname.getName().contains("_") )
					return true;
				// If made it here, don't accept it
				return false;
			}
		});

		// Top-level dataset is the directory itself
		datasets = catalog.getDatasets();
		assertEquals("Number of first-level datasets", 1, datasets.size());

		assertEquals("ID of first-level dataset", tmpdir.getPath(), datasets.get(0).getID());
		access = datasets.get(0).getAccess();
		assertEquals("Number of accesses for first-level dataset", 0, access.size());
		assertEquals("Number of datasets under the first-level dataset", 3, datasets.get(0).getDatasets().size());

		// Next level is the sorted contents of the directory
		datasets = datasets.get(0).getDatasets();

		assertEquals("ID of first second-level dataset", afile.getPath(), datasets.get(0).getID());
		access = datasets.get(0).getAccess();
		assertEquals("Number of accesses for first second-level dataset", 1, access.size());
		assertEquals("Number of datasets under the first second-level datset", 0, datasets.get(0).getDatasets().size());

		assertEquals("ID of second second-level dataset", subdir.getPath(), datasets.get(1).getID());
		access = datasets.get(1).getAccess();
		assertEquals("Number of accesses for second second-level dataset", 0, access.size());
		assertEquals("Number of datasets under the second second-level datset", 1, datasets.get(1).getDatasets().size());

		assertEquals("ID of third second-level dataset", zfile.getPath(), datasets.get(2).getID());
		access = datasets.get(2).getAccess();
		assertEquals("Number of accesses for third second-level dataset", 1, access.size());
		assertEquals("Number of datasets under the third second-level datset", 0, datasets.get(2).getDatasets().size());

		// Now the contents of subdir
		datasets = datasets.get(1).getDatasets();
		
		assertEquals("ID of first third-level dataset", asubfile.getPath(), datasets.get(0).getID());
		access = datasets.get(0).getAccess();
		assertEquals("Number of accesses for first third-level dataset", 1, access.size());
		assertEquals("Number of datasets under the first third-level datset", 0, datasets.get(0).getDatasets().size());

		asubfile.delete();
		subfile.delete();
		subdir.delete();
		zfile.delete();
		afile.delete();
		tmpdir.delete();
	}

}
