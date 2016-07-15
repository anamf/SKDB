/** \mainpage Gigal: An open source system for classification learning from very large data
 * Copyright (C) 2012 Geoffrey I Webb
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <new>

#include "instanceFile.h"
#include "instanceStreamDiscretiser.h"
#include "instanceStreamClassFilter.h"
#include "instanceStreamFilter.h"
#include "learner.h"
#include "mtrand.h"
#include "utils.h"
#include "globals.h"
#include "FILEtype.h"
#include "learnerRegistry.h"
#include "ALGLIB_ap.h"
#include "FilterSet.h"

// Train & test utilities
#include "trainTest.h"
#include "xVal.h"

/** 
 * Type of experiment and evaluation method.
 */
enum experimentType {
	etNone, /**< Nothing is done by default. */
	etTrainTest, /**< Use training set for testing (specified with -t) */
	etXVal, /**< Cross-validation (-x10 by default). */
};

/**
 * @param argv Options for the experiment
 * @param argc Number of options
 * @return An integer 0 upon exit success
 */
int main(int argc, char* const argv[]) {
	MTRand rand;
	char* testfilename = NULL;
	experimentType et = etNone;
	char* expArgs = NULL;
	std::vector<learner*> theLearners;
	char* const * eXValArgv = NULL;
	int eXValArgc = 0;
	char* const * argvEnd = argv + argc;
	FilterSet filters;
	TrainTestArgs ttArgs;

	// First parse the command line arguments
	try {
		printf("======================\n"
				"Gigal: the system for learning from big data\nVersion 0.2\n");
		for (int i = 0; i < argc; i++) {
			printf("%s ", argv[i]);
		}
		putchar('\n');
		putchar('\n');

		if (argc < 3) {
			error("Usage: %s <metafile> <trainingfile> [-p<posClassName>]"
					" [<test method args>] -l<learner> [<learner args>]",
					argv[0]);
		}

		InstanceFile instanceFile(argv[1], argv[2]);
		InstanceStream* instanceStream = &instanceFile;

		argv += 3; // skip the program name, the meta file name and the data file name

		while (argv != argvEnd) {
			if (**argv != '-') {
				error("Argument '%s' requires '-'", *argv);
			}

			char *p = argv[0] + 1;

			switch (*p) {
			case 'd':
				// discretise
				filters.push_back(
						new InstanceStreamDiscretiser(p + 1, ++argv, argvEnd));
				break;
			case 'l':
				// specify the learner

				// create the learner
				theLearners.push_back(createLearner(p + 1, ++argv, argvEnd));

				if (theLearners.back() == NULL) {
					error("Learner %s is not supported", p + 1);
				}
				break;
			case 'p':
				// filter the classes into binary classification
				instanceStream = new InstanceStreamClassFilter(instanceStream,
						p + 1, ++argv, argvEnd);
				break;
			case 't':
				// use a trainingfile-testfile experiment
				// the testfile name must follow the t
				et = etTrainTest;
				testfilename = p + 1;
				ttArgs.getArgs(++argv, argvEnd);
				break;
			case 'v':
				// set the verbosity level - the default is 1
				getUIntFromStr(p + 1, verbosity, "verbosity");
				++argv;
				break;
			case 'x':
				// use a cross validation experiment
				et = etXVal;
				expArgs = p + 1;
				++argv;
				break;
			default:
				error("-%c flag is not supported", *p);
			}
		}


                if (theLearners.empty()) {
                        error("No learner specified");
                }

                // perform the experiment
                switch (et) {
                case etTrainTest:
                        if (theLearners.size() > 1)
                                error("Train/test only accepts a single learner");

                        trainTest(theLearners[0], *instanceStream, instanceFile,
                                        filters, testfilename, ttArgs);
                        break;
                case etXVal:
                        if (theLearners.size() > 1)
                                error("Cross validation only accepts a single learner");

                        xVal(theLearners[0], *instanceStream, filters, expArgs);
                        break;
                default:
                        error("No action specified");
                        break;
                }

                for (std::vector<learner*>::iterator it = theLearners.begin();
                                it != theLearners.end(); it++) {
                        delete *it;
                }
		
	} catch (std::bad_alloc) {
		error("Out of memory");
	} catch (alglib::ap_error err) {
		error(err.msg.c_str());
	}

	if (verbosity >= 1)
		summariseUsage();

	return 0;
}

