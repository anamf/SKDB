/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "correlationMeasures.h"
#include "utils.h"
#include "nb.h"
#include "globals.h"

nb::nb(char*const*&, char*const*) : xyDist_(), trainingIsFinished_(false)
 {
	name_ = "Naive Bayes";
}


nb::~nb(void)
{
}

void  nb::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void nb::reset(InstanceStream &is) {
  xyDist_.reset(&is);
  trainingIsFinished_ = false;
  noCatAtts_=is.getNoCatAtts();
  noClasses_=is.getNoClasses();
  instanceStream_ = &is;
}


void nb::train(const instance &inst) {
  xyDist_.update(inst);
}


void nb::initialisePass() {
}


void nb::finalisePass() {    
    trainingIsFinished_ = true;
}


bool nb::trainingIsFinished() {
  return trainingIsFinished_;
}

void nb::classify(const instance &inst, std::vector<double> &classDist) {
	const unsigned int noClasses = xyDist_.getNoClasses();

	for (CatValue y = 0; y < noClasses; y++) {

		//double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
		// scale up by maximum possible factor to reduce risk of numeric underflow

		double p = xyDist_.p(y);

		if(verbosity>=4)
		{
			printf("%f,",p);
		}
		if (verbosity >= 4)
			printf("for classes: %u\n", y);
		for (CategoricalAttribute a = 0; a < xyDist_.getNoAtts(); a++) {

                        p *= xyDist_.p(a, inst.getCatVal(a), y);

                        if (verbosity >= 4)
                                printf("**%f**,", xyDist_.p(a, inst.getCatVal(a), y));

		}
		if (verbosity >= 4)
		{
			printf("\n");
			printf("y:%u, %f\n",y,p);
		}

		assert(p >= 0.0);
		classDist[y] = p;
	}

	normalise(classDist);
	if(verbosity>=4)
	{
		printf("\noutput the class distribution:\n");
		print(classDist);
		printf("\n");
	}
}



