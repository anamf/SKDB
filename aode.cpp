/* Gigal: An open source system for classification learning from very large data
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
 ** Please report any bugs to Shenglei Chen <tristan_chen@126.com>
 */

#include "aode.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include "crosstab.h"


aode::aode(char* const *& argv, char* const * end) {
	name_ = "AODE";



	// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else {
			error("Aode does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}

	trainingIsFinished_ = false;
}

aode::~aode(void) {
}

void aode::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aode::reset(InstanceStream &is) {
	xxyDist_.reset(is);
	trainingIsFinished_ = false;


	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();

	//selectedAtt.resize(noCatAtts_, true);

	instanceStream_ = &is;

}

void aode::initialisePass() {

}

void aode::train(const instance &inst) {
	xxyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()
bool aode::trainingIsFinished() {
	return trainingIsFinished_;
}

// creates a comparator for two attributes based on their
//relative value with the class,such as mutual information, symmetrical uncertainty

class valCmpClass {
public:
	valCmpClass(std::vector<float> *s) {
		val = s;
	}

	bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
		return (*val)[a] > (*val)[b];
	}

private:
	std::vector<float> *val;
};


void aode::finalisePass() {

	trainingIsFinished_ = true;
}



void aode::classify(const instance &inst, std::vector<double> &classDist) {
    
	const InstanceCount totalCount = xxyDist_.xyCounts.count;

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	CatValue delta = 0;


	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

                const CatValue parentVal = inst.getCatVal(parent);

                for (CatValue y = 0; y < noClasses_; y++) {
                        xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
                                        parentVal, y);
                }

                if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
                        delta++;
                        active[parent] = true;

                        for (CatValue y = 0; y < noClasses_; y++) {
                                spodeProbs[parent][y] = mEstimate(xyCount[parent][y], 
                                                  totalCount, noClasses_
                                                * xxyDist_.getNoValues(parent))
                                                * scaleFactor;
                        }

                } else if (verbosity >= 5)
                        printf("%d\n", parent);
		
	}

	if (delta == 0) {
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {

                const CatValue x1Val = inst.getCatVal(x1);
                const unsigned int noX1Vals = xxyDist_.getNoValues(x1);

                constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
                                noClasses_);

                for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

                    CatValue x2Val = inst.getCatVal(x2);
                    const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

                            InstanceCount x1x2Count = xySubDist.getCount(
                                            x2, x2Val, 0);
                            for (CatValue y = 1; y < noClasses_; y++) {
                                    x1x2Count += xySubDist.getCount(x2, x2Val, y);
                            }
                            const InstanceCount x2Count =
                                            xxyDist_.xyCounts.getCount(x2, x2Val);


                    for (CatValue y = 0; y < noClasses_; y++) {
                            const InstanceCount x1x2yCount = xySubDist.getCount(
                                            x2, x2Val, y);

                            spodeProbs[x1][y] *= mEstimate(x1x2yCount,
                                            xyCount[x1][y], noX2Vals);

                            spodeProbs[x2][y] *= mEstimate(x1x2yCount,
                                            xyCount[x2][y], noX1Vals);


                    }
                        
                }
	}

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {
		if (active[parent]) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[parent][y];
			}
		}
	}

	normalise(classDist);
}


void aode::nbClassify(const instance &inst, std::vector<double> &classDist,
		xyDist &xyDist_) {

	for (CatValue y = 0; y < noClasses_; y++) {
		double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
		// scale up by maximum possible factor to reduce risk of numeric underflow

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			p *= xyDist_.p(a, inst.getCatVal(a), y);
		}

		assert(p >= 0.0);
		classDist[y] = p;
	}
	normalise(classDist);
}
