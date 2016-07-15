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
#include "tan.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

TAN::TAN() :
		trainingIsFinished_(false) {
}

TAN::TAN(char* const *&, char* const *) :
	xxyDist_(), trainingIsFinished_(false) {
	name_ = "TAN";
}

TAN::~TAN(void) {}

void TAN::reset(InstanceStream &is) {
	instanceStream_ = &is;
	const unsigned int noCatAtts = is.getNoCatAtts();
	noCatAtts_ = noCatAtts;
	noClasses_ = is.getNoClasses();

	trainingIsFinished_ = false;

        parents_.resize(noCatAtts);
	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		parents_[a] = NOPARENT;
	}

	xxyDist_.reset(is);
}

void TAN::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void TAN::initialisePass() {
	assert(trainingIsFinished_ == false);
}

void TAN::train(const instance &inst) {
	xxyDist_.update(inst);
}

void TAN::classify(const instance &inst, std::vector<double> &classDist) {

	for (CatValue y = 0; y < noClasses_; y++) {
		classDist[y] = xxyDist_.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
	}

	for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
		const CategoricalAttribute parent = parents_[x1];

		if (parent == NOPARENT) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
			}
		} else {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
						inst.getCatVal(parent), y);
			}
		}
	}

	normalise(classDist);
}

void TAN::finalisePass() {
	assert(trainingIsFinished_ == false);

	crosstab<float> cmi = crosstab<float>(noCatAtts_);
	getCondMutualInf(xxyDist_, cmi);

	// find the maximum spanning tree

	CategoricalAttribute firstAtt = 0;

	parents_[firstAtt] = NOPARENT;

	float *maxWeight;
	CategoricalAttribute *bestSoFar;
	CategoricalAttribute topCandidate = firstAtt;
	std::set<CategoricalAttribute> available;

	safeAlloc(maxWeight, noCatAtts_);
	safeAlloc(bestSoFar, noCatAtts_);

	maxWeight[firstAtt] = -std::numeric_limits<float>::max();

	for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++) {
		maxWeight[a] = cmi[firstAtt][a];
		if (cmi[firstAtt][a] > maxWeight[topCandidate])
			topCandidate = a;
		bestSoFar[a] = firstAtt;
		available.insert(a);
	}

	while (!available.empty()) {
		const CategoricalAttribute current = topCandidate;
		parents_[current] = bestSoFar[current];
		available.erase(current);

		if (!available.empty()) {
			topCandidate = *available.begin();
			for (std::set<CategoricalAttribute>::const_iterator it =
					available.begin(); it != available.end(); it++) {
				if (maxWeight[*it] < cmi[current][*it]) {
					maxWeight[*it] = cmi[current][*it];
					bestSoFar[*it] = current;
				}

				if (maxWeight[*it] > maxWeight[topCandidate])
					topCandidate = *it;
			}
		}
	}

	delete[] bestSoFar;
	delete[] maxWeight;

	trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()
bool TAN::trainingIsFinished() {
	return trainingIsFinished_;
}
