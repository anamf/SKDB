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
#pragma once

#include "incrementalLearner.h"
#include "xxyDist.h"
#include <limits>
/**
<!-- globalinfo-start -->
 * Class for a Tree AugmeNted (TAN) Classifier .<br/>
 * <br/>
 * For more information on TAN classifiers, see:<br/>
 * <br/>
 * N. Friedman, D. Geiger and M. Goldszmidt: Bayesian network classifiers. In: 
 * Machine Learning, number 2-3, vol. 29, 131-163, 1997.
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * \@article{Friedman1997,
 *   author = {N. Friedman and D. Geiger and M. Goldszmidt},
 *   journal = {Machine Learning},
 *   number = {2-3},
 *   pages = {131-163},
 *   title = {Bayesian network classifiers},
 *   volume = {29},
 *   year = {1997}
 *}
 * </pre>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */
 
class TAN: public IncrementalLearner {
public:
	TAN();
	TAN(char* const *& argv, char* const * end);
	~TAN(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training
	void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
	void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
	void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
	void getCapabilities(capabilities &c);

	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
	unsigned int noCatAtts_;          ///< the number of categorical attributes.
	unsigned int noClasses_;                          ///< the number of classes

	InstanceStream* instanceStream_;
	std::vector<CategoricalAttribute> parents_;
	xxyDist xxyDist_;

	bool trainingIsFinished_; ///< true iff the learner is trained

	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; // cannot use std::numeric_limits<categoricalAttribute>::max() because some compilers will not allow it here
};
