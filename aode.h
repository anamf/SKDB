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

#pragma once

#include "incrementalLearner.h"
#include "xxyDist.h"
/**
<!-- globalinfo-start -->
 * Class for an Aggregating One-Dependence Estimators (AODE) classifier.<br/>
 * <br/>
 * For more information on AODE classifiers, see:<br/>
 * <br/>
 * Not So Naive Bayes: Aggregating One-Dependence Estimators. 
 * In: Machine Learning, number 1, vol. 58, 5-24, 2005.
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * \@article{Webb2005,
 *  author = {G. Webb and J. Boughton and Z. Wang},
 *  journal = {Machine Learning},
 *  number = {1},
 *  pages = {5-24},
 *  title = {Not So Naive Bayes: Aggregating One-Dependence Estimators},
 *  volume = {58},
 *  year = {2005}
 *}
 * </pre>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Shenglei Chen (tristan_chen@126.com)
 */

class aode: public IncrementalLearner {
public:
	/**
	 * @param argv Options for the aode classifier
	 * @param argc Number of options for aode
	 * @param m    Metadata information
	 */
	aode(char* const *& argv, char* const * end);

	virtual ~aode(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training

	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

	/**
	 * Inisialises the pass indicated by the parametre.
	 *
	 * @param pass  Current pass.
	 */
	void initialisePass();
	/**
	 * Train an aode with instance inst.
	 *
	 * @param inst Training instance
	 */
	void train(const instance &inst);

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param inst The instance to be classified
	 * @param classDist Predicted class probability distribution
	 */
	void classify(const instance &inst, std::vector<double> &classDist);
	/**
	 * Calculates the weight for waode
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:
	/**
	 * Naive Bayes classifier to which aode will deteriorate when there are no eligible parent attribute (also as SPODE)
	 *
	 *@param inst The instance to be classified
	 *@param classDist Predicted class probability distribution
	 *@param dist  class object pointer of xyDist describing the distribution of attribute and class
	 */
	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &xyDist_);

	InstanceStream* instanceStream_;

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
	bool trainingIsFinished_; ///< true iff the learner is trained
	xxyDist xxyDist_; ///< the xxy distribution that aode learns from the instance stream and uses for classification
};

